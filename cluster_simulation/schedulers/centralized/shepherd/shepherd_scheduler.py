import core.configs.gen_config as gcfg
from core.task import Task
from core.batch import Batch
from core.events.base import *
from core.events.centralized_scheduler_events import *
from core.events.worker_events import *
from core.network import *
from core.configs.workflow_config import *

from workers.worker import Worker

from schedulers.centralized.scheduler import Scheduler
from schedulers.algo.batching_policies import get_batch


class ShepherdScheduler(Scheduler):
    """
        Scheduler class for a Flex scheduler.
    """

    def __init__(self, simulation, herd_assignment=None):
        super().__init__(simulation, herd_assignment)

        self.model_queues = {}
        self.worker_instance_to_batch = {} # (worker_id, instance_id) -> batch
        self.last_worker_idx = 0 # for round robin tracking

    def update_herd_assignment(self, herd_assignment):
        super().update_herd_assignment(herd_assignment)

        # currently executing batch ids
        self.worker_states = {}
        for group in self.herd_assignment.worker_groups:
            for worker in group:
                self.worker_states[worker.worker_id] = None

    def drop_tasks(self, current_time: float):
        for model_id in self.model_queues.keys():
            for i in range(len(self.model_queues[model_id])-1, -1, -1):
                task = self.model_queues[model_id][i]

                # skip dropped tasks
                if (self.simulation.task_drop_log["job_id"]==task.job.id).any():
                    self.model_queues[model_id].pop(i)
                    continue

                if (gcfg.DROP_POLICY == "LATEST_POSSIBLE" and current_time >= task.get_task_deadline()) or \
                   (gcfg.DROP_POLICY == "OPTIMAL" and (current_time + task.job.get_min_remaining_processing_time()) >= task.get_task_deadline()):
                    
                    self.simulation.task_drop_log.loc[len(self.simulation.task_drop_log)] = {
                        "client_id": task.job.client_id,
                        "job_id": task.job.id, 
                        "workflow_id": task.job.job_type_id, 
                        "task_id": task.task_id,
                        "drop_time": current_time, 
                        "create_time": task.log.job_creation_timestamp,
                        "arrival_time": task.log.task_placed_on_worker_queue_timestamp,
                        "slo": task.slo if gcfg.SLO_GRANULARITY == "TASK" else task.job.slo, 
                        "deadline": task.get_task_deadline()
                    }
        
        for model_id in self.model_queues.keys():
            for i in range(len(self.model_queues[model_id])-1, -1, -1):
                task = self.model_queues[model_id][i]

                # skip dropped tasks
                if (self.simulation.task_drop_log["job_id"]==task.job.id).any():
                    self.model_queues[model_id].pop(i)
                    continue

    def schedule_job_on_arrival(self, job, current_time):
        return self.schedule_tasks_on_arrival(
            [t for t in job.tasks if len(t.required_task_ids) == 0], current_time)

    def schedule_tasks_on_queue(self, current_time):
        return self.schedule_tasks_on_arrival([], current_time)
    
    def schedule_tasks_on_arrival(self, tasks, current_time):
        super().schedule_tasks_on_arrival(tasks, current_time)

        for task in tasks:
            # skip any dropped tasks
            if (self.simulation.task_drop_log["job_id"] == task.job.id).any():
                continue

            if task.model_data.id not in self.model_queues:
                self.model_queues[task.model_data.id] = []
            
            self.model_queues[task.model_data.id].append(task)
        
        self.drop_tasks(current_time)

        events = []
        for worker_id in np.roll(self.simulation.worker_ids_by_creation, -self.last_worker_idx):
            worker = self.simulation.workers[worker_id]

            # sort idle instances first
            worker_state = sorted(worker.GPU_state.state_at(current_time), 
                                  key=lambda s: 1 if ((worker.id, s.model.id) in self.worker_instance_to_batch and \
                                                  self.worker_instance_to_batch[(worker.id, s.model.id)]) 
                                              else 0)
            for state in worker_state:
                if state.model.data.id not in self.model_queues or \
                   len(self.model_queues[state.model.data.id]) == 0:
                    
                    continue

                queued_batch = get_batch(current_time,
                                         worker.total_memory, 
                                         self.model_queues[state.model.data.id])
                
                if (worker.id, state.model.id) not in self.worker_instance_to_batch:
                    self.worker_instance_to_batch[(worker.id, state.model.id)] = None
                
                curr_batch = self.worker_instance_to_batch[(worker.id, state.model.id)]

                if not curr_batch:
                    self.worker_instance_to_batch[(worker.id, state.model.id)] = queued_batch
                    for task in queued_batch.tasks:
                        self.model_queues[state.model.data.id].remove(task)

                    events.append(EventOrders(
                        current_time + CPU_to_CPU_delay(sum(t.input_size for t in queued_batch.tasks)),
                        BatchArrivalAtWorker(self.simulation, worker, queued_batch, state.model.id)))
                
                elif queued_batch.size() >= gcfg.FLEX_LAMBDA * curr_batch.size():
                    self.worker_instance_to_batch[(worker.id, state.model.id)] = queued_batch
                    for task in queued_batch.tasks:
                        self.model_queues[state.model.data.id].remove(task)
                    
                    events.append(EventOrders(
                        current_time + CPU_to_CPU_delay(sum(t.input_size for t in queued_batch.tasks)), 
                        BatchPreemptionScheduledAtWorker(self.simulation, worker, state.model.id, queued_batch, curr_batch.id)))

        self.last_worker_idx = (self.last_worker_idx + 1) % len(self.simulation.workers)

        return events
    
    def schedule_on_batch_completion(self, worker: Worker, instance_id, completed_batch: Batch, current_time: float):
        # if alr. assigned to a new batch do nothing
        if self.worker_instance_to_batch[(worker.id, instance_id)].id != completed_batch.id:
            return []
        
        self.drop_tasks(current_time)
        
        self.worker_instance_to_batch[(worker.id, instance_id)] = None

        if len(self.model_queues[completed_batch.model_data.id]) == 0:
            return []

        candidate_batch = get_batch(current_time,
                                    worker.total_memory, 
                                    self.model_queues[completed_batch.model_data.id])
        
        if candidate_batch.size() > 0:
            self.worker_instance_to_batch[(worker.id, instance_id)] = candidate_batch
            for task in candidate_batch.tasks:
                self.model_queues[completed_batch.model_data.id].remove(task)
                
            return [EventOrders(
                current_time + CPU_to_CPU_delay(sum(t.input_size for t in candidate_batch.tasks)), 
                BatchArrivalAtWorker(self.simulation, worker, candidate_batch, instance_id))]
        return []