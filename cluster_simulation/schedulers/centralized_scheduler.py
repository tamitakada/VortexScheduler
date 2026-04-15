import numpy as np

import core.configs.gen_config as gcfg

from queue import PriorityQueue

from core.job import Job
from core.task import Task
from core.model import Model
from workers.worker import Worker
from workers.gpu_state import ModelState

from schedulers.scheduler import Scheduler
from queue_management.queued_task import QueuedTask
from queue_management.batching import TaskBatcher

from events.event_manager import EventManager
from events.event import *
from events.event_types import *


class CentralizedScheduler(Scheduler):

    def __init__(self, em: EventManager, workers: dict[UUID, Worker], scheduler_worker_id: UUID):
        super().__init__(em)

        self.workers = workers
        self.scheduler_worker_id = scheduler_worker_id

        # (worker ID, instance ID) -> list[(job ID, task ID)] to record scheduling decisions
        self.scheduled_batch_to_instance: dict[tuple[UUID, UUID], list[tuple[int, int]]] = {}

        # model ID -> model queue
        self.queues: dict[int, PriorityQueue] = {}

        # for round robin dispatch: model ID -> (worker ID, instance ID)
        self.last_sent_tasks_to: dict[int, tuple[UUID, UUID]] = {}


    def on_job_arrival(self, time: float, job: Job):
        return self.on_tasks_arrival(time, [t for t in job.tasks if len(t.required_task_ids) == 0])
    

    def on_tasks_arrival(self, time: float, tasks: list[Task]):
        print("SAW ", tasks)

        for task in tasks:
            if task.model_data.id not in self.queues:
                self.queues[task.model_data.id] = PriorityQueue()
            self.queues[task.model_data.id].put(QueuedTask(task))

        if gcfg.DISPATCH_POLICY == "ROUND_ROBIN":
            self._on_tasks_arrival_round_robin(time, tasks)
        elif gcfg.DISPATCH_POLICY == "SHEPHERD_PERFECT":
            self._on_tasks_arrival_shepherd(time, tasks, True)
        elif gcfg.DISPATCH_POLICY == "SHEPHERD_HALF_CAP":
            pass

        else:
            raise ValueError(f"Unknown dispatch policy {gcfg.DISPATCH_POLICY}")
        
    
    def _on_tasks_arrival_round_robin(self, time: float, tasks: list[Task]):
        tasks_to_send: dict[UUID, list[Task]] = {}
        for task in tasks:
            relevant_instances = [(w.id, s.model.id) 
                                  for w in sorted(self.workers.values(), key=lambda w: w.create_time)
                                  for s in sorted(w.GPU_state.state_at(time), key=s.model.created_at)
                                  if s.model.data.id == task.model_data.id]
            
            last_idx = relevant_instances.index(self.last_sent_tasks_to[task.model_data.id])
            (worker_id, instance_id) = relevant_instances[(last_idx + 1) % len(relevant_instances)]
            
            if worker_id not in tasks_to_send: tasks_to_send[worker_id] = []
            tasks_to_send[worker_id].append(task)

            # update round robin ptr
            self.last_sent_tasks_to[task.model_data.id] = (worker_id, instance_id)
        
        for worker_id, worker_tasks in tasks_to_send.items():
            self.em.add_event(
                Event(time,
                      EVENT_TYPES[EventIds.TASKS_SENT_TO_WORKER],
                      kwargs={"to_worker_id": worker_id,
                              "from_worker_id": self.scheduler_worker_id,
                              "tasks": worker_tasks,
                              "ignore_transfer_time": False}), 
                self.emitter_id)
    

    def _on_tasks_arrival_shepherd(self, time: float, tasks: list[Task], ignore_transfer_time: bool=False):
        """On task arrival, for each model required by given tasks, for each instance of
        the model, if the instance is idle, then assign it a batch according to the configured
        batching policy. Otherwise, if all instances are busy, preempt according to
        configured preemption policy (if any).
        """

        all_instance_states = [(w, s) for w in self.workers.values() for s in w.GPU_state.state_at(time)]

        for model_id in set([t.model_data.id for t in tasks]):
            # sort instances by availability, then create time, then ID
            relevant_instances: list[tuple[Worker, ModelState]] = sorted(
                [(w, s) for (w, s) in all_instance_states if s.model.data.id == model_id], 
                key=lambda k: (0 if (k[0].id, k[1].model.id) not in self.scheduled_batch_to_instance or \
                                    self.scheduled_batch_to_instance[(k[0].id, k[1].model.id)] == None else 1,
                               k[1].model.created_at,
                               k[1].model.id))
            
            for (worker, instance_state) in relevant_instances:
                self._schedule_instance_if_idle(
                    time, worker, instance_state.model.id, ignore_transfer_time)

                # TODO: preemption
                # curr_batch = None
                # if (worker.id, state.model.id) in self.scheduled_batch_to_instance:
                #     curr_batch = self.scheduled_batch_to_instance.inv[(worker.id, state.model.id)]

                # if not curr_batch:

                # elif gcfg.ENABLE_PREEMPTION and queued_batch.size() >= gcfg.FLEX_LAMBDA * curr_batch.size():
                #     self.scheduled_batch_to_instance[(worker.id, state.model.id)] = queued_batch
                #     for task in queued_batch.tasks:
                #         self.model_queues[state.model.data.id].remove(task)
                    
                #     events.append(EventOrders(
                #         current_time + CPU_to_CPU_delay(sum(t.input_size for t in queued_batch.tasks)), 
                #         BatchPreemptionScheduledAtWorker(self.simulation, worker, state.model.id, queued_batch, curr_batch.id)))
    

    def on_jobs_dropped(self, time: float, job_ids: list[int]):
        pass
    

    def on_batch_start(self, time: float, batch: Batch, worker_id: UUID):
        pass


    def on_batch_finish(self, time: float, batch: Batch, worker_id: UUID, instance_id: UUID):
        self.scheduled_batch_to_instance[(worker_id, instance_id)] = None

        if gcfg.DISPATCH_POLICY == "SHEPHERD_PERFECT":
            self._schedule_instance_if_idle(time, self.workers[worker_id], instance_id, True)


    def _schedule_instance_if_idle(self, time: float, worker: Worker, instance_id: UUID, ignore_transfer_time: bool):
        instance_state = worker.GPU_state.get_instance_state(instance_id, time)
        
        # skip if queue is empty
        if self.queues[instance_state.model.data.id].qsize() == 0:
            return
    
        # if instance is idle, send a new batch
        if (worker.id, instance_state.model.id) not in self.scheduled_batch_to_instance or \
            self.scheduled_batch_to_instance[(worker.id, instance_state.model.id)] == None:
            
            queued_batch = TaskBatcher.get_batch(
                time, 
                worker.total_memory_gb, 
                self.queues[instance_state.model.data.id], 
                False)
        
            if not queued_batch: # skip if cannot form batch
                return

            self.scheduled_batch_to_instance[(worker.id, instance_state.model.id)] = [
                (t.job.id, t.task_id) for t in queued_batch.tasks]
            
            TaskBatcher.dequeue_batch(queued_batch, self.queues[instance_state.model.data.id])

            self.em.add_event(Event(time, 
                                    EVENT_TYPES[EventIds.TASKS_SENT_TO_WORKER],
                                    kwargs={"to_worker_id": worker.id, 
                                            "force_instance_id": instance_state.model.id,
                                            "from_worker_id": self.scheduler_worker_id,
                                            "tasks": queued_batch.tasks,
                                            "ignore_transfer_time": ignore_transfer_time}), 
                                self.emitter_id)