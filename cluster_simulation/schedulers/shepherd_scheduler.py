import numpy as np
import pandas as pd

import core.configs.gen_config as gcfg

from queue import PriorityQueue

from core.job import Job
from core.task import Task
from core.model import Model
from core.data_models.workflow import Workflow
from core.allocation import ModelAllocation

from workers.worker import Worker
from workers.gpu_state import ModelState

from schedulers.scheduler import Scheduler
from schedulers.algo.nexus_algo import NexusSLOSplitter

from queue_management.queued_task import QueuedTask
from queue_management.batching import TaskBatcher

from events.event_manager import EventManager
from events.event import *
from events.event_types import *


class ShepherdScheduler(Scheduler):

    def __init__(self, em: EventManager, allocation: ModelAllocation, workers: dict[UUID, Worker], 
                 workflows: list[Workflow], scheduler_worker_id: UUID):
        super().__init__(em, allocation)

        self.workers = workers
        self.workflows = workflows
        self.scheduler_worker_id = scheduler_worker_id

        # (worker ID, instance ID) -> list[(job ID, task ID)] to record scheduling decisions
        self.scheduled_batch_to_instance: dict[tuple[UUID, UUID], list[tuple[int, int]]] = {}

        self.arrived_jobs: dict[int, Job] = {} # job ID -> Job instance
        self.arrived_tasks: pd.DataFrame = pd.DataFrame(columns=["time", "model_id"])
        self.dropped_jobs: set[int] = set()

        # model ID -> model queue
        self.queues: dict[int, PriorityQueue] = {}

        # (job ID, task ID) -> worker ID on which task result/output is stored
        self.output_locs: dict[tuple[int, int], UUID] = {}

        # for round robin dispatch: model ID -> (worker ID, instance ID)
        self.last_sent_tasks_to: dict[int, tuple[UUID, UUID]] = {}

        # Nexus task-level SLOs: workflow ID -> task ID -> (task level SLO, max batch size)
        self.workflow_task_slos: dict[int, dict[int, tuple[float, int]]] = {}


    def _set_nexus_task_level_slo(self, time: float, tasks: list[Task]):
        if gcfg.SLO_TYPE != "NEXUS": # not using nexus SLO splitter
            return
        
        if time < 1000: # not enough time to judge arrival rate
            return

        for task in tasks:
            if task.job.job_type_id not in self.workflow_task_slos:
                per_model_ar = {}
                for mid in set(self.arrived_tasks["model_id"]):
                    per_model_ar[mid] = self.arrived_tasks[(self.arrived_tasks["model_id"]==mid) & \
                                                           (self.arrived_tasks["time"] >= (time - 1000))].count().iloc[0] / \
                                        self.allocation.count(mid)

                slo_split = NexusSLOSplitter.generate_task_slos(
                    time, per_model_ar, self.workflows[task.job.job_type_id], task.job.slo)
                
                if not slo_split:
                    continue

                self.workflow_task_slos[task.job.job_type_id] = slo_split

            task.slo = self.workflow_task_slos[task.job.job_type_id][task.task_id][0]
            task.max_batch_size = self.workflow_task_slos[task.job.job_type_id][task.task_id][1]                


    def check_dropped_tasks(self, time: float):
        """NOTE: For CENTRAL, drops tasks immediately (since requesting scheduling 
        & executing scheduling are not disaggregated); for DECENTRAL, drops tasks
        only once JOBS_DROPPED event is received
        """

        if gcfg.DROP_POLICY == "NONE":
            return
        
        elif gcfg.DROP_POLICY == "LAZY":
            dropped: dict[int, Task] = {}
            for q in self.queues.values():
                filtered = []
                while q.qsize() > 0:
                    qt = q.get()
                    if qt.task.job.id in self.dropped_jobs:
                        continue
                    elif time >= qt.task.get_task_deadline():
                        if qt.task.job.id not in dropped or \
                            qt.task.get_task_deadline() < dropped[qt.task.job.id].get_task_deadline():
                            dropped[qt.task.job.id] = qt.task
                    else:
                        filtered.append(qt)

                for qt in filtered:
                    q.put(qt)

            if dropped:
                job_task_ids = [(jid, t.task_id) for jid, t in dropped.items()]
                for jid in dropped.keys():
                    self.dropped_jobs.add(jid)

                self.em.add_event(Event(time,
                                        EVENT_TYPES[EventIds.JOBS_DROPPED],
                                        kwargs={"job_task_ids": job_task_ids}), self.emitter_id) 


    def on_job_arrival(self, time: float, job: Job):
        if super().on_job_arrival(time, job):
            return

        self.arrived_jobs[job.id] = job
        return self.on_tasks_arrival(time, 
                                     [t for t in job.tasks if len(t.required_task_ids) == 0])
    

    def on_tasks_arrival(self, time: float, tasks: list[Task]):
        self.check_dropped_tasks(time)
        
        # update task arrival data
        for task in tasks:
            self.arrived_tasks.loc[len(self.arrived_tasks)] = (time, task.model_data.id)
            task.arrival_time = time

        self._set_nexus_task_level_slo(time, tasks)

        model_ids_to_check = set()
        for task in tasks:
            model_ids_to_check.add(task.model_data.id)
            if task.model_data.id not in self.queues:
                self.queues[task.model_data.id] = PriorityQueue()
            self.queues[task.model_data.id].put(QueuedTask(task))
        
        all_instance_states = [(w, s) for w in self.workers.values() for s in w.GPU_state.state_at(time)]
        for model_id in model_ids_to_check:
            # sort instances by availability, then create time, then ID
            relevant_instances: list[tuple[Worker, ModelState]] = sorted(
                [(w, s) for (w, s) in all_instance_states if s.model.data.id == model_id], 
                key=lambda k: (0 if (k[0].id, k[1].model.id) not in self.scheduled_batch_to_instance or \
                                    self.scheduled_batch_to_instance[(k[0].id, k[1].model.id)] == None else 1,
                               k[1].model.created_at,
                               k[1].model.id))
            
            for (worker, instance_state) in relevant_instances:
                self._schedule_instance_if_idle(time, worker.id, instance_state.model.id,
                                                not gcfg.ENABLE_NETWORKING_DELAYS)
    

    def on_jobs_dropped(self, time: float, job_task_ids: list[tuple[int, int]]):
        pass
    

    def on_batch_start(self, time, batch, worker_id, instance_id):
        pass


    def on_batch_finish(self, time: float, batch: Batch, worker_id: UUID, instance_id: UUID):
        # update scheduling state
        self.scheduled_batch_to_instance[(worker_id, instance_id)] = None
        for task in batch.tasks:
            assert((task.job.id, task.task_id) not in self.output_locs)
            self.output_locs[(task.job.id, task.task_id)] = worker_id

        # assign new batch if queue is not empty
        self._schedule_instance_if_idle(time, worker_id, instance_id, not gcfg.ENABLE_NETWORKING_DELAYS)
    

    def _schedule_instance_if_idle(self, time: float, worker_id: UUID, instance_id: UUID, ignore_transfer_time: bool):
        self.check_dropped_tasks(time)
        
        worker = self.workers[worker_id]
        instance_state = worker.GPU_state.get_instance_state(instance_id, time)

        # skip if queue is empty
        if self.queues[instance_state.model.data.id].qsize() == 0:
            return
    
        # if instance is idle, send a new batch
        queued_batch = None
        if (worker.id, instance_state.model.id) not in self.scheduled_batch_to_instance or \
            self.scheduled_batch_to_instance[(worker.id, instance_state.model.id)] == None:
            
            queued_batch = TaskBatcher.get_batch(
                time, worker.total_memory_gb, self.queues[instance_state.model.data.id], True,
                self.queues[instance_state.model.data.id].queue[0].task.max_batch_size)
            
            # skip if cannot form batch
            if not queued_batch: return
            
            # assign batch to worker
            self.em.add_event(
                Event(time,
                    EVENT_TYPES[EventIds.TASKS_ASSIGNED_TO_WORKER],
                    kwargs={"tasks": queued_batch.tasks,
                            "worker_id": worker.id,
                            "force_instance_id": instance_state.model.id}),
                self.emitter_id)
            
        elif gcfg.ENABLE_PREEMPTION:
            curr_batch = self.scheduled_batch_to_instance[(worker.id, instance_state.model.id)]
            queued_batch = TaskBatcher.get_batch(
                time, worker.total_memory_gb, self.queues[instance_state.model.data.id], False,
                instance_state.model.data.max_batch_size)
            
            # skip if cannot form batch
            if not queued_batch: return
            
            # don't allow preemption until batch actually begins execution to avoid
            # duplicate preemptions
            if not instance_state.reserved_batch or \
                [(t.job.id, t.task_id) for t in instance_state.reserved_batch.tasks] != curr_batch:
                return
            
            if queued_batch.size() > gcfg.FLEX_LAMBDA * len(curr_batch):
                queued_batch = TaskBatcher.get_batch(
                    time, worker.total_memory_gb, self.queues[instance_state.model.data.id], True,
                    instance_state.model.data.max_batch_size)
                
                self.em.add_event(
                    Event(time,
                        EVENT_TYPES[EventIds.BATCH_PREEMPTION_AT_WORKER],
                        kwargs={"replacement_batch": queued_batch.tasks,
                                "preempted_tasks": curr_batch,
                                "model_instance_id": instance_state.model.id,
                                "worker_id": worker.id}),
                    self.emitter_id)
                
            else:
                return
        
        else:
            return

        self.scheduled_batch_to_instance[(worker.id, instance_state.model.id)] = [
            (t.job.id, t.task_id) for t in queued_batch.tasks]
        
        # tell workers to send required outputs to chosen worker
        inputs_from_scheduler: list[Task] = []
        outputs_from_workers: dict[UUID, list[tuple[int, int]]] = {}
        for task in queued_batch.tasks:
            if len(task.required_task_ids) == 0:
                inputs_from_scheduler.append(task)

            else:
                for rt in task.required_task_ids:
                    worker_id = self.output_locs[(task.job.id, rt)]
                    if worker_id not in outputs_from_workers:
                        outputs_from_workers[worker_id] = []
                    outputs_from_workers[worker_id].append((task.job.id, rt))

        if inputs_from_scheduler:
            self.em.add_event(
                Event(time, 
                        EVENT_TYPES[EventIds.TASKS_INPUTS_SENT_TO_WORKER],
                        kwargs={"tasks": queued_batch.tasks,
                                "from_worker_id": self.scheduler_worker_id,
                                "to_worker_id": worker.id,
                                "force_instance_id": instance_id,
                                "ignore_transfer_time": ignore_transfer_time}),
                self.emitter_id)
        
        if outputs_from_workers:
            for (from_worker_id, job_task_ids) in outputs_from_workers.items():
                self.em.add_event(
                    Event(time, 
                        EVENT_TYPES[EventIds.TASKS_OUTPUTS_ASSIGNED_TO_WORKER],
                        kwargs={"job_task_ids": job_task_ids,
                                "from_worker_id": from_worker_id,
                                "to_worker_id": worker.id}),
                    self.emitter_id)
        
                    
    def get_qlen(self, model_id: int):
        return self.queues[model_id].qsize() if model_id in self.queues else 0