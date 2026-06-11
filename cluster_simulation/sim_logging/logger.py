from events.event_manager import EventManager
from events.event import *
from events.event_types import *

from core.data_models.workflow import Workflow

from workers.worker import Worker

import pandas as pd
import numpy as np


class Logger(EventListener):

    def __init__(self, em: EventManager, workers: dict[UUID, Worker], scheduler, workflows: list[Workflow], centralized):
        super().__init__(Agent.LOGGER)

        self.em = em
        self.workers = workers
        self.is_centralized = centralized
        self.scheduler = scheduler
        self.workflows = workflows

        self.em.register_listener(self, {
            EVENT_TYPES[EventIds.JOB_SENT_TO_SCHEDULER],
            EVENT_TYPES[EventIds.JOB_ARRIVAL_AT_SCHEDULER],
            EVENT_TYPES[EventIds.TASKS_ARRIVAL_AT_SCHEDULER],

            EVENT_TYPES[EventIds.TASKS_ASSIGNED_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_INPUTS_SENT_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_INPUTS_ARRIVAL_AT_WORKER],
            EVENT_TYPES[EventIds.TASKS_OUTPUTS_ASSIGNED_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_OUTPUTS_SENT_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_OUTPUTS_ARRIVAL_AT_WORKER],

            EVENT_TYPES[EventIds.JOBS_DROPPED],
            EVENT_TYPES[EventIds.BATCH_STARTED_AT_WORKER],
            EVENT_TYPES[EventIds.BATCH_FINISHED_AT_WORKER],
            EVENT_TYPES[EventIds.BATCH_PREEMPTION_AT_WORKER],
            EVENT_TYPES[EventIds.RESPONSE_SENT_TO_CLIENT],
            EVENT_TYPES[EventIds.RESPONSE_RECEIVED_AT_CLIENT]
        })

        self.task_log = pd.DataFrame(columns=["job_id", "task_id", "client_id", "workflow_id", "model_id", "executing_worker_id",
                                         "arrival_at_scheduler_timestamp", "last_dep_dispatch_timestamp", "arrival_at_worker_timestamp",
                                         "execution_start_timestamp", "execution_end_timestamp", "dropped_timestamp", "dropped_at_task_id",
                                         "curr_unfinished_jobs", "curr_idle_instances", "executing_worker_qlen_at_arrival"])
        self.worker_log = pd.DataFrame(columns=["worker_id", "instance_id", "model_id", "batch_id", "batched_job_task_ids", 
                                           "batch_size", "execution_start_timestamp", "execution_end_timestamp",
                                           "preempted_timestamp"])
        self.work_log = pd.DataFrame(columns=["time", "worker_id", "instance_id", "model_id",
                                              "total_incomplete_model_tasks", "worker_incomplete_tasks",
                                              "is_active"])
        
        self.unfinished_jobs: list[Job] = []

        model_ids = sorted(set(m.id for w in self.workflows.values() for m in w.get_models()))
        self.unfinished_tasks: dict[int, list[tuple[int, int]]] = {mid: [] for mid in model_ids}

        self.deps_to_task = {}

    def _get_curr_idle_instances(self, time: float):
        curr_idle_instances = 0
        for w in self.workers.values():
            for s in w.GPU_state.state_at(time):
                if not s.reserved_batch:
                    curr_idle_instances += 1
        return curr_idle_instances

    def on_event(self, event: Event):
        if event.type.id == EventIds.JOB_ARRIVAL_AT_SCHEDULER:
            job: Job = event.kwargs["job"]
            self.unfinished_jobs.append(job)
            for task in job.tasks:
                if len(task.required_task_ids) == 0:
                    self.task_log.loc[len(self.task_log)] = {
                        "job_id": job.id, "task_id": task.task_id, "client_id": job.client_id, 
                        "model_id": task.model_data.id, "workflow_id": job.job_type_id, "executing_worker_id": "N/A",
                        "arrival_at_scheduler_timestamp": event.time, "arrival_at_worker_timestamp": np.nan,
                        "last_dep_dispatch_timestamp": np.nan, "execution_start_timestamp": np.nan, 
                        "execution_end_timestamp": np.nan, "dropped_timestamp": np.nan, "dropped_at_task_id": np.nan,
                        "curr_unfinished_jobs": len(self.unfinished_jobs), 
                        "curr_idle_instances": self._get_curr_idle_instances(event.time),
                        "executing_worker_qlen_at_arrival": np.nan
                    }

                    self.unfinished_tasks[task.model_data.id].append((task.job.id, task.task_id))
            
        elif event.type.id == EventIds.TASKS_ARRIVAL_AT_SCHEDULER:
            tasks: list[Task] = event.kwargs["tasks"]
            for task in tasks:
                self.task_log.loc[len(self.task_log)] = {
                    "job_id": task.job.id, "task_id": task.task_id, "client_id": task.job.client_id, 
                    "model_id": task.model_data.id, "workflow_id": task.job.job_type_id, "executing_worker_id": "N/A",
                    "arrival_at_scheduler_timestamp": event.time, "last_dep_dispatch_timestamp": np.nan,
                    "arrival_at_worker_timestamp": np.nan, "execution_start_timestamp": np.nan, 
                    "execution_end_timestamp": np.nan, "dropped_timestamp": np.nan, "dropped_at_task_id": np.nan,
                    "curr_unfinished_jobs": len(self.unfinished_jobs), 
                    "curr_idle_instances": self._get_curr_idle_instances(event.time),
                    "executing_worker_qlen_at_arrival": np.nan
                }

                if self.is_centralized:
                    for worker in self.workers.values():
                        for state in worker.GPU_state.state_at(event.time):
                            self.work_log.loc[len(self.work_log)] = {
                                "time": event.time,
                                "worker_id": worker.id,
                                "model_id": state.model.data.id,
                                "instance_id": state.model.id,
                                "total_incomplete_model_tasks": len(
                                    self.unfinished_tasks[state.model.data.id]), 
                                "worker_incomplete_tasks": worker.get_remaining_work(
                                    event.time, state.model.data.id),
                                "is_active": state.reserved_batch != None 
                            }
                
                    self.unfinished_tasks[task.model_data.id].append((task.job.id, task.task_id))
        
        elif event.type.id == EventIds.TASKS_INPUTS_SENT_TO_WORKER:
            tasks: list[Task] = event.kwargs["tasks"]
            for task in tasks:
                self.task_log.loc[(self.task_log["job_id"]==task.job.id) & \
                                  (self.task_log["task_id"]==task.task_id), "last_dep_dispatch_timestamp"] = event.time
        
        elif event.type.id == EventIds.TASKS_INPUTS_ARRIVAL_AT_WORKER:
            tasks: list[Task] = event.kwargs["tasks"]
            for task in tasks:
                mask = (self.task_log["job_id"]==task.job.id) & (self.task_log["task_id"]==task.task_id)
                assert(not self.task_log.loc[mask].empty)

                self.task_log.loc[mask, "executing_worker_id"] = event.kwargs["to_worker_id"]
                self.task_log.loc[mask, "arrival_at_worker_timestamp"] = event.time
                self.task_log.loc[mask, "executing_worker_qlen_at_arrival"] = \
                    self.workers[event.kwargs["to_worker_id"]].get_qlen(task.model_data.id)
                
                if not self.is_centralized:
                    assert(self.workers[event.kwargs["to_worker_id"]].get_qlen(task.model_data.id) > 0)

        elif event.type.id == EventIds.TASKS_ASSIGNED_TO_WORKER:
            tasks: list[Task] = event.kwargs["tasks"]
            for task in tasks:
                mask = (self.task_log["job_id"]==task.job.id) & (self.task_log["task_id"]==task.task_id)
                if self.task_log.loc[mask].empty:
                    self.task_log.loc[len(self.task_log)] = {
                        "job_id": task.job.id, "task_id": task.task_id, "client_id": task.job.client_id, 
                        "model_id": task.model_data.id, "workflow_id": task.job.job_type_id, 
                        "executing_worker_id":  event.kwargs["worker_id"],
                        "arrival_at_scheduler_timestamp": np.nan, "last_dep_dispatch_timestamp": np.nan,
                        "arrival_at_worker_timestamp": event.time, "execution_start_timestamp": np.nan, 
                        "execution_end_timestamp": np.nan, "dropped_timestamp": np.nan, "dropped_at_task_id": np.nan,
                        "curr_unfinished_jobs": len(self.unfinished_jobs), 
                        "curr_idle_instances": self._get_curr_idle_instances(event.time),
                        "executing_worker_qlen_at_arrival": np.nan
                    }
                else:
                    self.task_log.loc[mask, "executing_worker_id"] = event.kwargs["worker_id"]

                for rt in task.required_task_ids:
                    if (task.job.id, rt) not in self.deps_to_task:
                        self.deps_to_task[(task.job.id, rt)] = []
                    self.deps_to_task[(task.job.id, rt)].append(task)
            
            if not self.is_centralized:
                for worker in self.workers.values():
                    for state in worker.GPU_state.state_at(event.time):
                        self.work_log.loc[len(self.work_log)] = {
                            "time": event.time,
                            "worker_id": worker.id,
                            "model_id": state.model.data.id,
                            "instance_id": state.model.id,
                            "total_incomplete_model_tasks": sum(
                                w.get_remaining_work(event.time, state.model.data.id)
                                for w in self.workers.values()),
                            "worker_incomplete_tasks": worker.get_remaining_work(
                                event.time, state.model.data.id),
                            "is_active": state.reserved_batch != None 
                        }

        elif event.type.id == EventIds.TASKS_OUTPUTS_SENT_TO_WORKER:
            tasks: list[Task] = event.kwargs["tasks"]
            for task in tasks:
                for succ in self.deps_to_task[(task.job.id, task.task_id)]:
                    mask = ((self.task_log["job_id"]==task.job.id) & 
                            (self.task_log["task_id"]==succ.task_id) & 
                            (self.task_log["executing_worker_id"]==event.kwargs["to_worker_id"]))

                    if self.task_log.loc[mask].empty or \
                        self.task_log.loc[mask].iloc[0]["executing_worker_id"] != event.kwargs["to_worker_id"]:
                        continue

                    self.task_log.loc[mask, "last_dep_dispatch_timestamp"] = event.time
        
        elif event.type.id == EventIds.TASKS_OUTPUTS_ARRIVAL_AT_WORKER:
            tasks: list[Task] = event.kwargs["tasks"]
            for task in tasks:
                for succ in self.deps_to_task[(task.job.id, task.task_id)]:
                    mask = ((self.task_log["job_id"]==task.job.id) & 
                            (self.task_log["task_id"]==succ.task_id) & 
                            (self.task_log["executing_worker_id"]==event.kwargs["to_worker_id"]))

                    if self.task_log.loc[mask].empty or \
                        self.task_log.loc[mask].iloc[0]["executing_worker_id"] != event.kwargs["to_worker_id"]:
                        
                        continue

                    self.task_log.loc[mask, "executing_worker_id"] = event.kwargs["to_worker_id"]
                    self.task_log.loc[mask, "arrival_at_worker_timestamp"] = event.time
                    self.task_log.loc[mask, "executing_worker_qlen_at_arrival"] = \
                        self.workers[event.kwargs["to_worker_id"]].get_qlen(succ.model_data.id)


        elif event.type.id == EventIds.BATCH_STARTED_AT_WORKER:
            batch: Batch = event.kwargs["batch"]
            for task in batch.tasks:
                self.task_log.loc[(self.task_log["job_id"]==task.job.id) & \
                                  (self.task_log["task_id"]==task.task_id), "execution_start_timestamp"] = event.time

            self.worker_log.loc[len(self.worker_log)] = {
                "worker_id": event.kwargs["worker_id"], 
                "instance_id": event.kwargs["model_instance_id"], 
                "model_id": batch.model_data.id, 
                "batch_id": batch.id, 
                "batched_job_task_ids": [(t.job.id, t.task_id) for t in batch.tasks], 
                "batch_size": batch.size(), 
                "execution_start_timestamp": event.time, 
                "execution_end_timestamp": np.nan,
                "preempted_timestamp": np.nan
            }

        elif event.type.id == EventIds.BATCH_FINISHED_AT_WORKER:
            batch: Batch = event.kwargs["batch"]
            for task in batch.tasks:
                if self.is_centralized:
                    self.unfinished_tasks[task.model_data.id].remove((task.job.id, task.task_id))
                self.task_log.loc[(self.task_log["job_id"]==task.job.id) & \
                                  (self.task_log["task_id"]==task.task_id), "execution_end_timestamp"] = event.time

            self.worker_log.loc[self.worker_log["batch_id"]==batch.id, "execution_end_timestamp"] = event.time

        elif event.type.id == EventIds.BATCH_PREEMPTION_AT_WORKER:
            tasks: list[Task] = event.kwargs["replacement_batch"]

            mask = (self.worker_log["instance_id"]==event.kwargs["model_instance_id"]) & \
                    (self.worker_log["execution_start_timestamp"] <= event.time) & \
                    (self.worker_log["execution_end_timestamp"]==np.nan)
            
            self.worker_log.loc[mask, "preempted_timestamp"] = event.time

            for task in tasks:
                for rt in task.required_task_ids:
                    if (task.job.id, rt) not in self.deps_to_task:
                        self.deps_to_task[(task.job.id, rt)] = []
                    self.deps_to_task[(task.job.id, rt)].append(task)
            
        elif event.type.id == EventIds.JOBS_DROPPED:
            for job_id, task_id in event.kwargs["job_task_ids"]:
                self.unfinished_jobs = [j for j in self.unfinished_jobs if j.id != job_id]
                self.task_log.loc[self.task_log["job_id"]==job_id, "dropped_timestamp"] = event.time
                self.task_log.loc[self.task_log["job_id"]==job_id, "dropped_at_task_id"] = task_id

        elif event.type.id == EventIds.RESPONSE_SENT_TO_CLIENT:
            if event.kwargs["job"] in self.unfinished_jobs:
                self.unfinished_jobs.remove(event.kwargs["job"])