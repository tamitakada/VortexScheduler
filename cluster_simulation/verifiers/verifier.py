from events.event_manager import EventManager
from events.event import *
from events.event_types import *

from workers.worker import Worker

from core.data_models.workflow import Workflow
from core.batch import Batch

from uuid import UUID

import core.configs.gen_config as gcfg
import core.configs.workflow_config as wcfg
import core.configs.model_config as mcfg

import numpy as np

from math import erf, sqrt
from scipy.stats import norm




class Verifier(EventListener):

    def __init__(self, em: EventManager, workers: dict[UUID, Worker]):
        super().__init__(Agent.VERIFIER)

        self.workers = workers
        self.em = em

        self.em.register_listener(self, {
            EVENT_TYPES[EventIds.JOB_SENT_TO_SCHEDULER],
            EVENT_TYPES[EventIds.JOB_ARRIVAL_AT_SCHEDULER],
            EVENT_TYPES[EventIds.TASKS_ARRIVAL_AT_SCHEDULER],
            EVENT_TYPES[EventIds.TASKS_SCHEDULED_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_SENT_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_SENT_TO_SCHEDULER],
            EVENT_TYPES[EventIds.TASKS_ARRIVAL_AT_WORKER],

            EVENT_TYPES[EventIds.JOBS_DROPPED],

            EVENT_TYPES[EventIds.BATCH_STARTED_AT_WORKER],
            EVENT_TYPES[EventIds.BATCH_FINISHED_AT_WORKER],
            EVENT_TYPES[EventIds.RESPONSE_SENT_TO_CLIENT],
            EVENT_TYPES[EventIds.RESPONSE_RECEIVED_AT_CLIENT]
        })

        # (job ID, task ID) -> {arrival at sched, arrival at worker, exec start, exec end}
        self.task_log: dict[tuple[int, int], dict[str, int]] = {}
        self.sampled_anomalies = 0
        self.total_samples = 0

    def on_event(self, event: Event):
        workflow_ids = set([k for ccfg in gcfg.CLIENT_CONFIGS for k in ccfg.keys()])
        workflow_cfgs = {wwcfg["JOB_TYPE"]: wwcfg for wwcfg in wcfg.WORKFLOW_LIST 
                         if wwcfg["JOB_TYPE"] in workflow_ids}

        if event.type.id == EventIds.BATCH_STARTED_AT_WORKER:
            batch: Batch = event.kwargs["batch"]

            assert(batch.size() <= mcfg.MODELS[batch.model_data.id]["MAX_BATCH_SIZE"])
            assert(all(workflow_cfgs[t.job.job_type_id]["TASKS"][t.task_id]["MODEL_ID"] == batch.model_data.id
                       for t in batch.tasks))
            
            for t in batch.tasks:
                for prev_id in workflow_cfgs[t.job.job_type_id]["TASKS"][t.task_id]["PREV_TASK_INDEX"]:
                    assert((t.job.id, prev_id) in self.task_log)
                    assert(self.task_log[(t.job.id, prev_id)]["exec_end_time"] <= event.time and \
                           self.task_log[(t.job.id, prev_id)]["exec_end_time"] >= 0)
                
                self.task_log[(t.job.id, t.task_id)] = {
                    "exec_start_time": event.time,
                    "exec_end_time": -1
                }

        elif event.type.id == EventIds.BATCH_FINISHED_AT_WORKER:
            batch: Batch = event.kwargs["batch"]

            for t in batch.tasks:
                assert((t.job.id, t.task_id) in self.task_log)
                assert(self.task_log[(t.job.id, t.task_id)]["exec_start_time"] < event.time)

                worker_size = self.workers[event.kwargs["worker_id"]].total_memory_gb
                expected_batch_exec_time = mcfg.MODELS[batch.model_data.id]["MIG_BATCH_EXEC_TIMES"][worker_size][batch.size()]
                actual_batch_exec_time = event.time - self.task_log[(t.job.id, t.task_id)]["exec_start_time"]
                expected_transfer_time = sum(workflow_cfgs[t1.job.job_type_id]["TASKS"][t1.task_id]["INPUT_SIZE"] 
                                             for t1 in batch.tasks) / 64000

                if not self.is_sampled_correctly(actual_batch_exec_time,
                                                 expected_batch_exec_time + expected_transfer_time,
                                                 mcfg.MODELS[batch.model_data.id]["EXEC_TIME_CVS"][worker_size]):
                    self.sampled_anomalies += 1
                self.total_samples += 1

                self.task_log[(t.job.id, t.task_id)]["exec_end_time"] = event.time

        elif event.type.id == EventIds.RESPONSE_SENT_TO_CLIENT:
            job: Job = event.kwargs["job"]

            assert(all(self.task_log[(job.id, task_id)]["exec_end_time"] >= 0 and \
                       self.task_log[(job.id, task_id)]["exec_end_time"] <= event.time 
                       for task_id in range(len(workflow_cfgs[job.job_type_id]["TASKS"]))))
            
    def is_sampled_correctly(self, x, mu, cv):
        """Calculate two sided p-value to test probability of being sampled
        from a given Normal distribution.
        """
        
        sigma = cv * mu
        z = (x - mu) / sigma
        p = 2 * norm.sf(abs(z))  # two-sided
        print(x, mu, cv, p)
        return p >= 0.05