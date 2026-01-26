from core.logging import *
from core.data_models.model_data import ModelData

import core.configs.gen_config as gcfg


class Task(object):
    def __init__(self, job, task_id: int, model_data: ModelData | None, 
                 input_size: float, result_size: float,
                 max_wait_time: float, max_emit_batch_size: int, 
                 slo: float | None=None):

        self.job = job
        self.task_id = task_id
        self.model_data = model_data
        self.input_size = input_size
        self.result_size = result_size
        self.max_wait_time = max_wait_time
        self.max_emit_batch_size = max_emit_batch_size
        self.slo = slo
        
        # list of Tasks (inputs) that this task requires ( list will be appended as the job generated)
        self.required_task_ids = []                        # list of task ids
        self.next_task_ids = []                            # list of task ids
        self.assigned_worker_id = None
        self.executing_worker_id = -1
        self.ADFG = {}                                  # ADFG assigned to the job that this task belongs to
        self.log = TaskLifeCycleTimestamp(
            self.job.id, self.task_id, is_initial_task=(len(self.required_task_ids) == 0))

    def get_task_deadline(self):
        if gcfg.SLO_GRANULARITY == "TASK":
            assert(self.slo != None)
            return self.log.task_placed_on_worker_queue_timestamp + self.slo * (1 + gcfg.SLO_SLACK)
        else:
            return self.log.job_creation_timestamp + self.job.slo * (1 + gcfg.SLO_SLACK)

    def __hash__(self):
        return hash((self.task_id, self.job.id))

    def __eq__(self, other):
        if (isinstance(other, Task)):
            return self.task_id == other.task_id and self.job.id == other.job.id
        return False

    def __ne__(self, other):
        return not (self.__eq__(other))

    def __str__(self):
        return "[JobID: {}, TaskID: {} placed on worker queue @ {}]".format(self.job.id, self.task_id, self.log.task_placed_on_worker_queue_timestamp)

    def __repr__(self):
        return self.__str__()

    def print_task_log(self):
        print(self.log.toString())
