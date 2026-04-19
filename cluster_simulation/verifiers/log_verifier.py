import numpy as np
import pandas as pd

from uuid import UUID


class LogVerifier:

    def __init__(self, job_log: pd.DataFrame, task_log: pd.DataFrame, batch_log: pd.DataFrame):
        self.job_log = job_log
        self.task_log = task_log
        self.batch_log = batch_log

    def run(self):
        self.verify_idle_workers()

    def verify_idle_workers(self):
        model_to_instances: dict[int, UUID] = {}
        # for mid in set(self.batch_log["model_id"]):
        #     instances = set(self.batch_log[self.batch_log["model_id"]==mid]["instance_id"])
        #     model_to_instances[mid] = instances

        # for i, row in self.task_log.iterrows():
        #     arrival_time = row["arrival_at_scheduler_timestamp"]
        #     dispatch_time = row["last_dep_dispatch_timestamp"]

        #     if arrival_time == dispatch_time:
        #         continue

        #     for instance_id in model_to_instances[row["model_id"]]:
        #         batch_at_arrival = self.batch_log[(self.batch_log["instance_id"]==instance_id) & \
        #                                           (self.batch_log["execution_start_timestamp"] <= arrival_time) & \
        #                                           (self.batch_log["execution_end_timestamp"] >= arrival_time)]
        #         assert(len(batch_at_arrival) == 1)

        #         batches_before_dispatch = self.batch_log[\
        #             (self.batch_log["instance_id"]==instance_id) & \
        #             (self.batch_log["execution_start_timestamp"] > arrival_time) & \
        #             (self.batch_log["execution_end_timestamp"] <= dispatch_time)]
                

        #         for _, batch_row in batches_before_dispatch.iterrows():

                

        #         instance_batch = self.batch_log[(self.batch_log["instance_id"]==instance_id) & \
        #                                         (self.batch_log["execution_start_timestamp"] <= arrival_time) & \
        #                                         (self.batch_log["execution_end_timestamp"] >= dispatch_time)]
                
        #         idle_time = 0
        #         prev_end = 0
        #         for _, batch_row in instance_batch.iterrows():
        #             if prev_end > 0:
        #                 idle_time += batch_row["execution_start_timestamp"] - prev_end
        #             prev_end = batch_row["execution_end_timestamp"]
                
        #         print(instance_id)
        #         print(row["job_id"], row["task_id"], arrival_time, dispatch_time)

        #         assert(len(instance_batch) > 0)