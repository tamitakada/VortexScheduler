import sys
import os
import ast
import json
import importlib.util

import numpy as np
import pandas as pd

from core.network import *


class LogVerifier:

    def __init__(self, job_log: pd.DataFrame, task_log: pd.DataFrame, batch_log: pd.DataFrame,
                 worker_log: pd.DataFrame, work_log: pd.DataFrame, client_log: pd.DataFrame,
                 slo_log: dict[int, dict[int, float]], centralized: bool, gcfg, mcfg, wcfg):
        
        self.job_log = job_log
        self.task_log = task_log
        self.batch_log = batch_log
        self.worker_log = worker_log
        self.work_log = work_log
        self.client_log = client_log
        self.slo_log = slo_log

        self.is_centralized = centralized

        self.gcfg = gcfg
        self.mcfg = mcfg
        self.wcfg = wcfg


    def run(self):
        self.verify_arrival_rates()
        self.verify_allocation()
        self.verify_batch_sizes()
        self.verify_instance_activity()
        self.verify_dropped_jobs()
        # self.trace_task_arrivals()


    def verify_arrival_rates(self):
        for i, client_id in enumerate(self.client_log["client_id"].unique()):
            client_jobs = self.job_log[self.job_log["client_id"]==client_id]
            
            for workflow_id, cfg in self.gcfg.CLIENT_CONFIGS[i].items():
                workflow_jobs = client_jobs[client_jobs["workflow_id"]==workflow_id]
                start_idx = 0
                for j, send_rate in enumerate(cfg["SEND_RATES"]):
                    num_jobs = cfg["JOBS_PER_SEND_RATE"][j]
                    create_intvl = workflow_jobs.iloc[start_idx:start_idx+num_jobs]["create_time"].diff().mean()
                    assert(abs(create_intvl - 1000 / send_rate) < 2) # must be within 2ms
                    start_idx += num_jobs

        assert(self.client_log["client_id"].nunique() == len(self.gcfg.CLIENT_CONFIGS))
        
        print("[PASS] Arrival rate verification")


    def verify_allocation(self):
        if self.gcfg.ALLOCATION_STRATEGY != "CUSTOM":
            return
        
        # check worker exists for each specified config
        worker_cfgs = {}
        for _, row in self.worker_log.iterrows():
            if row["instance_loaded_timestamp"] > 0:
                continue # only looking at init state

            if row["worker_id"] not in worker_cfgs:
                worker_cfgs[row["worker_id"]] = (row["worker_mem_size_gb"], [])
            worker_cfgs[row["worker_id"]][1].append(row["model_id"])

        for cfg in self.gcfg.CUSTOM_ALLOCATION:
            found = False
            for k, (mem_size, model_ids) in worker_cfgs.items():
                if mem_size == cfg[0] and sorted(model_ids) == sorted(cfg[1]):
                    worker_cfgs.pop(k)
                    found = True
                    break
            
            assert(found) # didn't find required worker config

        assert(self.worker_log["worker_id"].nunique() == len(self.gcfg.CUSTOM_ALLOCATION))

        # check batch never executes on worker without instance
        for _, row in self.batch_log.iterrows():
            assert(((self.worker_log["worker_id"]==row["worker_id"]) &
                   (self.worker_log["instance_id"]==row["instance_id"]) &
                   (self.worker_log["model_id"]==row["model_id"]) &
                   (self.worker_log["instance_loaded_timestamp"] <= row["execution_start_timestamp"])).any())

        print("[PASS] Configured allocation and worker instance/batch execution correspondence verification")


    def verify_batch_sizes(self):
        for i, cfg in enumerate(self.mcfg.MODELS):
            model_log = self.batch_log[self.batch_log["model_id"]==i]
            assert((model_log["batch_size"] <= cfg["MAX_BATCH_SIZE"]).all())
            assert((model_log["batch_size"] >= 1).all())

        print("[PASS] Min/max batch size verification")


    def verify_dropped_jobs(self):
        """Verify the following:
        * Drops match drop policy config
        * Jobs are dropped after their logged deadlines
        * Job deadlines match config SLO and (if Nexus) logged task level SLO split
        * Jobs are not batched/executed after being dropped
        """

        if self.gcfg.DROP_POLICY == "NONE":
            assert(self.job_log[self.job_log["was_completed"]==False].empty)
        
        else:
            dropped_jobs = self.job_log[self.job_log["was_completed"]==False]

            if self.gcfg.SLO_TYPE == "NEXUS":
                # dropped tasks are dropped after task SLO + task arrival time
                dropped_tasks = self.task_log.replace("", np.nan).dropna()
                dropped_tasks = dropped_tasks[dropped_tasks["task_id"]==dropped_tasks["dropped_at_task_id"]]
                for w in set(dropped_tasks["workflow_id"]):
                    assert((dropped_tasks["dropped_time"] >= dropped_tasks["arrival_at_scheduler_timestamp"] + \
                            dropped_tasks["dropped_at_task_id"].map(lambda tid: self.slo_log[w][tid]))
                            .all())
                    
                # completed tasks were never started after their task deadlines
                complete_tasks = self.task_log.replace("", np.nan)
                complete_tasks = complete_tasks[complete_tasks["dropped_timestamp"]==np.nan]
                for w in set(complete_tasks["workflow_id"]):
                    assert(complete_tasks["execution_start_timestamp"] <= 
                           complete_tasks["arrival_at_scheduler_timestamp"] + 
                           complete_tasks["task_id"].map(lambda tid: self.slo_log[w][tid]))
            
            elif self.gcfg.SLO_TYPE == "JOB_LEVEL":
                for _, row in dropped_jobs.iterrows():
                    # dropped jobs all dropped after deadline
                    assert(row["create_time"] + row["response_time"] >= row["deadline"])

                    # logged deadline == client SLO + job create time
                    slo = self.client_log[(self.client_log["client_id"]==row["client_id"]) & 
                                          (self.client_log["workflow_id"]==row["workflow_id"])]["slo"].iloc[0]
                    assert(row["deadline"] == row["create_time"] + slo)

            # dropped jobs are never batched after being dropped
            for _, row in dropped_jobs.iterrows():
                mask = (self.batch_log["batched_job_task_ids"].str.contains(f"\({row['job_id']},")) &\
                        (self.batch_log["execution_start_timestamp"] >= row["create_time"] + row["response_time"])
                assert(self.batch_log[mask].empty)

        print("[PASS] Job drop verification")


    def verify_instance_activity(self):
        """Cross-verify [is_active] column of work log with batch log.
        """

        for _, row in self.work_log.iterrows():
            mask = ((self.batch_log["instance_id"]==row["instance_id"]) & 
                    (self.batch_log["execution_start_timestamp"] < row["time"]) &
                    (self.batch_log["execution_end_timestamp"] > row["time"]))

            assert(len(self.batch_log[mask]) <= 1)
            assert(row["is_active"] == (not self.batch_log[mask].empty))

        print("[PASS] Work log activity column verification")


    def verify_remaining_work(self):
        if self.is_centralized:
            for _, row in self.work_log.iterrows():
                mask = ((self.batch_log["instance_id"]==row["instance_id"]) & 
                    (self.batch_log["execution_start_timestamp"] < row["time"]) &
                    (self.batch_log["execution_end_timestamp"] > row["time"]))
            
                assert(len(self.batch_log[mask]) <= 1)
                assert(row["num_incomplete_assigned_jobs"] == 
                       self.batch_log[mask]["batch_size"].sum())
                
        else:
            for i, row in self.work_log.iterrows():
                batch_mask = ((self.batch_log["instance_id"]==row["instance_id"]) & 
                              (self.batch_log["execution_start_timestamp"] < row["time"]) &
                              (self.batch_log["execution_end_timestamp"] > row["time"]))
                
                task_mask = ((self.task_log["executing_worker_id"]==row["worker_id"]) &
                             (self.task_log["model_id"]==row["model_id"]) & 
                             (self.task_log["arrival_at_worker_timestamp"] < row["time"]) &
                             (self.task_log["execution_start_timestamp"] >= row["time"]))

                assert(len(self.batch_log[batch_mask]) <= 1)
                assert(row["num_incomplete_assigned_jobs"] == 
                       (self.batch_log[batch_mask]["batch_size"].sum() + len(self.task_log[task_mask])))

        print("Successfully verified work log!")


    def trace_task_arrivals(self):
        # trace state
        worker_model_qs = {} # worker ID -> model ID -> awaiting tasks [(job ID, task ID)]
        worker_instances = {} # worker ID -> model ID -> [instance ID]

        # parse configs
        sched_worker_id = self.worker_log["worker_id"][0]
        wf_cfgs = {}
        for cfg in self.wcfg.WORKFLOW_LIST:
            wf_cfgs[cfg["JOB_TYPE"]] = {tcfg["TASK_INDEX"]: tcfg 
                                        for tcfg in cfg["TASKS"]}
            
        for _, wrow in self.worker_log.iterrows():
            if wrow["worker_id"] not in worker_instances:
                worker_instances[wrow["worker_id"]] = {}

            if wrow["model_id"] not in worker_instances[wrow["worker_id"]]:
                worker_instances[wrow["worker_id"]][wrow["model_id"]] = []

            worker_instances[wrow["worker_id"]][wrow["model_id"]].append(
                wrow["instance_id"])
            
        total_rows = len(self.task_log)

        prev_time = {(wid, mid): 0 for wid in worker_instances.keys()
                     for mid in set(self.task_log["model_id"])}
        
        for i, row in self.task_log.sort_values(["arrival_at_worker_timestamp", 
                                                 "executing_worker_qlen_at_arrival"]).iterrows():
            print(f"{i} / {total_rows} = {i / total_rows * 100:.1f}% done...")
            print(row["job_id"], row["task_id"])
            print()

            tcfg = wf_cfgs[row["workflow_id"]][row["task_id"]]

            if len(tcfg["PREV_TASK_INDEX"]) == 0:
                # if initial task, check arrival at scheduler timestamp
                if (not self.is_centralized) or (not self.gcfg.ENABLE_NETWORKING_DELAYS) or \
                    (sched_worker_id == row["executing_worker_id"]):
                    assert(row["arrival_at_scheduler_timestamp"] == row["arrival_at_worker_timestamp"])
                else:
                    assert(row["arrival_at_worker_timestamp"] ==
                           (row["arrival_at_scheduler_timestamp"] + CPU_to_CPU_delay(tcfg["INPUT_SIZE"])))

            if not self.gcfg.ENABLE_NETWORKING_DELAYS:
                assert(row["last_dep_dispatch_timestamp"]==row["arrival_at_worker_timestamp"])

            if row["executing_worker_id"] not in worker_model_qs:
                worker_model_qs[row["executing_worker_id"]] = {}
            
            if row["model_id"] not in worker_model_qs[row["executing_worker_id"]]:
                worker_model_qs[row["executing_worker_id"]][row["model_id"]] = []

            # should not duplicate tasks
            assert((row["job_id"], row["task_id"]) not in 
                   worker_model_qs[row["executing_worker_id"]][row["model_id"]])

            worker_model_qs[row["executing_worker_id"]][row["model_id"]].append(
                (row["job_id"], row["task_id"]))
            
            # update q based on batch exec history
            started_batches = self.batch_log[(self.batch_log["worker_id"] == row["executing_worker_id"]) &
                                             (self.batch_log["model_id"] == row["model_id"]) &
                                             (self.batch_log["execution_start_timestamp"] < row["arrival_at_worker_timestamp"])]
            for _, brow in started_batches.iterrows():
                for (j, t) in ast.literal_eval(brow["batched_job_task_ids"]):
                    if (j, t) in worker_model_qs[row["executing_worker_id"]][row["model_id"]]:
                        worker_model_qs[row["executing_worker_id"]][row["model_id"]].remove((j, t))
            
            prev_time[(row["executing_worker_id"], row["model_id"])] = row["arrival_at_worker_timestamp"]

            print(row["executing_worker_qlen_at_arrival"])
            print(worker_model_qs[row["executing_worker_id"]][row["model_id"]])

            # check qlen logging
            assert(row["executing_worker_qlen_at_arrival"] == 
                   len(worker_model_qs[row["executing_worker_id"]][row["model_id"]]))

            # verify exec against batch log
            mask = ((self.batch_log["model_id"]==row["model_id"]) & 
                    (self.batch_log["execution_start_timestamp"] == row["execution_start_timestamp"]) &
                    (self.batch_log["execution_end_timestamp"] == row["execution_end_timestamp"]))

            assert(not self.batch_log.loc[mask].empty)

            s = f"({row['job_id']}, {row['task_id']})"
            assert(any(s in r["batched_job_task_ids"] for  _, r in self.batch_log.loc[mask].iterrows()))

            # if qlen <= max bsize && idle instance exists, should start task right away
            should_start_exec_immediately = False
            if len(worker_model_qs[row["executing_worker_id"]][row["model_id"]]) <= \
                self.mcfg.MODELS[row["model_id"]]["MAX_BATCH_SIZE"]:

                for instance_id in worker_instances[row["executing_worker_id"]][row["model_id"]]:
                    mask = ((self.batch_log["instance_id"]==instance_id) & 
                            (self.batch_log["execution_start_timestamp"] < row["arrival_at_worker_timestamp"]) &
                            (self.batch_log["execution_end_timestamp"] > row["arrival_at_worker_timestamp"]))
                    
                    # if no batch is being executed currently
                    if self.batch_log.loc[mask].empty:
                        should_start_exec_immediately = True

            if should_start_exec_immediately:
                assert(row["arrival_at_worker_timestamp"] == row["execution_start_timestamp"])


if __name__ == "__main__":
    results_dir = sys.argv[1]
    is_centralized = sys.argv[2] == "True"

    gcfg_path = os.path.join(results_dir, "configs/gen_config.py")
    mcfg_path = os.path.join(results_dir, "configs/model_config.py")
    wcfg_path = os.path.join(results_dir, "configs/workflow_config.py")

    modules = {}

    for path in [gcfg_path, mcfg_path, wcfg_path]:
        spec = importlib.util.spec_from_file_location(f"results_{path.replace('/', '_')}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[path] = module

    slo_log = None
    if os.path.exists(os.path.join(results_dir, "sim_logs/nexus_task_slo_log.json")):
        slo_log = json.load(open(os.path.join(results_dir, "sim_logs/nexus_task_slo_log.json")))
        slo_log = {int(k1): {int(k2): v2 for k2, v2 in v1.items()} for k1, v1 in slo_log.items()}

    exec_verifier = LogVerifier(
        pd.read_csv(os.path.join(results_dir, "sim_logs/job_log.csv")),
        pd.read_csv(os.path.join(results_dir, "sim_logs/task_log.csv")),
        pd.read_csv(os.path.join(results_dir, "sim_logs/worker_batch_log.csv")),
        pd.read_csv(os.path.join(results_dir, "sim_logs/worker_config_log.csv")),
        pd.read_csv(os.path.join(results_dir, "sim_logs/work_log.csv")),
        pd.read_csv(os.path.join(results_dir, "sim_logs/client_config_log.csv")),
        slo_log,
        is_centralized,
        modules[gcfg_path],
        modules[mcfg_path],
        modules[wcfg_path]
    )

    exec_verifier.run()