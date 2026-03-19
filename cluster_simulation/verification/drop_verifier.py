from verification.verifier import Verifier


class DropVerifier(Verifier):
    
    def run_verifier(self):
        self.verify_task_level_slos()
        self.verify_job_level_slos()

    def verify_task_level_slos(self):
        if self.gcfg.SLO_TYPE not in ["NEXUS", "NEXUS_DYNAMIC"]:
            return
        
        # Early drop policy does not work for task level SLO
        assert(self.gcfg.DROP_POLICY != "OPTIMAL")
        
        for i, row in self.dfs["drop_log"].iterrows():
            task_arrival_time = self.dfs["arrival_log"][(self.dfs["arrival_log"]["job_id"]==row["job_id"]) & \
                                                        (self.dfs["arrival_log"]["task_id"]==row["task_id"])]["time"].max()
            task_slos = self.dfs["slo_log"][(self.dfs["slo_log"]["workflow_id"]==row["workflow_id"]) & \
                                            (self.dfs["slo_log"]["task_id"]==row["task_id"]) & \
                                            (self.dfs["slo_log"]["time"] < row["drop_time"])]
            expected_slo = task_slos[task_slos["time"]==task_slos["time"].max()].iloc[0]["slo"] \
                if len(task_slos) > 0 else self.gcfg.CLIENT_CONFIGS[int(row["client_id"])][int(row["workflow_id"])]["SLO"]
            
            assert(row["drop_time"] >= task_arrival_time + expected_slo)
        
        for i, row in self.dfs["exec_log"].iterrows():
            task_arrival_time = self.dfs["arrival_log"][(self.dfs["arrival_log"]["job_id"]==row["job_id"]) & \
                                                        (self.dfs["arrival_log"]["task_id"]==row["task_id"])]["time"].max()
            task_slos = self.dfs["slo_log"][(self.dfs["slo_log"]["workflow_id"]==row["workflow_id"]) & \
                                            (self.dfs["slo_log"]["task_id"]==row["task_id"]) & \
                                            (self.dfs["slo_log"]["time"] <= row["exec_start_time"])]
            expected_slo = task_slos[task_slos["time"]==task_slos["time"].max()].iloc[0]["slo"] \
                if len(task_slos) > 0 else self.gcfg.CLIENT_CONFIGS[int(row["client_id"])][int(row["workflow_id"])]["SLO"]
            
            wcfg = [wcfg for wcfg in self.wcfg.WORKFLOW_LIST if wcfg["JOB_TYPE"] == row["workflow_id"]][0]
            min_runtime = self.mcfg.MODELS[wcfg["TASKS"][row["task_id"]]["MODEL_ID"]]["MIG_BATCH_EXEC_TIMES"][24][1]

            assert(min_runtime <= expected_slo)
        
    def verify_job_level_slos(self):
        if self.gcfg.SLO_TYPE != "JOB_LEVEL":
            return
        
        if self.gcfg.DROP_POLICY == "LATEST_POSSIBLE":
            # all tasks should still be valid before being batched
            job_deadlines = {}
            for i, row in self.dfs["batch_log"].iterrows():
                job_ids = row["job_ids"] if type(row["job_ids"]) == list else \
                    [int(sjid) for sjid in row["job_ids"].strip("[] ").split(",")]
                for job_id in job_ids:
                    if job_id not in job_deadlines:
                        job_info = self._get_job_details(job_id)
                        slo = self.gcfg.CLIENT_CONFIGS[int(job_info["client_id"])][job_info["workflow_type"]]["SLO"]
                        job_deadlines[job_id] = slo + job_info["job_create_time"]

                assert(row["start_time"] <= job_deadlines[job_id])

            # all dropped tasks must have been invalid at drop
            for i, row in self.dfs["drop_log"].iterrows():
                job_id = int(row["job_id"])
                if job_id not in job_deadlines:
                    job_info = self._get_job_details(job_id)
                    slo = self.gcfg.CLIENT_CONFIGS[int(job_info["client_id"])][job_info["workflow_type"]]["SLO"]
                    job_deadlines[job_id] = slo + job_info["job_create_time"]

                assert(row["drop_time"] >= job_deadlines[job_id])
        
        elif self.gcfg.DROP_POLICY == "OPTIMAL":
            # all tasks should still be valid before being batched
            job_deadlines = {}
            job_exec_statuses = {} # job id -> task id -> completion time
            for i, row in self.dfs["batch_log"].sort_values(by="start_time").iterrows():
                job_ids = row["job_ids"] if type(row["job_ids"]) != str else [int(sjid) for sjid in row["job_ids"].strip("[] ").split(",")]
                for job_id in job_ids:
                    if job_id not in job_exec_statuses:
                        job_exec_statuses[job_id] = {}

                    job_info = self._get_job_details(job_id)
                    if job_id not in job_deadlines:
                        slo = self.gcfg.CLIENT_CONFIGS[int(job_info["client_id"])][job_info["workflow_type"]]["SLO"]
                        job_deadlines[job_id] = slo + job_info["job_create_time"]
    
                    min_rem_proc_time = self.workflows[job_info["workflow_type"]].get_processing_time(
                        lambda at: 0 if (at.model_data.id in job_exec_statuses[job_id] and \
                                         job_exec_statuses[job_id][at.model_data.id] <= row["start_time"]) \
                                     else at.model_data.batch_exec_times[24][1])
                    
                    assert(row["start_time"] + min_rem_proc_time <= job_deadlines[job_id])

                    job_exec_statuses[job_id][row["model_id"]] = row["end_time"]

            # all dropped tasks must have been invalid at drop
            for i, row in self.dfs["drop_log"].iterrows():
                job_id = int(row["job_id"])
                job_info = self._get_job_details(job_id)
                if job_id not in job_deadlines:
                    slo = self.gcfg.CLIENT_CONFIGS[int(job_info["client_id"])][job_info["workflow_type"]]["SLO"]
                    job_deadlines[job_id] = slo + job_info["job_create_time"]

                min_rem_proc_time = self.workflows[job_info["workflow_type"]].get_processing_time(
                    lambda at: 0 if (job_id in job_exec_statuses and at.model_data.id in job_exec_statuses[job_id] and \
                                     job_exec_statuses[job_id][at.model_data.id] <= row["drop_time"]) \
                                 else at.model_data.batch_exec_times[24][1])

                assert(row["drop_time"] + min_rem_proc_time > job_deadlines[job_id])