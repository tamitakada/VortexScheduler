from queue import PriorityQueue

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
from schedulers.centralized.hashtask_scheduler import HashTaskScheduler

import pandas as pd
import math


class NexusScheduler(HashTaskScheduler):
    """
        Scheduler class for Nexus:
        https://homes.cs.washington.edu/~arvind/papers/nexus.pdf
    """

    def __init__(self, simulation, herd_assignment=None):
        super().__init__(simulation, herd_assignment)

        assert(gcfg.SLO_GRANULARITY == "TASK") # Nexus uses per-task SLO

        self.expected_workflow_arrival_rates = {} # workflow id -> arrival rate used to set slo below
        self.workflow_task_slos = {} # workflow id -> task id -> (slo, bsize)

        self.wf_arrival_rate_log = pd.DataFrame(columns=["time", "workflow_id", "arrival_rate_past_5s"])
        self.task_slo_log = pd.DataFrame(columns=["time", "workflow_id", "task_id", "slo", "bsize"])

        # for round robin scheduling
        self.last_worker_idx = {}
    
    def get_task_slos(self, time: float, measurement_interval: float, job: Job):
        """
        Return map [task_slos: dict[int, int]] based on [arrival_rate] for each
        task in [job.tasks].
        """

        # granularity of SLOs
        TIME_STEP = 5

        task_model_arrival_rates = {}
        for task in job.tasks:
            # arrived_jobs = self.arrived_task_log[(self.arrived_task_log["workflow_id"]==job.job_type_id) & \
            #                       (self.arrived_task_log["time"] <= time) & \
            #                       (self.arrived_task_log["time"] > (time - measurement_interval))]
            arrived_job_count = self.simulation.task_arrival_log[\
                (self.simulation.task_arrival_log["model_id"]==task.model_data.id) & \
                (self.simulation.task_arrival_log["time"] <= time) & \
                (self.simulation.task_arrival_log["time"] > (time - measurement_interval))]["job_id"].nunique()
            num_model_workers = len([w for w in self.simulation.workers.values() 
                                     if any(s.model.data.id == task.model_data.id for s in w.GPU_state.state_at(time))])
            task_model_arrival_rates[task.task_id] = arrived_job_count / num_model_workers / measurement_interval * 1000 #len(set(arrived_jobs["job_id"])) / measurement_interval * 1000  

        # task id -> task SLO -> (min # gpus, optimal batch size, (SLO for curr task, SLO for remainder of pipeline/subtree))
        min_gpus = {task.task_id: {} for task in job.tasks}
        
        # base case: exit points / leaf nodes
        final_tasks = [t for t in job.tasks if len(t.next_task_ids) == 0]
        for task in final_tasks:
            for t in range(TIME_STEP, job.slo + 1, TIME_STEP):
                min_gpus[task.task_id][t] = (np.inf, 1, (t, 0)) # init to inf

                sat_bsizes = [bsize for bsize in range(1,task.model_data.max_batch_size+1) 
                              if task.model_data.batch_exec_times[24][bsize] <= t]
                    
                # if min exec time violates SLO k, skip
                if len(sat_bsizes) == 0:
                    continue

                viable_bsizes = [b for b in sat_bsizes if b == 1 or b < task_model_arrival_rates[task.task_id] / 1000 * \
                                    task.model_data.batch_exec_times[24][b]]
                opt_bsize = min(viable_bsizes, key=lambda b: task_model_arrival_rates[task.task_id] * task.model_data.batch_exec_times[24][b] / b / 1000)
                # opt_bsize = min(sat_bsizes, key=lambda b: task_model_arrival_rates[task.task_id] * task.model_data.batch_exec_times[24][task.model_data.batch_sizes.index(b)] / b / 1000)
                
                min_gpus_k = task_model_arrival_rates[task.task_id] * task.model_data.batch_exec_times[24][opt_bsize] / opt_bsize / 1000
                if min_gpus_k < min_gpus[task.task_id][t][0]:
                    min_gpus[task.task_id][t] = (min_gpus_k, opt_bsize, (t, 0))

        traversed_task_ids = set([t.task_id for t in final_tasks])
        remaining_tasks = [t for t in job.tasks if t.task_id not in traversed_task_ids]
        next_tasks = [t for t in remaining_tasks if all(tid in traversed_task_ids for tid in t.next_task_ids)]
        while next_tasks:
            for task in next_tasks:
                for t in range(TIME_STEP, job.slo + 1, TIME_STEP):
                    min_gpus[task.task_id][t] = (np.inf, 1, (t, 0)) # init to inf
                
                    for k in range(TIME_STEP, t + 1, TIME_STEP):
                        sat_bsizes = [bsize for bsize in range(1,task.model_data.max_batch_size+1) 
                                      if task.model_data.batch_exec_times[24][bsize] <= k]
                    
                        # if min exec time violates SLO k, skip
                        if len(sat_bsizes) == 0:
                            continue
                        
                        # if there's not enough time left for another step with this SLO
                        if t - k < TIME_STEP:
                            continue
                        
                        min_rem_gpus = min(sum(min_gpus[v][t_prime][0] for v in task.next_task_ids) for t_prime in range(TIME_STEP, t - k + 1, TIME_STEP))

                        viable_bsizes = [b for b in sat_bsizes
                                         if b == 1 or b < task_model_arrival_rates[task.task_id] / 1000 * \
                                            task.model_data.batch_exec_times[24][b]]
                        
                        opt_bsize = min(viable_bsizes, key=lambda b: task_model_arrival_rates[task.task_id] * task.model_data.batch_exec_times[24][b] / b / 1000)
                        # opt_bsize = min(sat_bsizes, key=lambda b: task_model_arrival_rates[task.task_id] * task.model_data.batch_exec_times[24][task.model_data.batch_sizes.index(b)] / b / 1000)
                        
                        min_gpus_k = task_model_arrival_rates[task.task_id] * task.model_data.batch_exec_times[24][opt_bsize] / opt_bsize / 1000 + min_rem_gpus
                        if min_gpus_k < min_gpus[task.task_id][t][0]:
                            min_gpus[task.task_id][t] = (min_gpus_k, opt_bsize, (k, t-k))
                
                traversed_task_ids.add(task.task_id)
                remaining_tasks.remove(task)
            
            next_tasks = [t for t in remaining_tasks if all(tid in traversed_task_ids for tid in t.next_task_ids)]

        def _traverse_slo_tree(tasks, tree_slo, task_slos):
            for task in tasks:
                if tree_slo < TIME_STEP:
                    continue

                best_timestep = min(range(TIME_STEP, tree_slo+1, TIME_STEP), key=lambda t: (min_gpus[task.task_id][t][0], -t))
                (_, opt_bsize, (task_slo, subtree_slo)) = min_gpus[task.task_id][best_timestep]

                # always use the tighter SLO if multiple subtrees share a task node
                if task.task_id not in task_slos or task_slo < task_slos[task.task_id][0]:         
                    task_slos[task.task_id] = (task_slo, opt_bsize)

                _traverse_slo_tree([job.get_task_by_id(tid) for tid in task.next_task_ids], subtree_slo, task_slos)
            
            return task_slos

        root_tasks = [t for t in job.tasks if len(t.required_task_ids) == 0]
        proposed_slos = _traverse_slo_tree(root_tasks, job.slo, {})

        # if job.job_type_id 

        # # if job.job_type_id in [0,5]:
        # # TODO
        # total_used_slo = max(proposed_slos[0][0], proposed_slos[1][0]) + proposed_slos[2][0] + proposed_slos[3][0]
        # remaining_slo = job.slo - total_used_slo
        # # TODO
        # slo_slack = remaining_slo / 3

        # print("INITIAL PROP: ", proposed_slos)
        # print()
        # print("SLACK ", slo_slack)

        # original_slos = proposed_slos.copy()

        # for tid in proposed_slos.keys():
        #     # TODO
        #     if tid in [0,1]:
        #         adjusted_slo = max(original_slos[0][0], original_slos[1][0]) + slo_slack
        #     else:
        #         adjusted_slo = proposed_slos[tid][0] + slo_slack
            
        #     proposed_slos[tid] = (adjusted_slo, 
        #                         max(b for i, b in enumerate(job.get_task_by_id(tid).model.batch_sizes)
        #                             if job.get_task_by_id(tid).model.batch_exec_times[24][i] < adjusted_slo))

        # print(min_gpus)
        print()
        print(proposed_slos)

        # assert(False)

        return proposed_slos

    
    def redistribute_task_slos(self, time: float, tasks: list[Task], workflow_id: int, realloc_amt_ms: float):
        if workflow_id not in self.workflow_task_slos:
            return # no base SLO allocation to update
        
        last_slo_update_time = self.task_slo_log[self.task_slo_log["workflow_id"]==workflow_id]["time"].max()
        if time - last_slo_update_time < 1000:
            return # at least 1s must pass from last update

        drop_df = self.simulation.task_drop_log[(self.simulation.task_drop_log["workflow_id"]==workflow_id) & \
                                                (self.simulation.task_drop_log["drop_time"] <= time) & \
                                                (self.simulation.task_drop_log["drop_time"] > last_slo_update_time)]
        
        task_drop_rates = {}
        for task in tasks:
            drop_rate = (drop_df["task_id"]==task.task_id).sum() / (time - last_slo_update_time) * 1000
            task_drop_rates[task.task_id] = drop_rate

        print("WF DROP RATES: ", task_drop_rates)

        MAX_ALLOWABLE_DROP_RATE = 2
        # tasks with significant drop rates in desc magnitude of drop rate
        # TODO: pipeline

        if workflow_id == 1:
            unsat_tasks = sorted([task for task in tasks if task_drop_rates[task.task_id] > MAX_ALLOWABLE_DROP_RATE],
                             key=lambda task: task_drop_rates[task.task_id],
                             reverse=True)
            
            sat_tasks = sorted([task for task in tasks if task not in unsat_tasks],
                    key=lambda task: self.workflow_task_slos[workflow_id][task.task_id][0],
                    reverse=True)
        elif workflow_id == 4:
            unsat_tasks = sorted([task for task in tasks if task.task_id not in [3,4] and task_drop_rates[task.task_id] > MAX_ALLOWABLE_DROP_RATE],
                             key=lambda task: task_drop_rates[task.task_id] if task.task_id != 2 else max(task_drop_rates[3] + task_drop_rates[4], task_drop_rates[2]),
                             reverse=True)
            
            sat_tasks = sorted([task for task in tasks if task not in unsat_tasks and task.task_id not in [3,4]],
                    key=lambda task: self.workflow_task_slos[workflow_id][task.task_id][0],
                    reverse=True)
        elif workflow_id == 5 or workflow_id == 0:
            unsat_tasks = sorted([task for task in tasks if task.task_id != 0 and task_drop_rates[task.task_id] > MAX_ALLOWABLE_DROP_RATE],
                                key=lambda task: task_drop_rates[task.task_id] if task.task_id != 1 else max(task_drop_rates[0], task_drop_rates[1]),
                                reverse=True)
            
            # tasks with drop rate < threshold in desc magnitude of SLO
            sat_tasks = sorted([task for task in tasks if task not in unsat_tasks and task.task_id != 0],
                            key=lambda task: self.workflow_task_slos[workflow_id][task.task_id][0],
                            reverse=True)
        else:
            raise NotImplementedError()

        if not unsat_tasks and workflow_id != 4:
            return
        
        if not sat_tasks and workflow_id != 4:
            print("WARNING: NO TASKS ARE SATISFIED")
            return
        
        if workflow_id == 4:
            if not sat_tasks and task_drop_rates[3] > MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] > MAX_ALLOWABLE_DROP_RATE:
                return
            
            if not unsat_tasks and task_drop_rates[3] <= MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] <= MAX_ALLOWABLE_DROP_RATE:
                return
        
        for sat_task in sat_tasks:
            if self.workflow_task_slos[workflow_id][sat_task.task_id][0] < realloc_amt_ms:
                continue
            
            new_sat_task_slo = self.workflow_task_slos[workflow_id][sat_task.task_id][0] - realloc_amt_ms
            if sat_task.model_data.batch_exec_times[24][1] > new_sat_task_slo:
                continue

            self.workflow_task_slos[workflow_id][sat_task.task_id] = \
                (new_sat_task_slo,
                 max([bsize for bsize in range(1,sat_task.model_data.max_batch_size+1) 
                      if sat_task.model_data.batch_exec_times[24][bsize] <= new_sat_task_slo]))
            
            unsat_task = unsat_tasks.pop(0)

            new_unsat_task_slo = self.workflow_task_slos[workflow_id][unsat_task.task_id][0] + realloc_amt_ms
            self.workflow_task_slos[workflow_id][unsat_task.task_id] = \
                (new_unsat_task_slo,
                 max([bsize for bsize in range(1,unsat_task.model_data.max_batch_size+1) 
                      if unsat_task.model_data.batch_exec_times[24][bsize] <= new_unsat_task_slo]))
            
            # TODO
            if workflow_id in [0, 5]:
                if sat_task.task_id == 1:
                    task_0 = [t for t in tasks if t.task_id ==0][0]
                    self.workflow_task_slos[workflow_id][0] = \
                        (new_sat_task_slo,
                        max([bsize for bsize in range(1,task_0.model_data.max_batch_size+1) 
                            if task_0.model_data.batch_exec_times[24][bsize] <= new_sat_task_slo]))
                    
                    self.task_slo_log.loc[len(self.task_slo_log)] = [time, workflow_id, 0,
                                                                self.workflow_task_slos[workflow_id][0][0],
                                                                self.workflow_task_slos[workflow_id][0][1]]
                
                if unsat_task.task_id == 1:
                    task_0 = [t for t in tasks if t.task_id ==0][0]
                    self.workflow_task_slos[workflow_id][0] = \
                        (new_unsat_task_slo,
                        max([bsize for bsize in range(1,task_0.model_data.max_batch_size+1) 
                            if task_0.model_data.batch_exec_times[24][bsize] <= new_unsat_task_slo]))
                    
                    self.task_slo_log.loc[len(self.task_slo_log)] = [time, workflow_id, 0,
                                                                self.workflow_task_slos[workflow_id][0][0],
                                                                self.workflow_task_slos[workflow_id][0][1]]
            elif workflow_id == 4:
                if sat_task.task_id == 2:
                    task_3 = [t for t in tasks if t.task_id ==3][0]
                    slo3 = self.workflow_task_slos[workflow_id][3][0] - 2.5
                    self.workflow_task_slos[workflow_id][3] = \
                        (slo3,
                        max([bsize for bsize in range(1,task_3.model_data.max_batch_size+1) 
                            if task_3.model_data.batch_exec_times[24][bsize] <= slo3]))
                    
                    task_4 = [t for t in tasks if t.task_id ==4][0]
                    slo4 = self.workflow_task_slos[workflow_id][4][0] - 2.5
                    self.workflow_task_slos[workflow_id][4] = \
                        (slo4,
                        max([bsize for bsize in range(1,task_4.model_data.max_batch_size+1) 
                            if task_4.model_data.batch_exec_times[24][bsize] <= slo4]))
                    
                    self.task_slo_log.loc[len(self.task_slo_log)] = [time, workflow_id, 3,
                                                                self.workflow_task_slos[workflow_id][3][0],
                                                                self.workflow_task_slos[workflow_id][3][1]]
                    self.task_slo_log.loc[len(self.task_slo_log)] = [time, workflow_id, 4,
                                                                self.workflow_task_slos[workflow_id][4][0],
                                                                self.workflow_task_slos[workflow_id][4][1]]

                if unsat_task.task_id == 2:
                    task_3 = [t for t in tasks if t.task_id ==3][0]
                    slo3 = self.workflow_task_slos[workflow_id][3][0] + 2.5
                    self.workflow_task_slos[workflow_id][3] = \
                        (slo3,
                        max([bsize for bsize in range(1,task_3.model_data.max_batch_size+1) 
                            if task_3.model_data.batch_exec_times[24][bsize] <= slo3]))
                    
                    task_4 = [t for t in tasks if t.task_id ==4][0]
                    slo4 = self.workflow_task_slos[workflow_id][4][0] + 2.5
                    self.workflow_task_slos[workflow_id][4] = \
                        (slo4,
                        max([bsize for bsize in range(1,task_4.model_data.max_batch_size+1) 
                            if task_4.model_data.batch_exec_times[24][bsize] <= slo4]))
                    
                    self.task_slo_log.loc[len(self.task_slo_log)] = [time, workflow_id, 3,
                                                                self.workflow_task_slos[workflow_id][3][0],
                                                                self.workflow_task_slos[workflow_id][3][1]]
                    self.task_slo_log.loc[len(self.task_slo_log)] = [time, workflow_id, 4,
                                                                self.workflow_task_slos[workflow_id][4][0],
                                                                self.workflow_task_slos[workflow_id][4][1]]

            
            self.task_slo_log.loc[len(self.task_slo_log)] = [time, workflow_id, sat_task.task_id,
                                                             self.workflow_task_slos[workflow_id][sat_task.task_id][0],
                                                             self.workflow_task_slos[workflow_id][sat_task.task_id][1]]
            self.task_slo_log.loc[len(self.task_slo_log)] = [time, workflow_id, unsat_task.task_id,
                                                             self.workflow_task_slos[workflow_id][unsat_task.task_id][0],
                                                             self.workflow_task_slos[workflow_id][unsat_task.task_id][1]]
        
            if not unsat_tasks:
                break

        if workflow_id == 4:
            if task_drop_rates[3] <= MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] <= MAX_ALLOWABLE_DROP_RATE:
                return

            if task_drop_rates[3] > MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] > MAX_ALLOWABLE_DROP_RATE:
                return
            
            if task_drop_rates[3] > MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] <= MAX_ALLOWABLE_DROP_RATE:

                slo3 = self.workflow_task_slos[workflow_id][3][0] + realloc_amt_ms
                slo4 = self.workflow_task_slos[workflow_id][4][0] - realloc_amt_ms

            if task_drop_rates[3] <= MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] > MAX_ALLOWABLE_DROP_RATE:

                slo3 = self.workflow_task_slos[workflow_id][3][0] - realloc_amt_ms
                slo4 = self.workflow_task_slos[workflow_id][4][0] + realloc_amt_ms

            task_3 = [t for t in tasks if t.task_id ==3][0]
            task_4 = [t for t in tasks if t.task_id ==4][0]
            
            self.workflow_task_slos[workflow_id][3] = \
                (slo3,
                max([bsize for bsize in range(1, task_3.model_data.max_batch_size+1) 
                    if task_3.model_data.batch_exec_times[24][bsize] <= slo3]))
            self.workflow_task_slos[workflow_id][4] = \
                (slo4,
                max([bsize for bsize in range(1, task_4.model_data.max_batch_size+1) 
                    if task_4.model_data.batch_exec_times[24][bsize] <= slo4]))
            
            self.task_slo_log.loc[len(self.task_slo_log)] = [time, workflow_id, 3,
                                                        self.workflow_task_slos[workflow_id][3][0],
                                                        self.workflow_task_slos[workflow_id][3][1]]
            self.task_slo_log.loc[len(self.task_slo_log)] = [time, workflow_id, 4,
                                                        self.workflow_task_slos[workflow_id][4][0],
                                                        self.workflow_task_slos[workflow_id][4][1]]


    def update_task_slos_if_needed(self, job: Job, time: float):
        """
        If request arrival rate over past window differs by at least 5 QPS from 
        [expected_workflow_arrival_rates], then update the expected arrival rate
        and reassign workflow task SLOs.
        """

        measurement_interval = 1000 # 1s
        workflow_id = job.job_type_id

        # calculate/log arrival rate over past window
        arrival_rate = self.simulation.get_arrival_rate(time, workflow_id, measurement_interval, 1)

        if workflow_id not in self.expected_workflow_arrival_rates:
            self.expected_workflow_arrival_rates[workflow_id] = arrival_rate

            # task_slo = job.slo / 3

            # self.workflow_task_slos[workflow_id] = {
            #     0: (task_slo, max(b for i, b in enumerate(job.get_task_by_id(0).model.batch_sizes)
            #                           if job.get_task_by_id(0).model.batch_exec_times[24][i] < task_slo)),
            #     1: (task_slo, max(b for i, b in enumerate(job.get_task_by_id(1).model.batch_sizes)
            #                           if job.get_task_by_id(1).model.batch_exec_times[24][i] < task_slo)),
            #     2: (task_slo, max(b for i, b in enumerate(job.get_task_by_id(2).model.batch_sizes)
            #                           if job.get_task_by_id(2).model.batch_exec_times[24][i] < task_slo)),
            #     3: (task_slo, max(b for i, b in enumerate(job.get_task_by_id(3).model.batch_sizes)
            #                           if job.get_task_by_id(3).model.batch_exec_times[24][i] < task_slo))
            # }

            # for task in job.tasks:
            #     self.task_slo_log.loc[len(self.task_slo_log)] = [time, workflow_id, task.task_id, task_slo, self.workflow_task_slos[workflow_id][task.task_id][1]]

            new_task_slos = self.get_task_slos(time, measurement_interval, job)
        
            if workflow_id not in self.workflow_task_slos or \
                any(self.workflow_task_slos[workflow_id][task.task_id][0] != new_task_slos[task.task_id][0]
                    for task in job.tasks):
                # only update if SLOs are different from before
                self.workflow_task_slos[workflow_id] = new_task_slos

                # log SLO update
                for task in job.tasks:
                    self.task_slo_log.loc[len(self.task_slo_log)] = [time, workflow_id, task.task_id, new_task_slos[task.task_id][0], new_task_slos[task.task_id][1]]
                
                print(self.workflow_task_slos)
                #assert(False)
        
        self.redistribute_task_slos(time, job.tasks, workflow_id, 5)
        # assert(False)

    def schedule_job_on_arrival(self, job, current_time):
        if current_time > 1000:
            self.update_task_slos_if_needed(job, current_time)

        for task in job.tasks:
            if job.job_type_id in self.workflow_task_slos:
                task.slo = self.workflow_task_slos[job.job_type_id][task.task_id][0]
                task.model_data.max_batch_size = self.workflow_task_slos[job.job_type_id][task.task_id][1]
                self.simulation.models[task.model_data.id].max_batch_size = self.workflow_task_slos[job.job_type_id][task.task_id][1]
            else:
                task.slo = np.inf

        return super().schedule_job_on_arrival(job, current_time)