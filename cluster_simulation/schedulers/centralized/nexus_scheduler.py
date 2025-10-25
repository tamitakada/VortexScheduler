from queue import PriorityQueue

from core.configs.gen_config import *
from core.task import Task
from core.batch import Batch
from core.events.base import *
from core.events.centralized_scheduler_events import *
from core.events.worker_events import *
from core.network import *
from core.configs.workflow_config import *

from workers.worker import Worker

from schedulers.centralized.scheduler import Scheduler
from schedulers.centralized.shepherd.ordered_task import OrderedTask

import pandas as pd


class NexusScheduler(Scheduler):
    """
        Scheduler class for Nexus:
        https://homes.cs.washington.edu/~arvind/papers/nexus.pdf
    """

    def __init__(self, simulation, herd_assignment=None):
        super().__init__(simulation, herd_assignment)

        assert(SLO_GRANULARITY == "TASK") # Nexus uses per-task SLO

        self.expected_workflow_arrival_rates = {} # workflow id -> arrival rate used to set slo below
        self.workflow_task_slos = {} # workflow id -> task id -> (slo, bsize)

        self.arrived_task_log = pd.DataFrame(columns=["time", "workflow_id", "job_id", "task_id"])
        self.wf_arrival_rate_log = pd.DataFrame(columns=["time", "workflow_id", "arrival_rate_past_5s"])
        self.task_slo_log = pd.DataFrame(columns=["time", "workflow_id", "task_id", "slo", "bsize"])

        # for round robin scheduling
        self.next_worker_id = { jt: [0 for _ in get_task_types([jt])] for jt in self.simulation.job_types_list }

    # TODO: Use Nexus global allocation/scheduler?
    def nexus_schedule_saturate(sessions: list[tuple[Model, float, float]]):
        nodes, residual_rates = [], []
        for (model, slo, arrival_rate) in sessions:
            opt_bsize = max(model.batch_sizes, key=(lambda b: 2 * model.batch_exec_times[24][b] 
                                                    if (2 * model.batch_exec_times[24][b]) <= slo else (0 if b == 1 else -1)))
            opt_tput = opt_bsize / model.batch_exec_times[24][opt_bsize]

            n_nodes = arrival_rate // opt_tput
            nodes += [(model, slo, arrival_rate) for _ in range(n_nodes)]

            residual_rate = arrival_rate % opt_tput
            residual_rates.append((model, slo, residual_rate))
        return nodes, residual_rates

    # TODO: Use Nexus global allocation/scheduler?
    def nexus_schedule_residue(residue: list[tuple[Model, float, float]]):
        mod_residue: list[tuple[Model, float, float, int, float, float]] = []
        for (model, slo, arrival_rate) in residue:
            bsize = max(model.batch_sizes, key=(lambda b: model.batch_exec_times[24][b] + b / arrival_rate
                                                if (model.batch_exec_times[24][b] + b / arrival_rate) <= slo else (0 if b == 1 else -1)))
            duty_cycle = bsize / arrival_rate
            occupancy = model.batch_exec_times[24][bsize] / duty_cycle # frac. duty cycle occupied by residual load
            
            mod_residue.append((model, slo, arrival_rate, bsize, duty_cycle, occupancy))
        
        sorted_residue = sorted(mod_residue, key=lambda r: r[-1], reverse=True)

        # list of sessions (model, slo, arrival rate, bsize) and duty cycle
        nodes: list[tuple[list[tuple[Model, float, float, int]], float]] = []
        for (model, slo, arrival_rate, bsize, duty_cycle, occupancy) in sorted_residue:
            node_to_replace = -1
            max_occupancy = 0
            max_node = None

            for i, (sessions, node_duty_cycle) in enumerate(nodes):
                merged_duty_cycle = min(duty_cycle, node_duty_cycle)
                merged_bsize = merged_duty_cycle * arrival_rate
                if merged_bsize > model.batch_sizes[-1]:
                    continue

                total_exec_time = model.batch_exec_times[24][merged_bsize] + \
                    sum(s[0].batch_exec_times[24][s[3]] for s in sessions)
                if total_exec_time <= merged_duty_cycle: # then is a valid merge
                    # check for if best merge based on occupancy
                    merged_occupancy = total_exec_time / merged_duty_cycle
                    if merged_occupancy > max_occupancy:
                        node_to_replace = i
                        max_occupancy = merged_occupancy
                        max_node = (sessions + [(model, slo, arrival_rate, merged_bsize)], merged_duty_cycle)
            
            if max_node:
                nodes[node_to_replace] = max_node
            else:
                curr_session = (model, slo, arrival_rate, bsize)
                nodes.append(([curr_session], duty_cycle))
        
        return [sessions for sessions, _ in nodes]
    
    def get_task_slos(self, job: Job, arrival_rate: float):
        """
        Return map [task_slos: dict[int, int]] based on [arrival_rate] for each
        task in [job.tasks].
        """

        # granularity of SLOs
        TIME_STEP = 5

        # task id -> task SLO -> (min # gpus, (SLO for curr task, SLO for remainder of pipeline/subtree))
        min_gpus = {task.task_id: {} for task in job.tasks}
        
        # base case: exit points / leaf nodes
        final_tasks = [t for t in job.tasks if len(t.next_task_ids) == 0]
        for task in final_tasks:
            for t in range(TIME_STEP, job.slo + 1, TIME_STEP):
                min_gpus[task.task_id][t] = (np.inf, 1, (t, 0)) # init to inf

                sat_bsizes = [task.model.batch_sizes[i] for i in range(len(task.model.batch_sizes)) if task.model.batch_exec_times[24][i] <= t]
                    
                # if min exec time violates SLO k, skip
                if len(sat_bsizes) == 0:
                    continue
                
                opt_bsize = min(sat_bsizes, key=lambda b: arrival_rate * task.model.batch_exec_times[24][task.model.batch_sizes.index(b)] / b / 1000)
                min_gpus_k = arrival_rate * task.model.batch_exec_times[24][task.model.batch_sizes.index(opt_bsize)] / opt_bsize / 1000
                if min_gpus_k < min_gpus[task.task_id][t][0]:
                    min_gpus[task.task_id][t] = (min_gpus_k, opt_bsize, (t, 0))

        next_tasks = [t for t in job.tasks if len(t.next_task_ids) == 1 and any(ft.task_id == t.next_task_ids[0] for ft in final_tasks)]
        while next_tasks:
            for task in next_tasks:
                for t in range(TIME_STEP, job.slo + 1, TIME_STEP):
                    min_gpus[task.task_id][t] = (np.inf, 1, (t, 0)) # init to inf
                
                    for k in range(TIME_STEP, t + 1, TIME_STEP):
                        sat_bsizes = [task.model.batch_sizes[i] for i in range(len(task.model.batch_sizes)) if task.model.batch_exec_times[24][i] <= k]
                    
                        # if min exec time violates SLO k, skip
                        if len(sat_bsizes) == 0:
                            continue
                        
                        # if there's not enough time left for another step with this SLO
                        if t - k < TIME_STEP:
                            continue

                        min_rem_gpus = min(sum(min_gpus[v][t_prime][0] for v in task.next_task_ids) for t_prime in range(TIME_STEP, t - k + 1, TIME_STEP))
                        opt_bsize = min(sat_bsizes, key=lambda b: arrival_rate * task.model.batch_exec_times[24][task.model.batch_sizes.index(b)] / b / 1000)
                        min_gpus_k = arrival_rate * task.model.batch_exec_times[24][task.model.batch_sizes.index(opt_bsize)] / opt_bsize / 1000 + min_rem_gpus
                        if min_gpus_k < min_gpus[task.task_id][t][0]:
                            min_gpus[task.task_id][t] = (min_gpus_k, opt_bsize, (k, t-k))
                
            next_tasks = [job.get_task_by_id(id) for t in next_tasks for id in t.required_task_ids]

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
        return _traverse_slo_tree(root_tasks, job.slo, {})

    def update_task_slos_if_needed(self, job: Job, time: float):
        """
        If request arrival rate over past window differs by at least 5 QPS from 
        [expected_workflow_arrival_rates], then update the expected arrival rate
        and reassign workflow task SLOs.
        """

        measurement_interval = 5 * 1000 # 5s
        workflow_id = job.job_type_id

        # calculate/log arrival rate over past window
        arrived_jobs = self.arrived_task_log[(self.arrived_task_log["workflow_id"] == workflow_id) & \
                                                (self.arrived_task_log["time"] <= time) & \
                                                (self.arrived_task_log["time"] > (time - measurement_interval))]
        arrival_rate = len(set(arrived_jobs["job_id"])) / measurement_interval * 1000
        self.wf_arrival_rate_log.loc[len(self.wf_arrival_rate_log)] = [time, workflow_id, arrival_rate]

        if workflow_id not in self.expected_workflow_arrival_rates or \
            abs(self.expected_workflow_arrival_rates[workflow_id] - arrival_rate) > 5:

            self.expected_workflow_arrival_rates[workflow_id] = arrival_rate

            new_task_slos = self.get_task_slos(job, arrival_rate)
            self.workflow_task_slos[workflow_id] = new_task_slos

            # log SLO update
            for task in job.tasks:
                self.task_slo_log.loc[len(self.task_slo_log)] = [time, workflow_id, task.task_id, new_task_slos[task.task_id][0], new_task_slos[task.task_id][1]]

    def schedule_job_on_arrival(self, job, current_time):
        self._assign_adfg(job.tasks, current_time)

        for task in job.tasks:
            self.arrived_task_log.loc[len(self.arrived_task_log)] = \
                [current_time, task.job.job_type_id, task.job_id, task.task_id]
    
        # wait at least 5s
        if current_time > 5000:
            self.update_task_slos_if_needed(job, current_time)

        for task in job.tasks:
            if task.job.job_type_id in self.workflow_task_slos:
                task.slo = self.workflow_task_slos[task.job.job_type_id][task.task_id][0]
                task.max_batch_size = self.workflow_task_slos[task.job.job_type_id][task.task_id][1]
            else:
                task.slo = np.inf
        
        # from hashtask scheduler
        self._assign_adfg(job.tasks, current_time)

        task_arrival_events = []

        initial_tasks = [task for task in job.tasks if len(task.required_task_ids) == 0]
        for task in initial_tasks:
            task_arrival_time = current_time + CPU_to_CPU_delay(task.input_size)
            worker_index = task.ADFG[task.task_id]
            task_arrival_events.append(EventOrders(
                task_arrival_time, TaskArrival(self.simulation, self.simulation.workers[worker_index], task, task.job.id)))

        return task_arrival_events
    
    def schedule_tasks_on_arrival(self, tasks, current_time):
        for task in tasks:
            self.arrived_task_log.loc[len(self.arrived_task_log)] = \
                [current_time, task.job.job_type_id, task.job_id, task.task_id]
        
        # from hashtask_scheduler
        task_arrival_events = []

        self._assign_adfg(tasks, current_time)

        for task in tasks:
            task_arrival_time = current_time + CPU_to_CPU_delay(task.input_size)
            worker_index = task.ADFG[task.task_id]
            task_arrival_events.append(EventOrders(
                task_arrival_time, TaskArrival(self.simulation, self.simulation.workers[worker_index], task, task.job.id)))

        return task_arrival_events
    
    # from hashtask_scheduler
    def _assign_adfg(self, tasks, current_time):
        for task in tasks:
            if ENABLE_DYNAMIC_MODEL_LOADING:
                if ALLOCATION_STRATEGY == "HERD":
                    # don't choose worker that is not in the correct group
                    while task.model and task.model.model_id not in self.herd_assignment.group_models[self.simulation.workers[self.next_worker_id[task.task_type[0]][task.task_id]].group_id] and \
                        self.simulation.workers[self.next_worker_id[task.task_type[0]][task.task_id]].total_memory * 10**6 < task.model.model_size:
                        self.next_worker_id[task.task_type[0]][task.task_id] = (self.next_worker_id[task.task_type[0]][task.task_id] + 1) % len(self.simulation.workers)
                else:
                    # don't choose partition that is too small
                    while task.model and self.simulation.workers[self.next_worker_id[task.task_type[0]][task.task_id]].total_memory * 10**6 < task.model.model_size:
                        self.next_worker_id[task.task_type[0]][task.task_id] = (self.next_worker_id[task.task_type[0]][task.task_id] + 1) % len(self.simulation.workers)
            else:
                # don't choose worker without the required model
                while task.model and all(m.model_id != task.model.model_id for m in self.simulation.workers[self.next_worker_id[task.task_type[0]][task.task_id]].GPU_state.placed_models(current_time)):
                    self.next_worker_id[task.task_type[0]][task.task_id] = (self.next_worker_id[task.task_type[0]][task.task_id] + 1) % len(self.simulation.workers)
            task.ADFG[task.task_id] = self.next_worker_id[task.task_type[0]][task.task_id]
            task.job.ADFG[task.task_id] = task.ADFG
            self.next_worker_id[task.task_type[0]][task.task_id] = (self.next_worker_id[task.task_type[0]][task.task_id] + 1) % len(self.simulation.workers)