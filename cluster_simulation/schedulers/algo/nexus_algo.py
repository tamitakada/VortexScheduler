import numpy as np

from core.data_models.workflow import Workflow


class NexusSLOSplitter:
    """SLO split algorithm adapted from pg. 330 (sec 6.2): 
    https://homes.cs.washington.edu/~arvind/papers/nexus.pdf
    """

    @classmethod
    def generate_task_slos(cls, time: float, measurement_interval: float, 
                           simulation, workflow: Workflow, slo: float) -> dict[int, tuple[int, int]]:
        """Split job SLO over workflow tasks.

        Args:
            time: Current simulation time in ms
            measurement_interval: Length of interval to consider arrivals over (ms)
            simulation: Current simulation
            workflow: Workflow to split SLO for
            slo: Job-level SLO to split across tasks
        
        Returns:
            task_slos: (SLO, max batch size) for each task ID in workflow
        """

        # granularity of SLOs in ms
        TIME_STEP = 5

        task_model_arrival_rates = {}
        for task in workflow.tasks.values():
            arrived_job_count = simulation.task_arrival_log[\
                (simulation.task_arrival_log["model_id"]==task.model_data.id) & \
                (simulation.task_arrival_log["time"] <= time) & \
                (simulation.task_arrival_log["time"] > (time - measurement_interval))]["job_id"].nunique()
            num_model_workers = len([w for w in simulation.workers.values() 
                                     if any(s.model.data.id == task.model_data.id for s in w.GPU_state.state_at(time))])
            task_model_arrival_rates[task.id] = arrived_job_count / num_model_workers / measurement_interval * 1000 

        # task id -> task SLO -> (min # gpus, max batch size, (SLO for curr task, SLO for subtree))
        min_gpus = {task.id: {} for task in workflow.tasks.values()}
        
        # base case: exit points / leaf nodes
        final_tasks = [t for t in workflow.tasks.values() if len(t.next_tasks) == 0]
        assert(len(final_tasks) == 1) # NOTE: algorithm is for fork-join graphs
        final_task = final_tasks[0]

        def _min_gpu_single(model, req_rate, k):
            bsizes = [b for b in range(1, model.max_batch_size + 1) if model.batch_exec_times[24][b] <= k]
            if not bsizes:
                return np.inf
            return min([req_rate * model.batch_exec_times[24][bsize] / bsize / 1000 for bsize in bsizes])

        for t in range(TIME_STEP, slo + 1, TIME_STEP):
            min_gpus[final_task.id][t] = min(
                [(k, _min_gpu_single(final_task.model_data, task_model_arrival_rates[final_task.id], k)) 
                 for k in range(TIME_STEP, t + 1, TIME_STEP)],
                key=lambda x: x[1])
        
        # reverse traverse tree to find remaining SLO splits
        computed_task_ids = set([final_task.id])
        rem_tasks = [t for t in final_task.prev_tasks if all(nt.id in computed_task_ids for nt in t.next_tasks)]
        while rem_tasks:
            for task in rem_tasks:
                for t in range(TIME_STEP, slo + 1, TIME_STEP):
                    min_gpus[task.id][t] = min(
                        [(k, 
                          _min_gpu_single(task.model_data, task_model_arrival_rates[task.id], k) + \
                          (np.inf if t - k < TIME_STEP else
                           min(sum(min_gpus[v.id][t_prime][1] for v in task.next_tasks)
                               for t_prime in range(TIME_STEP, t - k + 1, TIME_STEP))))
                         for k in list(range(TIME_STEP, t + 1, TIME_STEP))],
                        key=lambda x: x[1])
                computed_task_ids.add(task.id)
            rem_tasks = set([pt for t in rem_tasks for pt in t.prev_tasks 
                             if pt.id not in computed_task_ids and all(pt_nt.id in computed_task_ids for pt_nt in pt.next_tasks)])

        slos = {}
        
        def _traverse_slo_tree(tasks, subtree_slo):
            for task in tasks:
                slo = min_gpus[task.id][subtree_slo // TIME_STEP * TIME_STEP][0]
                slos[task.id] = (slo,
                                 max([b for b in range(1, task.model_data.max_batch_size + 1)
                                      if task.model_data.batch_exec_times[24][b] <= slo]))
                if task.next_tasks:
                    _traverse_slo_tree([t for t in task.next_tasks if all(pt.id in slos for pt in t.prev_tasks)], 
                                       subtree_slo - slos[task.id][0])
        
        _traverse_slo_tree(workflow.initial_tasks, slo)
        
        return slos

    @classmethod
    def redistribute_task_slos(cls, time: float, simulation, workflow: Workflow, slos: dict[int, int], realloc_amt_ms: float):
        last_slo_update_time = simulation.scheduler.task_slo_log[simulation.scheduler.task_slo_log["workflow_id"]==workflow.id]["time"].max()
        if time - last_slo_update_time < 1000:
            return # at least 1s must pass from last update

        drop_df = simulation.task_drop_log[(simulation.task_drop_log["workflow_id"]==workflow.id) & \
                                                (simulation.task_drop_log["drop_time"] <= time) & \
                                                (simulation.task_drop_log["drop_time"] > last_slo_update_time)]
        
        task_drop_rates = {}
        for task in workflow.tasks.values():
            drop_rate = (drop_df["task_id"]==task.id).sum() / (time - last_slo_update_time) * 1000
            task_drop_rates[task.id] = drop_rate

        print("WF DROP RATES: ", task_drop_rates)

        MAX_ALLOWABLE_DROP_RATE = 2

        # TODO
        if workflow.id == 1:
            unsat_tasks = sorted([task for task in workflow.tasks.values() if task_drop_rates[task.id] > MAX_ALLOWABLE_DROP_RATE],
                             key=lambda task: task_drop_rates[task.id],
                             reverse=True)
            
            sat_tasks = sorted([task for task in workflow.tasks.values() if task not in unsat_tasks],
                    key=lambda task: slos[task.id][0],
                    reverse=True)
        elif workflow.id == 4:
            unsat_tasks = sorted([task for task in workflow.tasks.values() if task.id not in [3,4] and task_drop_rates[task.id] > MAX_ALLOWABLE_DROP_RATE],
                             key=lambda task: task_drop_rates[task.id] if task.id != 2 else max(task_drop_rates[3] + task_drop_rates[4], task_drop_rates[2]),
                             reverse=True)
            
            sat_tasks = sorted([task for task in workflow.tasks.values() if task not in unsat_tasks and task.id not in [3,4]],
                    key=lambda task: slos[task.id][0],
                    reverse=True)
        elif workflow.id == 5 or workflow.id == 0:
            unsat_tasks = sorted([task for task in workflow.tasks.values() if task.id != 0 and task_drop_rates[task.id] > MAX_ALLOWABLE_DROP_RATE],
                                key=lambda task: task_drop_rates[task.id] if task.id != 1 else max(task_drop_rates[0], task_drop_rates[1]),
                                reverse=True)
            
            # tasks with drop rate < threshold in desc magnitude of SLO
            sat_tasks = sorted([task for task in workflow.tasks.values() if task not in unsat_tasks and task.id != 0],
                            key=lambda task: slos[task.id][0],
                            reverse=True)
        else:
            raise NotImplementedError()

        if not unsat_tasks and workflow.id != 4:
            return
        
        if not sat_tasks and workflow.id != 4:
            return
        
        if workflow.id == 4:
            if not sat_tasks and task_drop_rates[3] > MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] > MAX_ALLOWABLE_DROP_RATE:
                return
            
            if not unsat_tasks and task_drop_rates[3] <= MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] <= MAX_ALLOWABLE_DROP_RATE:
                return
        
        for sat_task in sat_tasks:
            if slos[sat_task.id][0] < realloc_amt_ms:
                continue
            
            new_sat_task_slo = slos[sat_task.id][0] - realloc_amt_ms
            if sat_task.model_data.batch_exec_times[24][1] > new_sat_task_slo:
                continue

            slos[sat_task.id] = (new_sat_task_slo,
                 max([bsize for bsize in range(1,sat_task.model_data.max_batch_size+1) 
                      if sat_task.model_data.batch_exec_times[24][bsize] <= new_sat_task_slo]))
            
            unsat_task = unsat_tasks.pop(0)

            new_unsat_task_slo = slos[unsat_task.id][0] + realloc_amt_ms
            slos[unsat_task.id] = \
                (new_unsat_task_slo,
                 max([bsize for bsize in range(1,unsat_task.model_data.max_batch_size+1) 
                      if unsat_task.model_data.batch_exec_times[24][bsize] <= new_unsat_task_slo]))
            
            # TODO
            if workflow.id in [0, 5]:
                if sat_task.id == 1:
                    task_0 = [t for t in workflow.tasks.values() if t.id ==0][0]
                    slos[0] = \
                        (new_sat_task_slo,
                        max([bsize for bsize in range(1,task_0.model_data.max_batch_size+1) 
                            if task_0.model_data.batch_exec_times[24][bsize] <= new_sat_task_slo]))
                
                if unsat_task.id == 1:
                    task_0 = [t for t in workflow.tasks.values() if t.id ==0][0]
                    slos[0] = \
                        (new_unsat_task_slo,
                        max([bsize for bsize in range(1,task_0.model_data.max_batch_size+1) 
                            if task_0.model_data.batch_exec_times[24][bsize] <= new_unsat_task_slo]))
                    
            elif workflow.id == 4:
                if sat_task.id == 2:
                    task_3 = [t for t in workflow.tasks.values() if t.id ==3][0]
                    slo3 = slos[3][0] - 2.5
                    slos[3] = \
                        (slo3,
                        max([bsize for bsize in range(1,task_3.model_data.max_batch_size+1) 
                            if task_3.model_data.batch_exec_times[24][bsize] <= slo3]))
                    
                    task_4 = [t for t in workflow.tasks.values() if t.id ==4][0]
                    slo4 = slos[4][0] - 2.5
                    slos[4] = \
                        (slo4,
                        max([bsize for bsize in range(1,task_4.model_data.max_batch_size+1) 
                            if task_4.model_data.batch_exec_times[24][bsize] <= slo4]))

                if unsat_task.id == 2:
                    task_3 = [t for t in workflow.tasks.values() if t.id ==3][0]
                    slo3 = slos[3][0] + 2.5
                    slos[3] = \
                        (slo3,
                        max([bsize for bsize in range(1,task_3.model_data.max_batch_size+1) 
                            if task_3.model_data.batch_exec_times[24][bsize] <= slo3]))
                    
                    task_4 = [t for t in workflow.tasks.values() if t.id ==4][0]
                    slo4 = slos[4][0] + 2.5
                    slos[4] = \
                        (slo4,
                        max([bsize for bsize in range(1,task_4.model_data.max_batch_size+1) 
                            if task_4.model_data.batch_exec_times[24][bsize] <= slo4]))
        
            if not unsat_tasks:
                break

        if workflow.id == 4:
            if task_drop_rates[3] <= MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] <= MAX_ALLOWABLE_DROP_RATE:
                return

            if task_drop_rates[3] > MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] > MAX_ALLOWABLE_DROP_RATE:
                return
            
            if task_drop_rates[3] > MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] <= MAX_ALLOWABLE_DROP_RATE:

                slo3 = slos[3][0] + realloc_amt_ms
                slo4 = slos[4][0] - realloc_amt_ms

            if task_drop_rates[3] <= MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] > MAX_ALLOWABLE_DROP_RATE:

                slo3 = slos[3][0] - realloc_amt_ms
                slo4 = slos[4][0] + realloc_amt_ms

            task_3 = [t for t in workflow.tasks.values() if t.id ==3][0]
            task_4 = [t for t in workflow.tasks.values() if t.id ==4][0]
            
            slos[3] = \
                (slo3,
                max([bsize for bsize in range(1, task_3.model_data.max_batch_size+1) 
                    if task_3.model_data.batch_exec_times[24][bsize] <= slo3]))
            slos[4] = \
                (slo4,
                max([bsize for bsize in range(1, task_4.model_data.max_batch_size+1) 
                    if task_4.model_data.batch_exec_times[24][bsize] <= slo4]))