import numpy as np

from core.data_models.workflow import Workflow


class NexusSLOSplitter:
    """SLO split algorithm adapted from pg. 330 (sec 6.2): 
    https://homes.cs.washington.edu/~arvind/papers/nexus.pdf
    """

    @classmethod
    def generate_task_slos(cls, time: float, arrival_rates: dict[int, float], workflow: Workflow, slo: float) -> dict[int, tuple[int, int]]:
        """Split job SLO over workflow tasks.

        Args:
            time: Current simulation time in ms
            arrival_rates: model ID -> request arrival rate per instance (QPS)
            workflow: Workflow to split SLO for
            slo: Job-level SLO to split across tasks
        
        Returns:
            task_slos: (SLO, max batch size) for each task ID in workflow
        """
        
        if any(m.id not in arrival_rates for m in workflow.get_models()):
            return {}

        # granularity of SLOs in ms
        TIME_STEP = 5

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
                [(k, _min_gpu_single(final_task.model_data, arrival_rates[final_task.model_data.id], k)) 
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
                          _min_gpu_single(task.model_data, arrival_rates[task.model_data.id], k) + \
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