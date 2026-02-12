from core.data_models.model_data import ModelData
from core.data_models.task_data import TaskData
from core.allocation import ModelAllocation


class Workflow:
    """
    Represents a DAG pipeline.
    """

    def __init__(self, workflow_cfg, models, slo_granularity):
        self.id = workflow_cfg["JOB_TYPE"]
        
        self.initial_tasks: list[TaskData] = []
        self.tasks = {}
        for cfg in workflow_cfg["TASKS"]:
            task = TaskData(
                cfg["TASK_INDEX"],
                models[cfg["MODEL_ID"]],
                cfg["INPUT_SIZE"],
                cfg["OUTPUT_SIZE"],
                cfg["MAX_WAIT_TIME"],
                cfg["MAX_EMIT_BATCH_SIZE"],
                cfg["SLO"] if slo_granularity == "TASK" else None)
            
            self.tasks[task.id] = task

            if len(cfg["PREV_TASK_INDEX"]) == 0:
                self.initial_tasks.append(task)
        
        for cfg in workflow_cfg["TASKS"]:
            task = self.tasks[cfg["TASK_INDEX"]]
            task.prev_tasks = [self.tasks[id] 
                               for id in cfg["PREV_TASK_INDEX"]]
            task.next_tasks = [self.tasks[id]
                               for id in cfg["NEXT_TASK_INDEX"]]
            
    def get_models(self) -> list[ModelData]:
        """
        Returns all models used by any task in workflow.
        """
        return list(set([task.model_data for task in self.tasks.values() if task.model_data]))
    
    def get_max_batch_sizes(self, allocation: ModelAllocation, arrival_rate: float) -> dict[int, int]:
        throughputs = {} # task ID -> max throughput
        max_batch_sizes = {}
        completed_tasks = set()
        available_tasks = [t for t in self.initial_tasks]
        remaining_tasks = [t for t in self.tasks.values()]

        # 1) best achievable throughput given arrival rate per step
        # 2) 

        while remaining_tasks:
            for task in available_tasks:
                model_count = allocation.count(task.model_data.id)
                
                total_input_rate = min(throughputs[t.id] for t in task.prev_tasks) \
                    if task.prev_tasks else arrival_rate
                input_rate = total_input_rate / model_count
                refill_time = 1 / input_rate * 1000

                for bsize in range(allocation.models[task.model_data.id].max_batch_size, -1, -1):
                    bexec_time = allocation.models[task.model_data.id].batch_exec_times[24][bsize]
                    if bexec_time / refill_time < bsize: # doesn't refill fast enough to regularly run this size
                        break
                
                

                
                expected_bsize = max(b for b in range(1, allocation.models[task.model_data.id].max_batch_size+1, 1) 
                                        if b == 1 or input_rate >= b / task.model_data.batch_exec_times[24][b] * 1000)
                throughputs[task.id] = min(input_rate, expected_bsize / task.model_data.batch_exec_times[24][expected_bsize] * 1000) * model_count
                
                print("IN: ", input_rate, " OUT: ", throughputs[task.id], " COUNT: ", model_count)

                remaining_tasks.remove(task)
                completed_tasks.add(task.id)

            max_throughput = min(throughputs[t.id] for t in available_tasks)

            available_tasks = []
            for task in remaining_tasks:
                if all(t.id in completed_tasks for t in task.prev_tasks):
                    available_tasks.append(task)  

    def get_min_processing_time(self) -> float:
        return self.get_processing_time(
            lambda t: t.model_data.batch_exec_times[24][1])

    def get_processing_time(self, get_exec_time) -> float:
        dependencies: dict[int, set[int]] = {}
        dependents: dict[int, set[int]] = {}
        available_tasks: list[TaskData] = []
        for task in self.tasks.values():
            dependencies[task.id] = set([t.id for t in task.prev_tasks])
            dependents[task.id] = set([t.id for t in task.next_tasks])
            
            if len(dependencies[task.id]) == 0:
                available_tasks.append(task)
        
        max_cum_processing_time = 0
        while available_tasks:
            next_available_tasks = []
            max_cum_processing_time += max([
                get_exec_time(task) for task in available_tasks
            ])
            for task in available_tasks:
                for dep in dependents[task.id]:
                    dependencies[dep].remove(task.id)
                    if len(dependencies[dep]) == 0:
                        next_available_tasks.append(self.tasks[dep])

            available_tasks = next_available_tasks
        return max_cum_processing_time