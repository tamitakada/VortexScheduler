from core.model import Model
from core.task import Task
from core.configs.workflow_config import *
import core.configs.gen_config as gcfg


class AbstractTask:
    """
    Represents one node in a DAG pipeline.
    """

    def __init__(self, id: int, model: Model | None, input_size: float, output_size: float,
                 max_wait_time: float, max_emit_batch_size: int, slo: float | None):

        self.id = id
        self.model = model
        self.input_size = input_size
        self.output_size = output_size
        self.max_wait_time = max_wait_time
        self.max_emit_batch_size = max_emit_batch_size
        self.slo = slo

        self.prev_tasks: list[AbstractTask] = []
        self.next_tasks: list[AbstractTask] = []

    def __eq__(self, value):
        if type(value) != AbstractTask:
            return False
        
        if value.id != self.id:
            return False
        
        if value.model.model_id != self.model.model_id:
            return False
        
        return True
    
    def create_task(self, job) -> Task:
        task = Task(job,
                    job.id,
                    self.id,  # taskID
                    (job.job_type_id, self.id), # task type
                    0, 
                    self.model, 
                    self.input_size,
                    self.output_size,
                    self.model.max_batch_size if self.model else 1,
                    self.max_wait_time,
                    self.slo if gcfg.SLO_GRANULARITY == "TASK" else 0,
                    self.max_emit_batch_size)
        
        for prev_task in self.prev_tasks:
            task.required_task_ids.append(prev_task.id)
        
        for next_task in self.next_tasks:
            task.next_task_ids.append(next_task.id)

        return task


class Workflow:
    """
    Represents a DAG pipeline.
    """

    def __init__(self, simulation, workflow_cfg):
        self.id = workflow_cfg["JOB_TYPE"]
        
        self.initial_tasks = []
        self.tasks = {}
        for cfg in workflow_cfg["TASKS"]:
            task = AbstractTask(
                cfg["TASK_INDEX"],
                simulation.get_model_from_id(cfg["MODEL_ID"]),
                cfg["INPUT_SIZE"],
                cfg["OUTPUT_SIZE"],
                cfg["MAX_WAIT_TIME"],
                cfg["MAX_EMIT_BATCH_SIZE"],
                cfg["SLO"] if gcfg.SLO_GRANULARITY == "TASK" else None)
            
            self.tasks[task.id] = task

            if len(cfg["PREV_TASK_INDEX"]) == 0:
                self.initial_tasks.append(task)
        
        for cfg in workflow_cfg["TASKS"]:
            task = self.tasks[cfg["TASK_INDEX"]]
            task.prev_tasks = [self.tasks[id] 
                               for id in cfg["PREV_TASK_INDEX"]]
            task.next_tasks = [self.tasks[id]
                               for id in cfg["NEXT_TASK_INDEX"]]
            
    def get_models(self) -> list[Model]:
        """
        Returns all models used by any task in workflow.
        """
        return [task.model for task in self.tasks.values() if task.model]
    
    def get_processing_time(self, get_exec_time) -> float:
        dependencies: dict[int, list[AbstractTask]] = {}
        dependents: dict[int, list[AbstractTask]] = {}
        available_tasks: list[AbstractTask] = self.initial_tasks

        for task in self.tasks.values():
            dependencies[task.id] = task.prev_tasks
            dependents[task.id] = task.next_tasks
        
        max_cum_processing_time: float = 0
        while available_tasks:
            next_available_tasks = []
            max_cum_processing_time += max([
                get_exec_time(task) for task in available_tasks
            ])
            for task in available_tasks:
                for dep in dependents[task.id]:
                    dependencies[dep.id].remove(task)
                    if len(dependencies[dep.id]) == 0:
                        next_available_tasks.append(dep)

            available_tasks = next_available_tasks
        return max_cum_processing_time