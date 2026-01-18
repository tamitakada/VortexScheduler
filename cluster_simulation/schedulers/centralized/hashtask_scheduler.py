import core.configs.gen_config as gcfg
from core.events.base import *
from core.events.centralized_scheduler_events import *
from core.events.worker_events import *

from schedulers.centralized.scheduler import Scheduler
# from schedulers.algo.inferline_planner_algo import Inferline

import pandas as pd


class HashTaskScheduler(Scheduler):
    """
        Scheduler class for a round robin centralized task scheduler.
    """

    def __init__(self, simulation, herd_assignment=None):
        super().__init__(simulation, herd_assignment)
        self.next_worker_id = { jt: [0 for _ in get_task_types([jt])] for jt in self.simulation.job_types_list }
        # self.last_scale = 0

    def schedule_job_on_arrival(self, job, current_time):
        super().schedule_job_on_arrival(job, current_time)

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
        super().schedule_tasks_on_arrival(tasks, current_time)

        task_arrival_events = []

        self._assign_adfg(tasks, current_time)
        
        for task in tasks:
            self.arrived_task_log.loc[len(self.arrived_task_log)] = [current_time, task.job.id, task.task_id]

            task_arrival_time = current_time + CPU_to_CPU_delay(task.input_size)
            worker_index = task.ADFG[task.task_id]
            task_arrival_events.append(EventOrders(
                task_arrival_time, TaskArrival(self.simulation, self.simulation.workers[worker_index], task, task.job.id)))

        return task_arrival_events
    
    def _assign_adfg(self, tasks, current_time):
        for task in tasks:
            # don't choose worker without the required model
            while task.model and all(m.model_id != task.model.model_id for m in self.simulation.workers[self.next_worker_id[task.task_type[0]][task.task_id]].GPU_state.placed_models(current_time)):
                self.next_worker_id[task.task_type[0]][task.task_id] = (self.next_worker_id[task.task_type[0]][task.task_id] + 1) % len(self.simulation.workers)
            
            task.ADFG[task.task_id] = self.next_worker_id[task.task_type[0]][task.task_id]
            task.job.ADFG[task.task_id] = task.ADFG
            self.next_worker_id[task.task_type[0]][task.task_id] = (self.next_worker_id[task.task_type[0]][task.task_id] + 1) % len(self.simulation.workers)