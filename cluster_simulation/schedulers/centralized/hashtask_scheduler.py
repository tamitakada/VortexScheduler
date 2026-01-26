import core.configs.gen_config as gcfg
from core.events.base import *
from core.events.centralized_scheduler_events import *
from core.events.worker_events import *

from schedulers.centralized.scheduler import Scheduler

import pandas as pd


class HashTaskScheduler(Scheduler):
    """
        Scheduler class for a round robin centralized task scheduler.
    """

    def __init__(self, simulation, herd_assignment=None):
        super().__init__(simulation, herd_assignment)

        self.last_worker_idx = {}

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
            task_arrival_time = current_time + CPU_to_CPU_delay(task.input_size)
            worker_index = task.ADFG[task.task_id]
            task_arrival_events.append(EventOrders(
                task_arrival_time, TaskArrival(self.simulation, self.simulation.workers[worker_index], task, task.job.id)))

        return task_arrival_events
    
    def _assign_adfg(self, tasks, current_time):
        for task in tasks:
            candidate_worker_idx = 0
            if task.model_data.id in self.last_worker_idx:
                candidate_worker_idx = (self.last_worker_idx[task.model_data.id] + 1) % len(self.simulation.worker_ids_by_creation)

            candidate_worker_id = self.simulation.worker_ids_by_creation[candidate_worker_idx]

            # don't choose worker without the required model
            while task.model_data and \
                all(s.model.data.id != task.model_data.id for s in
                    self.simulation.workers[candidate_worker_id].GPU_state.state_at(current_time)):

                candidate_worker_idx = (candidate_worker_idx + 1) % len(self.simulation.worker_ids_by_creation)
                candidate_worker_id = self.simulation.worker_ids_by_creation[candidate_worker_idx]

            task.ADFG[task.task_id] = candidate_worker_id
            task.job.ADFG[task.task_id] = task.ADFG

            self.last_worker_idx[task.model_data.id] = candidate_worker_idx