from workers.worker import *
from workers.taskworker import *

from core.network import *
from core.events.base import *
from core.events.centralized_scheduler_events import *
from core.events.worker_events import *


class ShepherdWorker(TaskWorker):
    
    def free_slot(self, current_time, batch: Batch):
        """ Attempts to launch another task. """
        events = super().free_slot(current_time, batch)
        events += self.simulation.scheduler.flex_schedule_on_batch_completion(
            self, batch, current_time)
        return events
    
    def maybe_start_batch(self, batch, current_time):
        events = super().maybe_start_batch(batch, current_time)
        if not events:
            return [EventOrders(
                current_time + CPU_to_CPU_delay(batch.size()*batch.tasks[0].input_size),
                BatchRejectionAtWorker(self.simulation, self, batch))]
        return events
    
    def preempt_batch(self, old_batch_id: int, new_batch: Batch, current_time: float):
        evicted_batch = self.evict_batch(old_batch_id, current_time)

        if self.can_run_task(current_time, new_batch.model) == self._CAN_RUN_ON_EVICT:
            current_time += self.evict_models_from_GPU_until(
                current_time, new_batch.model.model_size, self.FCFS_EVICTION)

        events, _ = self.batch_execute(new_batch, current_time)
        assert(events)
        events.append(EventOrders(
            current_time + CPU_to_CPU_delay(evicted_batch.tasks[0].input_size * evicted_batch.size()), 
            TasksArrivalAtScheduler(self.simulation, evicted_batch.tasks)))
        return events

    #  ---------------------------  Subsequent TASK Transfer   --------------------

    def send_results_to_next_workers(self, current_time: float, batch: Batch) -> list:
        """
        Send the result of a task to the next worker in the inference pipeline (it may be the same worker)
        """
        new_tasks = []
        for task in batch.tasks:
            new_tasks += task.job.newly_available_tasks(task)
        if len(new_tasks) > 0:
            return [EventOrders(
                current_time + CPU_to_CPU_delay(task.result_size), 
                TasksArrivalAtScheduler(self.simulation, new_tasks))]
        return []