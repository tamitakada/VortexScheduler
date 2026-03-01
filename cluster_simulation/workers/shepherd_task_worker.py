from workers.worker import *
from workers.taskworker import *

from core.network import *
from core.events.base import *
from core.events.centralized_scheduler_events import *
from core.events.worker_events import *


class ShepherdWorker(TaskWorker):
    
    def free_slot(self, current_time, batch: Batch):
        """ Attempts to launch another task. """
        instance_id = None
        for state in self.GPU_state.state_at(current_time):
            if state.reserved_batch == batch:
                instance_id = state.model.id
                break
        assert(instance_id)

        events = super().free_slot(current_time, batch)
        events += self.simulation.scheduler.schedule_on_batch_completion(
            self, instance_id, batch, current_time)
        return events
    
    def preempt_batch(self, instance_id, old_batch_id: int, new_batch: Batch, current_time: float):
        instance_state = self.GPU_state.get_instance_state(instance_id, current_time)
        assert(instance_state.reserved_batch == None or instance_state.reserved_batch.id == old_batch_id)
        
        evicted_batch = self.evict_batch(old_batch_id, current_time)
        events, _ = self.batch_execute(new_batch, current_time, instance_id=instance_id)
        assert(events)
        
        events.append(EventOrders(
            current_time, 
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