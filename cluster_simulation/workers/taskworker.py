from workers.worker import *

from core.network import *
from core.batch import Batch

from core.events.base import *
from core.events.centralized_scheduler_events import *
from core.events.worker_events import *


class TaskWorker(Worker):
    def __init__(self, simulation, id, total_memory, group_id=-1, created_at=0):
        super().__init__(simulation, id, total_memory, group_id=group_id, created_at=created_at)
        self.involved = False

    def add_task(self, current_time, task):
        """
        Add task into the local task queue
        """
        raise NotImplementedError()

    def free_slot(self, current_time, batch: Batch):
        """ Attempts to launch another task. """
        assert(not self.did_abandon_batch(batch.id))

        # finished batch clean up & logging
        if batch.model_data != None:
            self.GPU_state.release_busy_model(batch.id, current_time)

        for task in batch.tasks:
            self.simulation.add_job_completion_time(
                task.job.id, task.task_id, current_time)
            task.log.task_execution_end_timestamp = current_time

            self.simulation.add_task_exec_to_worker_metrics(
                task, self, task.get_task_deadline())

        self.simulation.batch_exec_log.loc[len(self.simulation.batch_exec_log)] = {
            "batch_id": batch.id,
            "start_time": batch.tasks[0].log.task_execution_start_timestamp,
            "end_time": current_time,
            "worker_id": self.id,
            "model_id": batch.model_data.id,
            "batch_size": batch.size(),
            "job_ids": batch.job_ids
        }

        events = self.send_results_to_next_workers(current_time, batch)
        return events

    #  ---------------------------  TASK EXECUTION  ----------------------

    def maybe_start_batch(self, batch: Batch, current_time: float) -> list[EventOrders]:
        """Start batch if exists a fully fetched idle copy of required model.
        """
        # TODO: future feat: handle can run on evict

        if any(s.state == ModelState.PLACED and s.model.data.id == batch.model_data.id and not s.reserved_batch
               for s in self.GPU_state.state_at(current_time)):
            
            batch_end_events, task_end_time = self.batch_execute(batch, current_time)
            return batch_end_events
        
        return []

    def batch_execute(self, batch: Batch, current_time: float):
        """
            Fetches a new copy or reserves an idle copy of any required GPU models
            and executes the batch [tasks]. Returns a list containing the 
            BatchEndEvent and the batch execution end time.
        """
        self.involved = True

        for task in batch.tasks:
            task.executing_worker_id = self.id

        batch_exec_time = batch.model_data.get_randomized_exec_time(batch.size(), self.total_memory)
        model_fetch_time = 0

        if batch.model_data != None:
            if self.GPU_state.does_have_idle_copy(batch.model_data.id, current_time):
                reserved_instance_id = self.GPU_state.reserve_idle_copy(
                   batch.model_data, current_time, batch, batch_exec_time)

                # verify reserved instance
                reserved_state = [s for s in self.GPU_state.state_at(current_time)
                                  if s.model.id == reserved_instance_id][0]
                assert(reserved_state.reserved_until >= (current_time + batch_exec_time))
                assert(reserved_state.reserved_batch == batch)
                assert(reserved_state.model.data.id == batch.model_data.id)
                assert(reserved_state.state == ModelState.PLACED)

                batch_exec_time = reserved_state.reserved_until - current_time
            else:
                assert(False) # not allowing currently
                model_fetch_time = self.fetch_model(
                    batch.model, batch, current_time, exec_time=batch_exec_time)
        
        for task in batch.tasks:
            task.log.task_front_queue_timestamp = current_time
            task.log.task_execution_start_timestamp = current_time + model_fetch_time

        task_end_time = current_time + model_fetch_time + batch_exec_time + \
            SameMachineCPUtoGPU_delay(batch.tasks[0].input_size * batch.size()) + \
            SameMachineGPUtoCPU_delay(batch.tasks[0].result_size * batch.size())
        task_end_events = []

        task_end_events.append(EventOrders(current_time, BatchStartEvent(
            self, batch.model_data.id, batch_id=batch.id, job_ids=batch.job_ids)))
        task_end_events.append(EventOrders(task_end_time, BatchEndEvent(
            self, batch, job_ids=batch.job_ids)))
        return task_end_events, task_end_time

    #  ---------------------------  Subsequent TASK Transfer   --------------------

    def send_result_to_next_workers(self, current_time, task) -> list:
        """
        Send the result of a task to the next worker in the inference pipeline (it may be the same worker)
        """
        raise NotImplementedError()

    def get_task_queue_waittime(self, current_time, task_type, info_staleness=0, requiring_worker_id=None):
        if requiring_worker_id != None and requiring_worker_id != self.id:
            info_staleness = 0

        task_model_id = WORKFLOW_LIST[task_type[0]]["TASKS"][task_type[1]]["MODEL_ID"]
        if task_model_id < 0:
            return 0
        
        task_model_states = list(filter(lambda s: s.model.model_id == task_model_id, 
                                        self.GPU_state.placed_model_states(current_time)))
        if len(task_model_states) == 0:
            return np.inf

        if self.GPU_state.does_have_idle_copy(task_model_states[0].model, current_time):
            return 0
        
        return min(s.reserved_until for s in task_model_states) - current_time