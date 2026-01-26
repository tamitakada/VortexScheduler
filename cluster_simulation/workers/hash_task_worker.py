from collections import defaultdict

from workers.taskworker import *
from workers.worker import *

import core.configs.gen_config as gcfg
from core.configs.workflow_config import *
from core.network import *
from core.events.base import *
from core.events.centralized_scheduler_events import *
from core.events.worker_events import *
from core.batch import Batch

from schedulers.algo.batching_policies import get_batch


class HashTaskWorker(TaskWorker):
    def __init__(self, simulation, id, total_memory, group_id=-1, created_at=0):
        super().__init__(simulation, id, total_memory, group_id=group_id, created_at=created_at)
        # {task_obj1:[(preq_task_id0,arrival_time0), (preq_taks_id0, arrival_time1), ...], task2:[( ...],}
        self.waiting_tasks_buffer = defaultdict(lambda: [])
        # keep track of the queue information at time:  [ (time1,[task0,task1,]), (time2,[task1,...]),...]
        self.queue_history = {}
        self.involved = False

        # round robin tracking
        self.next_task_type_idx = 0
        self.last_worker_idx = {}

    def evict_models(self, time: float, models: list[ModelData]):
        # TODO: batched send
        model_ids = [m.id for m in models]
        events = []
        for model_id in model_ids:
            evicted_batch = self.evict_model(time, model_id)
            if evicted_batch:
                events.append(EventOrders(
                    time + CPU_to_CPU_delay(sum(t.input_size for t in evicted_batch.tasks)), 
                    TasksArrivalAtScheduler(self.simulation, evicted_batch.tasks)))
        
        for model_id in set(model_ids):
            if not self.GPU_state.does_have_copy(model_id, time):
                task_queue = self.get_queue_history(time, model_id, info_staleness=0)
                if task_queue:
                    events.append(EventOrders(
                        time + CPU_to_CPU_delay(sum(t.input_size for t in task_queue)), 
                        TasksArrivalAtScheduler(self.simulation, task_queue)))
                
        return events
    
    def add_tasks(self, current_time, tasks):
        """
        Add tasks into the local task queue
        """
        if len(tasks) == 0:
            return []
        
        assert(all(task.model_data == None or \
                   any(s.model.data.id == task.model_data.id for s in self.GPU_state.state_at(current_time))
                   for task in tasks))

        # add tasks to worker queue
        for task in tasks:
            self.add_task_to_queue_history(task, current_time)

        # if concurrency disabled and occupied, do nothing
        if (not gcfg.ENABLE_MULTITHREADING) and \
            any(s.reserved_batch and s.state == ModelState.PLACED for s in self.GPU_state.state_at(current_time)):
            
            return []

        events = []

        model_ids = list(set(task.model_data.id for task in tasks))
        while True:
            no_batches_launched = True

            for model_id in model_ids:
                new_events = self.check_task_queue(model_id, current_time)
                events += new_events
                
                if len(new_events) > 0:
                    # only start at most 1 batch if singlethreaded
                    if not gcfg.ENABLE_MULTITHREADING:
                        assert(len([s for s in self.GPU_state.state_at(current_time) 
                                    if s.reserved_batch and s.state == ModelState.PLACED]) == 1)
                        return events

                    # otherwise keep checking queues until no more batches are launched
                    no_batches_launched = False

            if no_batches_launched:
                return events

    def free_slot(self, current_time, batch: Batch):
        """ Attempts to launch another task. """
        events = super().free_slot(current_time, batch)

        # start next batch
        if gcfg.ENABLE_MULTITHREADING:
            while True:
                new_events = self.check_task_queue(batch.model_data.id, current_time)
                events += new_events
                # keep checking queue until no more batches are launched
                if len(new_events) == 0:
                    return events
        else:
            # try launching tasks round robin until models have been exhausted
            all_mids = self.queue_history.keys()
            mids_left = len(all_mids)
            batch_end_events = []
            while not batch_end_events and mids_left:
                if all_mids[self.next_task_type_idx] in self.queue_history:
                    batch_end_events = self.check_task_queue(all_mids[self.next_task_type_idx], current_time)
                    events += batch_end_events
                self.next_task_type_idx = (self.next_task_type_idx + 1) % len(all_mids)
                mids_left -= 1
        return events

    #  --------------------------- DECENTRALIZED WORKER SCHEDULING  ----------------------
    
    def check_task_queue(self, model_id: int, current_time: float) -> list[EventOrders]:
        """Launch up to 1 new batch for a given model drawing from tasks in
        model queue.

        Args:
            model_id: ID of model to launch (not instance ID)
            current_time: ms since simulation start

        Returns:
            events: Batch start and end events if batch is launched,
            otherwise empty.
        """
        model_state = [s for s in self.GPU_state.state_at(current_time)
                       if s.model.data.id == model_id]
        
        assert(len(model_state) >= 1)

        if all(s.reserved_batch for s in model_state):
            return []

        # drop tasks if necessary
        task_queue = self.get_queue_history(current_time, model_id, info_staleness=0)
        for task in task_queue:
            # skip dropped tasks
            if (self.simulation.task_drop_log["job_id"]==task.job.id).any():
                self.rm_task_in_queue_history(task, current_time)
                continue

            if ((gcfg.DROP_POLICY == "LATEST_POSSIBLE" or gcfg.DROP_POLICY == "CLUSTER_ADMISSION_LIMIT") and current_time >= task.get_task_deadline()) or \
                (gcfg.DROP_POLICY == "OPTIMAL" and (current_time + task.job.get_min_remaining_processing_time()) >= task.get_task_deadline()):
                
                for job_task in task.job.tasks:
                    self.rm_task_in_queue_history(job_task, current_time)
                
                self.simulation.task_drop_log.loc[len(self.simulation.task_drop_log)] = {
                    "client_id": task.job.client_id,
                    "job_id": task.job.id, 
                    "workflow_id": task.task_type[0], 
                    "task_id": task.task_type[1],
                    "drop_time": current_time, 
                    "create_time": task.log.job_creation_timestamp,
                    "arrival_time": task.log.task_placed_on_worker_queue_timestamp,
                    "slo": task.slo if gcfg.SLO_GRANULARITY == "TASK" else task.job.slo, 
                    "deadline": task.get_task_deadline()
                }
                continue
        
        # get queue after drops
        task_queue = [task for task in self.get_queue_history(current_time, model_id, info_staleness=0)
                      if current_time >= task.log.task_placed_on_worker_queue_timestamp]
        
        assert((self.simulation.task_drop_log["job_id"]!=task.job.id).all() for task in task_queue)

        if len(task_queue) > 0:
            batch = get_batch(current_time, self.total_memory, task_queue)
            if batch:
                batch_events = self.maybe_start_batch(batch, current_time)
                if batch_events: # if did start batch
                    for task in batch.tasks:
                        self.rm_task_in_queue_history(task, current_time)
                return batch_events
        return []

    #  ---------------------------  Subsequent TASK Transfer   --------------------

    def send_results_to_next_workers(self, current_time: float, batch: Batch) -> list[EventOrders]:
        """For a given completed batch, if any new tasks are available to start
        execution, assign and send tasks to workers.

        Args:
            current_time: ms since simulation start
            batch: Batch completed on this worker

        Returns:
            events: TasksArrival events for any newly available tasks
        """
        events = []

        # TODO: diff emit sizes for diff workflows?
        emit_size = batch.tasks[0].max_emit_batch_size

        # model_id -> list[Task]
        ready_tasks_by_model = {}
        for task in batch.tasks:
            next_tasks = task.job.newly_available_tasks(task)
            for next_task in next_tasks:
                if next_task.model_data.id not in ready_tasks_by_model:
                    ready_tasks_by_model[next_task.model_data.id] = []
                ready_tasks_by_model[next_task.model_data.id].append(next_task)
        
        for mid, ready_tasks in ready_tasks_by_model.items():
            prev_curr = current_time
            for i in range(0, len(ready_tasks), emit_size):
                curr_send_batch = ready_tasks[i:(i+emit_size)]

                # choose worker to assign next DAG tasks to round robin
                candidate_worker_idx = 0
                if curr_send_batch[0].model_data.id in self.last_worker_idx:
                    candidate_worker_idx = (self.last_worker_idx[curr_send_batch[0].model_data.id] + 1) % len(self.simulation.worker_ids_by_creation)
                else:
                    self.last_worker_idx[curr_send_batch[0].model_data.id] = 0

                candidate_worker_id = self.simulation.worker_ids_by_creation[candidate_worker_idx]

                # don't choose worker without the required model
                while curr_send_batch[0].model_data and \
                    all(s.model.data.id != curr_send_batch[0].model_data.id for s in
                        self.simulation.workers[candidate_worker_id].GPU_state.state_at(current_time)):
                    
                    candidate_worker_idx = (candidate_worker_idx + 1) % len(self.simulation.worker_ids_by_creation)
                    candidate_worker_id = self.simulation.worker_ids_by_creation[candidate_worker_idx]

                self.last_worker_idx[curr_send_batch[0].model_data.id] = candidate_worker_idx

                transfer_delay = 0
                if candidate_worker_id != self.id:  # The next worker on the pipeline is NOT the same node
                    transfer_delay = CPU_to_CPU_delay(curr_send_batch[0].result_size * len(curr_send_batch))

                events.append(EventOrders(prev_curr + transfer_delay, TasksArrival(
                    self.simulation, self.simulation.workers[candidate_worker_id], curr_send_batch)))
                
                prev_curr = prev_curr + transfer_delay

        return events
    
    # ------------------------- queue history update helper functions ---------------

    def add_task_to_queue_history(self, task, current_time):
        # 0. Base case (first entry)
        if task.model_data.id not in self.queue_history:
            self.queue_history[task.model_data.id] = [(current_time, [task])]
            return

        # 1. Find the time_stamp place to add this queue information
        last_index = len(self.queue_history[task.model_data.id]) - 1
        while last_index >= 0:
            if self.queue_history[task.model_data.id][last_index][0] == current_time:
                if task not in self.queue_history[task.model_data.id][last_index][1]:
                    self.queue_history[task.model_data.id][last_index][1].append(task)
                break
            if self.queue_history[task.model_data.id][last_index][0] < current_time:
                if task not in self.queue_history[task.model_data.id][last_index][1]:
                    next_queue = self.queue_history[task.model_data.id][last_index][1].copy()
                    next_queue.append(task)
                    last_index += 1
                    self.queue_history[task.model_data.id].insert(
                        last_index, (current_time, next_queue))
                break
            # check the previous entry
            last_index -= 1

        # 2. added the task to all the subsequent timestamp tuples
        while last_index < len(self.queue_history[task.model_data.id]):
            if task not in self.queue_history[task.model_data.id][last_index][1]:
                self.queue_history[task.model_data.id][last_index][1].append(task)
            last_index += 1

    def rm_task_in_queue_history(self, task, current_time):
        if task.model_data.id not in self.queue_history:
            AssertionError("rm model cached location to an empty list")
            return

        last_index = len(self.queue_history[task.model_data.id]) - 1
        
        # 1. find the place to add this remove_event to the tuple list
        while last_index >= 0:
            if self.queue_history[task.model_data.id][last_index][0] == current_time:
                if task in self.queue_history[task.model_data.id][last_index][1]:
                    self.queue_history[task.model_data.id][last_index][1].remove(task)
                break
            if self.queue_history[task.model_data.id][last_index][0] < current_time:
                if task in self.queue_history[task.model_data.id][last_index][1]:
                    next_tasks_in_queue = self.queue_history[task.model_data.id][last_index][1].copy()
                    next_tasks_in_queue.remove(task)
                    last_index = last_index + 1
                    self.queue_history[task.model_data.id].insert(
                        last_index, (current_time, next_tasks_in_queue))
                break
            last_index -= 1  # go to prev time
        # 2. remove the task from all the subsequent tuple
        while last_index < len(self.queue_history[task.model_data.id]):
            if task in self.queue_history[task.model_data.id][last_index]:
                self.queue_history[task.model_data.id][last_index][1].remove(task)
            last_index += 1  # do this for the remaining element after

    def get_queue_history(self, current_time, model_id, info_staleness=0) -> list:
        return self.get_history(self.queue_history[model_id], current_time, info_staleness)