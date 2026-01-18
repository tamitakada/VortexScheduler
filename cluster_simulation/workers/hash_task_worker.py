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
    def __init__(self, simulation, worker_id, total_memory, group_id=-1):
        super().__init__(simulation, worker_id, total_memory, group_id=group_id)
        # {task_obj1:[(preq_task_id0,arrival_time0), (preq_taks_id0, arrival_time1), ...], task2:[( ...],}
        self.waiting_tasks_buffer = defaultdict(lambda: [])
        # keep track of the queue information at time:  [ (time1,[task0,task1,]), (time2,[task1,...]),...]
        self.queue_history = {}
        self.involved = False

        self.next_task_type_idx = 0
        self.next_worker_id = { mid: 0 for mid in set([get_model_id_for_task_type(tt) for tt in get_task_types(self.simulation.job_types_list) ]) }
    
    def add_tasks(self, current_time, tasks):
        """
        Add tasks into the local task queue
        """
        if len(tasks) == 0:
            return []
        
        assert(all(task.model == None or task.model in self.GPU_state.placed_models(current_time) for task in tasks))

        # add tasks to worker queue
        for task in tasks:
            self.add_task_to_queue_history(task, current_time)

        # if concurrency disabled and occupied, do nothing
        if (not gcfg.ENABLE_MULTITHREADING) and any(s.reserved_batch for s in self.GPU_state.state_at(current_time)):
            return []

        events = []

        i = 0
        model_ids = list(set(task.model.model_id for task in tasks))
        while True:
            model_id = model_ids[i]
            new_events = self.check_task_queue(model_id, current_time)
            events += new_events

            if len(new_events) == 0:
                return events

            i = (i + 1) % len(model_ids)

    def free_slot(self, current_time, batch: Batch):
        """ Attempts to launch another task. """
        events = super().free_slot(current_time, batch)

        # start next batch
        if gcfg.ENABLE_MULTITHREADING:
            for mid in self.queue_history.keys():
                batch_end_events = self.check_task_queue(mid, current_time)
                events += batch_end_events
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
    
    def check_task_queue(self, model_id, current_time):
        model_state = [s for s in self.GPU_state.state_at(current_time)
                       if s.model.model_id == model_id]
        assert(len(model_state) >= 1)

        if len(model_state) > 0 and all(s.reserved_batch for s in model_state):
            # print(f"[Worker {self.worker_id}] Skip scheduling for model ID {model_id}, alr. executing batch")
            return []

        task_queue = self.get_queue_history(current_time, model_id, info_staleness=0)
        
        for task in task_queue:
            # skip dropped tasks
            if (self.simulation.task_drop_log["job_id"]==task.job_id).any():
                self.rm_task_in_queue_history(task, current_time)
                continue

            # get correct task deadline
            if gcfg.SLO_GRANULARITY == "TASK":
                task_deadline = task.log.task_placed_on_worker_queue_timestamp + task.slo * (1 + gcfg.SLO_SLACK)
                task_arrival = task.log.task_placed_on_worker_queue_timestamp
            else:
                task_deadline = task.log.job_creation_timestamp + task.job.slo * (1 + gcfg.SLO_SLACK)
                task_arrival = task.log.job_creation_timestamp

            if ((gcfg.DROP_POLICY == "LATEST_POSSIBLE" or gcfg.DROP_POLICY == "CLUSTER_ADMISSION_LIMIT") and current_time >= task_deadline) or \
                (gcfg.DROP_POLICY == "OPTIMAL" and (current_time + task.job.get_min_remaining_processing_time()) >= task_deadline):
                
                for job_task in task.job.tasks:
                    self.rm_task_in_queue_history(job_task, current_time)
                
                self.simulation.task_drop_log.loc[len(self.simulation.task_drop_log)] = {
                    "client_id": task.job.client_id,
                    "job_id": task.job_id, 
                    "workflow_id": task.task_type[0], 
                    "task_id": task.task_type[1],
                    "drop_time": current_time, 
                    "create_time": task.log.job_creation_timestamp,
                    "arrival_time": task_arrival,
                    "slo": task.slo if gcfg.SLO_GRANULARITY == "TASK" else task.job.slo, 
                    "deadline": task_deadline
                }
                continue
        
        # get queue after drops
        task_queue = [task for task in self.get_queue_history(current_time, model_id, info_staleness=0)
                      if current_time >= task.log.task_placed_on_worker_queue_timestamp]
        
        assert((self.simulation.task_drop_log["job_id"]!=task.job_id).all() for task in task_queue)

        if len(task_queue) > 0:
            batch = get_batch(current_time, self.total_memory, task_queue)
            if batch:
                batch_events = self.maybe_start_batch(batch, current_time)

                # if did start batch
                if batch_events:
                    # update queue history
                    for task in batch.tasks: # rm all tasks in batch
                        self.rm_task_in_queue_history(task, current_time)
                
                return batch_events

        return []

    #  ---------------------------  Subsequent TASK Transfer   --------------------

    def send_results_to_next_workers(self, current_time, batch) -> list:
        """
        Send the result of a task to the next worker in the inference pipeline (it may be the same worker)
        """
        events = []

        # TODO: diff emit sizes for diff workflows?
        emit_size = batch.tasks[0].max_emit_batch_size

        # model_id -> list[Task]
        ready_tasks_by_model = {}
        for task in batch.tasks:
            next_tasks = task.job.newly_available_tasks(task)
            for next_task in next_tasks:
                if next_task.model.model_id not in ready_tasks_by_model:
                    ready_tasks_by_model[next_task.model.model_id] = []
                ready_tasks_by_model[next_task.model.model_id].append(next_task)
        
        for mid, ready_tasks in ready_tasks_by_model.items():
            prev_curr = current_time
            for i in range(0, len(ready_tasks), emit_size):
                curr_send_batch = ready_tasks[i:(i+emit_size)]

                transfer_delay = 0
                while curr_send_batch[0].model and all(m.model_id != curr_send_batch[0].model.model_id for m in self.simulation.workers[self.next_worker_id[curr_send_batch[0].model.model_id]].GPU_state.placed_models(current_time)):
                    self.next_worker_id[curr_send_batch[0].model.model_id] = (self.next_worker_id[curr_send_batch[0].model.model_id] + 1) % len(self.simulation.workers)
                
                if self.next_worker_id[curr_send_batch[0].model.model_id] != self.worker_id:  # The next worker on the pipeline is NOT the same node
                    transfer_delay = CPU_to_CPU_delay(task.result_size * len(curr_send_batch))

                events.append(EventOrders(prev_curr + transfer_delay, TasksArrival(
                    self.simulation, self.simulation.workers[self.next_worker_id[curr_send_batch[0].model.model_id]], curr_send_batch)))
                
                prev_curr = prev_curr + transfer_delay

                self.next_worker_id[curr_send_batch[0].model.model_id] = (self.next_worker_id[curr_send_batch[0].model.model_id] + 1) % len(self.simulation.workers)
       
        return events

    def receive_intermediate_result(self, current_time, prev_task, cur_task) -> list:
        """
        event handling when taskworker receives the result of prev_task and put it to the waiting_buffer of cur_task
        @param: current_time: the time when the prev_task result arrives at this worker
        @param: prev_task: the task that has been executed and sent the result to this worker
        @param: cur_task: the task that is waiting for the result of prev_task, to be put to the waiting_buffer
        """
        events = []
        # 0. add this arrived prev_task result to the buffer of cur_task
        if prev_task != None:
            self.waiting_tasks_buffer[cur_task].append(
                [prev_task.task_id, current_time])
        # check if we have collected all the preq_tasks for current_task to be put on the task queue
        prev_arrived_list = self.waiting_tasks_buffer[cur_task]
        if(len(prev_arrived_list) != len(cur_task.required_task_ids)):
            # the cur_task hasn't received all the prerequisite task it needs to execute. SKIP this round
            return events
        # time when all the pre-requisite tasks have arrived
        receive_time = current_time
        for element in prev_arrived_list:
            receive_time = max(receive_time, element[1])
        events.append(EventOrders(
            receive_time, TaskArrival(self.simulation, self, cur_task, cur_task.job_id)))
        return events
    
    # ------------------------- queue history update helper functions ---------------

    def add_task_to_queue_history(self, task, current_time):
        # 0. Base case (first entry)
        if task.model.model_id not in self.queue_history:
            self.queue_history[task.model.model_id] = [(current_time, [task])]
            return

        # 1. Find the time_stamp place to add this queue information
        last_index = len(self.queue_history[task.model.model_id]) - 1
        while last_index >= 0:
            if self.queue_history[task.model.model_id][last_index][0] == current_time:
                if task not in self.queue_history[task.model.model_id][last_index][1]:
                    self.queue_history[task.model.model_id][last_index][1].append(task)
                break
            if self.queue_history[task.model.model_id][last_index][0] < current_time:
                if task not in self.queue_history[task.model.model_id][last_index][1]:
                    next_queue = self.queue_history[task.model.model_id][last_index][1].copy()
                    next_queue.append(task)
                    last_index += 1
                    self.queue_history[task.model.model_id].insert(
                        last_index, (current_time, next_queue)
                    )
                break
            # check the previous entry
            last_index -= 1

        # 2. added the task to all the subsequent timestamp tuples
        while last_index < len(self.queue_history[task.model.model_id]):
            if task not in self.queue_history[task.model.model_id][last_index][1]:
                self.queue_history[task.model.model_id][last_index][1].append(task)
            last_index += 1

    def rm_task_in_queue_history(self, task, current_time):
        # 0. base case: shouldn't happen
        if task.model.model_id not in self.queue_history:
            AssertionError("rm model cached location to an empty list")
            return

        last_index = len(self.queue_history[task.model.model_id]) - 1
        
        # 1. find the place to add this remove_event to the tuple list
        while last_index >= 0:
            if self.queue_history[task.model.model_id][last_index][0] == current_time:
                if task in self.queue_history[task.model.model_id][last_index][1]:
                    self.queue_history[task.model.model_id][last_index][1].remove(task)
                break
            if self.queue_history[task.model.model_id][last_index][0] < current_time:
                if task in self.queue_history[task.model.model_id][last_index][1]:
                    next_tasks_in_queue = self.queue_history[task.model.model_id][last_index][1].copy()
                    next_tasks_in_queue.remove(task)
                    last_index = last_index + 1
                    self.queue_history[task.model.model_id].insert(
                        last_index, (current_time, next_tasks_in_queue)
                    )
                break
            last_index -= 1  # go to prev time
        # 2. remove the task from all the subsequent tuple
        while last_index < len(self.queue_history[task.model.model_id]):
            if task in self.queue_history[task.model.model_id][last_index]:
                self.queue_history[task.model.model_id][last_index][1].remove(task)
            last_index += 1  # do this for the remaining element after

    def get_queue_history(self, current_time, model_id, info_staleness=0) -> list:
        return self.get_history(self.queue_history[model_id], current_time, info_staleness)