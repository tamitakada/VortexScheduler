from queue import PriorityQueue

from core.config import *
from core.task import Task
from core.batch import Batch
from core.events.base import *
from core.events.centralized_scheduler_events import *
from core.events.worker_events import *
from core.network import *
from core.workflow import *

from workers.worker import Worker

from schedulers.centralized.scheduler import Scheduler
from schedulers.centralized.shepherd.ordered_task import OrderedTask

import pandas as pd


class ShepherdScheduler(Scheduler):
    """
        Scheduler class for a Flex scheduler.
    """

    def __init__(self, simulation, herd_assignment=None):
        super().__init__(simulation, herd_assignment)
        
        self.model_queues = { mid: PriorityQueue() for mid in set(get_model_id_for_task_type(tt) 
                                                                  for tt in get_task_types(self.simulation.job_types_list)) }

        # init currently executing batch ids
        self.worker_states = {}
        for group in self.herd_assignment.worker_groups:
            for worker in group:
                self.worker_states[worker.worker_id] = None

        # for round-robin worker ordering
        self.next_worker_idxs = [0 for _ in self.herd_assignment.worker_groups]

        self.sys_limit = -1
        self.last_change = -1

        self.tput_gput_log = pd.DataFrame(columns=["time", "ar", "tput", "gput"])
        self.limit_log = pd.DataFrame(columns=["time", "limit"])

        self.arrived_task_log = pd.DataFrame(columns=["time", "job_id", "task_id"])
        self.per_task_limit_log = pd.DataFrame(columns=["time", "workflow_id", "task_id", "limit"])
        self.per_task_metrics_log = pd.DataFrame(columns=["time", "workflow_id", "task_id", "arrival_rate", "throughput"])

    def update_herd_assignment(self, herd_assignment):
        super().update_herd_assignment(herd_assignment)

        # currently executing batch ids
        self.worker_states = {}
        for group in self.herd_assignment.worker_groups:
            for worker in group:
                self.worker_states[worker.worker_id] = None

        # for round-robin worker ordering
        self.next_worker_idxs = [0 for _ in self.herd_assignment.worker_groups]

    def get_job_stats_over_past(self, time, time_frame=1000, samples=10):
        job_arrivals = self.arrived_task_log.loc[self.arrived_task_log.groupby("job_id")["task_id"].idxmin()]

        arrival_rates = []
        throughputs = []
        goodputs = []

        for i in range(samples):
            sample_start = time - (i+1) * time_frame
            sample_end = time - i * time_frame

            if sample_start < 0:
                continue

            arrived_jobs = len(job_arrivals[(job_arrivals["time"] <= sample_end) & (job_arrivals["time"] > sample_start)])
            arrival_rate = arrived_jobs / time_frame * 1000

            complete_jobs = [j for j in self.simulation.jobs.values() if len(j.completed_tasks) == len(j.tasks) and j.end_time > sample_start and j.end_time <= sample_end]
            if len(complete_jobs) < 10:
                continue
            
            time_since_first_complete = sample_end - min(j.end_time for j in complete_jobs)
            throughput = len(complete_jobs) / time_since_first_complete * 1000
            goodput = len([j for j in complete_jobs if j.end_time <= j.create_time + j.slo]) / time_since_first_complete * 1000

            arrival_rates.append(arrival_rate)
            throughputs.append(throughput)
            goodputs.append(goodput)
        
        return (np.mean(arrival_rates), np.mean(throughputs), np.mean(goodputs))
    
    def get_task_stats_over_past(self, time, task_id, time_frame=1000, samples=10):
        task_arrivals = self.arrived_task_log[self.arrived_task_log["task_id"] == task_id]
        all_complete_tasks = [t for j in self.simulation.jobs.values() for t in j.tasks 
                          if t.task_id == task_id and t.task_id in j.completed_tasks]

        arrival_rates = []
        throughputs = []

        for i in range(samples):
            sample_start = time - (i+1) * time_frame
            sample_end = time - i * time_frame

            if sample_start < 0:
                continue

            arrival_rate = len(task_arrivals[(task_arrivals["time"] <= sample_end) & (task_arrivals["time"] > sample_start)]) / time_frame * 1000

            complete_tasks = [t for t in all_complete_tasks if t.log.task_execution_start_timestamp > sample_start and t.log.task_execution_end_timestamp <= sample_end]
            # if len(complete_tasks) < 10:
            #     continue
            
            time_since_first_complete = sample_end - min(t.log.task_execution_start_timestamp for t in complete_tasks)
            throughput = len(complete_tasks) / time_since_first_complete * 1000

            arrival_rates.append(arrival_rate)
            throughputs.append(throughput)
        
        return (np.mean(arrival_rates), np.mean(throughputs))

    def schedule_job_on_arrival(self, job, current_time):
        if DROP_POLICY == "CLUSTER_ADMISSION_LIMIT":
            if current_time > 5000:
                curr_ar, curr_tp, curr_gp = self.get_job_stats_over_past(current_time, samples=1)
                ar, tput, gput = self.get_job_stats_over_past(current_time)

                self.tput_gput_log.loc[len(self.tput_gput_log)] = [current_time, ar, tput, gput]

                if self.sys_limit <= 0 or (current_time - self.last_change > 1000 and ar < self.sys_limit): 
                    if ar - tput > 5:
                        self.sys_limit = ar - 0.25 * (ar - tput)
                        self.limit_log.loc[len(self.limit_log)] = {"time": current_time, "limit": self.sys_limit}
                        self.last_change = -1
                    elif tput - gput > 5:
                        self.sys_limit = ar - 0.25 * (ar - gput)
                        self.limit_log.loc[len(self.limit_log)] = {"time": current_time, "limit": self.sys_limit}
                        self.last_change = -1
                
                # set last_change = first time arrival rate is detected to have responded to latest system limit
                if curr_ar < self.sys_limit and self.last_change < 0:
                    self.last_change = current_time
        
                if self.sys_limit > 0 and curr_ar > self.sys_limit:
                    return [EventOrders(
                        current_time, 
                        SchedulerDropJob(
                            self.simulation, 
                            job, 
                            job.tasks[0], 
                            job.create_time + job.slo,
                            reason=SchedulerDropJob._ARRIVAL_RATE_CAP))]
    
        return self.schedule_tasks_on_arrival([t for t in job.tasks if len(t.required_task_ids) == 0], current_time)
    
    def schedule_tasks_on_arrival(self, tasks, current_time):
        """
            While there are unchecked workers and queued tasks, creates the largest batch
            possible across all models and attempts to assign a worker to the batch in round 
            robin order.
        """
        per_task_ars = {}

        arrived_groups = set()
        for task in tasks:
            if (self.simulation.task_drop_log["job_id"] == task.job_id).any():
                continue

            if DROP_POLICY == "TASK_ADMISSION_LIMIT":
                if current_time > 5000:
                    ar, tput = self.get_task_stats_over_past(current_time, task.task_id)
                    self.per_task_metrics_log.loc[len(self.per_task_metrics_log)] = [current_time, task.job.job_type_id, task.task_id, ar, tput]

                    task_limit_hist = self.per_task_limit_log[(self.per_task_limit_log["workflow_id"] == task.job.job_type_id) & (self.per_task_limit_log["task_id"] == task.task_id)]
                    curr_limit, curr_limit_set_time = (task_limit_hist.tail(1)["limit"].values[0], task_limit_hist.tail(1)["time"].values[0]) if len(task_limit_hist) > 0 else (-1, -1)
                   
                    if curr_limit <= 0 or (current_time - curr_limit_set_time > 1000 and ar < curr_limit): 
                        if ar - tput > 5:
                            task_limit = ar - 0.25 * (ar - tput)
                            curr_limit = task_limit
                            self.per_task_limit_log.loc[len(self.per_task_limit_log)] = [current_time, task.job.job_type_id, task.task_id, task_limit]
            
                    if task.task_type in per_task_ars:
                        ar_with_task = per_task_ars[task.task_type] + 1
                    else:
                        task_arrivals = self.arrived_task_log[(self.arrived_task_log["task_id"] == task.task_id) & (self.arrived_task_log["time"] <= current_time) & (self.arrived_task_log["time"] > (current_time - 1000))]
                        ar_with_task = (len(task_arrivals) + 1)
                    
                    if curr_limit > 0 and ar_with_task > curr_limit:
                        self.simulation.task_drop_log.loc[len(self.simulation.task_drop_log)] = {
                            "client_id": task.job.client_id,
                            "job_id": task.job_id,
                            "workflow_id": task.task_type[0],
                            "task_id": task.task_type[1],
                            "drop_time": current_time,
                            "create_time": task.log.job_creation_timestamp,
                            "arrival_time": current_time,
                            "slo": task.slo,
                            "deadline": task.job.slo + task.job.create_time
                        }
                        continue
                    else:
                        per_task_ars[task.task_type] = ar_with_task
            
            if task.model.model_id not in self.model_queues:
                self.model_queues[task.model.model_id] = PriorityQueue()
            
            self.arrived_task_log.loc[len(self.arrived_task_log)] = [current_time, task.job.id, task.task_id]
            self.model_queues[task.model.model_id].put(OrderedTask(task, current_time))
            arrived_groups.add(self.herd_assignment.task_type_to_group[task.task_type])
        
        events = []
        for group in arrived_groups:
            events += self._flex_schedule_tasks_on_arrival(group, current_time)
        return events

    def schedule_tasks_on_queue(self, current_time):
        events = []
        for group in range(len(self.herd_assignment.worker_groups)):
            events += self._flex_schedule_tasks_on_arrival(group, current_time)
        return events
    
    # STATE MANAGEMENT --------------------------------------------------------------------
    
    def worker_completed_batch(self, worker_id: int, batch: Batch):
        assert(self.worker_states[worker_id] == batch)
        self.worker_states[worker_id] = None

    def worker_rejected_batch(self, worker_id: int, batch: Batch, current_worker_batch: Batch):
        # if alr. scheduled a different batch, do NOT update state
        if self.worker_states[worker_id] != batch:
            return
        self.worker_states[worker_id] = current_worker_batch

    def preempt_batch_on_worker(self, worker_id: int, new_batch: Batch):
        assert(self.worker_states[worker_id] is not None)
        self.worker_states[worker_id] = new_batch

    def assign_batch_to_worker(self, worker_id: int, batch: Batch):
        assert(self.worker_states[worker_id] is None)
        self.worker_states[worker_id] = batch
        
    # FLEX ALGO ----------------------------------------------------------------------------

    def _flex_schedule_tasks_on_arrival(self, group, current_time):
        if DROP_POLICY not in ["NONE", "OPTIMAL"]:
            self._drop_currently_late_tasks(current_time)
        
        events = []
        group_workers = self.herd_assignment.worker_groups[group].copy()
        ordered_group_workers = group_workers[self.next_worker_idxs[group]:] + group_workers[:self.next_worker_idxs[group]]
        for worker in ordered_group_workers:
            curr_batch = self.worker_states[worker.worker_id]
            curr_batch_size = curr_batch.size() if not curr_batch is None else 0

            largest_batch_model_id, largest_batch_size = self._flex_get_largest_candidate_batch(group, current_time, for_worker=worker)

            if curr_batch_size == 0 and largest_batch_size > 0:
                batch = self._flex_form_largest_batch(largest_batch_model_id, current_time)

                assert(batch.size() == largest_batch_size)

                if DROP_POLICY == "OPTIMAL":
                    self._drop_tasks(current_time, batch)
                
                self.assign_batch_to_worker(worker.worker_id, batch)
                events.append(EventOrders(
                    current_time + CPU_to_CPU_delay(batch.size()*batch.tasks[0].input_size), 
                    BatchArrivalAtWorker(self.simulation, worker, batch)))
            elif largest_batch_size > 0 and largest_batch_size >= FLEX_LAMBDA * curr_batch_size:
                batch = self._flex_form_largest_batch(largest_batch_model_id, current_time)
                
                assert(batch.size() == largest_batch_size)
                
                if DROP_POLICY == "OPTIMAL":
                    self._drop_tasks(current_time, batch)
                
                old_batch_id = self.worker_states[worker.worker_id].id
                self.preempt_batch_on_worker(worker.worker_id, batch)
                events.append(EventOrders(
                    current_time + CPU_to_CPU_delay(batch.size()*batch.tasks[0].input_size), 
                    BatchPreemptionScheduledAtWorker(self.simulation, worker, batch, old_batch_id)))
        
        # next time start from next worker in group
        self.next_worker_idxs[group] = (self.next_worker_idxs[group] + 1) % len(group_workers)
        return events
    
    def _drop_tasks(self, time: float, curr_batch: Batch):
        for model_queue in self.model_queues.values():
            skipped_tasks = []
            while model_queue.qsize() > 0:
                ot = model_queue.get()
                if time < ot.task.log.task_arrival_at_scheduler_timestamp:
                    skipped_tasks.append(ot)
                    continue

                if any(t.job_id == ot.task.job_id and t.task_id == ot.task.task_id for t in curr_batch.tasks):
                    # batched tasks should have been popped
                    assert(False)

                # drop tasks whose SLOs can't be satisfied within a grace period
                curr_batch_exec_time = curr_batch.model.batch_exec_times[24][curr_batch.model.batch_sizes.index(len(curr_batch.tasks))]
                if (time + curr_batch_exec_time + ot.task.job.get_min_remaining_processing_time()) > ot.deadline:
                    self.simulation.task_drop_log.loc[len(self.simulation.task_drop_log)] = {
                        "client_id": ot.task.job.client_id,
                        "job_id": ot.task.job_id,
                        "workflow_id": ot.task.task_type[0],
                        "task_id": ot.task.task_type[1],
                        "drop_time": time,
                        "create_time": ot.task.log.job_creation_timestamp,
                        "arrival_time": ot.task_arrival_time,
                        "slo": ot.task.slo,
                        "deadline": ot.deadline
                    }
                else:
                    skipped_tasks.append(ot)
            
            for ot in skipped_tasks:
                model_queue.put(ot)

    def _form_largest_batch(self, model_id: int, time: float, pop=False) -> list[Task]:
        model_queue = self.model_queues[model_id]
        if model_queue.qsize() == 0:
            return []
        tasks = []
        max_bsize = model_queue.queue[0].task.max_batch_size
        if SHEPHERD_BATCHING_POLICY == "OPTIMAL":
            sorted_queue = sorted(model_queue.queue, key=lambda ot: ot.deadline - ot.task.job.get_min_remaining_processing_time())
            bsizes = [[0 for j in range(max_bsize)] for i in range(model_queue.qsize())]
            for i in range(model_queue.qsize()-1, -1, -1):
                for j in range(max_bsize-1, 0, -1):
                    ot = sorted_queue[i]
                    min_rem_ppl_time = ot.task.job.get_min_remaining_processing_time(init_proc_times={ot.task.task_id: ot.task.model.batch_exec_times[24][ot.task.model.batch_sizes.index(j)]})
                    if time + min_rem_ppl_time > ot.deadline:
                        # on SLO violation, task [i] cannot be incl. in batch of size [j]
                        bsizes[i][j] = 0
                    elif i == (model_queue.qsize()-1): # if on last task and no SLO violation, always 1
                        bsizes[i][j] = 1
                    else:
                        bsizes[i][j] = max(bsizes[i+1][j-1] + 1, # either task [i] in batch of size [j]
                                           bsizes[i+1][j])       # or not
            
            max_i, max_j, max_formable_bsize = 0, 0, 0
            for i in range(len(bsizes)):
                for j in range(len(bsizes[i])):
                    if bsizes[i][j] > max_formable_bsize:
                        max_formable_bsize = bsizes[i][j]
                        max_i = i
                        max_j = j
            
            if max_formable_bsize == 0:
                # if all SLOs are violated, default to largest batch size
                tasks = sorted_queue[:max_bsize]
            else:
                counter = max_i
                while counter < (max_i + max_j):
                    tasks.append(sorted_queue[counter])
                    counter += 1
            
            if pop:
                for task in tasks:
                    model_queue.queue.remove(task)
                model_queue.queue.sort()

            return [ot.task for ot in tasks]

        first_deadline = 0
        skipped_tasks = []
        model_queue = self.model_queues[model_id]
        curr_queue_idx = 0
        while ((pop and model_queue.qsize() > 0) or (not pop and curr_queue_idx < model_queue.qsize())) and len(tasks) < max_bsize:
            ot = model_queue.get() if pop else model_queue.queue[curr_queue_idx]

            if SHEPHERD_BATCHING_POLICY == "BEST_EXEC_TIME_ONLY":
                tasks.append(ot.task)
            elif SHEPHERD_BATCHING_POLICY == "FIRST_TASK_DEADLINE":
                if len(tasks) == 0:
                    tasks.append(ot.task)
                    first_deadline = ot.deadline
                else:
                    # break once first task SLO cannot be satisfied anymore
                    if (time + ot.task.model.batch_exec_times[24][ot.task.model.batch_sizes.index(len(tasks))]) > first_deadline:
                        if pop: skipped_tasks.append(ot)
                        break
                    else:
                        tasks.append(ot.task)
                        
            curr_queue_idx += 1
        
        for ot in skipped_tasks:
            model_queue.put(ot)
        return tasks

    def _flex_form_largest_batch(self, model_id: int, time: float) -> Batch:
        """
            Form the largest batch available in [self.model_queues[model_id]].
        """
        return Batch(self._form_largest_batch(model_id, time, pop=True))

    def _flex_get_largest_candidate_batch(self, group: int, current_time: float, for_worker: Worker=None):
        """
            Returns (model_id, batch_size) of the largest batch that can be formed from currently 
            queued tasks across all of [model_queues] where model_id is required by some task in
            [task_types].
        """
        task_types = self.herd_assignment.group_task_types[group]
        largest_batch_model_id = -1
        largest_batch_size = 0
        for task_type in task_types:
            mid = get_model_id_for_task_type(task_type)
            if mid not in self.model_queues or self.model_queues[mid].qsize() == 0:
                continue # no tasks queued

            if for_worker:
                if not ENABLE_DYNAMIC_MODEL_LOADING and all(s.model.model_id != mid for s in for_worker.GPU_state.state_at(current_time)):
                    continue
                    
                if ENABLE_DYNAMIC_MODEL_LOADING and self.simulation.get_model_from_id(mid).model_size > for_worker.GPU_state._total_memory:
                    continue

            largest_batch = self._form_largest_batch(mid, current_time)
            if len(largest_batch) > largest_batch_size:
                largest_batch_size = len(largest_batch)
                largest_batch_model_id = mid
        return (largest_batch_model_id, largest_batch_size)
    
    def _drop_currently_late_tasks(self, time: float):
        for model_queue in self.model_queues.values():
            skipped_tasks = []
            while model_queue.qsize() > 0:
                ot = model_queue.get()
                if time < ot.task.log.task_arrival_at_scheduler_timestamp:
                    skipped_tasks.append(ot)
                    continue

                if time > ot.deadline:
                    self.simulation.task_drop_log.loc[len(self.simulation.task_drop_log)] = {
                        "client_id": ot.task.job.client_id,
                        "job_id": ot.task.job_id,
                        "workflow_id": ot.task.task_type[0],
                        "task_id": ot.task.task_type[1],
                        "drop_time": time,
                        "create_time": ot.task.log.job_creation_timestamp,
                        "arrival_time": ot.task_arrival_time,
                        "slo": ot.task.slo,
                        "deadline": ot.deadline
                    }
                else:
                    skipped_tasks.append(ot)
            
            for ot in skipped_tasks:
                model_queue.put(ot)

    def flex_schedule_on_batch_completion(self, worker: Worker, completed_batch: Batch, current_time: float):
        if DROP_POLICY not in ["NONE", "OPTIMAL"]:
            self._drop_currently_late_tasks(current_time)
        
        # if alr. assigned to a new batch do nothing
        if self.worker_states[worker.worker_id].id != completed_batch.id:
            return []

        self.worker_completed_batch(worker.worker_id, completed_batch)
        
        largest_batch_model_id, largest_batch_size = self._flex_get_largest_candidate_batch(
            self.herd_assignment.task_type_to_group[completed_batch.tasks[0].task_type], 
            current_time,
            for_worker=worker)

        if largest_batch_size > 0:
            batch = self._flex_form_largest_batch(largest_batch_model_id, current_time)

            if DROP_POLICY == "OPTIMAL":
                self._drop_tasks(current_time, batch)
            
            self.assign_batch_to_worker(worker.worker_id, batch)
            return [EventOrders(
                current_time + CPU_to_CPU_delay(batch.size()*batch.tasks[0].input_size), 
                BatchArrivalAtWorker(self.simulation, worker, batch))]
        return []