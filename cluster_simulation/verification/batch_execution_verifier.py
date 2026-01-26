from core.network import *

import core.configs.gen_config as gcfg
import core.configs.model_config as mcfg

from core.configs.workflow_config import *

from core.simulation import Simulation
from verification.verifier import Verifier

from collections import Counter

import re
import uuid


class BatchExecutionVerifier(Verifier):

    def __init__(self, simulation: Simulation):
        super().__init__(simulation)
    
    def run_verifier(self):
        self.verify_batch_sizes()
        self.verify_event_log()

    def _get_job_details(self, job_id: int):
        return self.simulation.result_to_export[self.simulation.result_to_export["job_id"]==int(job_id)].iloc[0]

    def _tasks_are_returned_to_scheduler(self, time: float, max_transfer_time: float, tasks: list[tuple[int, int]]):
        """Verify that given tasks all arrive at scheduler within some time interval.

        Args:
            time: Interval start (excl.)
            max_transfer_time: Interval duration
            tasks: [(job ID, task ID)] of tasks to check
        """
        
        task_arrivals = self.simulation.event_log[(self.simulation.event_log["time"] > time) & \
                    (self.simulation.event_log["time"] <= time + max_transfer_time) & \
                    (self.simulation.event_log["event"].str.contains("Tasks Arrival at Scheduler"))]
        arrived_tasks = []
        for _, ta in task_arrivals.iterrows():
            res = re.search(r"Job ID, Task ID: \[([^\]]*)\]", ta["event"])
            arrived_tasks += [[int(sid) for sid in s.strip("() ").split(", ")] for s in res.group(1).split("), ")]

        # verify tasks sent back
        for (job_id, task_id) in tasks:
            self.debug_assert(
                [job_id, task_id] in arrived_tasks,
                f"Job {job_id} Task {task_id} should be sent back to scheduler")

    def verify_event_log(self):
        worker_queues = {}
        for i, row in self.simulation.worker_log[self.simulation.worker_log["time"]==0].iterrows():
            if row["add_or_remove"] == "add":
                worker_queues[row["worker_id"]] = {}
            elif row["add_or_remove"] == "remove":
                self.debug_assert(
                    row["worker_id"] in worker_queues,
                    f"Removed worker {row['worker_id']} before adding worker")
                worker_queues.pop(row["worker_id"])
            else:
                raise AssertionError(f"Unrecognized value {row['add_or_remove']} in worker_log")
        
        worker_models = {wid: [] for wid in worker_queues.keys()}
        for i, row in self.simulation.worker_model_log[self.simulation.worker_model_log["start_time"]==0].iterrows():
            self.debug_assert(
                row["worker_id"] in worker_models,
                f'Worker {row["worker_id"]} not logged in worker_log')
            self.debug_assert(
                row["placed_or_evicted"]=="placed",
                "Model eviction at sim initialization")
            worker_models[row["worker_id"]].append((row["model_id"], row["instance_id"], None))
            worker_queues[row["worker_id"]][row["model_id"]] = []
        
        arrived_jobs = {} # job id -> task id -> 0 | 1 | 2 (not started, started, complete)
        max_transfer_time = 5 # TODO: more acc transfer time

        # (time, worker ID, model ID, instance ID)
        model_fetch_events = []

        for i, row in self.simulation.event_log.sort_values(by="time", kind="mergesort").iterrows():  
            if gcfg.ENABLE_VERIFICATION_DEBUG_LOGGING:
                print(f"[STEP {i} OF TRACE]\n\tTIME: {row['time']}\n")
                print(f"\tCURRENT WORKER QUEUES: {worker_queues}\n")
                print(f"\tCURRENT WORKER MODELS: {worker_models}\n")
                print(f"\tNEXT EVENT TO PROCESS: {row['event']}")

            # model fetch events may need to be preempted due to lack of separate logging
            for j in range(len(model_fetch_events)-1, -1, -1):
                (time, wid, mid, iid) = model_fetch_events[j]
                if time <= row["time"]:
                    worker_models[wid].append((mid, iid, None))
                    model_fetch_events.pop(j)

            if "Job Arrival" in row["event"]:
                job_id = int(re.search(r"Job ([0-9]+)", row["event"]).group(1))
                
                # don't log duplicate jobs
                self.debug_assert(
                    job_id not in arrived_jobs,
                    f"Duplicate Job {job_id} arrival events")
                
                # arrival at sim should be later than create time at client
                job_details = self._get_job_details(job_id)
                self.debug_assert(
                    row["time"] > job_details["job_create_time"],
                    f'Job created at {job_details["job_create_time"]} <= arrival at {row["time"]}')

                job_workflow = self.simulation.workflows[job_details["workflow_type"]]
                arrived_jobs[job_id] = { # log completion status of job tasks
                    tid: 0 for tid in job_workflow.tasks.keys()
                }

                # all initial tasks should arrive at workers soon after
                expected_initial_task_ids = [t.id for t in self.simulation.workflows[job_details["workflow_type"]].initial_tasks]
                for id in expected_initial_task_ids:
                    event_name = f"Task Arrival \\(Job {job_id} - Task {id}\\)"
                    self.debug_assert(
                        ((self.simulation.event_log["time"] > row["time"]) & \
                         (self.simulation.event_log["time"] <= row["time"] + max_transfer_time) & \
                         (self.simulation.event_log["event"].str.contains(event_name))).sum() > 0,
                        f"Job {job_id} initial task {id} arrival event not found"
                    )

            elif "Task Arrival " in row["event"]:
                res = re.search(r"Job ([0-9]+) - Task ([0-9]+)", row["event"])
                
                job_id = int(res.group(1))
                workflow_id = self._get_job_details(job_id)["workflow_type"]
                task_id = int(res.group(2))
                model_id = self.simulation.workflows[workflow_id].tasks[task_id].model_data.id

                # checks if worker has model/will fetch model
                # if not assume outdated arrival, tasks will be sent back (should only happen due to downscale)
                if (row["worker_id"] not in worker_queues) or \
                    (all(m != model_id for (m, _, _) in worker_models[row["worker_id"]]) and \
                     all(m != model_id for (_, wid, m, _) in model_fetch_events if wid == row["worker_id"])):

                    self._tasks_are_returned_to_scheduler(row["time"], max_transfer_time, [(job_id, task_id)])
                    continue

                # task arrival should not precede job arrival
                self.debug_assert(
                    job_id in arrived_jobs,
                    f"Job {job_id} arrival not logged by time of task {task_id} arrival")

                # dependencies should all be completed already
                if self.simulation.workflows[workflow_id].tasks[task_id].prev_tasks:
                    self.debug_assert(
                        all(arrived_jobs[job_id][t.id] == 2
                            for t in self.simulation.workflows[workflow_id].tasks[task_id].prev_tasks),
                        f'Job {job_id} Task {task_id} arrived but deps are missing')

                worker_queues[row["worker_id"]][model_id].append((job_id, task_id))
                
                # if available instance exists, should begin batch immediately
                if any(m == model_id and batch == None for (m, _, batch) in worker_models[row["worker_id"]]):
                    self.debug_assert(
                        ((self.simulation.event_log["worker_id"]==row["worker_id"]) & \
                         (self.simulation.event_log["time"]==row["time"]) & \
                         (self.simulation.event_log["event"].str.contains(f"Batch .* Start .* Model {model_id}"))).sum() > 0,
                        f"Idle model {model_id} on worker {row['worker_id']}: \
                            {worker_models[row['worker_id']]}\n{worker_queues[row['worker_id']][model_id]}")
            
            elif "Tasks Arrival at Worker" in row["event"]:
                res = re.search(r"Job ID, Task ID: \[([^\]]*)\]", row["event"])
                task_details = [[int(sid) for sid in s.strip("() ").split(", ")] for s in res.group(1).split("), ")]

                # outdated arrival in event of downscale
                if row["worker_id"] not in worker_queues:
                    self._tasks_are_returned_to_scheduler(
                        row["time"], 
                        max_transfer_time, 
                        [(task[0], task[1]) for task in task_details])
                    continue

                for task in task_details:
                    job_id = task[0]
                    task_id = task[1]

                    workflow_id = self._get_job_details(job_id)["workflow_type"]
                    model_id = self.simulation.workflows[workflow_id].tasks[task_id].model_data.id

                    # checks if worker has model/will fetch model
                    # if not assume outdated arrival in event of downscale (per model)
                    if all(m != model_id for (m, _, _) in worker_models[row["worker_id"]]) and \
                        all(m != model_id for (_, wid, m, _) in model_fetch_events if wid == row["worker_id"]):

                        self._tasks_are_returned_to_scheduler(
                            row["time"], max_transfer_time, [(job_id, task_id)])
                        continue

                    # task arrival should not precede job arrival
                    self.debug_assert(
                        job_id in arrived_jobs,
                        f"Job {job_id} arrival not logged by time of task {task_id} arrival")

                    # task should not be started
                    self.debug_assert(
                        arrived_jobs[job_id][task_id] == 0,
                        f"Job {job_id} Task {task_id} is either in progress or complete already")

                    # dependencies should all be completed already
                    if self.simulation.workflows[workflow_id].tasks[task_id].prev_tasks:
                        self.debug_assert(
                            all(arrived_jobs[job_id][t.id] == 2
                                for t in self.simulation.workflows[workflow_id].tasks[task_id].prev_tasks),
                            f'Job {job_id} Task {task_id} arrived but deps are missing')

                    worker_queues[row["worker_id"]][model_id].append((job_id, task_id))

                    # if available instance exists, should begin batch immediately
                    if any(m == model_id and batch == None for (m, _, batch) in worker_models[row["worker_id"]]):
                        self.debug_assert(
                            ((self.simulation.event_log["worker_id"]==row["worker_id"]) & \
                            (self.simulation.event_log["time"]==row["time"]) & \
                            (self.simulation.event_log["event"].str.contains(f"Batch .* Start .* Model {model_id}"))).sum() > 0,
                            f"Idle model {model_id} on worker {row['worker_id']}: \
                                {worker_models[row['worker_id']]}\n{worker_queues[row['worker_id']][model_id]}")

            elif "Add Worker" in row["event"]:
                worker_id = uuid.UUID(re.search(r"ID: (.*), Size", row["event"]).group(1))

                assert(worker_id not in worker_queues)

                worker_models[worker_id] = []
                worker_queues[worker_id] = {}
                
                model_loads = self.simulation.worker_model_log[\
                    (self.simulation.worker_model_log["worker_id"]==worker_id) & \
                    (self.simulation.worker_model_log["placed_or_evicted"]=="placed") & \
                    (self.simulation.worker_model_log["start_time"]==row["time"])]
                
                for _, mrow in model_loads.iterrows():
                    model_fetch_events.append(
                        (mrow["end_time"], worker_id, mrow["model_id"], mrow["instance_id"]))
                    if mrow["model_id"] not in worker_queues[worker_id]:
                        worker_queues[worker_id][mrow["model_id"]] = []

            elif "Remove Worker" in row["event"]:
                worker_id = uuid.UUID(re.search(r"Worker (.*)", row["event"]).group(1))

                # worker exists
                assert(worker_id in worker_queues)
                assert(worker_id in worker_models)

                # removal is successful: never executes any more batches
                assert(((self.simulation.batch_exec_log["worker_id"]==worker_id) & \
                        (self.simulation.batch_exec_log["start_time"] > row["time"])).sum() == 0)

                tasks_to_send_back = []
                for queued_tids in worker_queues[worker_id].values():
                    tasks_to_send_back += queued_tids

                for (_, _, batch) in worker_models[worker_id]:
                    if batch != None:
                        tasks_to_send_back += batch
                
                self._tasks_are_returned_to_scheduler(
                    row["time"], max_transfer_time, tasks_to_send_back)
                
                worker_queues.pop(worker_id)
                worker_models.pop(worker_id)

            elif "Update Worker" in row["event"]:
                res = re.search(r"Add \[(.*)\], Rm \[(.*)\]\]", row["event"])
                added_models = [int(x) for x in res.group(1).split(", ") if x]
                rm_models = [int(x) for x in res.group(2).split(", ") if x]

                tasks_to_send_back = []
                worker_updates = self.simulation.worker_model_log[\
                    (self.simulation.worker_model_log["worker_id"]==row["worker_id"]) & \
                    (self.simulation.worker_model_log["start_time"]==row["time"])]
                for _, mrow in worker_updates.iterrows():
                    if mrow["placed_or_evicted"] == "placed":
                        # should match with event log
                        assert(mrow["model_id"] in added_models)

                        model_fetch_events.append(
                            (mrow["end_time"], 
                             row["worker_id"], 
                             mrow["model_id"], 
                             mrow["instance_id"]))
                        if mrow["model_id"] not in worker_queues[row["worker_id"]]:
                            worker_queues[row["worker_id"]][mrow["model_id"]] = []
                        
                    elif mrow["placed_or_evicted"] == "evicted":
                        # should match with event log
                        assert(mrow["model_id"] in rm_models)

                        did_find_instance = False
                        for j, (wmid, iid, batch) in enumerate(worker_models[row["worker_id"]]):
                            if wmid == mrow["model_id"] and iid == mrow["instance_id"]:
                                worker_models[row["worker_id"]].pop(j)
                                if batch:
                                    tasks_to_send_back += batch
                                did_find_instance = True
                                break
                        assert(did_find_instance)

                    else:
                        raise AssertionError(f"Unrecognized value {mrow['placed_or_evicted']}")
                
                if tasks_to_send_back:
                    self._tasks_are_returned_to_scheduler(
                        row["time"], max_transfer_time, tasks_to_send_back)
                    
            elif "Fetch Model" in row["event"]:
                res = re.search(r"Fetch Model ([0-9]+) Finished on Worker (.*)\]", row["event"])
                model_id = int(res.group(1))
                worker_id = res.group(2)

                if uuid.UUID(worker_id) not in worker_queues:
                    continue

                if len(worker_queues[uuid.UUID(worker_id)][model_id]) > 0:
                    self.debug_assert(
                        ((self.simulation.event_log["worker_id"]==row["worker_id"]) & \
                         (self.simulation.event_log["time"]==row["time"]) & \
                         (self.simulation.event_log["event"].str.contains(f"Batch .* Start .* Model {model_id}"))).sum() > 0,
                        f"Idle model {model_id} on worker {row['worker_id']}: \
                            {worker_models[row['worker_id']]}\n{worker_queues[row['worker_id']][model_id]}")

            elif "Batch" in row["event"]:
                res = re.search(r"Batch ([^ ]+) (Start|End) \(Jobs (.*)\) at Worker", row["event"])
                batch_id = uuid.UUID(res.group(1))
                batched_jobs = [int(x) for x in res.group(3).split(",")]
                batch_log = self.simulation.batch_exec_log[self.simulation.batch_exec_log["batch_id"]==batch_id]
                model_id = batch_log.iloc[0]["model_id"] if len(batch_log) > 0 else int(re.search(r"Model ([0-9]+)", row["event"]).group(1))

                if len(batch_log) == 0:
                    assert("Start" in row["event"]) # should always be logged if allowed to finish

                    # batch was abandoned, tasks should have been returned to scheduler during exec
                    # model assigned to batch must be evicted at some point
                    model_eviction_event = ((self.simulation.worker_model_log["worker_id"] == row["worker_id"]) & \
                        (self.simulation.worker_model_log["model_id"] == model_id) & \
                        (self.simulation.worker_model_log["placed_or_evicted"] == "evicted") & \
                        (self.simulation.worker_model_log["start_time"] > row["time"])).sum() > 0
                    worker_removal_event = ((self.simulation.worker_log["worker_id"] == row["worker_id"]) & \
                        (self.simulation.worker_log["time"] > row["time"]) & \
                        (self.simulation.worker_log["add_or_remove"] == "remove")).sum() > 0
                    
                    assert(model_eviction_event or worker_removal_event)

                    task_arrivals = self.simulation.event_log[(self.simulation.event_log["time"] > row["time"]) & \
                                                              (self.simulation.event_log["event"].str.contains("Tasks Arrival at Scheduler"))]
                    future_arrived_jobs = []
                    for _, ta in task_arrivals.iterrows():
                        res1 = re.search(r"Job ID, Task ID: \[([^\]]*)\]", ta["event"])
                        future_arrived_jobs += [int(s.strip("() ").split(", ")[0]) for s in res1.group(1).split("), ")]

                    # tasks must be returned to scheduler
                    for jid in batched_jobs:
                        assert(jid in future_arrived_jobs)
                else:
                    batch_log = batch_log.iloc[0]

                if "Start" in row["event"]:
                    batched_tasks = []
                    for jid in batched_jobs:
                        # arrival should be logged before execution
                        assert(int(jid) in arrived_jobs)

                        idx = -1
                        for j, (jid_1, tid_1) in enumerate(worker_queues[row["worker_id"]][model_id]):
                            if jid_1 == int(jid):
                                idx = j

                                # task is batched for correct model
                                assert(self.simulation.workflows[\
                                    self._get_job_details(int(jid))["workflow_type"]].tasks[tid_1].model_data.id == \
                                    model_id)
                                
                                arrived_jobs[jid_1][tid_1] = 1
                                batched_tasks.append((jid_1, tid_1))

                                break
                        
                        # must be on worker queue
                        self.debug_assert(
                            idx >= 0,
                            f'Job {jid} not found on worker {row["worker_id"]} model {model_id} queue: \
                                {worker_queues[row["worker_id"]][model_id]}')

                        worker_queues[row["worker_id"]][model_id].pop(idx)
                    
                    found_unoccupied_model = False
                    for j, (mid, iid, curr_batch) in enumerate(worker_models[row["worker_id"]]):
                        if mid == model_id and curr_batch == None:
                            worker_models[row["worker_id"]][j] = (mid, iid, batched_tasks)
                            found_unoccupied_model = True
                            break
                    
                    assert(found_unoccupied_model)
                
                elif "End" in row["event"]:
                    found_occupied_model = False

                    for j, (mid, iid, curr_batch) in enumerate(worker_models[row["worker_id"]]):
                        if mid == batch_log["model_id"] and curr_batch != None:
                            if Counter([bj for (bj, _) in curr_batch]) == Counter([int(k) for k in batched_jobs]):
                                worker_models[row["worker_id"]][j] = (mid, iid, None)
                                found_occupied_model = True
                                break
                    
                    self.debug_assert(
                        found_occupied_model,
                        f'Batch not found on worker {row["worker_id"]}, current state: \
                            {worker_models[row["worker_id"]]}')

                    # if queued tasks for model, should start next batch immediately
                    if len(worker_queues[row["worker_id"]][batch_log["model_id"]]) > 0:
                        self.debug_assert(
                            ((self.simulation.event_log["worker_id"]==row["worker_id"]) & \
                            (self.simulation.event_log["time"]==row["time"]) & \
                            (self.simulation.event_log["event"].str.contains(f"Batch .* Start .* Model {batch_log['model_id']}"))).sum() > 0,
                            f"Idle model {batch_log['model_id']} on worker {row['worker_id']}: \
                                {worker_models[row['worker_id']]}\n{worker_queues[row['worker_id']][batch_log['model_id']]}")

                    for jid in batched_jobs:
                        found_job_task = False
                        workflow_id = self._get_job_details(int(jid))["workflow_type"]
                        for tid in arrived_jobs[int(jid)].keys():
                            if arrived_jobs[int(jid)][tid] == 1 and \
                                self.simulation.workflows[workflow_id].tasks[tid].model_data.id == batch_log["model_id"]:

                                arrived_jobs[int(jid)][tid] = 2
                                found_job_task = True
                                break
                        assert(found_job_task)

            elif gcfg.ENABLE_VERIFICATION_DEBUG_LOGGING:
                print(f"Unchecked event: {row['event']}")       
        
        if gcfg.DROP_POLICY == "NONE":
            assert(all(t_stat == 2 for t_stat in j_stat.values()) for j_stat in arrived_jobs.values())
        
        assert(all(batch == None for (_, _, batch) in worker_models[wid]) for wid in worker_models.keys())

    def verify_batch_sizes(self):
        model_ids = set(self.simulation.batch_exec_log["model_id"])
        for model_id in model_ids:
            batch_log = self.simulation.batch_exec_log[self.simulation.batch_exec_log["model_id"]==model_id]

            self.debug_assert(
                (batch_log["batch_size"] <= self.simulation.models[model_id].max_batch_size).all(),
                f"Some batch for model {model_id} had batch size > {self.simulation.models[model_id].max_batch_size}")
            
            self.debug_assert(
                (batch_log["batch_size"] > 0).all(),
                f"Some batch for model {model_id} ran with <= 0 batch size")