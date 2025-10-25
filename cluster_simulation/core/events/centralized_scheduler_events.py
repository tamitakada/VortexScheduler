from core.job import *
from core.network import *
from core.configs.gen_config import *

from core.events.base import *


class JobArrivalAtScheduler(Event):
    """
    Event signifying that a Job arrived to a Centralized scheduler.
    Only for Centralized Schedulers
    """

    def __init__(self, simulation, job):
        self.simulation = simulation
        self.job = job

    def run(self, current_time):
        # Schedule job
        if self.simulation.job_split == "PER_TASK":
            for task in self.job.tasks:
                task.log.task_arrival_at_scheduler_timestamp = current_time
            new_events = self.simulation.schedule_job_and_send_tasks(
                self.job, current_time)
        # elif self.simulation.job_split == "PER_JOB":
        #     new_events = self.simulation.schedule_job_and_send_job(
        #         self.job, current_time)
        return new_events
    
    def to_string(self):
        return "[Job Arrival at Scheduler (Job {})] ++".format(self.job.id)
    
    def is_worker_event():
        return False
    

class TasksArrivalAtScheduler(Event):
    """
    Event signifying that Task(s) arrived at a Centralized scheduler.
    Only for Centralized Schedulers
    """

    def __init__(self, simulation, tasks):
        assert(len(tasks) > 0)
        self.simulation = simulation
        self.tasks = tasks

    def run(self, current_time):
        # leave out dropped tasks
        self.tasks = [task for task in self.tasks 
                      if not (self.simulation.task_drop_log["job_id"] == task.job_id).any()]
        if not self.tasks:
            return []

        for task in self.tasks:
            # only set if not set already (avoid changing order for preempted tasks)
            if task.log.task_arrival_at_scheduler_timestamp == 0:
                task.log.task_arrival_at_scheduler_timestamp = current_time

        return self.simulation.schedule_tasks_on_arrival(self.tasks, current_time)
   
    def to_string(self):
        return f"[Tasks Arrival at Scheduler (Type: {self.tasks[0].task_type}, Job IDs: {list(map(lambda t: t.job_id, self.tasks))})] ++"
    
    def is_worker_event():
        return False
    

class SchedulerDropJob(Event):
    """
        Event signifying that a job has been dropped by the centralized scheduler.
    """

    # Drop reasons
    _SLO_VIOLATION_DETECTED = 0
    _ARRIVAL_RATE_CAP = 1

    def __init__(self, simulation, job, task, deadline, reason=_SLO_VIOLATION_DETECTED):
        self.simulation = simulation
        self.job = job
        self.at_task = task
        self.deadline = deadline
        self.reason = reason

    def run(self, current_time):
        if (self.simulation.task_drop_log["job_id"] != self.job.id).all():
            self.simulation.task_drop_log.loc[len(self.simulation.task_drop_log)] = {
                "client_id": self.job.client_id,
                "job_id": self.job.id,
                "workflow_id": self.job.job_type_id,
                "task_id": self.at_task.task_id,
                "drop_time": current_time,
                "create_time": self.job.create_time,
                "arrival_time": self.at_task.log.task_arrival_at_scheduler_timestamp,
                "slo": self.job.slo,
                "deadline": self.deadline
            }
        return []

    def to_string(self):
        return f"[Drop Job (ID: {self.job.id})] Due to {SchedulerDropJob.get_reason_str(self.reason)}"
    
    def is_worker_event():
        return False
    
    @classmethod
    def get_reason_str(cls, reason):
        if reason == cls._SLO_VIOLATION_DETECTED:
            return "SLO violation"
        elif reason == cls._ARRIVAL_RATE_CAP:
            return "Arrival rate cap reached"
        else:
            return "Unknown"


class AbortAllJobsEvent(Event):
    """
        Event signifying that all workers should immediately abort all
        currently executing batches and send them back to the Centralized
        scheduler for rescheduling.
    """

    def __init__(self, simulation, run_herd_sched=False):
        self.simulation = simulation
        self.run_herd_sched = run_herd_sched

    def run(self, current_time):
        # NOTE: assumes abort is costless

        events = []
        for worker in self.simulation.workers:
            curr_batch_ids = [s.reserved_batch.id for s in worker.GPU_state.state_at(current_time) if s.reserved_batch]
            for batch_id in curr_batch_ids:
                evicted_batch = worker.evict_batch(batch_id, current_time)
                if self.simulation.centralized_scheduler:
                    events.append(EventOrders(
                        current_time + CPU_to_CPU_delay(evicted_batch.tasks[0].input_size * evicted_batch.size()), 
                        TasksArrivalAtScheduler(self.simulation, evicted_batch.tasks)))
                else:
                    # TODO: decentral + HERD case
                    raise NotImplementedError("Job abort for decentralized scheduler")
            
            if self.simulation.simulation_name == "shepherd":
                assigned_batch = self.simulation.scheduler.worker_states[worker.worker_id]
                if assigned_batch and not worker.did_abandon_batch(assigned_batch.id):
                    Worker._abandoned_batches.append(assigned_batch.id)

        assert(all(all(not s.reserved_batch for s in w.GPU_state.state_at(current_time)) 
                    for w in self.simulation.workers))
        
        if self.run_herd_sched:
            events.append(EventOrders(current_time, RerunHerdScheduler(self.simulation)))

        return events

    def to_string(self):
        return "[Abort All Jobs]"
    
    def is_worker_event():
        return False


class StartHerdSchedulerRerun(Event):
    """
        Event signifying that the HERD scheduler should be run again to reallocate GPUs.
        Triggers job abortion and scheduler rerun.
    """

    def __init__(self, simulation):
        self.simulation = simulation

    def run(self, current_time):
        if self.simulation.remaining_jobs:
            return [EventOrders(current_time, 
                                AbortAllJobsEvent(self.simulation, run_herd_sched=True))]
        return []

    def to_string(self):
        return "[HERD Scheduler Rerun Queued]"
    
    def is_worker_event():
        return False


class RerunHerdScheduler(Event):
    """
        Event signifying that the HERD scheduler will be run again to
        reallocate GPUs.
    """

    def __init__(self, simulation):
        self.simulation = simulation

    def run(self, current_time):
        self.simulation.run_herd_scheduler(current_time)
        self.simulation.scheduler.update_herd_assignment(self.simulation.herd_assignment)
        events = []
        if self.simulation.centralized_scheduler:
            events = self.simulation.scheduler.schedule_tasks_on_queue(current_time)

        if HERD_PERIODICITY == np.inf:
            return events
        return events + [EventOrders(current_time + HERD_PERIODICITY,
                                     StartHerdSchedulerRerun(self.simulation))]
    
    def to_string(self):
        return "[HERD Scheduler Rerun]"
    
    def is_worker_event():
        return False


class SampleWorkerMetrics(Event):
    """
        Event signifying that the simulator should sample and log worker metrics.
    """

    def __init__(self, simulation, interval=500):
        self.simulation = simulation
        self.interval = interval

    def run(self, current_time):
        self.simulation.add_worker_metrics_sample(current_time, self.interval)

        if self.simulation.event_queue.qsize() > 0:
            return [EventOrders(current_time + self.interval, SampleWorkerMetrics(self.simulation, self.interval))]

        return []
    
    def to_string(self):
        return "[Sample Worker Metrics]"
    
    def is_worker_event():
        return False