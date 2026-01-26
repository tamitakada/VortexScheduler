from core.job import *
from core.network import *
import core.configs.gen_config as gcfg

from core.events.base import *
from core.events.centralized_scheduler_events import TasksArrivalAtScheduler


class JobArrivalAtWorker(Event):
    """
    Event signifying that a Job arrived to a Cascade node (a Worker).
    Only for Decentralized Schedulers
    """

    def __init__(self, simulation, job, worker_id):
        self.simulation = simulation
        self.worker = self.simulation.workers[worker_id]
        self.worker_id = worker_id
        self.job = job

    def run(self, current_time):
        new_events = self.simulation.workers[self.worker_id].schedule_job_heft(
            current_time, self.job)
        
        for task in self.job.tasks:
            if len(task.required_task_ids) == 0:
                self.simulation.add_task_arrival_to_worker_metrics(
                    current_time, task, self.worker)

        return new_events

    def to_string(self):
        return "[Job Arrival at Worker (Job {})] ++".format(self.job.id)
    
    def is_worker_event():
        return True


# for PER_TASK scheduler
class TaskArrival(Event):
    """ Event to signify a TASK arriving at a WORKER. """

    def __init__(self, simulation, worker, task, job_id):
        self.simulation = simulation
        self.worker = worker
        self.task = task
        self.job_id = job_id

    def run(self, current_time):
        # log tracking for this task
        self.task.log.set_task_placed_on_worker_queue_timestamp(current_time)
        self.simulation.add_task_arrival_to_worker_metrics(current_time, self.task, self.worker)
        
        if gcfg.AUTOSCALING_POLICY != "NONE" and (self.worker.id not in self.simulation.workers or \
            (self.task.model_data != None and \
             all(s.model.data.id != self.task.model_data.id for s in self.worker.GPU_state.state_at(current_time)))):
            # outdated decision, send back tasks
            return [EventOrders(
                current_time + CPU_to_CPU_delay(self.task.input_size),
                TasksArrivalAtScheduler(self.simulation, [self.task]))]
        
        return self.worker.add_tasks(current_time, [self.task])

    def to_string(self):
        return "[Task Arrival (Job {} - Task {}) at {}] ---".format(self.job_id, self.task.task_id, self.worker)
    
    def is_worker_event():
        return True
    

# for PER_TASK scheduler
class TasksArrival(Event):
    """ Event to signify TASKs arriving at a WORKER. """

    def __init__(self, simulation, worker, tasks):
        self.simulation = simulation
        self.worker = worker
        self.tasks = tasks

    def run(self, current_time):
        # log tracking for this task
        drop_log = self.worker.simulation.task_drop_log
        relevant_tasks = [t for t in self.tasks if not (drop_log[current_time >= drop_log["drop_time"]]["job_id"] == t.job.id).any()]
        for task in relevant_tasks:
            task.log.set_task_placed_on_worker_queue_timestamp(current_time)
            self.simulation.add_task_arrival_to_worker_metrics(current_time, task, self.worker)

        if gcfg.AUTOSCALING_POLICY != "NONE":
            events = []
            curr_state = self.worker.GPU_state.state_at(current_time)
            
            executable_tasks = [] if self.worker.id not in self.simulation.workers else [
                task for task in self.tasks if any(s.model.data.id == task.model_data.id for s in curr_state)]
            if executable_tasks:
                events += self.worker.add_tasks(current_time, executable_tasks)
            
            tasks_to_return = [task for task in self.tasks if task not in executable_tasks]
            if tasks_to_return:
                events.append(
                    EventOrders(
                        current_time + sum(CPU_to_CPU_delay(t.input_size) for t in tasks_to_return),
                        TasksArrivalAtScheduler(self.simulation, tasks_to_return)))
            return events
        
        return self.worker.add_tasks(current_time, self.tasks)

    def to_string(self):
        return f"[Tasks Arrival at Worker (Job ID, Task ID: {[(t.job.id, t.task_id) for t in self.tasks]}) at {self.worker.id}] ---"
    
    def is_worker_event():
        return True


class InterResultArrival(Event):
    """ Event to signify a TASK arriving at a WORKER. (?) """

    def __init__(self, worker, prev_task, cur_task):
        self.worker = worker
        self.prev_task = prev_task
        self.cur_task = cur_task

    def run(self, current_time):
        self.cur_task.log.set_task_arrival_at_worker_buffer_timestamp(
            current_time)
        return self.worker.receive_intermediate_result(current_time, self.prev_task, self.cur_task)

    def to_string(self):
        return "[Intermediate Results Arrival]: worker:" + str(self.worker.id) + ", prev_task_id:" + str(self.prev_task.task_id) + ", cur_task_id:" + str(self.cur_task.task_id)
    
    def is_worker_event():
        return True
    
class BatchArrivalAtWorker(Event):
    """
    Event signifying that a batch of tasks arrived for execution at a worker.
    """

    def __init__(self, simulation, worker, batch):
        self.simulation = simulation
        self.worker = worker
        self.batch = batch

    def run(self, current_time):
        self.batch.tasks = [task for task in self.batch.tasks 
                            if not (self.simulation.task_drop_log["job_id"] == task.job.id).any()]
        if (not self.worker.GPU_state.does_have_idle_copy(self.batch.model_data.id, current_time)) or \
            (not gcfg.ENABLE_MULTITHREADING and any(s.reserved_batch for s in self.worker.GPU_state.state_at(current_time))) or \
                self.worker.did_abandon_batch(self.batch.id) or self.batch.size() == 0:
            current_batches = [s.reserved_batch for s in self.worker.GPU_state.state_at(current_time) if s.reserved_batch]
            transfer_delay = CPU_to_CPU_delay(self.batch.size()*self.batch.tasks[0].input_size) if self.batch.size() > 0 else 0
            return [EventOrders(current_time + transfer_delay, 
                                BatchRejectionAtWorker(self.simulation, self.worker, self.batch,
                                                       current_worker_batch=(current_batches[0] if current_batches else None)))]
        for task in self.batch.tasks:
            task.log.set_task_placed_on_worker_queue_timestamp(current_time)
            self.simulation.add_task_arrival_to_worker_metrics(current_time, task, self.worker)
        return self.worker.maybe_start_batch(self.batch, current_time)

    def to_string(self):
        return f"[Batch {self.batch.id} Arrival at Worker {self.worker.id} (Type: {self.batch.tasks[0].task_type}, Job IDs: {self.batch.job_ids})] ++"
    
    def is_worker_event():
        return True

class BatchRejectionAtWorker(Event):
    """
    Event signifying that a worker was busy when a batch was sent for execution, and
    the batch has been sent back to the Centralized scheduler for rescheduling.
    """

    def __init__(self, simulation, worker, batch, current_worker_batch=None):
        self.simulation = simulation
        self.worker = worker
        self.batch = batch
        self.current_worker_batch = current_worker_batch

    def run(self, current_time):
        if self.simulation.simulation_name == "shepherd":
            self.simulation.scheduler.worker_rejected_batch(self.worker.id, self.batch, self.current_worker_batch)
        if self.batch.size() == 0:
            return [] # possible if all tasks were dropped
        return [EventOrders(current_time, TasksArrivalAtScheduler(self.simulation, self.batch.tasks))] # reschedule batch

    def to_string(self):
        return f"[Batch {self.batch.id} Sent Back by Worker {self.worker.id}]"
    
    def is_worker_event():
        return True


class BatchPreemptionScheduledAtWorker(Event):
    """
    Event signifying that a batch should be preempted at a worker.
    """

    def __init__(self, simulation, worker, batch, old_batch_id):
        self.simulation = simulation
        self.worker = worker
        self.batch = batch # replacement batch
        self.old_batch_id = old_batch_id # preempted batch

    def run(self, current_time):
        self.batch.tasks = [task for task in self.batch.tasks 
                            if not (self.simulation.task_drop_log["job_id"] == task.job.id).any()]

        # check if batch to be preempted still exists/is actively executing
        old_batch = [s.reserved_batch for s in self.worker.GPU_state.state_at(current_time)
                     if s.reserved_batch and s.reserved_batch.id == self.old_batch_id]
        if len(self.batch.tasks) > 0 and old_batch:
            old_batch = old_batch[0]

            for task in self.batch.tasks:
                task.log.set_task_placed_on_worker_queue_timestamp(current_time)
                self.simulation.add_task_arrival_to_worker_metrics(current_time, task, self.worker)

            return self.worker.preempt_batch(self.old_batch_id, self.batch, current_time)
        else:
            # NOTE: marks as abandoned anyway in case prior assigned batch has not yet arrived
            self.worker._abandoned_batches.append(self.old_batch_id)
            # if outdated decision, or all batch tasks were dropped, send back tasks for rescheduling
            current_batches = [s.reserved_batch for s in self.worker.GPU_state.state_at(current_time) if s.reserved_batch]
            return [EventOrders(
                current_time + (0 if self.batch.size() == 0 else CPU_to_CPU_delay(self.batch.size()*self.batch.tasks[0].input_size)),
                BatchRejectionAtWorker(self.simulation, self.worker, self.batch, 
                                       current_worker_batch=(current_batches[0] if current_batches else None)))]
    
    def to_string(self):
        return f"[Batch Preemption Scheduled at Worker {self.worker.id} (Batch {self.old_batch_id} to be preempted)]"
    
    def is_worker_event():
        return True


class BatchStartEvent(Event):
    """ Event to signify that a BATCH has been started by the WORKER. """

    def __init__(self, worker, model_id, batch_id=-1, job_ids=[]):
        self.worker = worker
        self.model_id = model_id
        self.batch_id = batch_id
        self.job_ids = job_ids    # integers representing the job_ids

    def run(self, current_time):
        return []
    
    def should_abandon_event(self, current_time, kwargs: dict):
        return self.worker.did_abandon_batch(self.batch_id)

    def to_string(self):
        jobs = ",".join([str(id) for id in self.job_ids])
        return f"[Batch {self.batch_id} Start (Jobs {jobs}) at Worker {self.worker.id}, Model {self.model_id}]"
    
    def is_worker_event():
        return True


class BatchEndEvent(Event):
    """ Event to signify that a BATCH has been performed by the WORKER. """

    def __init__(self, worker, batch, job_ids=[]):
        self.worker = worker
        self.batch = batch
        self.job_ids = job_ids    # integers representing the job_ids

    def run(self, current_time):
        if self.worker.did_abandon_batch(self.batch.id):
            return []
        return self.worker.free_slot(current_time, self.batch)

    def should_abandon_event(self, current_time, kwargs: dict):
        return self.worker.did_abandon_batch(self.batch.id)

    def to_string(self):
        jobs = ",".join([str(id) for id in self.job_ids])
        return f"[Batch {self.batch.id} End (Jobs {jobs}) at Worker {self.worker.id}]"
    
    def is_worker_event():
        return True


class WorkerDropJob(Event):
    """
        Event signifying that a job has been dropped by a worker.
    """

    # Drop reasons
    _SLO_VIOLATION_DETECTED = 0

    def __init__(self, simulation, worker, job, task, deadline, reason=_SLO_VIOLATION_DETECTED):
        self.simulation = simulation
        self.worker = worker
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
                "arrival_time": self.task.log.task_placed_on_worker_queue_timestamp,
                "slo": self.job.slo,
                "deadline": self.deadline
            }
        return []

    def to_string(self):
        return f"[Drop Job (ID: {self.job.id}) at Worker (ID: {self.worker.id})] Due to {WorkerDropJob.get_reason_str(self.reason)}"
    
    def is_worker_event():
        return True
    
    @classmethod
    def get_reason_str(cls, reason):
        if reason == cls._SLO_VIOLATION_DETECTED:
            return "SLO violation"
        else:
            return "Unknown"


class UpdateLoadedModelsEvent(Event):
    """ Event to signify that a worker's loaded models should be updated. """

    def __init__(self, simulation, worker, models_to_add, models_to_rm):
        self.simulation = simulation
        self.worker = worker
        self.models_to_add = models_to_add
        self.models_to_rm = models_to_rm

    def run(self, current_time):
        events = self.worker.evict_models(current_time, self.models_to_rm)
        
        for model in self.models_to_add:
            events += self.worker.fetch_model(model, None, current_time)
        
        return events

    def to_string(self):
        return f"[Update Worker {self.worker.id}: Add {[m.id for m in self.models_to_add]}, Rm {[m.id for m in self.models_to_rm]}]"
    
    def is_worker_event():
        return True


class WorkerFinishedModelFetchEvent(Event):
    """ Event to signify that a worker finished loading a model and should check
    the task queue. """

    def __init__(self, simulation, worker, model_id):
        self.simulation = simulation
        self.worker = worker
        self.model_id = model_id

    def run(self, current_time):
        # may occur due to autoscaling
        if self.worker.id not in self.simulation.workers:
            return []

        if not gcfg.ENABLE_MULTITHREADING and \
            any(s.reserved_batch for s in self.worker.GPU_state.state_at(current_time)):
            return []
        
        return self.worker.check_task_queue(self.model_id, current_time)

    def to_string(self):
        return f"[Fetch Model {self.model_id} Finished on Worker {self.worker.id}]"
    
    def is_worker_event():
        return True