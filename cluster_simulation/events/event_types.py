from core.job import Job
from core.task import Task
from core.batch import Batch
from events.event import EventType, Agent
from uuid import UUID


class EventIds:
    JOB_SENT_TO_SCHEDULER           = 0
    JOB_ARRIVAL_AT_SCHEDULER        = 1
    TASKS_ARRIVAL_AT_SCHEDULER      = 2
    TASKS_SCHEDULED_TO_WORKER       = 3
    TASKS_SENT_TO_WORKER            = 4
    TASKS_SENT_TO_SCHEDULER         = 5
    TASKS_ARRIVAL_AT_WORKER         = 6

    JOBS_DROPPED                    = 7

    BATCH_STARTED_AT_WORKER         = 8
    BATCH_FINISHED_AT_WORKER        = 9
    RESPONSE_SENT_TO_CLIENT         = 10
    RESPONSE_RECEIVED_AT_CLIENT     = 11


EVENT_TYPES: dict[int, EventType] = {
    EventIds.JOB_SENT_TO_SCHEDULER: EventType(
        EventIds.JOB_SENT_TO_SCHEDULER, "Job Sent to Scheduler",
        kwargs={"job": (True, Job), "from_client_id": (True, UUID), 
                "ignore_transfer_time": (True, bool)},
        emitter_types=[Agent.CLIENT],
        listener_types=[Agent.NETWORK, Agent.LOGGER, Agent.VERIFIER]),
    
    EventIds.JOB_ARRIVAL_AT_SCHEDULER: EventType(
        EventIds.JOB_ARRIVAL_AT_SCHEDULER, "Job Arrival at Scheduler", 
        kwargs={"job": (True, Job)},
        emitter_types=[Agent.NETWORK],
        listener_types=[Agent.SCHEDULER, Agent.LOGGER, Agent.VERIFIER]),

    EventIds.TASKS_ARRIVAL_AT_SCHEDULER: EventType(
        EventIds.TASKS_ARRIVAL_AT_SCHEDULER, "Tasks Arrival at Scheduler", 
        kwargs={"tasks": (True, list[Task])},
        emitter_types=[Agent.NETWORK],
        listener_types=[Agent.SCHEDULER, Agent.LOGGER, Agent.VERIFIER]),

    EventIds.TASKS_SCHEDULED_TO_WORKER: EventType(
        EventIds.TASKS_SCHEDULED_TO_WORKER, "Tasks Scheduled to Worker", 
        kwargs={"job_task_ids": (True, list[tuple[int, int]]), "worker_id": (True, UUID)},
        emitter_types=[Agent.SCHEDULER],
        listener_types=[Agent.WORKER, Agent.LOGGER, Agent.VERIFIER]),

    EventIds.TASKS_SENT_TO_WORKER: EventType(
        EventIds.TASKS_SENT_TO_WORKER, "Tasks Sent to Worker", 
        kwargs={"tasks": (True, list[Task]), "force_instance_id": (True, UUID), 
                "from_worker_id": (True, UUID), "to_worker_id": (True, UUID), 
                "ignore_transfer_time": (True, bool)},
        emitter_types=[Agent.SCHEDULER, Agent.WORKER],
        listener_types=[Agent.NETWORK, Agent.LOGGER, Agent.VERIFIER]),

    EventIds.TASKS_SENT_TO_SCHEDULER: EventType(
        EventIds.TASKS_SENT_TO_SCHEDULER, "Tasks Sent to Scheduler", 
        kwargs={"tasks": (True, list[Task]), "from_worker_id": (True, UUID),
                "ignore_transfer_time": (True, bool)},
        emitter_types=[Agent.WORKER],
        listener_types=[Agent.NETWORK, Agent.LOGGER, Agent.VERIFIER]),
    
    EventIds.TASKS_ARRIVAL_AT_WORKER: EventType(
        EventIds.TASKS_ARRIVAL_AT_WORKER, "Tasks Arrival at Worker", 
        kwargs={"tasks": (True, list[Task]), "force_instance_id": (True, UUID),
                "worker_id": (True, UUID)},
        emitter_types=[Agent.NETWORK],
        listener_types=[Agent.WORKER, Agent.LOGGER, Agent.VERIFIER]),
    
    EventIds.JOBS_DROPPED: EventType(
        EventIds.JOBS_DROPPED, "Jobs Dropped",
        kwargs={"job_ids": (True, list[int])},
        emitter_types=[Agent.SCHEDULER, Agent.WORKER],
        listener_types=[Agent.SCHEDULER, Agent.WORKER, Agent.CLIENT, Agent.LOGGER, Agent.VERIFIER]),
    
    EventIds.BATCH_STARTED_AT_WORKER: EventType(
        EventIds.BATCH_STARTED_AT_WORKER, "Batch Started at Worker",
        kwargs={"batch": (True, Batch), "model_instance_id": (True, UUID), "worker_id": (True, UUID)},
        emitter_types=[Agent.WORKER],
        listener_types=[Agent.SCHEDULER, Agent.LOGGER, Agent.VERIFIER]),
    
    EventIds.BATCH_FINISHED_AT_WORKER: EventType(
        EventIds.BATCH_FINISHED_AT_WORKER, "Batch Finished at Worker",
        kwargs={"batch": (True, Batch), "model_instance_id": (True, UUID), "worker_id": (True, UUID)},
        emitter_types=[Agent.WORKER],
        listener_types=[Agent.SCHEDULER, Agent.WORKER, Agent.LOGGER, Agent.VERIFIER]),

    EventIds.RESPONSE_SENT_TO_CLIENT: EventType(
        EventIds.RESPONSE_SENT_TO_CLIENT, "Response Sent to Client",
        kwargs={"job": (True, Job), "client_id": (True, UUID), "worker_id": (True, UUID),
                "ignore_transfer_time": (True, bool)},
        emitter_types=[Agent.WORKER],
        listener_types=[Agent.NETWORK, Agent.LOGGER, Agent.VERIFIER]),

    EventIds.RESPONSE_RECEIVED_AT_CLIENT: EventType(
        EventIds.RESPONSE_RECEIVED_AT_CLIENT, "Response Received at Client",
        kwargs={"job": (True, Job), "client_id": (True, UUID), "worker_id": (True, UUID)},
        emitter_types=[Agent.NETWORK],
        listener_types=[Agent.CLIENT, Agent.LOGGER, Agent.VERIFIER]),
}