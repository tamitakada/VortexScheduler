import core.configs.gen_config as gcfg

from queue import PriorityQueue
from queue_management.queued_task import QueuedTask
from queue_management.batching import TaskBatcher

from events.event_manager import EventManager
from events.event import *
from events.event_types import *

from core.network import *
from core.data_models.model_data import ModelData
from core.batch import Batch

from workers.gpu_state import ModelState, GPUState
from schedulers.algo.batching_policies import get_batch

from uuid import UUID


class Worker(EventListener):

    _abandoned_batches = []

    def __init__(self, id: UUID, em: EventManager, total_memory_gb: int, create_time: float):
        super().__init__(Agent.WORKER)

        self.id = id
        self.em = em
        self.total_memory_gb = total_memory_gb
        self.create_time = create_time
        self.GPU_state = GPUState(total_memory_gb * (10**6))

        self.em.register_listener(self, {
            EVENT_TYPES[EventIds.TASKS_SCHEDULED_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_ARRIVAL_AT_WORKER],
            EVENT_TYPES[EventIds.BATCH_FINISHED_AT_WORKER],
            EVENT_TYPES[EventIds.JOBS_DROPPED],
        })

        self.emitter_id = self.em.register_emitter(Agent.WORKER, {
            EVENT_TYPES[EventIds.TASKS_SENT_TO_SCHEDULER],
            EVENT_TYPES[EventIds.TASKS_SENT_TO_WORKER],
            EVENT_TYPES[EventIds.BATCH_STARTED_AT_WORKER],
            EVENT_TYPES[EventIds.BATCH_FINISHED_AT_WORKER],
            EVENT_TYPES[EventIds.JOBS_DROPPED],
            EVENT_TYPES[EventIds.RESPONSE_SENT_TO_CLIENT]
        })

        self.queues: dict[int, PriorityQueue] = {}
        self.completed_tasks: dict[tuple[int, int], Task] = {}
        self.scheduled_task_to_worker: dict[tuple[int, int], UUID] = {} # (job ID, task ID) -> worker ID

    def on_event(self, event: Event):
        if event.type.id == EventIds.TASKS_SCHEDULED_TO_WORKER:
            if event.kwargs["worker_id"] != self.id:
                return

            self.on_tasks_scheduled(event.time, event.kwargs["job_task_ids"], 
                                    event.kwargs["worker_id"])
        
        elif event.type.id == EventIds.TASKS_ARRIVAL_AT_WORKER:
            if event.kwargs["worker_id"] != self.id:
                return

            self.on_tasks_arrival(event.time, event.kwargs["tasks"], event.kwargs["force_instance_id"])

        elif event.type.id == EventIds.JOBS_DROPPED:
            self._drop_tasks(event.kwargs["job_ids"])

        elif event.type.id == EventIds.BATCH_FINISHED_AT_WORKER:
            if event.kwargs["worker_id"] != self.id:
                return
            
            self.on_batch_finish(event.time, event.kwargs["batch"], event.kwargs["model_instance_id"])

        else:
            raise ValueError(f"Worker received unregistered event: {event}")
        
    def _drop_tasks(self, job_ids: list[int]):
        """Remove all tasks associated with given jobs from queues. Does not
        affect executing batches.

        Args:
            job_ids: All IDs of jobs to drop
        """
        for q in self.queues.values():
            filtered = []
            while q.qsize() > 0:
                task: Task = q.get()
                if task.job.id not in job_ids: 
                    filtered.append(task)
            
            for t in filtered: q.put(t)
        
    def on_tasks_scheduled(self, time: float, job_task_ids: list[tuple[int, int]], 
                           worker_id: UUID):
        
        tasks_to_send = []
        for (jid, tid) in job_task_ids:
            self.scheduled_task_to_worker[(jid, tid)] = worker_id
            if (jid, tid) in self.completed_tasks:
                tasks_to_send.append((jid, tid))
        
        if tasks_to_send:
            self.em.add_event(
                Event(time, 
                      EVENT_TYPES[EventIds.TASKS_SENT_TO_WORKER],
                      kwargs={"tasks": tasks_to_send, 
                              "from_worker_id": self.id, 
                              "to_worker_id": worker_id}),
                self.emitter_id)
    
    def on_tasks_arrival(self, time: float, tasks: list[Task], force_exec_on_instance_id: UUID | None):
        """On arrival of tasks to execute, attempts to start batch execution if idle models
        exist.

        Args:
            time: Current time
            tasks: Tasks that arrived at worker at given time
            force_exec_on_instance_id: If specified, disables worker queueing and attempts to
            execute arrived tasks as a single batch on the required model instance.
        """
        
        if force_exec_on_instance_id:
            state = self.GPU_state.get_instance_state(force_exec_on_instance_id, time)
            assert(state.reserved_batch == None)

            self.exec_batch(time, Batch(tasks), force_exec_on_instance_id)
        
        else:
            for task in tasks:
                if task.model_data.id not in self.queues:
                    self.queues[task.model_data.id] = PriorityQueue()
                self.queues[task.model_data.id].put(QueuedTask(task))

            models_needed = [mid for mid in self.queues.keys() if self.queues[mid].qsize() > 0]

            states = self.GPU_state.state_at(time)
            for state in states:
                # if idle copy of required model exists start batch
                if state.model.data.id in models_needed and not state.reserved_batch:
                    batch = TaskBatcher.get_batch(time, 
                                                self.total_memory_gb, 
                                                self.queues[state.model.data.id],
                                                False)
                    TaskBatcher.dequeue_batch(batch, self.queues[state.model.data.id])

                    self.exec_batch(time, batch, instance_id=state.model.id)
                    
                    if self.queues[state.model.data.id].qsize() == 0:
                        models_needed.remove(state.model.data.id)


    def on_batch_finish(self, time: float, batch: Batch, instance_id):
        assert(not self.did_abandon_batch(batch.id))

        # release model
        if batch.model_data != None:
            self.GPU_state.release_busy_model(batch.id, time)

        # if any jobs are complete, send response to client
        for task in batch.tasks:
            task.job.set_completion_time(time, task.task_id)
            if task.job.is_complete():
                self.em.add_event(
                    Event(time,
                          EVENT_TYPES[EventIds.RESPONSE_SENT_TO_CLIENT],
                          kwargs={"job": task.job,
                                  "ignore_transfer_time": gcfg.DISPATCH_POLICY == "SHEPHERD_PERFECT",
                                  "client_id": task.job.client_id,
                                  "worker_id": self.id}),
                    self.emitter_id)

        # for perfect scheduler, send new available tasks at no cost to scheduler
        if gcfg.DISPATCH_POLICY == "SHEPHERD_PERFECT":
            all_available = [nt for task in batch.tasks for nt in task.job.newly_available_tasks(task)]
            if all_available:
                self.em.add_event(
                    Event(time,
                        EVENT_TYPES[EventIds.TASKS_SENT_TO_SCHEDULER],
                        kwargs={"tasks": all_available, 
                                "from_worker_id": self.id,
                                "ignore_transfer_time": True}),
                    self.emitter_id)
            
        else:
            # if any tasks have been scheduled, send to next worker
            tasks_to_send: dict[UUID, list[Task]] = {}
            for task in batch.tasks:
                
                self.completed_tasks[(task.job.id, task.task_id)] = task
                if (task.job.id, task.task_id) in self.scheduled_task_to_worker:
                    worker_id = self.scheduled_task_to_worker[(task.job.id, task.task_id)]
                    if worker_id not in tasks_to_send: tasks_to_send[worker_id] = []
                    tasks_to_send[worker_id].append(task)
                
            for (worker_id, tasks) in tasks_to_send.items():
                self.em.add_event(
                    Event(time, 
                        EVENT_TYPES[EventIds.TASKS_SENT_TO_WORKER],
                        kwargs={"tasks": tasks, 
                                "from_worker_id": self.id, 
                                "to_worker_id": worker_id}),
                    self.emitter_id)
            
            # if queue is not empty, start a new batch
            instance_state = self.GPU_state.get_instance_state(instance_id, time)
            if self.queues[instance_state.model.data.id].qsize() > 0:
                batch = TaskBatcher.get_batch(time, 
                                            self.total_memory_gb,
                                            self.queues[instance_state.model.data.id],
                                            False)
                TaskBatcher.dequeue_batch(batch, self.queues[instance_state.model.data.id])
                
                self.exec_batch(time, batch, instance_id)

    def exec_batch(self, time: float, batch: Batch, instance_id=None):
        """Reserves an idle copy of any required GPU models and executes the given batch.

        Args:
            time: Time to start batch execution
            batch: Batch to execute
            instance_id: Optional ID of model instance to execute
        """

        batch_exec_time = batch.model_data.get_randomized_exec_time(
            batch.size(), self.total_memory_gb)
        reserved_instance_id = None

        if batch.model_data != None:
            assert(self.GPU_state.does_have_idle_copy(batch.model_data.id, time))

            if instance_id:
                reserved_instance_id = instance_id
                self.GPU_state.reserve_instance(instance_id, time, batch, batch_exec_time)
            else:
                reserved_instance_id = self.GPU_state.reserve_idle_copy(
                    batch.model_data, time, batch, batch_exec_time)

            # verify reserved instance
            reserved_state = [s for s in self.GPU_state.state_at(time)
                                if s.model.id == reserved_instance_id][0]
            assert(reserved_state.reserved_until >= (time + batch_exec_time))
            assert(reserved_state.reserved_batch == batch)
            assert(reserved_state.model.data.id == batch.model_data.id)
            assert(reserved_state.state == ModelState.PLACED)

            batch_exec_time = reserved_state.reserved_until - time
        
        for task in batch.tasks:
            task.log.task_front_queue_timestamp = time
            task.log.task_execution_start_timestamp = time

        task_end_time = time + batch_exec_time + \
            SameMachineCPUtoGPU_delay(sum(t.input_size for t in batch.tasks)) + \
            SameMachineGPUtoCPU_delay(sum(t.result_size for t in batch.tasks))
        
        self.em.add_event(
            Event(time, 
                  EVENT_TYPES[EventIds.BATCH_STARTED_AT_WORKER],
                  kwargs={"batch": batch, 
                          "model_instance_id": reserved_instance_id, 
                          "worker_id": self.id}),
            self.emitter_id)
        
        self.em.add_event(
            Event(task_end_time, 
                  EVENT_TYPES[EventIds.BATCH_FINISHED_AT_WORKER],
                  kwargs={"batch": batch, 
                          "model_instance_id": reserved_instance_id, 
                          "worker_id": self.id}),
            self.emitter_id)
        
    def did_abandon_batch(self, batch_id: int):
        return batch_id in Worker._abandoned_batches

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"[WORKER {self.id}] [STATE: {self.GPU_state}]"
    
    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, Worker) and self.id == other.id