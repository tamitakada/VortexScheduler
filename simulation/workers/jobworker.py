from queue import Queue
from workers.worker import *
from core.events import *


""" ---------  ---------------------------------  Worker for   JOB SCHEDULER   ------------------------- ----------"""

class JobWorker(Worker):
    def __init__(self, simulation, num_free_slots, worker_id):
        super().__init__(simulation, num_free_slots, worker_id)
        self.slots_timer = [0 for i in range(num_free_slots)]
        # [(time1,working slot1 num), (time2, working slots2 num), ... ]
        self.slots_track = []
        self.queued_jobs = Queue()  # create a FIFO queue
        self.num_queued_jobs = 0
        self.queue_track = [(0, 0)]

    def add_job(self, current_time, job):
        self.queued_jobs.put(job)
        self.num_queued_jobs += 1

        self.queue_track_update(current_time)

        return self.maybe_start_job(current_time)

    def free_slot(self, current_time, job):
        """ Frees a slot on the worker and attempts to launch another task in that slot. """
        self.free_slots = self.num_free_slots
        self.update_slots_track(current_time)
        get_job_events = self.maybe_start_job(current_time)
        return get_job_events

    def maybe_start_job(self, current_time):
        while not self.queued_jobs.empty() and self.free_slots > 0:
            # Account for "running" task
            job = self.queued_jobs.get()
            self.num_queued_jobs -= 1
            completion_time = self.local_schedule_job(
                job, current_time)
            self.free_slots = 0
            self.update_slots_track(current_time)
            job.end_time = completion_time
            self.queue_track_update(completion_time)
            self.simulation.remaining_jobs -= 1
            return [EventOrders(completion_time, JobEndEvent(self, job))]
        return []

    def local_schedule_job(self, job, current_time):
        """
        Implementation of a local greedy scheduler
        """
        task_slot = {i: None for i in range(self.total_num_slots)}
        incomplete_tasks = [j for j in job.tasks]
        cur_id = 0
        while len(job.completed_tasks) != len(job.tasks):
            task = job.tasks[cur_id]
            pre_time = task.acquired_pre_tasks(
                current_time, job.completed_tasks)

            if (not job.finished_task(task)) and pre_time != -1:
                earliest_slot = self.slots_timer.index(min(self.slots_timer))
                time = max(current_time, pre_time, min(self.slots_timer))
                objects_fetching = self.fetch_model(
                    task.model, time) + self.fetch_task(task, True)
                task_time = task.task_exec_duration + objects_fetching
                task_end_time = time + task.task_exec_duration + objects_fetching
                if min(self.slots_timer) < time:
                    task_end_time = time + task_time
                else:
                    task_end_time = self.slots_timer[earliest_slot] + task_time
                self.slots_timer[earliest_slot] = task_end_time
                job.completed_tasks.append((task, task_end_time))
                task_slot[task.task_id] = earliest_slot
            cur_id += 1

            if cur_id > len(incomplete_tasks) - 1:
                cur_id = 0
        return max(self.slots_timer)

    def queue_track_update(self, current_time):
        assert self.num_queued_jobs >= 0
        for i in range(len(self.queue_track)-1, -1, -1):
            if self.queue_track[i][0] == current_time:
                self.queue_track[i] = (current_time, self.num_queued_jobs)
                return
            if self.queue_track[i][0] < current_time:
                self.queue_track.insert(
                    i+1, (current_time, self.num_queued_jobs))
                return
        self.queue_track.append((current_time, self.num_queued_jobs))
        return

    def update_slots_track(self, current_time):
        for i in range(len(self.slots_track)):
            if current_time == self.slots_track[i][0]:
                self.slots_track[i] = (
                    self.slots_track[i][0], self.total_num_slots - self.free_slots)
                return
        self.slots_track.append(
            (current_time, self.total_num_slots - self.free_slots))
        self.slots_track.sort(key=lambda x: x[0])

    def find_finish_time(self, task, finished_tasks):
        for f_t in finished_tasks:
            if f_t[0].task_id == task.task_id:
                return f_t[1]
        raise Exception("task not finished")

    def worker_queue_num(self, current_time, info_staleness):
        delayed_time = current_time - info_staleness
        if len(self.queue_track) == 0:
            return 0
        for i in range(len(self.queue_track)-1, -1, -1):
            if self.queue_track[i][0] == current_time or self.queue_track[i][0] == delayed_time:
                return self.queue_track[i][1]
            if self.queue_track[i][0] < delayed_time:
                return self.queue_track[i][1]
        return self.queue_track[-1][1]
