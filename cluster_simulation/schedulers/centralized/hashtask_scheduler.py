import core.configs.gen_config as gcfg
from core.events.base import *
from core.events.centralized_scheduler_events import *
from core.events.worker_events import *

from schedulers.centralized.scheduler import Scheduler
from schedulers.algo.nexus_algo import NexusSLOSplitter

import numpy as np
import scipy

rng = np.random.default_rng(seed=42)


class HashTaskScheduler(Scheduler):
    """
        Scheduler class for a round robin centralized task scheduler.
    """

    def __init__(self, simulation, herd_assignment=None):
        super().__init__(simulation, herd_assignment)

        # for round robin tracking
        self.last_worker_idx = {}

        self.last_change = 0
        self.last_update = 0

    def schedule_job_on_arrival(self, job, current_time):
        if gcfg.DROP_RATE and job.job_type_id in gcfg.DROP_RATE:
            if rng.binomial(n=1, p=gcfg.DROP_RATE[job.job_type_id]):
                return [EventOrders(
                        current_time, 
                        SchedulerDropJob(
                            self.simulation, 
                            job, 
                            job.tasks[0], 
                            job.tasks[0].get_task_deadline()))]

        if gcfg.SLO_TYPE == "NEXUS" or gcfg.SLO_TYPE == "NEXUS_DYNAMIC":
            if current_time > 1000:
                if job.job_type_id not in self.workflow_task_slos:
                    self.workflow_task_slos[job.job_type_id] = NexusSLOSplitter.generate_task_slos(
                        current_time, 1000, self.simulation, self.simulation.workflows[job.job_type_id], job.slo)

                    for task_id in sorted(self.workflow_task_slos[job.job_type_id].keys()):
                        self.task_slo_log.loc[len(self.task_slo_log)] = [current_time, 
                                                                         job.job_type_id, 
                                                                         task_id,
                                                                         self.workflow_task_slos[job.job_type_id][task_id][0],
                                                                         self.workflow_task_slos[job.job_type_id][task_id][1]]

                elif gcfg.SLO_TYPE == "NEXUS_DYNAMIC":
                    prev_slos = self.workflow_task_slos[job.job_type_id].copy()

                    NexusSLOSplitter.redistribute_task_slos(
                        current_time, self.simulation, self.simulation.workflows[job.job_type_id],
                        self.workflow_task_slos[job.job_type_id], 15)
                    
                    for task_id in sorted(self.workflow_task_slos[job.job_type_id].keys()):
                        if self.workflow_task_slos[job.job_type_id][task_id] != prev_slos[task_id]:
                            self.task_slo_log.loc[len(self.task_slo_log)] = [
                                current_time, job.job_type_id, task_id, 
                                self.workflow_task_slos[job.job_type_id][task_id][0],
                                self.workflow_task_slos[job.job_type_id][task_id][1]]

            for task in job.tasks:
                if job.job_type_id in self.workflow_task_slos:
                    task.slo = self.workflow_task_slos[job.job_type_id][task.task_id][0]
                else:
                    task.slo = job.slo

        super().schedule_job_on_arrival(job, current_time)

        self._assign_adfg(job.tasks, current_time)

        task_arrival_events = []

        initial_tasks = [task for task in job.tasks if len(task.required_task_ids) == 0]

        if gcfg.DROP_POLICY == "CLUSTER_ADMISSION_LIMIT" and current_time - self.last_update > 1000:
            for task in initial_tasks:
                worker_index = task.ADFG[task.task_id]

                samples = self.simulation.queue_size_samples[(self.simulation.queue_size_samples["model_id"]==task.model_data.id) & \
                                                             (self.simulation.queue_size_samples["worker_id"]==self.simulation.workers[worker_index].id) & \
                                                             (current_time - self.simulation.queue_size_samples["time"] <= 1000)]
                if len(samples) < 2:
                    break

                slope, intercept, r, p, se = scipy.stats.linregress(samples["time"], samples["queue_size"])
                is_overloaded = (slope > 0) and (p < 0.05)

                if is_overloaded:
                    if job.job_type_id not in gcfg.DROP_RATE:
                        gcfg.DROP_RATE[job.job_type_id] = 0.01
                    else:
                        gcfg.DROP_RATE[job.job_type_id] *= 2
                    
                    self.last_update = current_time

        for task in initial_tasks:
            task_arrival_time = current_time # + CPU_to_CPU_delay(task.input_size)
            worker_index = task.ADFG[task.task_id]
            task_arrival_events.append(EventOrders(
                task_arrival_time, TaskArrival(self.simulation, self.simulation.workers[worker_index], task, task.job.id)))

        return task_arrival_events

    def schedule_tasks_on_arrival(self, tasks, current_time):
        super().schedule_tasks_on_arrival(tasks, current_time)

        task_arrival_events = []

        self._assign_adfg(tasks, current_time)
        
        for task in tasks:
            task_arrival_time = current_time # + CPU_to_CPU_delay(task.input_size)
            worker_index = task.ADFG[task.task_id]
            task_arrival_events.append(EventOrders(
                task_arrival_time, TaskArrival(self.simulation, self.simulation.workers[worker_index], task, task.job.id)))

        return task_arrival_events
    
    def _assign_adfg(self, tasks, current_time):
        for task in tasks:
            if gcfg.DISPATCH_POLICY == "ROUND_ROBIN":
                candidate_worker_idx = 0
                if task.model_data.id in self.last_worker_idx:
                    candidate_worker_idx = (self.last_worker_idx[task.model_data.id] + 1) % len(self.simulation.worker_ids_by_creation)

                candidate_worker_id = self.simulation.worker_ids_by_creation[candidate_worker_idx]

                # don't choose worker without the required model
                while task.model_data and \
                    all(s.model.data.id != task.model_data.id for s in
                        self.simulation.workers[candidate_worker_id].GPU_state.state_at(current_time)):

                    candidate_worker_idx = (candidate_worker_idx + 1) % len(self.simulation.worker_ids_by_creation)
                    candidate_worker_id = self.simulation.worker_ids_by_creation[candidate_worker_idx]

                self.last_worker_idx[task.model_data.id] = candidate_worker_idx

            elif gcfg.DISPATCH_POLICY == "HEFT":
                candidate_worker_id = min(self.simulation.worker_ids_by_creation,
                                          key=lambda wid: self.simulation.workers[wid].get_avg_model_queue_len(
                                            current_time, 
                                            task.model_data.id,
                                            info_staleness=gcfg.LOAD_INFORMATION_STALENESS))

            task.ADFG[task.task_id] = candidate_worker_id
            task.job.ADFG[task.task_id] = task.ADFG