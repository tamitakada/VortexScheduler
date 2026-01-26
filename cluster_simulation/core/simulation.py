'''
This simulation experiment framework of event generation 
is referenced from Sparrow: https://github.com/radlab/sparrow 
'''

import numpy as np
import pandas as pd

from queue import PriorityQueue
from collections import Counter

import core.configs.model_config as mcfg
import core.configs.gen_config as gcfg
from core.print_utils import *
from core.external_client import *
from core.data_models.workflow import Workflow
from core.data_models.model_data import ModelData

from core.events.base import *
from core.events.centralized_scheduler_events import *
from core.events.worker_events import *

from workers.heft_task_worker import *
from workers.shepherd_task_worker import *
from workers.hash_task_worker import *

from schedulers.algo.herd_algo import get_herd_assignment
from schedulers.centralized.shepherd.herd_assignment import HerdAssignment
from core.allocation import ModelAllocation

import uuid


class Simulation(object):
    def __init__(
            self,
            simulation_name, 
            centralized_scheduler=False,  
            dynamic_adjust=False,
            job_types_list=[0],
            produce_breakdown=False,
            inferline=None
    ):
        self.inferline = inferline

        self.simulation_name = simulation_name
        self.centralized_scheduler = centralized_scheduler
        self.job_types_list = job_types_list
        self.dynamic_adjust = dynamic_adjust

        self.workers = {} # uuid -> Worker
        self.worker_ids_by_creation = [] # global ordering of workers

        self.external_clients = []

        self.models: list[ModelData] = self.generate_models()
        self.workflows: list[Workflow] = self.generate_workflows()
        
        self.remaining_jobs = sum(cc[jt]["NUM_JOBS"] for cc in gcfg.CLIENT_CONFIGS for jt in cc.keys())
        self.event_queue = PriorityQueue()

        self.jobs = {}
        
        # Tracking measurements
        self.result_to_export = pd.DataFrame()
        self.tasks_logging_times = pd.DataFrame()
        self.event_log = pd.DataFrame(columns=["time", "worker_id", "event"])
        self.batch_exec_log = pd.DataFrame(columns=["batch_id","start_time", "end_time", "worker_id", "model_id", "batch_size", "job_ids"])
        self.task_drop_log = pd.DataFrame(columns=["client_id", "job_id", "workflow_id", "task_id", "drop_time", "create_time",
                                                   "arrival_time", "slo", "deadline"])
        
        # TODO: client ID to worker mets
        self.task_exec_log = pd.DataFrame(columns=["client_id", "workflow_id", "job_id", "task_id", "worker_id", 
                                                   "worker_arrival_time", "exec_start_time", "exec_end_time", "deadline"])
        self.task_arrival_log = pd.DataFrame(columns=["time", "client_id", "workflow_id", "job_id", "task_id", "model_id", "worker_id"])
        self.worker_metrics_log = pd.DataFrame(columns=["time", "sampled_interval_ms", "worker_id", "workflow_id",
                                                        "task_id", "task_arrival_rate", "task_throughput"])
        self.worker_log = pd.DataFrame(columns=["time", "add_or_remove", "worker_id"])
        self.worker_model_log = pd.DataFrame(columns=["start_time", "end_time", "worker_id", "model_id", "instance_id", "placed_or_evicted"])

        self.allocation_logs = []

        print("---- SIMULATION : " + self.simulation_name + "----")
        self.produce_breakdown =  produce_breakdown

    def add_task_arrival_to_worker_metrics(self, time: float, task: Task, worker: Worker | None):
        """
        Update task arrival metrics by logging newly arrived [task] at [worker].
        """
        logged_worker_id = "N/A"
        if worker:
            logged_worker_id = worker.id

        if ((self.task_arrival_log["job_id"]==task.job.id) & \
            (self.task_arrival_log["task_id"]==task.task_id) & \
            (self.task_arrival_log["worker_id"]==logged_worker_id)).any():

            print(f"[Simulation] Already logged Job {task.job.id}, Task {task.task_id} arrival @ Worker {logged_worker_id}")
            return
        
        self.task_arrival_log.loc[len(self.task_arrival_log)] = \
            [time, task.job.client_id, task.job.job_type_id, task.job.id, task.task_id, task.model_data.id, logged_worker_id]

    def add_task_exec_to_worker_metrics(self, task: Task, worker: Worker, deadline: float):
        """
        Update task exec metrics by logging newly arrived [task] at [worker].
        """
        if ((self.task_exec_log["job_id"]==task.job.id) & (self.task_exec_log["task_id"]==task.task_id)).any():
            print(f"[Simulation] Already logged Job {task.job.id}, Task {task.task_id} exec")
            return
        
        self.task_exec_log.loc[len(self.task_exec_log)] = \
            [task.job.client_id, task.job.job_type_id, task.job.id, task.task_id, worker.id,
             task.log.task_placed_on_worker_queue_timestamp, task.log.task_execution_start_timestamp, task.log.task_execution_end_timestamp, deadline]
        
    def add_worker_metrics_sample(self, time: float, interval: float):
        """
        Log a new sample of worker metrics over the past [interval] ms.
        """
        for worker in self.workers.values():
            exec_log = self.task_exec_log[(self.task_exec_log["worker_id"]==worker.id) & (self.task_exec_log["exec_end_time"] <= time) & (self.task_exec_log["exec_end_time"] > time-interval)]
            arrival_log = self.task_arrival_log[(self.task_arrival_log["worker_id"]==worker.id) & (self.task_arrival_log["time"] <= time) & (self.task_arrival_log["time"] > time-interval)]
            arrived_workflows = set(arrival_log["workflow_id"])

            for wf in arrived_workflows:
                wf_arrival_log = arrival_log[arrival_log["workflow_id"]==wf]
                arrived_tasks = set(wf_arrival_log["task_id"])

                for task in arrived_tasks:
                    worker_task_arrival_log = wf_arrival_log[wf_arrival_log["task_id"]==task]
                    arrival_rate = len(worker_task_arrival_log) / interval * 1000
                    throughput = ((exec_log["workflow_id"]==wf) & (exec_log["task_id"]==task)).sum() / interval * 1000
                    self.worker_metrics_log.loc[len(self.worker_metrics_log)] = \
                        [time, interval, worker.id, wf, task, arrival_rate, throughput]

    def _get_avg_over_time(self, time: float, time_frame: float, samples: int, get_sample):
        avg = 0
        for i in range(samples):
            sample_start = time - time_frame * (i + 1)
            sample_end = sample_start + time_frame

            if sample_start < 0:
                return avg / i if i > 0 else 0

            avg += get_sample(sample_start, sample_end)
        return avg / samples

    def get_drop_rate(self, time: float, workflow_id: int, time_frame: float, samples: int):
        def _get_drop_rate_at(start, end):
            dropped_jobs = ((self.task_drop_log["workflow_id"]==workflow_id) & \
                            (self.task_drop_log["drop_time"] >= start) & \
                            (self.task_drop_log["drop_time"] < end)).sum()
            drop_rate = dropped_jobs / time_frame * 1000
            return drop_rate
        
        return self._get_avg_over_time(
            time, time_frame, samples, _get_drop_rate_at)
    
    def get_throughput(self, time: float, workflow_id: int, time_frame: float, samples: int):
        def _get_tput_at(start, end):
            complete_jobs = ((self.task_exec_log["workflow_id"]==workflow_id) & \
                            (self.task_exec_log["exec_end_time"] >= start) & \
                            (self.task_exec_log["exec_end_time"] < end)).sum()
            tput = complete_jobs / time_frame * 1000
            return tput
        
        return self._get_avg_over_time(
            time, time_frame, samples, _get_tput_at)
    
    def get_goodput(self, time: float, workflow_id: int, time_frame: float, samples: int):
        def _get_gput_at(start, end):
            nontardy_jobs = ((self.task_exec_log["workflow_id"]==workflow_id) & \
                            (self.task_exec_log["exec_end_time"] >= start) & \
                            (self.task_exec_log["exec_end_time"] < end) & \
                            (self.task_exec_log["exec_end_time"] < self.task_exec_log["deadline"])).sum()
            gput = nontardy_jobs / time_frame * 1000
            return gput
        
        return self._get_avg_over_time(
            time, time_frame, samples, _get_gput_at)
    
    def get_goodput_throughput_diff(self, time: float, workflow_id: int, time_frame: float, samples: int):
        def _get_diff_at(start, end):
            nontardy_jobs = ((self.task_exec_log["workflow_id"]==workflow_id) & \
                            (self.task_exec_log["exec_end_time"] >= start) & \
                            (self.task_exec_log["exec_end_time"] < end) & \
                            (self.task_exec_log["exec_end_time"] < self.task_exec_log["deadline"])).sum()
            gput = nontardy_jobs / time_frame * 1000

            complete_jobs = ((self.task_exec_log["workflow_id"]==workflow_id) & \
                            (self.task_exec_log["exec_end_time"] >= start) & \
                            (self.task_exec_log["exec_end_time"] < end)).sum()
            tput = complete_jobs / time_frame * 1000

            return tput - gput
        
        return self._get_avg_over_time(
            time, time_frame, samples, _get_diff_at)
    
    def add_worker(self, time: float, id: str, memory_size: int, model_ids_to_load: list[int]) -> list[EventOrders]:
        new_worker = HashTaskWorker(self, id, memory_size, created_at=time)
        self.workers[new_worker.id] = new_worker
        self.worker_ids_by_creation.append(new_worker.id)

        events = []
        for model_id in model_ids_to_load:
            events += new_worker.fetch_model(self.models[model_id], None, time)

        self.worker_log.loc[len(self.worker_log)] = {
            "time": time,
            "add_or_remove": "add",
            "worker_id": new_worker.id
        }
        return events
    
    def remove_worker(self, time: float, worker_id: str):
        worker = self.workers[worker_id]
        
        events = []

        # handle currently executing tasks
        curr_batches = [s.reserved_batch for s in worker.GPU_state.state_at(time)
                        if s.reserved_batch != None]
        for batch in curr_batches:
            evicted_batch = worker.evict_batch(batch.id, time)
            
            events.append(EventOrders(
                time + CPU_to_CPU_delay(evicted_batch.tasks[0].input_size * evicted_batch.size()), 
                TasksArrivalAtScheduler(self, evicted_batch.tasks)))
        
        # handle currently queued tasks
        model_ids = worker.queue_history.keys()
        for mid in model_ids:
            task_queue = worker.get_queue_history(time, mid, info_staleness=0)
            if task_queue:
                events.append(EventOrders(
                    time + CPU_to_CPU_delay(sum(t.input_size for t in task_queue)),
                    TasksArrivalAtScheduler(self, task_queue)))

        self.worker_ids_by_creation.remove(worker_id)
        self.workers.pop(worker_id)

        self.worker_log.loc[len(self.worker_log)] = {
            "time": time,
            "add_or_remove": "remove",
            "worker_id": worker_id
        }

        return events

    def run_herd_scheduler(self, current_time: float):
        # TODO: fix

        """
            Initializes a new HerdAssignment and stores in [self.herd_assignment].
        """
        curr_send_rates = {}
        for jt in self.job_types_list:
            jobs_since_last_sched = [j for j in self.jobs.values() if j.job_type_id == jt and \
                                        j.tasks[0].log.task_arrival_at_scheduler_timestamp > (current_time - HERD_PERIODICITY) and \
                                        j.tasks[0].log.task_arrival_at_scheduler_timestamp <= current_time]
            curr_send_rates[jt] = 5 if current_time == 0 else (len(jobs_since_last_sched) / HERD_PERIODICITY * 1000 + 5) 
            # TODO: minimum send rate? 
        self.workers = {}

        task_types = get_task_types(self.job_types_list)
        print(self.job_types_list)
        print(task_types)

        #TODO TODO TODO
        all_models = []
            # get_model_for_task(self, 0, 0),
            # get_model_for_task(self, 0, 1),
            # get_model_for_task(self, 0, 2),
            # get_model_for_task(self, 0, 3)]
        # all_models = [m for jt in self.job_types_list for m in models_by_wf[jt]]
        task_tputs = {(0,0): 270, (0,1): 45, (0,2): 270, (0,3): 70,
                        (1,0): 125, (1,1): 7555, (1,2): 92, (1,3): 4.82}
        (group_sizes, stream_groups) = get_herd_assignment(
            task_types, all_models, task_tputs, curr_send_rates)
        
        worker_groups = []
        worker_counter = 0
        for i, group_size in enumerate(group_sizes):
            group_workers = []
            if self.simulation_name == "shepherd":
                group_workers = [ShepherdWorker(self, worker_counter+j, 24, i) for j in range(int(group_size))]
            elif self.simulation_name in ["nexus", "hashtask"]:
                group_workers = [HashTaskWorker(self, worker_counter+j, 24, group_id=i) for j in range(int(group_size))]
            else:
                group_workers = [HeftTaskWorker(self, worker_counter+j, 24, group_id=i) for j in range(int(group_size))]

            worker_counter += len(group_workers)

            for grp in group_workers:
                for w in grp:
                    self.workers[w.id] = w
            
            worker_groups.append(group_workers)

        task_type_assignments = {}
        for (sid, group_id) in stream_groups:
            task_type_assignments[task_types[sid]] = group_id

        self.herd_assignment = HerdAssignment(worker_groups, task_type_assignments)
        self.allocation_logs.append((current_time, str(self.herd_assignment)))

        # TODO load all models in group
        # if current_time == 0 and ENABLE_MODEL_PREFETCH:
        #     for worker in self.workers:
        #         # randomly choose a model to prefetch
        #         group_model_ids = []
        #         if self.simulation_name == "shepherd":
        #             group_model_ids = self.herd_assignment.group_models[worker.group_id]
        #         else:
        #             group_model_ids = [get_model_id_for_task_type(tt) for tt in task_types]

        #         if group_model_ids:
        #             preloaded_model_id = np.random.choice(list(group_model_ids))
        #             preloaded_model = [m for ms in models_by_wf for m in ms if m.model_id == preloaded_model_id][0]
        #             print(f"W{worker.worker_id} PRELOADED {preloaded_model_id}")
        #             worker.GPU_state.prefetch_model(preloaded_model)
                

    def initialize_model_placement_at_workers(self) -> ModelAllocation:
        """
            Returns a list of worker configurations (partition size, [model ids])
            based on [gcfg.ALLOCATION_STRATEGY].
        """
        if gcfg.ALLOCATION_STRATEGY == "CUSTOM":
            assert(all((psize*(10**6)) in gcfg.VALID_WORKER_SIZES for psize, _ in gcfg.CUSTOM_ALLOCATION))
            return ModelAllocation(
                self,
                {uuid.uuid4(): cfg for cfg in gcfg.CUSTOM_ALLOCATION},
                reset_batch_sizes=False)
        
        elif gcfg.ALLOCATION_STRATEGY == "INFERLINE":
            # TODO: add support for multitenant
            assert(len(gcfg.CLIENT_CONFIGS) == 1 and len(gcfg.CLIENT_CONFIGS[0].keys()) == 1)

            slo = list(gcfg.CLIENT_CONFIGS[0].values())[0]["SLO"]
            alloc = self.inferline.planner_minimize_cost(self, 0, self.workflows[0], slo)
            print("ALLOCATION IS ", alloc.worker_cfgs)

            return alloc
            
        return None
    
    def initialize_workers(self):
        """
            Creates all workers and initializes model placement.
        """
        if gcfg.ALLOCATION_STRATEGY == "HERD":
            self.run_herd_scheduler(0)
            self.initialize_external_clients()
            if gcfg.HERD_PERIODICITY > 0:
                self.event_queue.put(EventOrders(
                    gcfg.HERD_PERIODICITY, StartHerdSchedulerRerun(self)))
        else:
            self.allocation = self.initialize_model_placement_at_workers()
            self.workers = {}
            self.worker_ids_by_creation = []
            for (wid, _) in self.allocation.worker_ids_by_create_time:
                cfg = self.allocation.worker_cfgs[wid]
                worker = HashTaskWorker(self, wid, cfg[0], created_at=0)
                self.workers[wid] = worker
                self.worker_ids_by_creation.append(wid)

                self.worker_log.loc[len(self.worker_log)] = {
                    "time": 0,
                    "add_or_remove": "add",
                    "worker_id": wid
                }

                for mid in cfg[1]:
                    self.models[mid].max_batch_size = self.allocation.models[mid].max_batch_size
                    instance_id = self.workers[wid].GPU_state.prefetch_model(self.models[mid])

                    self.worker_model_log.loc[len(self.worker_model_log)] = {
                        "start_time": 0,
                        "end_time": 0, # prefetch special case assume no overhead
                        "worker_id": wid,
                        "model_id": mid,
                        "instance_id": instance_id,
                        "placed_or_evicted": "placed"
                    }

            for model in self.workflows[0].get_models():
                mcfg.MODELS[model.id]["MAX_BATCH_SIZE"] = self.models[mid].max_batch_size
            
            self.herd_assignment = HerdAssignment(
                [self.workers.values()],
                { tt: 0 for tt in get_task_types(self.job_types_list) })
            self.initialize_external_clients()

    def initialize_external_clients(self):
        for i, client_config in enumerate(gcfg.CLIENT_CONFIGS):
            self.external_clients.append(
                ExternalClient(self, i, client_config))
            
    def generate_models(self):
        """Initializes ModelData objects for all models used by jobs in [self.job_types_list].
        """
        model_ids: set[int] = set()
        for workflow_cfg in WORKFLOW_LIST:
            if workflow_cfg["JOB_TYPE"] not in self.job_types_list:
                continue
            
            for task_cfg in workflow_cfg["TASKS"]:
                if task_cfg["MODEL_ID"] >= 0:
                    model_ids.add(task_cfg["MODEL_ID"])

        return {id: ModelData(id,
                              mcfg.MODELS[id]["MODEL_SIZE"],
                              mcfg.MODELS[id]["MAX_BATCH_SIZE"],
                              mcfg.MODELS[id]["MIG_BATCH_EXEC_TIMES"],
                              mcfg.MODELS[id]["EXEC_TIME_CVS"]) 
                              for id in model_ids}

    
    def generate_workflows(self) -> list[Workflow]:
        """Initializes Workflow objects for all job types in [self.job_types_list].
        """
        return {
            cfg["JOB_TYPE"] : Workflow(self, cfg) for cfg in WORKFLOW_LIST
            if cfg["JOB_TYPE"] in self.job_types_list}
            
    def generate_all_jobs(self):
        """
            Generates exactly the number of jobs specified in [gcfg.CLIENT_CONFIGS]
            for each client at the designated send rates and places all creation
            events on the event queue.
        """

        job_id = 0
        for client in self.external_clients:
            for job_type in client.job_types:
                curr_time = 0

                curr_send_rate = client.per_job_config[job_type]["SEND_RATES"][0]
                curr_send_rate_idx = 0

                for i in range(client.per_job_config[job_type]["NUM_JOBS"]):
                    if curr_send_rate_idx < len(client.per_job_config[job_type]["SEND_RATES"]) - 1:
                        if i == sum(client.per_job_config[job_type]["SEND_RATE_CHANGE_INTERVALS"][:curr_send_rate_idx+1]):
                            curr_send_rate_idx += 1
                            curr_send_rate = client.per_job_config[job_type]["SEND_RATES"][curr_send_rate_idx]
                    
                    next_job = client.create_job(job_type, job_id, curr_send_rate, curr_time)
                    self.jobs[next_job.id] = next_job
                    curr_time = next_job.create_time + CPU_to_CPU_delay(next_job.tasks[0].input_size)

                    if self.centralized_scheduler:
                        self.event_queue.put(EventOrders(
                            curr_time, JobArrivalAtScheduler(self, next_job)))
                    else:
                        initial_worker_id = client.select_initial_worker_id()
                        self.event_queue.put(EventOrders(
                            curr_time, JobArrivalAtWorker(self, next_job, initial_worker_id)))
                        
                    job_id += 1

    def scale_cluster(self, current_time):
        if gcfg.AUTOSCALING_POLICY == "INFERLINE":
            print(f"{current_time} Running InferLine autoscaler...")

            final_delta_workers = {}
            for workflow in sorted(self.workflows.values()):
                delta_workers = self.inferline.check_scale_up(current_time, workflow, self)
                
                for (id, cfg) in delta_workers:
                    final_delta_workers[id] = cfg
                    self.last_scale = current_time

            print("AFTER SCALE UPS: ", final_delta_workers)

            if current_time - self.last_scale > gcfg.INFERLINE_TUNING_INTERVAL:
                # check scale down after all scale up checks to prevent removal
                # of necessary nodes
                for workflow in sorted(self.workflows.values()):
                    delta_workers = self.inferline.check_scale_down(current_time, workflow, self)
                    
                    for (id, cfg) in delta_workers:
                        final_delta_workers[id] = cfg

                print("AFTER SCALE DOWNS: ", final_delta_workers)

            events = []
            for id, cfg in final_delta_workers.items():
                if id in self.workers and cfg: # change loaded models
                    new_models = Counter(cfg[1])
                    curr_models = Counter([s.model.data.id for s in 
                                           self.workers[id].GPU_state.state_at(current_time)])
                    add_models = list((new_models - curr_models).elements())
                    rm_models = list((curr_models - new_models).elements())
                    
                    events.append(EventOrders(
                        current_time, 
                        UpdateLoadedModelsEvent(
                            self,
                            self.workers[id],
                            [self.models[mid] for mid in add_models],
                            [self.models[mid] for mid in rm_models])))
                elif id in self.workers: # cfg=None -> rm worker
                    events.append(EventOrders(
                        current_time,
                        RemoveWorkerFromCluster(self, id)))
                else: # add worker
                    events.append(EventOrders(
                        current_time,
                        AddWorkerToCluster(self, id, cfg[0], cfg[1])))

            if not events:
                print("No allocation changes required")

            assert(self.allocation.count(m.id) > 0 for w in self.workflows.values() for m in w.get_models())
            return events
        
        return []

    
    """
     --------------    Printing out the simulation results     --------------
    """


    def run_finish(self, last_time, by_job_type=False):
        def _get_jobs_per_sec(jobs):
            if len(jobs) == 0: return 0
            completion_time = max(j.end_time for j in jobs) - min(j.create_time for j in jobs)
            return len(jobs) / completion_time * 1000

        stats_dict = { "clients": [] } 
        for client in self.external_clients:
            stats_dict["clients"].append({})
            for job_type in client.job_types:
                stats_dict["clients"][-1][job_type] = {
                    "client_id": client.id, 
                    "slo": client.per_job_config[job_type]["SLO"] if "SLO" in client.per_job_config[job_type] else -1,
                    "num_jobs": client.per_job_config[job_type]["NUM_JOBS"]}
                
                completed_jobs = [j for j in self.jobs.values() if j.client_id == client.id and len(j.completed_tasks) == len(j.tasks)]
                stats_dict["clients"][-1][job_type]["throughput_qps"] = _get_jobs_per_sec(completed_jobs)
                stats_dict["clients"][-1][job_type]["total_num_complete"] = len(completed_jobs)

                if gcfg.SLO_GRANULARITY == "JOB" or self.simulation_name == "nexus":
                    nontardy_jobs = [j for j in completed_jobs if j.end_time <= j.create_time + j.slo]
                    stats_dict["clients"][-1][job_type]["goodput_qps"] = _get_jobs_per_sec(nontardy_jobs)

                    tardy_jobs = [j for j in completed_jobs if j.end_time > (j.create_time + j.slo)]
                    
                    stats_dict["clients"][-1][job_type]["total_num_tardy"] = len(tardy_jobs)

                    if tardy_jobs:
                        job_tardiness = [j.end_time - (j.create_time + j.slo) for j in tardy_jobs]

                        stats_dict["clients"][-1][job_type]["median_tardiness_ms"] = np.median(job_tardiness)
                        stats_dict["clients"][-1][job_type]["mean_tardiness_ms"] = np.mean(job_tardiness)
                        stats_dict["clients"][-1][job_type]["std_tardiness_ms"] = np.std(job_tardiness)

                if completed_jobs:
                    job_latencies = [j.end_time - j.create_time for j in completed_jobs]

                    stats_dict["clients"][-1][job_type]["median_latency_ms"] = np.median(job_latencies)
                    stats_dict["clients"][-1][job_type]["mean_latency_ms"] = np.mean(job_latencies)
                    stats_dict["clients"][-1][job_type]["std_latency_ms"] = np.std(job_latencies)
                    stats_dict["clients"][-1][job_type]["p99_latency_ms"] = np.percentile(job_latencies, 99)
                
                client_dropped_jobs = set(self.task_drop_log[self.task_drop_log["client_id"]==client.id]["job_id"])
                stats_dict["clients"][-1][job_type]["total_num_dropped"] = len(client_dropped_jobs)
                stats_dict["clients"][-1][job_type]["drop_rate_qps"] = len(client_dropped_jobs) / \
                    (max(j.end_time for j in completed_jobs) - min(j.create_time for j in completed_jobs)) * 1000 \
                    if len(completed_jobs) > 0 else 0
        
        self.sim_stats_log = stats_dict

        if self.produce_breakdown:
            all_completed_jobs = [j for j in self.jobs.values() if len(j.completed_tasks) == len(j.tasks)]
            self.produce_time_breakdown_results(all_completed_jobs)

    def produce_time_breakdown_results(self, completed_jobs):

        dataframe = pd.DataFrame(columns=["job_id", "client_id", "load_info_staleness", "placement_info_staleness",
                                          "workflow_type", "job_create_time", "scheduler_type", "slowdown", "response_time"])
        dataframe_tasks_log = pd.DataFrame(columns=["workflow_type", "task_id", "worker_id", "task_arrival_time", "task_start_exec_time", "time_to_buffer", "dependency_wait_time",
                                                    "time_spent_in_queue", "model_fetching_time", "execution_time"])

        for index, completed_job in enumerate(completed_jobs):

            if index < 0:  # ignore the first 50 jobs
                continue

            slowdown = (completed_job.end_time - completed_job.create_time) / \
                WORKFLOW_LIST[completed_job.job_type_id]["BEST_EXEC_TIME"]
            response_time = completed_job.end_time - completed_job.create_time
            dataframe.loc[index] = [completed_job.id, completed_job.client_id, gcfg.LOAD_INFORMATION_STALENESS, gcfg.PLACEMENT_INFORMATION_STALENESS, completed_job.job_type_id,
                                    completed_job.create_time, self.simulation_name, slowdown, response_time]

        task_index = 0
        for job in completed_jobs:
            for task in job.tasks:
                time_to_buffer = task.log.task_arrival_at_worker_buffer_timestamp - \
                    task.log.job_creation_timestamp
                dependency_wait_time = task.log.task_placed_on_worker_queue_timestamp - \
                    task.log.task_arrival_at_worker_buffer_timestamp
                time_spent_in_queue = task.log.task_front_queue_timestamp - \
                    task.log.task_placed_on_worker_queue_timestamp
                model_fetching_time = task.log.get_model_fetch_time()
                execution_time = task.log.task_execution_end_timestamp - \
                    task.log.task_execution_start_timestamp

                assert time_to_buffer >= 0
                assert dependency_wait_time >= 0
                assert time_spent_in_queue >= 0
                assert model_fetching_time >= 0
                assert execution_time >= 0

                dataframe_tasks_log.loc[task_index] = [job.job_type_id, task.task_id, task.executing_worker_id, task.log.task_arrival_at_worker_buffer_timestamp, 
                                                       task.log.task_execution_start_timestamp,time_to_buffer, dependency_wait_time, 
                                                       time_spent_in_queue, model_fetching_time, execution_time]
                task_index += 1

        self.tasks_logging_times = dataframe_tasks_log
        self.result_to_export = dataframe