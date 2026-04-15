import pandas as pd

import core.configs.gen_config as gcfg
import core.configs.workflow_config as wcfg
import core.configs.model_config as mcfg

from core.data_models.workflow import Workflow
from core.data_models.model_data import ModelData

from client.client import Client
from network.network import Network
from schedulers.centralized_scheduler import CentralizedScheduler
from workers.worker import Worker

from core.allocation import ModelAllocation
from schedulers.algo.vortex_planner_algo import VortexPlanner

from verifiers.verifier import Verifier

from events.event_manager import EventManager
from events.event import *
from events.event_types import *

from uuid import uuid4


class Simulation:

    def __init__(self, centralized: bool):
        self.em = EventManager()

        self.models = self._generate_models()
        self.workflows = self._generate_workflows()
        self.clients = self._generate_clients()

        self.allocation = self._generate_model_allocation()
        self.workers = self._generate_workers()

        self.scheduler = None
        if centralized:
            scheduler_worker_id = list(self.workers.keys())[0]
            self.scheduler = CentralizedScheduler(self.em, self.workers, scheduler_worker_id)

        self.network = Network(self.em, scheduler_worker_id)
        self.verifier = Verifier(self.em, self.workers)


    def _generate_clients(self) -> list[Client]:
        """Generates clients from client config and enqueues all
        job creation/send events.

        Returns:
            clients: List of generated clients
        """
        clients: list[Client] = []
        created_job_count = 0
        for cfg in gcfg.CLIENT_CONFIGS:
            for wid, cfgw in cfg.items():
                clients.append(Client(uuid4(), self.em))

                prev_create_time = 0
                for i, send_rate in enumerate(cfgw["SEND_RATES"]):
                    n_jobs = cfgw["JOBS_PER_SEND_RATE"][i]
                    prev_create_time = clients[-1].generate_jobs(
                        self.workflows[wid], 
                        n_jobs, 
                        prev_create_time, 
                        send_rate,
                        cfgw["SLO"],
                        created_job_count)
                    created_job_count += n_jobs
        return clients
    

    def _generate_models(self) -> dict[int, ModelData]:
        """Initializes ModelData objects for all models used by jobs issued by configured
        clients.

        Returns:
            models: Map of model ID -> abstract model representation
        """
        all_workflow_ids = set([k for c in gcfg.CLIENT_CONFIGS for k in c.keys()])
        model_ids: set[int] = set()
        for cfg in wcfg.WORKFLOW_LIST:
            if cfg["JOB_TYPE"] not in all_workflow_ids:
                continue
            
            for task_cfg in cfg["TASKS"]:
                if task_cfg["MODEL_ID"] >= 0:
                    model_ids.add(task_cfg["MODEL_ID"])

        return {id: ModelData(id,
                              mcfg.MODELS[id]["MODEL_SIZE"],
                              mcfg.MODELS[id]["MAX_BATCH_SIZE"],
                              mcfg.MODELS[id]["MIG_BATCH_EXEC_TIMES"],
                              mcfg.MODELS[id]["EXEC_TIME_CVS"]) 
                              for id in model_ids}

    
    def _generate_workflows(self) -> dict[int, Workflow]:
        """Initializes Workflow objects for all workflow types required by
        configured clients.

        Returns:
            workflows: Map of workflow ID -> abstract workflow representation
        """
        return {
            cfg["JOB_TYPE"] : Workflow(cfg, self.models, gcfg.SLO_TYPE) for cfg in wcfg.WORKFLOW_LIST
            if cfg["JOB_TYPE"] in set([k for c in gcfg.CLIENT_CONFIGS for k in c.keys()])}


    def _generate_model_allocation(self) -> ModelAllocation:
        if gcfg.ALLOCATION_STRATEGY == "CUSTOM":
            assert(all((psize*(10**6)) in gcfg.VALID_WORKER_SIZES for psize, _ in gcfg.CUSTOM_ALLOCATION))
            return ModelAllocation(
                self,
                {uuid4(): cfg for cfg in gcfg.CUSTOM_ALLOCATION},
                reset_batch_sizes=False)
        
        elif gcfg.ALLOCATION_STRATEGY == "INFERLINE":
            # TODO: add support for multitenant
            assert(len(self.workflows) == 1)
            assert(len(gcfg.CLIENT_CONFIGS) == 1 and len(gcfg.CLIENT_CONFIGS[0].keys()) == 1)

            slo = list(gcfg.CLIENT_CONFIGS[0].values())[0]["SLO"]
            alloc = self.inferline.planner_minimize_cost(self, 0, self.workflows[0], slo)

            return alloc
        
        elif gcfg.ALLOCATION_STRATEGY == "VORTEX":
            # NOTE: no support for multitenant yet
            assert(len(self.workflows) == 1)
            assert(len(gcfg.CLIENT_CONFIGS) == 1 and len(gcfg.CLIENT_CONFIGS[0].keys()) == 1)

            slo = list(gcfg.CLIENT_CONFIGS[0].values())[0]["SLO"]
            return VortexPlanner.get_allocation(self,
                                                (gcfg.MIN_NUM_NODES + gcfg.MAX_NUM_NODES) // 2,
                                                self.workflows[0],
                                                slo)


    def _generate_workers(self) -> dict[UUID, Worker]:
        """Generates and initializes model placements for initial worker objects.

        Returns:
            workers: Map of worker ID -> worker object
        """
        workers = {}
        for (wid, _) in self.allocation.worker_ids_by_create_time:
            cfg = self.allocation.worker_cfgs[wid]
            worker = Worker(wid, self.em, cfg[0], 0)
            
            workers[wid] = worker

            # self.worker_log.loc[len(self.worker_log)] = {
            #     "time": 0,
            #     "add_or_remove": "add",
            #     "worker_id": wid
            # }

            for mid in cfg[1]:
                self.models[mid].max_batch_size = self.allocation.models[mid].max_batch_size
                instance_id = workers[wid].GPU_state.prefetch_model(self.models[mid])

                # self.worker_model_log.loc[len(self.worker_model_log)] = {
                #     "start_time": 0,
                #     "end_time": 0, # prefetch special case assume no overhead
                #     "worker_id": wid,
                #     "model_id": mid,
                #     "instance_id": instance_id,
                #     "placed_or_evicted": "placed"
                # }
        return workers


    def run(self):
        while self.em.has_events():
            self.em.process_next_event()
        
        sampled_anomalies = self.verifier.sampled_anomalies / self.verifier.total_samples
        if sampled_anomalies > 0.05:
            RED_BOLD = "\033[1;31m"
            RESET = "\033[0m"
            print(f"{RED_BOLD}[VERIFIER WARNING] {sampled_anomalies * 100:.2f}% of batch execution time samples ({self.verifier.sampled_anomalies}/{self.verifier.total_samples}) showed significant deviation (p < 0.05) given configured mean and CV{RESET}")

        self._get_client_data()


    def _get_client_data(self):
        jobs_df = pd.DataFrame(columns=["client_id", "workflow_id", "job_id", "was_completed",
                                        "deadline", "create_time", "response_time"])
        for client in self.clients:
            for jid, (create_time, finish_time, was_completed, job) in client.jobs.items():
                jobs_df.loc[len(jobs_df)] = {
                    "client_id": client.id,
                    "workflow_id": job.job_type_id,
                    "job_id": jid,
                    "was_completed": was_completed,
                    "deadline": -1, #TODO FIX FIX
                    "create_time": create_time,
                    "response_time": finish_time - create_time
                }
        jobs_df.to_csv("jobs.csv")
