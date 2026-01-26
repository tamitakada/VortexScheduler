import core.configs.gen_config as gcfg
import core.configs.model_config as mcfg

from core.data_models.model_data import ModelData
from core.data_models.workflow import Workflow
from core.allocation import ModelAllocation, AllocationUpdateStrategy
from simulations.simulation_central import Simulation_central

import math
import os
import contextlib
import pandas as pd
import copy


class Inferline:

    # workflow ID -> [sample rate per window]
    sample_workload_arrival_logs: dict[int, pd.DataFrame] = {}

    # model ID -> mpr
    max_provisioning_ratios: dict[int, float] = {}

    @classmethod
    def is_feasible(cls, workflow: Workflow, allocation: ModelAllocation, slo: float) -> bool:
        """Estimator simulating a real workload trace given model replica counts
        and a job-level SLO-constrained workflow.

        Args:
            workflow: Workflow to measure feasibility for
            allocation: Allocation to simulate

        Returns:
            is_feasible: P99 request latency <= slo satisfied during
            simulation with given allocation
        """

        prev_cfg = copy.deepcopy(gcfg.CLIENT_CONFIGS)
        prev_scaling_strat = gcfg.AUTOSCALING_POLICY
        prev_alloc_strat = gcfg.ALLOCATION_STRATEGY
        prev_custom_alloc = gcfg.CUSTOM_ALLOCATION.copy()
        prev_models = copy.deepcopy(mcfg.MODELS)

        gcfg.CLIENT_CONFIGS = gcfg.ESTIMATOR_CLIENT_CONFIGS
        gcfg.ALLOCATION_STRATEGY = "CUSTOM"
        gcfg.AUTOSCALING_POLICY = "NONE"

        gcfg.CUSTOM_ALLOCATION = [allocation.worker_cfgs[id] for (id, _) in allocation.worker_ids_by_create_time]
        while len(gcfg.CUSTOM_ALLOCATION) < gcfg.MIN_NUM_NODES:
            gcfg.CUSTOM_ALLOCATION.append((24, []))

        for model in workflow.get_models():
            mcfg.MODELS[model.id]["MAX_BATCH_SIZE"] = allocation.models[model.id].max_batch_size

        # ignore stdout for estimator sim runs
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            sim = Simulation_central(simulation_name="hashtask",
                                    job_types_list=[workflow.id],
                                    produce_breakdown=True)
            sim.run()

        for client in sim.sim_stats_log["clients"]:
            p99 = client[workflow.id]["p99_latency_ms"]
            if p99 > slo:
                print(f"Estimator ended with P99 latency={p99}ms : {p99-slo}ms violation for client {client[workflow.id]['client_id']}")
                
                gcfg.CLIENT_CONFIGS = prev_cfg
                gcfg.ALLOCATION_STRATEGY = prev_alloc_strat
                gcfg.CUSTOM_ALLOCATION = prev_custom_alloc
                mcfg.MODELS = prev_models
                gcfg.AUTOSCALING_POLICY = prev_scaling_strat

                return False
            
        gcfg.CLIENT_CONFIGS = prev_cfg
        gcfg.ALLOCATION_STRATEGY = prev_alloc_strat
        gcfg.CUSTOM_ALLOCATION = prev_custom_alloc
        mcfg.MODELS = prev_models
        gcfg.AUTOSCALING_POLICY = prev_scaling_strat
        
        return True

    @classmethod
    def planner_init(cls, simulation, time: float, workflow: Workflow, slo: float) -> tuple[bool, ModelAllocation]:
        """Produce initial InferLine allocation under given constraints.

        Args:
            workflow: Workflow to be deployed over allocation

        Returns:
            did_sat_slo: Allocation does satisfy P99 latency <= slo
            based on estimator.

            allocation: Initial InferLine allocation
        """

        allocation = ModelAllocation(simulation)
        
        for model in sorted(workflow.get_models(), key=lambda m: m.size):
            if not allocation.add_model(time, model.id, strategy=AllocationUpdateStrategy.SORT_AND_PACK):
                raise RuntimeError("Not enough nodes to fit all models")
        
        min_service_time = workflow.get_processing_time(
            get_exec_time=lambda t: t.model_data.batch_exec_times[24][1])
        
        if min_service_time >= slo:
            raise RuntimeError(f"Minimum pipeline={workflow.id} execution time violates SLO={slo}")

        while not cls.is_feasible(workflow, allocation, slo):
            model = min(workflow.get_models(),
                        key=lambda m: allocation.count(m) / m.batch_exec_times[24][1] * 1000)
            
            if not allocation.add_model(time, model.id, AllocationUpdateStrategy.SORT_AND_PACK):
                print("[WARNING] Best allocation P99 latency still violates SLO")
                return False, allocation

        return True, allocation
    
    @classmethod
    def planner_minimize_cost(cls, simulation, time: float, workflow: Workflow, slo: float) -> ModelAllocation:
        did_sat_slo, allocation = cls.planner_init(simulation, time, workflow, slo)

        if not did_sat_slo:
            print("[WARNING] Skip minimize cost : Best allocation P99 latency still violates SLO")
            return allocation

        while True:
            did_change = False

            for model in sorted(workflow.get_models(), key=lambda m: m.size, reverse=True):
                # try increasing max batch size
                prev_max_bsize = allocation.models[model.id].max_batch_size
                allocation.models[model.id].max_batch_size = min(
                    model.max_batch_size, prev_max_bsize * 2)
                if allocation.models[model.id].max_batch_size == prev_max_bsize:
                    continue
                
                if cls.is_feasible(workflow, allocation, slo):
                    did_change = True
                else:
                    allocation.models[model.id].max_batch_size = prev_max_bsize

                # try removing replica only if batch size was modified
                # and at least 2 copies exist
                if did_change and allocation.count(model.id) > 1:
                    allocation.remove_model(model.id, AllocationUpdateStrategy.SORT_AND_PACK)
                    if not cls.is_feasible(workflow, allocation, slo):
                        allocation.add_model(time, model.id, AllocationUpdateStrategy.SORT_AND_PACK)
                        break
                
                if did_change:
                    break

            if not did_change:
                return allocation
            
    @classmethod
    def get_sample_rates(cls, workflow: Workflow, allocation: ModelAllocation, window_size):
        if workflow.id not in cls.sample_workload_arrival_logs:
            prev_cfg = copy.deepcopy(gcfg.CLIENT_CONFIGS)
            prev_alloc_strat = gcfg.ALLOCATION_STRATEGY
            prev_custom_alloc = gcfg.CUSTOM_ALLOCATION.copy()
            prev_models = copy.deepcopy(mcfg.MODELS)
            prev_scaling_strat = gcfg.AUTOSCALING_POLICY

            gcfg.CLIENT_CONFIGS = gcfg.ESTIMATOR_CLIENT_CONFIGS
            gcfg.ALLOCATION_STRATEGY = "CUSTOM"
            gcfg.AUTOSCALING_POLICY = "NONE"

            gcfg.CUSTOM_ALLOCATION = [allocation.worker_cfgs[id] for (id, _) in allocation.worker_ids_by_create_time]
            while len(gcfg.CUSTOM_ALLOCATION) < gcfg.MIN_NUM_NODES:
                gcfg.CUSTOM_ALLOCATION.append((24, []))

            for model in workflow.get_models():
                mcfg.MODELS[model.id]["MAX_BATCH_SIZE"] = allocation.models[model.id].max_batch_size

            # ignore stdout for estimator sim runs
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                sim = Simulation_central(simulation_name="hashtask",
                                        job_types_list=[workflow.id],
                                        produce_breakdown=True)
                sim.run()
            
            gcfg.CLIENT_CONFIGS = prev_cfg
            gcfg.ALLOCATION_STRATEGY = prev_alloc_strat
            gcfg.CUSTOM_ALLOCATION = prev_custom_alloc
            mcfg.MODELS = prev_models
            gcfg.AUTOSCALING_POLICY = prev_scaling_strat

            cls.sample_workload_arrival_logs[workflow.id] = \
                sim.task_arrival_log[sim.task_arrival_log["workflow_id"]==workflow.id]

        arrival_log = cls.sample_workload_arrival_logs[workflow.id]
        arrival_rates = []

        window_end = arrival_log["time"].max()
        window_start = window_end - window_size
        while window_start >= 0:
            observed_jobs = len(set(arrival_log[(arrival_log["time"] >= window_start) & \
                                    (arrival_log["time"] < window_end)]["job_id"]))
            arrival_rates.append(observed_jobs / window_size)

            window_end -= window_size
            window_start -= window_size

        return arrival_rates
    
    @classmethod
    def get_max_provisioning_ratio(cls, workflow: Workflow, model: ModelData):
        if model.id in cls.max_provisioning_ratios:
            return cls.max_provisioning_ratios[model.id]
        
        if workflow.id in cls.sample_workload_arrival_logs:
            arrival_log = cls.sample_workload_arrival_logs[workflow.id]
            arrived_jobs = len(set(arrival_log[arrival_log["workflow_id"] == workflow.id]["job_id"]))
            arrival_rate = arrived_jobs / arrival_log["time"].max() * 1000
            max_tput = model.max_batch_size / model.batch_exec_times[24][model.max_batch_size] * 1000
            cls.max_provisioning_ratios[model.id] = arrival_rate / max_tput

            print(f"MPR M{model.id}: ", cls.max_provisioning_ratios[model.id])
            return cls.max_provisioning_ratios[model.id]

        raise NotImplementedError("Must init sample workload results first")

    @classmethod
    def check_scale_up(cls, time: float, workflow: Workflow, simulation) -> list[tuple[int, tuple[int, list[int]]]]:
        """Returns a list of (worker ID, worker config) to modify/add/delete from the
        given allocation to accomodate the given workflow arrival rate.
        """
        
        arrival_log = simulation.task_arrival_log[(simulation.task_arrival_log["workflow_id"]==workflow.id) & \
                                                  (simulation.task_arrival_log["worker_id"]=="N/A")]
        
        r_max = -1

        service_time = workflow.get_processing_time(lambda t: t.model_data.batch_exec_times[24][1])
        window_size = min(60 * 1000, service_time * 2)

        sample_rates = cls.get_sample_rates(workflow, simulation.allocation, window_size)

        window_start = max(0, time - len(sample_rates) * window_size)
        
        for sample_rate in sample_rates:
            window_end = window_start + window_size
            observed_jobs = len(set(arrival_log[(arrival_log["time"] >= window_start) & \
                                                (arrival_log["time"] < window_end)]["job_id"]))
            r_obs = observed_jobs / window_size

            if r_obs > sample_rate:
                r_max = max(r_max, r_obs)

            window_start = window_end
        
        delta_workers = []

        r_max *= 1000

        if r_max > 0:
            # NOTE: Prioritizes smaller models for limited cluster setting
            for model in sorted(workflow.get_models(), key=lambda m: m.size):
                # TODO: Set params
                scale_factor = 1
                max_provision_ratio = cls.get_max_provisioning_ratio(workflow, model)
                max_model_throughput = model.max_batch_size * simulation.allocation.count(model.id) / \
                    model.batch_exec_times[24][model.max_batch_size] * 1000
                k = r_max * scale_factor / (max_provision_ratio * max_model_throughput)
                reps = math.ceil(k) - simulation.allocation.count(model.id)
                print(f"M{model.id} ADD REPS: ", reps)
                if reps > 0:
                    for _ in range(reps):
                        delta_rep = simulation.allocation.add_model(time, model.id, AllocationUpdateStrategy.FIRST_VALID_WORKER)
                        if delta_rep:
                            delta_workers += delta_rep
                        else: # exit early if out of space
                            return delta_workers
        
        return delta_workers

    @classmethod
    def check_scale_down(cls, time: float, workflow: Workflow, simulation) -> list[tuple[int, tuple[int, list[int]]]]:
        arrival_log = simulation.task_arrival_log[(simulation.task_arrival_log["workflow_id"]==workflow.id) & \
                                                  (simulation.task_arrival_log["worker_id"]=="N/A")]
        
        # arrival rates over 5s windows for past 30s
        arrival_rates = []
        for i in range(0, 30, 5):
            window_end = time - i
            window_start = window_end - 5

            arrivals = ((arrival_log["time"] > window_start) & (arrival_log["time"] <= window_end)).sum()
            arrival_rates.append(arrivals / 5)
        
        max_req_rate = max(arrival_rates)
        delta_workers = []

        min_max_provision_ratio = min(cls.get_max_provisioning_ratio(workflow, m) for m in workflow.get_models())

        for model in workflow.get_models():
            # TODO: Set params
            scale_factor = 1
            max_model_throughput = model.max_batch_size * simulation.allocation.count(model.id) / \
                model.batch_exec_times[24][model.max_batch_size] * 1000
            k = max_req_rate * scale_factor / (min_max_provision_ratio * max_model_throughput)
            
            extra_reps = simulation.allocation.count(model.id) - math.ceil(k)
            print(f"M{model.id} Excess of ", extra_reps)
            if extra_reps > 0:
                for _ in range(extra_reps):
                    # NOTE: require at least 1 copy of each model at all times
                    if simulation.allocation.count(model.id) > 1:
                        delta_workers += simulation.allocation.remove_model(
                            model.id, AllocationUpdateStrategy.LAST_VALID_WORKER)

        return delta_workers