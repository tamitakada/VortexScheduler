import core.configs.gen_config as gcfg
import core.configs.model_config as mcfg

from core.simulation import Simulation
from verification.verifier import Verifier


class AutoscalingVerifier(Verifier):

    def __init__(self, simulation: Simulation):
        super().__init__(simulation)
    
    def run_verifier(self):
        self.verify_worker_logs_match_initial_allocation()
        self.verify_worker_logs_match_autoscaling_policy()
        self.verify_worker_model_execution()

    def verify_worker_logs_match_initial_allocation(self):
        initial_workers = self.simulation.worker_log[self.simulation.worker_log["time"]==0]["worker_id"]

        assert(len(initial_workers) >= gcfg.MIN_NUM_NODES)
        assert(len(initial_workers) <= gcfg.MAX_NUM_NODES)

        initial_models = self.simulation.worker_model_log[self.simulation.worker_model_log["start_time"]==0]

        if gcfg.ALLOCATION_STRATEGY == "CUSTOM":
            assert(len(initial_workers) == len(gcfg.CUSTOM_ALLOCATION))
            
            remaining_allocs = [sorted(mids) for (_, mids) in gcfg.CUSTOM_ALLOCATION]
            for wid in set(initial_workers):
                initial_w_models = initial_models[initial_models["worker_id"]==wid]["model_id"]
                initial_w_models_lst = sorted(initial_w_models.to_list())

                # for every initial worker, exists a corresponding alloc
                i = remaining_allocs.index(initial_w_models_lst)
                remaining_allocs.pop(i)
    

    def verify_worker_model_execution(self):
        for worker_id in set(self.simulation.worker_log["worker_id"]):
            worker_rows = self.simulation.worker_log[self.simulation.worker_log["worker_id"]==worker_id]
            worker_start = worker_rows[worker_rows["add_or_remove"]=="add"]["time"].values[0]
            
            worker_batches = self.simulation.batch_exec_log[self.simulation.batch_exec_log["worker_id"]==worker_id]

            # no batches should execute before start or after end
            assert((worker_batches["start_time"] < worker_start).sum() == 0)
            
            worker_end = worker_rows[worker_rows["add_or_remove"]=="remove"]["time"]
            if worker_end.sum() > 0:
                assert((worker_batches["start_time"] >= worker_end.values[0]).sum() == 0)

            worker_models = self.simulation.worker_model_log[self.simulation.worker_model_log["worker_id"]==worker_id]
            all_worker_models = set(worker_models["model_id"])

            # no models never loaded to worker should be executed
            assert((worker_batches["model_id"].isin(all_worker_models)).all())

    def verify_worker_logs_match_autoscaling_policy(self):
        if gcfg.AUTOSCALING_POLICY == "NONE":
            assert((self.simulation.worker_log["time"] > 0).sum() == 0)

        elif gcfg.AUTOSCALING_POLICY == "INFERLINE":
            scale_up_times = set(
                self.simulation.worker_log[self.simulation.worker_log["add_or_remove"]=="add"]["time"])
            scale_down_times = set(
                self.simulation.worker_log[self.simulation.worker_log["add_or_remove"]=="remove"]["time"])
            
            for scale_down_time in scale_down_times:
                prev_scale_ups = [t for t in scale_up_times if t <= scale_down_time]
                last_scale_up = max(prev_scale_ups) if prev_scale_ups else 0
                assert(scale_down_time >= last_scale_up + gcfg.INFERLINE_TUNING_INTERVAL)