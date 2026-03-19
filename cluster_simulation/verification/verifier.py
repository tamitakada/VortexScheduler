from core.simulation import Simulation
import pandas as pd

from core.data_models.workflow import Workflow
from core.data_models.model_data import ModelData


class Verifier:

    def __init__(self, simulation: Simulation):
        self.simulation = simulation

    def __init__(self, dfs: dict[str, pd.DataFrame], 
                 gen_cfg_module, model_cfg_module, workflow_cfg_module):
        
        required_dfs = [
            "worker_log",
            "model_log",
            "batch_log",
            "drop_log",
            "event_log",
            "job_log"
        ]
        assert(all(df_name in dfs for df_name in required_dfs))
        assert(len(dfs["batch_log"]) > 0)
        assert(len(dfs["worker_log"]) > 0)
        assert(len(dfs["model_log"]) > 0)
        assert(len(dfs["event_log"]) > 0)
        assert(len(dfs["job_log"]) > 0)

        self.dfs = dfs
        self.gcfg = gen_cfg_module
        self.mcfg = model_cfg_module
        self.wcfg = workflow_cfg_module

        workflow_ids = set(k for client_cfg in self.gcfg.CLIENT_CONFIGS for k in client_cfg.keys())
        
        model_ids: set[int] = set()
        for workflow_cfg in self.wcfg.WORKFLOW_LIST:
            if workflow_cfg["JOB_TYPE"] in workflow_ids:
                for task_cfg in workflow_cfg["TASKS"]:
                    if task_cfg["MODEL_ID"] >= 0:
                        model_ids.add(task_cfg["MODEL_ID"])

        self.models = {id: ModelData(id,
                            self.mcfg.MODELS[id]["MODEL_SIZE"],
                            self.mcfg.MODELS[id]["MAX_BATCH_SIZE"],
                            self.mcfg.MODELS[id]["MIG_BATCH_EXEC_TIMES"],
                            self.mcfg.MODELS[id]["EXEC_TIME_CVS"]) for id in model_ids}
        
        self.workflows = {
            cfg["JOB_TYPE"] : Workflow(cfg, self.models, self.gcfg.SLO_TYPE) for cfg in self.wcfg.WORKFLOW_LIST
            if cfg["JOB_TYPE"] in workflow_ids}

    @classmethod
    def init_from(cls, simulation: Simulation, gen_cfg_module, model_cfg_module, workflow_cfg_module):
        return cls(
            {
                "worker_log": simulation.worker_log,
                "model_log": simulation.worker_model_log,
                "batch_log": simulation.batch_exec_log,
                "drop_log": simulation.task_drop_log,
                "event_log": simulation.event_log,
                "job_log": simulation.result_to_export
            },
            gen_cfg_module, model_cfg_module, workflow_cfg_module)

    def run_verifier(self):
        raise NotImplementedError()
    
    def debug_assert(self, predicate: bool, msg: str):
        if not predicate:
            print(f"[ASSERTION FAILED] {msg}")
        
        assert(predicate)