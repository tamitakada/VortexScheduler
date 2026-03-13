import sys
import os

import pandas as pd
import importlib.util

from verification.batch_execution_verifier import BatchExecutionVerifier


if __name__ == "__main__":
    results_dir = sys.argv[1]

    gcfg_path = os.path.join(results_dir, "configs/gen_config.py")
    mcfg_path = os.path.join(results_dir, "configs/model_config.py")
    wcfg_path = os.path.join(results_dir, "configs/workflow_config.py")

    modules = {}

    for path in [gcfg_path, mcfg_path, wcfg_path]:
        spec = importlib.util.spec_from_file_location(f"results_{path.replace('/', '_')}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[path] = module

    exec_verifier = BatchExecutionVerifier(
        {
            "worker_log": pd.read_csv(os.path.join(results_dir, "worker_log.csv")),
            "model_log": pd.read_csv(os.path.join(results_dir, "model_history_log.csv")),
            "batch_log": pd.read_csv(os.path.join(results_dir, "batch_log.csv")),
            "drop_log": pd.read_csv(os.path.join(results_dir, "drop_log.csv")),
            "event_log": pd.read_csv(os.path.join(results_dir, "events_by_time.csv")),
            "job_log": pd.read_csv(os.path.join(results_dir, "job_breakdown.csv"))
        },
        modules[gcfg_path],
        modules[mcfg_path],
        modules[wcfg_path]
    )

    exec_verifier.run_verifier()
