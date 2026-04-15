import os
import sys
import argparse
import shutil
import numpy as np

# from schedulers.algo.inferline_planner_algo import Inferline
from simulations.simulation import Simulation

import core.configs.gen_config as gcfg
import core.configs.model_config as mcfg
import core.configs.workflow_config as wcfg

sys.dont_write_bytecode = True
np.random.seed(42)

def run_experiment(is_centralized: bool, out_path: str):
    # copy exp configs
    shutil.copytree(
        os.path.join(os.environ.get("SIMULATION_DIR"), "core", "configs"), 
        os.path.join(out_path, "configs"),
        dirs_exist_ok=True, 
        ignore=lambda dp, names: ["__pycache__", ".DS_Store"])

    sim = Simulation(is_centralized)
    sim.run()


if __name__ == "__main__":
    produce_breakdown = True

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--scheduler-type", type=str, required=True, choices=["central", "decentral"])
    parser.add_argument("-o", "--out", type=str, default="results", help="Path to output directory")
    
    args = parser.parse_args()

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    run_experiment(args.scheduler_type == "central", args.out)