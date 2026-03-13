import os
import sys
import argparse
import json
import shutil

from bidict import bidict

from simulations.simulation_central import *
from simulations.simulation_decentral import *
from schedulers.algo.inferline_planner_algo import Inferline

from verification.autoscaling import AutoscalingVerifier
from verification.batch_execution_verifier import BatchExecutionVerifier

import core.configs.gen_config as gcfg
import core.configs.model_config as mcfg
import core.configs.workflow_config as wcfg

sys.dont_write_bytecode = True
np.random.seed(42)

# Scheduler options
DECENTRALHEFT = 1
CENTRALHEFT = 2
HASHTASK = 3
SHEPHERD = 4
NEXUS = 5

SCHEDULER_NAMES = bidict({
    DECENTRALHEFT: "decentralheft",
    CENTRALHEFT: "centralheft",
    HASHTASK: "hashtask",
    SHEPHERD: "shepherd",
    NEXUS: "nexus"
})

def run_experiment(scheduler_type: int, job_types: list[int], out_path_root: str):
    assert(scheduler_type in [DECENTRALHEFT, CENTRALHEFT, HASHTASK, SHEPHERD, NEXUS])
    assert(scheduler_type != NEXUS or gcfg.SLO_GRANULARITY == "TASK") # Nexus requires task-level SLO split
    assert(gcfg.BOOST_POLICY == "EDF" or gcfg.BATCH_POLICY != "OPTIMAL") # Optimal policy sorts by deadline

    out_path = os.path.join(out_path_root, SCHEDULER_NAMES[scheduler_type])
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Created directory at {out_path}")
    else:
        print(f"Directory at {out_path} exists")

    # copy exp configs
    shutil.copytree(
        os.path.join(os.environ.get("SIMULATION_DIR"), "core", "configs"), 
        os.path.join(out_path, "configs"),
        dirs_exist_ok=True, 
        ignore=lambda dp, names: ["__pycache__", ".DS_Store"])

    sim = None
    if scheduler_type == CENTRALHEFT:
        sim = Simulation_central(simulation_name="centralheft",
                                 job_types_list=job_types, 
                                 produce_breakdown=True, 
                                 inferline=Inferline)
    elif scheduler_type == DECENTRALHEFT:
        sim = Simulation_decentral(simulation_name="decentralheft",
                                   job_types_list=job_types,
                                   dynamic_adjust=False, 
                                   consider_load=True, consider_cache=True, 
                                   produce_breakdown=True)
    elif scheduler_type == HASHTASK:
        sim = Simulation_central(simulation_name="hashtask",
                                 job_types_list=job_types, 
                                 produce_breakdown=True, 
                                 inferline=Inferline)
    elif scheduler_type == SHEPHERD:
        sim = Simulation_central(simulation_name="shepherd",
                                 job_types_list=job_types, 
                                 produce_breakdown=True, 
                                 inferline=Inferline)
    elif scheduler_type == NEXUS:
        sim = Simulation_central(simulation_name="nexus",
                                 job_types_list=job_types, 
                                 produce_breakdown=True, 
                                 inferline=Inferline)

    sim.run()

    if gcfg.PRODUCE_EVENT_LOG:
        sim.event_log.to_csv(os.path.join(out_path, "events_by_time.csv"))
    
    sim.worker_log.to_csv(os.path.join(out_path, "worker_log.csv"))
    sim.result_to_export.to_csv(os.path.join(out_path, "job_breakdown.csv"))
    sim.batch_exec_log.to_csv(os.path.join(out_path, "batch_log.csv"))
    sim.task_drop_log.to_csv(os.path.join(out_path, "drop_log.csv"))
    sim.worker_model_log.to_csv(os.path.join(out_path, "model_history_log.csv"))

    if gcfg.DROP_POLICY == "CLUSTER_ADMISSION_LIMIT":
        sim.tput_gput_log.to_csv(os.path.join(out_path, "throughput_goodput_over_time.csv"))
        sim.limit_log.to_csv(os.path.join(out_path, "arrival_rate_limits.csv"))

    if scheduler_type == NEXUS:
        sim.scheduler.wf_arrival_rate_log.to_csv(os.path.join(out_path, "nexus_job_arrival_rate_log.csv"))
        sim.scheduler.task_slo_log.to_csv(os.path.join(out_path, "nexus_task_slo_log.csv"))

    with open(os.path.join(out_path, "stats.json"), "w") as f:
        f.write(json.dumps(sim.sim_stats_log))
        f.close()

    print("** Simulation stats saved to stats.json **")

    if gcfg.ALLOCATION_STRATEGY == "HERD":
        os.makedirs(os.path.join(out_path, "allocations"), exist_ok=True)
        for time, alog in sim.allocation_logs:
            with open(os.path.join(out_path, "allocations", f"herd_allocation_{time}.json"), "w") as f:
                f.write(alog)
                f.close()

    if gcfg.ENABLE_VERIFICATION:
        #AutoscalingVerifier(sim).run_verifier()

        BatchExecutionVerifier(
            {
                "worker_log": sim.worker_log,
                "model_log": sim.worker_model_log,
                "batch_log": sim.batch_exec_log,
                "drop_log": sim.task_drop_log,
                "event_log": sim.event_log,
                "job_log": sim.result_to_export
            },
            gcfg, mcfg, wcfg
        ).run_verifier()


if __name__ == "__main__":
    produce_breakdown = True
    job_types = [jt for c in gcfg.CLIENT_CONFIGS for jt, v in c.items() if v["NUM_JOBS"] > 0]

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--scheduler-type", type=str, required=True, choices=["centralheft", "decentralheft", "shepherd", "hashtask", "nexus"])
    parser.add_argument("-o", "--out", type=str, default="results", help="Path to output directory")
    
    args = parser.parse_args()

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    scheduler_type = SCHEDULER_NAMES.inv[args.scheduler_type]
    run_experiment(scheduler_type, job_types, args.out)