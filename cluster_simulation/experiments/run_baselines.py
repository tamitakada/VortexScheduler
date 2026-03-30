import argparse
import os
import core.configs.gen_config as gcfg

from run_experiment import run_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", type=str, default="results", help="Path to output directory")
    
    args = parser.parse_args()

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        
    for send_rate in [32, 40, 48, 56]:
        base_path = os.path.join(args.out, f"{send_rate}")
        os.makedirs(base_path, exist_ok=True)
    
        for sched_type in [3, 4]:
            # FCFS
            gcfg.CLIENT_CONFIGS = [
                {6: {"NUM_JOBS": 500,
                    "SEND_RATES": [send_rate],
                     "SEND_RATE_CHANGE_INTERVALS": [],
                     "SLO": int(62.48 * 0)}},
                {7: {"NUM_JOBS": 500,
                    "SEND_RATES": [send_rate],
                     "SEND_RATE_CHANGE_INTERVALS": [],
                     "SLO": int(70.48 * 0)}},
                {8: {"NUM_JOBS": 500,
                    "SEND_RATES": [send_rate],
                     "SEND_RATE_CHANGE_INTERVALS": [],
                     "SLO": int(80.48 * 0)}}
            ]
            gcfg.ENABLE_PREEMPTION = False
            gcfg.BOOST_POLICY = "FCFS"
            gcfg.BATCH_POLICY = "LARGEST"
            gcfg.DROP_POLICY = "NONE"
            gcfg.SLO_TYPE = "JOB_LEVEL"
            
            policy_name = {3: "decentral", 4: "central"}
            os.makedirs(os.path.join(base_path, policy_name[sched_type] + "-fcfs"), exist_ok=True)
            run_experiment(sched_type, [6,7,8], os.path.join(base_path, policy_name[sched_type] + "-fcfs"))
            
            for multiplier in [2, 4, 5]:
                # EDF
                gcfg.CLIENT_CONFIGS = [
                    {6: {"NUM_JOBS": 500,
                         "SEND_RATES": [send_rate],
                         "SEND_RATE_CHANGE_INTERVALS": [],
                         "SLO": int(62.48 * multiplier)}},
                    {7: {"NUM_JOBS": 500,
                         "SEND_RATES": [send_rate],
                         "SEND_RATE_CHANGE_INTERVALS": [],
                         "SLO": int(70.48 * multiplier)}},
                    {8: {"NUM_JOBS": 500,
                         "SEND_RATES": [send_rate],
                         "SEND_RATE_CHANGE_INTERVALS": [],
                         "SLO": int(80.48 * multiplier)}}
                ]
                gcfg.ENABLE_PREEMPTION = False
                gcfg.BOOST_POLICY = "EDF"
                gcfg.BATCH_POLICY = "LARGEST"
                gcfg.DROP_POLICY = "NONE"
                gcfg.SLO_TYPE = "JOB_LEVEL"
                
                os.makedirs(os.path.join(base_path, policy_name[sched_type] + f"-edf-{multiplier}xslo"), exist_ok=True)
                run_experiment(sched_type, [6,7,8], os.path.join(base_path, policy_name[sched_type] + f"-edf-{multiplier}xslo"))
                
                if sched_type == 4:
                    # Shepherd
                    gcfg.ENABLE_PREEMPTION = True
                    gcfg.BOOST_POLICY = "EDF"
                    gcfg.BATCH_POLICY = "OPTIMAL"
                    gcfg.DROP_POLICY = "LATEST_POSSIBLE"
                    gcfg.SLO_TYPE = "JOB_LEVEL"
                    
                    os.makedirs(os.path.join(base_path, policy_name[sched_type] + f"-shepherd-{multiplier}xslo"), exist_ok=True)
                    run_experiment(sched_type, [6,7,8], os.path.join(base_path, policy_name[sched_type] + f"-shepherd-{multiplier}xslo"))
                else:
                    # Nexus
                    gcfg.ENABLE_PREEMPTION = False
                    gcfg.BOOST_POLICY = "EDF"
                    gcfg.BATCH_POLICY = "OPTIMAL"
                    gcfg.DROP_POLICY = "LATEST_POSSIBLE"
                    gcfg.SLO_TYPE = "NEXUS"
                    
                    os.makedirs(os.path.join(base_path, policy_name[sched_type] + f"-nexus-{multiplier}xslo"), exist_ok=True)
                    run_experiment(sched_type, [6,7,8], os.path.join(base_path, policy_name[sched_type] + f"-nexus-{multiplier}xslo"))
