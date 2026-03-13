import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_response_time_tail_cdf(srcs: list[tuple[str, str]], split_by_workflow: bool):
    workflow_colors = [
        sns.light_palette("purple", n_colors=len(srcs)+1)[1:],
        sns.light_palette("orange", n_colors=len(srcs)+1)[1:],
        sns.light_palette("red", n_colors=len(srcs)+1)[1:],
        sns.light_palette("blue", n_colors=len(srcs)+1)[1:]
    ]

    workflow2color = {}
    
    max_res = 0
    for (dir, _) in srcs:
        data = pd.read_csv(os.path.join(dir, "job_breakdown.csv"))
        max_res = max(max_res, int(data["response_time"].max()) + 1)

    thresholds = np.arange(0, max_res, 1)

    plt.figure(figsize=(8, 6))

    for i, (dir, name) in enumerate(srcs):
        data = pd.read_csv(os.path.join(dir, "job_breakdown.csv"))

        if split_by_workflow:
            workflows = sorted(set(data["workflow_type"]))
            for wf in workflows:
                if wf not in workflow2color:
                    if workflow2color:
                        workflow2color[wf] = max(workflow2color.values())+1
                    else:
                        workflow2color[wf] = 0

                cdf = np.array([(data[data["workflow_type"]==wf]["response_time"] > t).mean() for t in thresholds])

                log_cdf = np.log(cdf)
                log_cdf[cdf == 0] = np.nan

                plt.plot(thresholds, log_cdf, 
                         label=f"{name}: Workflow {wf}", 
                         color=workflow_colors[workflow2color[wf]][i])
        else:
            cdf = np.array([(data["response_time"] > t).mean() for t in thresholds])

            log_cdf = np.log(cdf)
            log_cdf[cdf == 0] = np.nan

            plt.plot(thresholds, log_cdf, label=name)

    plt.xlabel('Response time')
    plt.ylabel('log(tail CDF)')
    plt.title('Tail CDF comparison')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_tardiness_tail_cdf(srcs: list[tuple[str, str]], slos: dict[int, float], split_by_workflow: bool):
    """
    Args:
        srcs: List of [root directory of sim exp results, name of experiment]
        slos: client ID -> slo
    """

    workflow_colors = [
        sns.light_palette("purple", n_colors=len(srcs)+1)[1:],
        sns.light_palette("orange", n_colors=len(srcs)+1)[1:],
        sns.light_palette("red", n_colors=len(srcs)+1)[1:],
        sns.light_palette("blue", n_colors=len(srcs)+1)[1:]
    ]

    workflow2color = {}

    max_res = 0
    min_res = None
    for (dir, _) in srcs:
        data = pd.read_csv(os.path.join(dir, "job_breakdown.csv"))
        data["slo"] = data["client_id"].map(slos)
        max_res = max(max_res, int((data["response_time"]-data["slo"]).max()) + 1)
        min_res = int((data["response_time"]-data["slo"]).min()) \
            if not min_res else min(min_res, int((data["response_time"]-data["slo"]).min()) )

    thresholds = np.arange(min_res, max_res, 1)

    plt.figure(figsize=(8, 6))

    for i, (dir, name) in enumerate(srcs):
        data = pd.read_csv(os.path.join(dir, "job_breakdown.csv"))
        data["slo"] = data["client_id"].map(slos)
        
        if split_by_workflow:
            workflows = sorted(set(data["workflow_type"]))
            for wf in workflows:
                if wf not in workflow2color:
                    if workflow2color:
                        workflow2color[wf] = max(workflow2color.values())+1
                    else:
                        workflow2color[wf] = 0

                wf_data = data[data["workflow_type"]==wf]
                cdf = np.array([(((wf_data["response_time"]-wf_data["slo"])) > t).mean() for t in thresholds])

                log_cdf = np.log(cdf)
                log_cdf[cdf == 0] = np.nan

                plt.plot(thresholds, log_cdf, 
                         label=f"{name}: Workflow {wf}", 
                         color=workflow_colors[workflow2color[wf]][i])
        else:
            cdf = np.array([(((data["response_time"]-data["slo"])) > t).mean() for t in thresholds])

            log_cdf = np.log(cdf)
            log_cdf[cdf == 0] = np.nan

            plt.plot(thresholds, log_cdf, label=name)

    plt.xlabel('SLO - response time')
    plt.ylabel('log(CDF)')
    plt.title('Tardiness CDF comparison')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    srcs = [
        (sys.argv[1], "FCFS"),
        (sys.argv[2], "EDF"),
        (sys.argv[3], "Boost: Job Size"),
        # (sys.argv[4], "Boost: Remaining exec time"),
        # (sys.argv[5], "Boost: Remaining time to deadline"),
    ]

    plot_response_time_tail_cdf(srcs, True)
    # plot_tardiness_tail_cdf(srcs, {0: 507, 1: 1727, 2: 768}, True)