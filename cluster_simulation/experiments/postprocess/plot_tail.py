import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import argparse

from scipy.stats import linregress


def plot_response_time_tail_cdf(srcs: list[tuple[str, str]], split_by_workflow: bool, 
                                save_fig: bool, out_path: str):
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

        for _, dropped_row in pd.read_csv(os.path.join(dir, "drop_log.csv")).iterrows():
            data.loc[len(data)] = {
                "workflow_type": dropped_row["workflow_id"],
                "job_create_time": dropped_row["create_time"],
                "response_time": np.inf
            }

        if split_by_workflow:
            workflows = sorted(set(data["workflow_type"]))
            for wf in workflows:
                if wf not in workflow2color:
                    if workflow2color:
                        workflow2color[wf] = max(workflow2color.values())+1
                    else:
                        workflow2color[wf] = 0

                cdf = np.array([(data[data["workflow_type"]==wf]["response_time"] > t).mean() for t in thresholds])

                log_cdf = np.log10(cdf)
                log_cdf[cdf == 0] = np.nan

                plt.plot(thresholds, log_cdf, 
                         label=f"{name}: Workflow {wf}", 
                         color=workflow_colors[workflow2color[wf]][i])
        else:
            cdf = np.array([(data["response_time"] > t).mean() for t in thresholds])

            log_cdf = np.log(cdf)
            log_cdf[cdf == 0] = np.nan

            plt.plot(thresholds, log_cdf, label=name)

            # thresh_masked = np.arange(150, 200, 10)
            # m, b, _, _, _ = linregress(
            #     thresh_masked, 
            #     np.array([(data["response_time"] > t).mean() for t in thresh_masked]))
            # print("SLOPE IS ", m)

    plt.xlabel('Response time (ms)')
    plt.ylabel('log10(tail CDF)')
    plt.title('Tail CDF')
    plt.grid(True)
    plt.legend()

    if save_fig: plt.savefig(out_path)
    else: plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--srcs", type=str, nargs="+", required=True, 
                        help="Root directories of simulation results to compare")
    parser.add_argument("--labels", type=str, nargs="+", required=True, 
                        help="Names to give each simulation run")
    parser.add_argument("--pdf", action="store_true",
                        help="Save as PDF instead of launching plot")
    parser.add_argument("--split", action="store_true",
                        help="Split by workflow")
    parser.add_argument("--out", type=str,
                        help="Output directory path for saved figures")

    args = parser.parse_args()

    srcs = [(args.srcs[i], args.labels[i]) for i in range(len(args.srcs))]

    plot_response_time_tail_cdf(srcs, args.split, args.pdf, 
                                args.out if args.out else ("tail_by_workflow.pdf" if args.split else "tail_agg.pdf"))
