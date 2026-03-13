import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_workers_over_time(save_dir: str, workers_df: pd.DataFrame, jobs_df: pd.DataFrame | None):
    times = []
    worker_counts = []
    for _, row in workers_df.sort_values(by="time", kind="mergesort").iterrows():
        if row["add_or_remove"] == "add":
            if len(times) > 0 and row["time"] == times[-1]:
                worker_counts[-1] += 1
            else:
                times.append(row["time"])
                if len(worker_counts) == 0:
                    worker_counts.append(1)
                else:
                    worker_counts.append(worker_counts[-1] + 1)
        
        elif row["add_or_remove"] == "remove":
            if len(times) > 0 and row["time"] == times[-1]:
                worker_counts[-1] -= 1
            else:
                times.append(row["time"])
                worker_counts.append(worker_counts[-1] - 1)

        else:
            raise ValueError(f"Unrecognized value {row['add_or_remove']}")

    # if jobs_df != None:
    #     # arrival rates per 5s interval
    #     times = np.arange(5000, jobs_df["job_create_time"].max(), 5000)
    #     for wf in sorted(set(jobs_df["workflow_type"])):
    #         wf_jobs = jobs_df[jobs_df["workflow_type"]==wf]

    #         arrival_rates = []

    #         for time in times:
    #             created_jobs = ((wf_jobs["job_create_time"] >= time - 5000) & \
    #                             (wf_jobs["job_create_time"] < time))
    #             arrival_rates.append(created_jobs / 5)
    #             plt.plot(times, arrival_rates, label=f"Workflow {wf} Job Send Rate (past 5s)")
    
    times = [t / 1000 for t in times]

    plt.figure(figsize=(8, 6))
    plt.plot(times, worker_counts)
    plt.xlabel('Time (s)')
    plt.ylabel('Number of workers')
    plt.title('Inferline Pipeline 1 95 -> 55 QPS Send Rate Cluster Size Over Time')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "workers_over_time.pdf"))
    # plt.show()


def plot_model_counts_over_time(save_dir: str, model_df: pd.DataFrame):
    times = {} # model ID -> [time]
    model_counts = {} # model ID -> [count]
    for _, row in model_df.sort_values(by="end_time", kind="mergesort").iterrows():
        if row["model_id"] not in times:
            times[row["model_id"]] = []
            model_counts[row["model_id"]] = []

        if row["placed_or_evicted"] == "placed":
            if len(times[row["model_id"]]) > 0 and row["end_time"] == times[row["model_id"]][-1]:
                model_counts[row["model_id"]][-1] += 1
            else:
                times[row["model_id"]].append(row["end_time"])
                if len(model_counts[row["model_id"]]) == 0:
                    model_counts[row["model_id"]].append(1)
                else:
                    model_counts[row["model_id"]].append(model_counts[row["model_id"]][-1] + 1)
        
        elif row["placed_or_evicted"] == "evicted":
            if len(times[row["model_id"]]) > 0 and row["end_time"] == times[row["model_id"]][-1]:
                model_counts[row["model_id"]][-1] -= 1
            else:
                times[row["model_id"]].append(row["end_time"])
                model_counts[row["model_id"]].append(model_counts[row["model_id"]][-1] - 1)

        else:
            raise ValueError(f"Unrecognized value {row['placed_or_evicted']}")

    plt.figure(figsize=(8, 6))

    for model_id in sorted(times.keys()):
        mtimes = [t / 1000 for t in times[model_id]]
        plt.plot(mtimes, model_counts[model_id], label=f"Model {model_id}")
    
    plt.xlabel('Time (s)')
    plt.ylabel('Model count')
    plt.title('Inferline Pipeline 1 95 -> 55 QPS Send Rate Model Counts Over Time')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(os.path.join(save_dir, "model_counts_time.pdf"))


# def plot_model_utility_over_time(model_df: pd.DataFrame, batch_df: pd.DataFrame):
#     end_time = batch_df["end_time"].max()

#     model_ids = sorted(set(model_df["model_id"]))
#     for model_id in model_ids:
#         mdf = model_df[model_df["model_id"]==model_id]
#         for instance_id in set(mdf["instance_id"]):
#             idf = mdf[mdf["instance_id"]==instance_id]
#             start_time = idf[idf["placed_or_evicted"]=="placed"]["end_time"]
#             end_time = batch_df["end_time"].max() if (idf["placed_or_evicted"]=="evicted").sum() == 0 else \
#                 idf[idf["placed_or_evicted"]=="evicted"]["end_time"]
#             active_duration = end_time - start_time

#             bdf = batch_df[batch_df["instance_id"]==instance_id]
#             total_batch_runtime = (bdf["end_time"] - bdf["start_time"]).sum()

#             instance_utility = total_batch_runtime / active_duration


if __name__ == "__main__":
    root_results_dir = sys.argv[1]
    workers_df = pd.read_csv(os.path.join(root_results_dir, "worker_log.csv"))
    model_df = pd.read_csv(os.path.join(root_results_dir, "model_history_log.csv"))

    plot_workers_over_time(root_results_dir, workers_df, None)
    plot_model_counts_over_time(root_results_dir, model_df)
