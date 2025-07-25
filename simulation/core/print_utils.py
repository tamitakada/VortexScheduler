#  Helper function for printing and ploting of each simulator
import numpy as np
from core.workflow import WORKFLOW_LIST


def print_end_jobs(last_time, complete_jobs, jobs):
    print("Complete jobs: {}, Included {} jobs".format(
        len(complete_jobs), len(complete_jobs)))
    throughput = 1000 * len(complete_jobs) / last_time
    print("Simulation ended after {0:.2f} ms ({1} jobs started)".format(
        last_time, len(jobs)))
    print("Throughput: {0:.2f} jobs/s".format(throughput))

def print_response_time(response_times):
    print("~~~~~~  RESPONSE TIME  ~~~~~~~~")
    print("|     Avg: {0:.2f} ms".format(np.mean(response_times)))
    print("+     Max: {0:.2f} ms".format(np.max(response_times)))
    print("-     Min: {0:.2f} ms".format(np.min(response_times)))


def print_slowdown(slow_down_rate):
    print("~~~~~~  SLOW DOWN RATE   ~~~~~~~~")
    print("|     Avg: {0:.2f}  ".format(np.mean(slow_down_rate)))
    print("|     Med: {0:.2f}  ".format(np.median(slow_down_rate)))
    print("|     Std: {0:.2f}  ".format(np.std(slow_down_rate)))
    print("+     Max: {0:.2f}  ".format(np.max(slow_down_rate)))
    print("-     Min: {0:.2f}  ".format(np.min(slow_down_rate)))



def print_stats_by_job_type(response_time_per_type, slow_down_per_type):
    """
    Function to print the stats of the jobs by job type
    @param
        response_time_per_type: dict of job type and list of jobs
        slow_down_per_type: dict of job type and list of slow down
    """
    width = 20
    print("------------------------- End to End Latency by Job Type -------------------------")
    workflows = "workflow_type:  "
    means = "mean(ms):       "
    medians = "median(ms):     "
    maxs = "max:            "
    mins = "min:            "
    stds = "std:            "
    for job_type_id, response_times in response_time_per_type.items():
        workflows += "{:<{width}}".format(job_type_id, width=width)
        mean = "{0:.2f}".format(np.mean(response_times))
        means += f"{mean:{width}}"
        median = "{0:.2f}".format(np.median(response_times))
        medians += f"{median:{width}}"
        max = "{0:.2f}".format(np.max(response_times))
        maxs += f"{max:{width}}"
        min = "{0:.2f}".format(np.min(response_times))
        mins += f"{min:{width}}"
        std = "{0:.3f}".format(np.std(response_times))
        stds += f"{std:{width}}"
    print(workflows)
    print(means)
    print(medians)
    print(maxs)
    print(mins)
    print(stds)
    print("-------------------------- Slow Down Factor by Job Type --------------------------")
    workflows = "workflow_type:  "
    means = "mean:           "
    medians = "median:         "
    maxs = "max:            "
    mins = "min:            "
    stds = "std:            "
    for job_type_id, slow_downs in slow_down_per_type.items():
        workflows += "{:<{width}}".format(job_type_id, width=width)
        mean = "{0:.2f}".format(np.mean(slow_downs))
        means += f"{mean:{width}}"
        median = "{0:.2f}".format(np.median(slow_downs) )
        medians += f"{median:{width}}"
        max = "{0:.2f}".format(np.max(slow_downs) )
        maxs += f"{max:{width}}"
        min = "{0:.2f}".format(np.min(slow_downs) )
        mins += f"{min:{width}}"
        std = "{0:.3f}".format(np.std(slow_downs))
        stds += f"{std:{width}}"
    print(workflows)
    print(means)
    print(medians)
    print(maxs)
    print(mins)
    print(stds)



# the total numbers of model transfer in the whole system on every worker
def compute_worker_models_fetch(simulator):
    print("~~~      Whole System model transfer times        ~~~")
    worker_models_fetching_times = []
    worker_model_PtoC_transfer_time = []
    total_model_moves = 0
    for worker in simulator.workers:
        worker_models_fetching_times.append(len(worker.model_fetch_track))
        worker_model_PtoC_transfer_time.append(
            len(worker.model_local_transfer_p2c_track))
        total_model_moves += len(worker.model_fetch_track)
        total_model_moves += len(worker.model_local_transfer_p2c_track)
    print("   Total number of model transfer per worker is : {}".format(
        total_model_moves))
    print("   Models fetching times per worker are : {}".format(
        worker_models_fetching_times))
    print("   Models local transfer times per worker are : {}".format(
        worker_model_PtoC_transfer_time))


def print_involved_workers(workers):
    num_involved = 0
    for worker in workers:
        if worker.involved:
            num_involved += 1
    print("   Total number of involved workers is : {}".format(num_involved))