import numpy as np

from core.config import *
from core.task import Task


def get_task_boost(task):
    """
    Compute a boost for the job based on the total processing time needed to
    traverse its ADFG.
    """
    # Compute the dependents for each task, and populate the initial set of
    # tasks that can be processed (i.e. those with no dependents).
    dependencies: dict[int, set[int]] = {}
    dependents: dict[int, set[int]] = {}
    available_tasks: list[Task] = []
    for t in task.job.tasks:
        dependencies[t.task_id] = set(t.required_task_ids)
        dependents[t.task_id] = set(t.next_task_ids)
        if len(t.required_task_ids) == 0:
            available_tasks.append(t)

    max_cum_processing_time = 0
    while available_tasks:
        next_available_tasks = []
        # TODO: Figure out the right way to compute "processing time" here.
        # Do we include things like GPU_to_GPU_delay? Does it make sense to
        # use execution_time from a previous run to start?
        max_cum_processing_time += max([
            t.task_exec_duration for t in available_tasks
        ])
        for t in available_tasks:
            for dep in dependents[t.task_id]:
                dependencies[dep].remove(t.task_id)
                if len(dependencies[dep]) == 0:
                    next_available_tasks.append(task.job.get_task_by_id(dep))

        available_tasks = next_available_tasks

    job_size = max_cum_processing_time
    return task.job.create_time - 1 / BOOST_PARAMETER * np.log(
            1 / (1 - np.exp(-BOOST_PARAMETER * job_size)))