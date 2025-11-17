from core.configs.gen_config import *
from core.task import Task
from core.batch import Batch

from schedulers.algo.boost_algo import get_task_priority_by_boost, BoostPolicy


def get_batch(time: float, partition_size: int, task_queue: list[Task], preserve_order=False) -> Batch | None:
    """
    Returns a batch drawn from [task_queue] formed according to the batch
    policy specified in the config file.
    """
    if BOOST_POLICY == "JOB_SIZE":
        task_queue = sorted(task_queue, key=lambda t: get_task_priority_by_boost(time, t, BoostPolicy.TOTAL_JOB_TIME))
    elif BOOST_POLICY == "REMAINING_JOB_TIME":
        task_queue = sorted(task_queue, key=lambda t: get_task_priority_by_boost(time, t, BoostPolicy.REMAINING_JOB_TIME))
    elif BOOST_POLICY == "REMAINING_TO_DEADLINE":
        task_queue = sorted(task_queue, key=lambda t: get_task_priority_by_boost(time, t, BoostPolicy.REMAINING_TIME_TO_DEADLINE))
    else:
        task_queue = sorted(task_queue, key=lambda t: t.job.create_time + t.job.slo)

    if BATCH_POLICY == "LARGEST":
        return get_largest_batch(task_queue)
    elif BATCH_POLICY == "OPTIMAL":
        return get_optimal_batch(time, partition_size, task_queue, preserve_order)
    elif BATCH_POLICY == "FIRST_TASK":
        return get_largest_batch_with_first_task(time, partition_size, task_queue, preserve_order)


def get_largest_batch(task_queue: list[Task]) -> Batch | None:
    """
    Returns largest batch <= task max batch size drawn from [task_queue]
    in FIFO order.
    """
    tasks = []
    for task in task_queue:
        tasks.append(task)
        if len(tasks) >= task.max_batch_size:
            break

    if len(tasks) == 0:
        return None
    
    return Batch(tasks)


def get_largest_batch_with_first_task(time: float, partition_size: int, task_queue: list[Task], preserve_order: bool) -> Batch | None:
    """
    Returns largest batch drawn from [task_queue] s.t. no task
    deadline in the batch is violated and the task with the earliest
    satisfiable SLO is included in the batch.
    """
    if not preserve_order:
        for task in task_queue:
            # get correct task deadline
            if SLO_GRANULARITY == "TASK":
                task.deadline = task.log.task_placed_on_worker_queue_timestamp + task.slo * (1 + SLO_SLACK)
            else:
                task.deadline = task.log.job_creation_timestamp + task.job.slo * (1 + SLO_SLACK)

    tasks = []
    for task in task_queue:
        if time + task.model.batch_exec_times[partition_size][len(tasks)] <= task.deadline:
            tasks.append(task)
            if len(tasks) >= task.max_batch_size:
                return Batch(tasks)
    
    return Batch(tasks) if tasks else None


def get_optimal_batch(time: float, partition_size: int, task_queue: list[Task], preserve_order: bool) -> Batch | None:
    """
    Returns largest batch drawn from [task_queue] s.t. no task
    deadline in the batch is violated.
    """
    if preserve_order:
        assert(hasattr(t, "deadline") for t in task_queue)
        assert(task_queue == sorted(task_queue, key=lambda t: t.deadline))
    else:
        for task in task_queue:
            # get correct task deadline
            if SLO_GRANULARITY == "TASK":
                task.deadline = task.log.task_placed_on_worker_queue_timestamp + task.slo * (1 + SLO_SLACK)
            else:
                task.deadline = task.log.job_creation_timestamp + task.job.slo * (1 + SLO_SLACK)

        task_queue = sorted(task_queue, key=lambda t: t.deadline)

    max_bsize = task_queue[0].max_batch_size
    if max_bsize == 1:
        for task in task_queue: 
            if time + task.model.batch_exec_times[partition_size][0] <= task.deadline:
                return Batch([task])
        return None

    tasks = []
    bsizes = [[0 for _ in range(max_bsize+1)] for _ in range(len(task_queue))]

    for i in range(len(task_queue) - 1, -1, -1):
        for j in range(max_bsize, 0, -1):
            task = task_queue[i]

            if time + task.model.batch_exec_times[partition_size][task.model.batch_sizes.index(j)] > task.deadline:
                # on SLO violation, task [i] cannot be incl. in batch of size [j]
                bsizes[i][j] = 0
            elif i == (len(task_queue)-1): # if on last task and no SLO violation, always 1
                bsizes[i][j] = 1
            else:
                bsizes[i][j] = max(bsizes[i+1][j-1] + 1, # either task [i] in batch of size [j]
                                    bsizes[i+1][j])      # or not
    
    max_i, max_j, max_formable_bsize = 0, 0, 0
    for i in range(len(bsizes)):
        for j in range(len(bsizes[i])):
            if bsizes[i][j] > max_formable_bsize:
                max_formable_bsize = bsizes[i][j]
                max_i = i
                max_j = j
    
    # print()
    # print(f"M{task_queue[0].model.model_id} MAX FORMABLE IS {max_formable_bsize}")
    # print(f"FROM QUEUE: {[(t, t.deadline) for t in task_queue]}")
    # print()
    
    counter = max_i
    while counter < (max_i + max_formable_bsize):
        tasks.append(task_queue[counter])
        counter += 1

    if len(tasks) > 0:
        proposed_exec_time = task_queue[0].model.batch_exec_times[partition_size][task_queue[0].model.batch_sizes.index(len(tasks))]
        assert(all(t.deadline > time + proposed_exec_time for t in tasks))

    if len(tasks) < max_bsize:
        larger_exec_time = task_queue[0].model.batch_exec_times[partition_size][task_queue[0].model.batch_sizes.index(len(tasks)+1)]
        assert(all(t in tasks or t.deadline < time + larger_exec_time for t in task_queue))

    if len(tasks) == 0:
        return None
    
    return Batch(tasks)