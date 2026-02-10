import core.configs.gen_config as gcfg
from core.task import Task
from core.batch import Batch

from schedulers.algo.boost_algo import get_task_priority_by_boost, BoostPolicy


def get_batch(time: float, partition_size: int, task_queue: list[Task], preserve_order=False) -> Batch | None:
    """
    Returns a batch drawn from [task_queue] formed according to the batch
    policy specified in the config file.
    """
    if gcfg.BOOST_POLICY == "JOB_SIZE":
        task_queue = sorted(task_queue, key=lambda t: get_task_priority_by_boost(time, t, BoostPolicy.TOTAL_JOB_TIME))
    elif gcfg.BOOST_POLICY == "REMAINING_JOB_TIME":
        task_queue = sorted(task_queue, key=lambda t: get_task_priority_by_boost(time, t, BoostPolicy.REMAINING_JOB_TIME))
    # elif gcfg.BOOST_POLICY == "REMAINING_TO_DEADLINE":
    #     task_queue = sorted(task_queue, key=lambda t: get_task_priority_by_boost(time, t, BoostPolicy.REMAINING_TIME_TO_DEADLINE))
    elif gcfg.BOOST_POLICY == "FCFS":
        task_queue = sorted(task_queue, key=lambda t: t.job.create_time)
    elif gcfg.BOOST_POLICY == "EDF":
        task_queue = sorted(task_queue, key=lambda t: t.get_task_deadline())
    else:
        raise RuntimeError("Unknown queue sort policy ", gcfg.BOOST_POLICY)
    
    # print(f"TASK QUEUE: {task_queue}\n")

    if gcfg.BATCH_POLICY == "LARGEST":
        batch = get_largest_batch(task_queue)
    elif gcfg.BATCH_POLICY in ["OPTIMAL", "OPT_PREEMPT"]:
        batch = get_optimal_batch(time, partition_size, task_queue, preserve_order)
    else:
        raise RuntimeError("Unknown batch policy ", gcfg.BATCH_POLICY)
    
    # queue ordering policy check
    if batch and gcfg.BOOST_POLICY in ["FCFS", "EDF"]:
        nonbatched_tasks = [t for t in task_queue 
                            if all([bt.job.id != t.job.id for bt in batch.tasks])]
        
        if nonbatched_tasks:
            if gcfg.BOOST_POLICY == "FCFS":
                # tasks that were not chosen must have been created later
                # than tasks that were chosen for the batch
                # or, for optimal policy, inlcuding the task would cause an SLO violation for this batch size

                latest_create_time = max([t.job.create_time for t in batch.tasks])

                if gcfg.BATCH_POLICY == "LARGEST":
                    assert(all([t.job.create_time >= latest_create_time for t in nonbatched_tasks]))
                elif gcfg.BATCH_POLICY == "OPTIMAL":
                    assert(all([t.job.create_time >= latest_create_time or \
                                t.deadline < time + t.model.data.batch_exec_times[partition_size][batch.size()] 
                                for t in nonbatched_tasks]))
            
            elif gcfg.BOOST_POLICY == "EDF":
                # tasks that were not chosen must have a later deadline
                # than tasks that were chosen for the batch
                # or, for optimal policy, inlcuding the task would cause an SLO violation for this batch size

                latest_deadline = max([t.job.create_time + t.job.slo for t in batch.tasks])

                if gcfg.BATCH_POLICY == "LARGEST":
                    assert(all([t.job.create_time + t.job.slo >= latest_deadline for t in nonbatched_tasks]))
                elif gcfg.BATCH_POLICY in ["OPTIMAL", "OPT_PREEMPT"]:
                    assert(all([t.job.create_time + t.job.slo >= latest_deadline or \
                                t.get_task_deadline() < time + t.model_data.batch_exec_times[partition_size][batch.size()] 
                                for t in nonbatched_tasks]))

    if not batch and gcfg.BATCH_POLICY != "LARGEST" and gcfg.FALLBACK_TO_LARGEST_BATCH:
        return get_largest_batch(task_queue)

    return batch


def get_largest_batch(task_queue: list[Task]) -> Batch | None:
    """
    Returns largest batch <= task max batch size drawn from [task_queue]
    in FIFO order.
    """
    tasks = []
    for task in task_queue:
        tasks.append(task)
        if len(tasks) >= task.model_data.max_batch_size:
            break

    if len(tasks) == 0:
        return None
    
    return Batch(tasks)


def get_optimal_batch(time: float, partition_size: int, task_queue: list[Task], preserve_order: bool) -> Batch | None:
    """
    Returns largest batch drawn from [task_queue] s.t. no task
    deadline in the batch is violated.
    """
    assert(all(task.model_data.id == task_queue[0].model_data.id for task in task_queue))

    if preserve_order:
        assert(task_queue == sorted(task_queue, key=lambda t: t.get_task_deadline()))
    else:
        task_queue = sorted(task_queue, key=lambda t: t.get_task_deadline())

    max_bsize = task_queue[0].model_data.max_batch_size

    # if no valid batch possible, return None
    if all((time + task.model_data.batch_exec_times[partition_size][1] > task.get_task_deadline())
           for task in task_queue):
        return None

    if max_bsize == 1:
        for task in task_queue: 
            if time + task.model_data.batch_exec_times[partition_size][1] <= task.get_task_deadline():
                return Batch([task])
        return None

    tasks = []
    bsizes = [[0 for _ in range(max_bsize+1)] for _ in range(len(task_queue))]

    for i in range(len(task_queue) - 1, -1, -1):
        task = task_queue[i]
        for j in range(1, max_bsize+1):
            if time + task.model_data.batch_exec_times[partition_size][j] > task.get_task_deadline():
                # on SLO violation, task [i] cannot be incl. in batch of size [j]
                bsizes[i][j] = 0
            elif i == (len(task_queue)-1): # if on last task and no SLO violation, always 1
                bsizes[i][j] = 1
            else:
                bsizes[i][j] = max(bsizes[i+1][j-1] + 1, # either task [i] in batch of size [j]
                                    bsizes[i+1][j])      # or not
    
    max_i, max_formable_bsize = 0, 0
    for i in range(len(bsizes)):
        for j in range(len(bsizes[i])):
            if bsizes[i][j] > max_formable_bsize:
                max_formable_bsize = bsizes[i][j]
                max_i = i
    
    counter = max_i
    while counter < (max_i + max_formable_bsize):
        tasks.append(task_queue[counter])
        counter += 1

    # batched tasks should not violate deadline
    if len(tasks) > 0:
        proposed_exec_time = task_queue[0].model_data.batch_exec_times[partition_size][len(tasks)]
        assert(all(t.get_task_deadline() > time + proposed_exec_time for t in tasks))

    if len(tasks) < max_bsize and len(task_queue) > len(tasks):
        if len(tasks) == 0:
            # if no valid batch found, all queued tasks should violate
            # slo even with min batch size (=1)
            assert(all(t.get_task_deadline() < time + task_queue[0].model_data.batch_exec_times[partition_size][1] for t in task_queue))
        else:
            # if batch size < max batch size and unbatched tasks exist,
            # then any larger batch must cause an SLO violation for some
            # task in the larger batch

            edf_sorted = sorted(task_queue, key=lambda t: t.get_task_deadline())
            min_larger_size = len(tasks) + 1
            larger_candidate_batch = edf_sorted[(len(edf_sorted)-min_larger_size):]
            larger_candidate_exec = task_queue[0].model_data.batch_exec_times[partition_size][min_larger_size]

            assert(any(t.get_task_deadline() < time + larger_candidate_exec 
                       for t in larger_candidate_batch))

    if len(tasks) == 0:
        return None
    
    return Batch(tasks)