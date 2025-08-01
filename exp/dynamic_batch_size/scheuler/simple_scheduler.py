from collections import deque
# from main import Request
import math
import itertools
from utils import SortedQueue


class SimpleScheduler:
    max_batch_size: int
    batch_runtimes: dict
    slo: float
    # base_latency: float

    def __init__(self, max_batch_size: int, batch_runtimes: dict):
        self.max_batch_size = max_batch_size
        self.batch_runtimes = batch_runtimes
        # self.slo = slo
        # self.base_latency = base_latency

    def preempt(self, current_batch: SortedQueue, queue: SortedQueue, current_time: float, batch_finish_time: float) -> bool:
        size = len(current_batch) + len(queue)
        # copy the current batch and queue
        all_queue = queue.copy()
        all_queue.extend(current_batch)

        new_batch = SortedQueue()
        self.schedule(new_batch, all_queue, current_time)

        # if len(new_batch) < 3.03 * len(current_batch):
        if len(new_batch) < 1.5 * len(current_batch):
            return False
        else:
            # preempt the current batch
            for req in current_batch:
                req.preempt()
            
            queue.extend(current_batch)
            current_batch.clear()
            current_batch.extend(new_batch)
            for req in current_batch:
                queue.remove(req)
                req.schedule(current_time, len(current_batch), self.batch_runtimes[len(current_batch)])
                
            assert len(queue) + len(current_batch) == size, f"Queue and current batch size is not equal to the original size: {len(queue)} + {len(current_batch)} and {size}"

            return True


    def schedule(self, current_batch: SortedQueue, queue: SortedQueue, current_time: float) -> float:
        """
        Find the largest feasible batch size where all requests can meet their latency SLOs.
        For each batch size, check all possible subsets of the queue to find a feasible subset.
        
        Args:
            current_batch: The current batch being built
            queue: The queue of pending requests
            current_time: Current simulation time
            
        Returns:
            float: Next check time (math.inf if no batch scheduled)
        """

        assert len(current_batch) == 0, f"Current batch is not empty: {current_batch}"
        
        # Find the largest feasible batch size
        max_feasible_size = 0
        best_subset = []
        
        # Convert queue to list for easier subset generation
        queue_list = queue.requests
        
        for batch_size in range(1, min(self.max_batch_size + 1, len(queue) + 1)):
            # Check if this batch size is feasible
            feasible = False
            
            # Calculate batch duration for this size
            batch_duration = self.batch_runtimes[batch_size]

            
            # Check all possible subsets of size batch_size
            for subset in itertools.combinations(queue_list, batch_size):
                subset_feasible = True
                
                # Check if all requests in this subset can meet their SLOs
                for request in subset:
                    # Calculate when this request would finish
                    request_finish_time = current_time + batch_duration
                    
                    # Check SLO violation: arrival_time + slo < finish_time
                    if request.deadline < request_finish_time:
                        subset_feasible = False
                        break
                
                if subset_feasible:
                    feasible = True
                    best_subset = list(subset)
                    break  # Found a feasible subset for this batch size
            
            if feasible:
                max_feasible_size = batch_size
            else:
                # Once we find an infeasible batch size, larger sizes will also be infeasible
                break
        
        # Build the batch using the best subset found
        if max_feasible_size > 0 and best_subset:
            # Remove the requests in best_subset from the queue and add to current_batch
            for request in best_subset:
                # Find and remove the request from the queue
                queue.remove(request)
                current_batch.append(request)
                
        return math.inf, None

    def offline_schedule(self, current_batch: SortedQueue, queue: SortedQueue, current_time: float, finished_reqs: list) -> float:
        cumulated_latency = 0
        num_batch = 0
        while len(queue) > 0:
            self.logger.info(f"####### Batch: {num_batch} Time: {cumulated_latency} #######")
            self.schedule(current_batch, queue, current_time)
            batch_time = self.batch_runtimes[len(current_batch)]
            deadline = {req.id: req.deadline - cumulated_latency for req in current_batch}
            self.logger.info(f"deadline: {deadline}")
            while len(current_batch) > 0:
                req = current_batch.pop()
                finished_reqs.append(req)
                self.logger.info(f"\tRequest {req.id}: time remaining: {req.deadline - cumulated_latency}")
            cumulated_latency += batch_time
            num_batch += 1

            self.logger.info(f"--------------------------------")

        return current_time