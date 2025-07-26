from collections import deque
# from main import Request
import math
class SimpleScheduler:
    max_batch_size: int

    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size


    def schedule(self, current_batch: deque, queue: deque, current_time: float) -> float:
        for i in range(self.max_batch_size):
            if len(queue) > 0:
                current_batch.append(queue.popleft())
            else:
                break
        return math.inf

