# Adapted from https://github.com/QLM-project/QLM/blob/main/qlm/queue/virtual_queue.py

import uuid
from collections import deque


class VirtualQueue:
    """
    A VirtualQueue is a queue that contains a list of task groups. Each task group is a list of tasks.
    """

    def __init__(self):
        self.vq_id = uuid.uuid4()
        self.groups = deque()

    def add_group(self, group):
        self.groups.append(group)

    def pop_group(self):
        return self.groups.popleft()

    def get_head_group(self):
        return self.groups[0]

    def __hash__(self):
        return hash(self.vq_id)
    
    def __str__(self):
        s = f"[VQ {self.vq_id}] [\n"
        for group in self.groups:
            s += f"\t[GROUP {group.group_id} : SLO {group.slo} : MID {group.model.model_id}]: <{[t.job_id for t in group.tasks]}>\n"
        s += "]"
        return s
    
    def __repr__(self):
        return self.__str__()