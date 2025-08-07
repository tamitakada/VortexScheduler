# Adapted from https://github.com/QLM-project/QLM/blob/main/qlm/queue/group.py

import uuid
from collections import deque


class Group:
    """
    Group class is used to store the tasks that are in the same task group.
    task group is a group of tasks that have the same model and similar clustered SLO.
    """

    def __init__(self, model, slo):
        self.group_id = uuid.uuid4()
        self.model = model
        self.slo = slo
        self.tasks = deque()

    def add_task(self, task):
        self.tasks.append(task)

    def pop_task(self):
        return self.tasks.popleft()

    def __hash__(self):
        return hash(self.group_id)