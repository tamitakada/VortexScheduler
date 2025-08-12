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
    
    def __eq__(self, value):
        return type(value) == Group and value.group_id == self.group_id
    
    def __str__(self):
        return f"[GROUP {self.group_id} : SLO {self.slo} : MID {self.model.model_id}]: <{[t.job_id for t in self.tasks]}>"
    
    def __repr__(self):
        return self.__str__()