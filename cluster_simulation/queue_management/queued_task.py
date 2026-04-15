from core.task import Task


class QueuedTask:
    def __init__(self, task: Task):
        self.task = task
        self.priority = task.job.create_time # TODO

    def __lt__(self, other):
        return self.priority < other.priority
    
    def __str__(self):
        return f"[PRIORITY: {self.priority}] [TASK: {self.task}]"
    
    def __repr__(self):
        return self.__str__()