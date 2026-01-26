import uuid

from core.task import Task


class Batch:
    """
        Represents a batch of Tasks executing on the same model.
    """
    def __init__(self, tasks: list[Task]):
        assert(len(tasks) > 0)
        assert(len(set(t.job.id for t in tasks)) == len(tasks))

        self.id = uuid.uuid4()
        self.tasks = tasks
        self.model_data = tasks[0].model_data
        self.job_ids = list(map(lambda t: t.job.id, tasks))

    def size(self) -> int:
        return len(self.tasks)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, value):
        return value is not None and type(value) == Batch and value.id == self.id
    
    def __str__(self):
        return f"[Batch {self.id} | Model ID {self.model_data.id}] <Jobs {self.job_ids}>"
    
    def __repr__(self):
        return self.__str__()