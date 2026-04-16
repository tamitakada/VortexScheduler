import core.configs.gen_config as gcfg

from core.task import Task


class QueuedTask:
    def __init__(self, task: Task):
        self.task = task

        if gcfg.BOOST_POLICY == "FCFS":
            self.priority = task.job.create_time
        elif gcfg.BOOST_POLICY == "EDF":
            self.priority = task.get_task_deadline()
        else:
            raise ValueError(f"Unrecognized queue ordering policy {gcfg.BOOST_POLICY}")

    def __lt__(self, other):
        return self.priority < other.priority
    
    def __str__(self):
        return f"[PRIORITY: {self.priority}] [TASK: {self.task}]"
    
    def __repr__(self):
        return self.__str__()