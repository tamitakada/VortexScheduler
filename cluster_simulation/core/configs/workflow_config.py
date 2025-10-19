import numpy as np


"""  --------       Workflow Parameters     --------  """
# https://keras.io/api/applications/

WORKFLOW_LIST = [
    {"JOB_TYPE": 0,         # ID of the type of workflow (dependency graph)
     "JOB_NAME": "textvision",
     # the minimum amount of time necessary to execute the whole job
     "BEST_EXEC_TIME": 51.7,
     "TASKS": [
        {"MODEL_ID": 0,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 1, # kB
         "OUTPUT_SIZE": 2,
         "MAX_EMIT_BATCH_SIZE": 4,
         "MAX_WAIT_TIME": 1,
         "SLO": 0}, # ms
        {"MODEL_ID": 1,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 10000, # kB
         "OUTPUT_SIZE": 1000,
         "MAX_EMIT_BATCH_SIZE": 4,
         "MAX_WAIT_TIME": 10,
         "SLO": 0},
        {"MODEL_ID": 2,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [0,1],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 1002, # kB
         "OUTPUT_SIZE": 10,
         "MAX_EMIT_BATCH_SIZE": 16,
         "MAX_WAIT_TIME": 1,
         "SLO": 0},
        {"MODEL_ID": 3,
         "TASK_INDEX": 3,
         "PREV_TASK_INDEX": [2],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 10, # kB
         "OUTPUT_SIZE": 10,
         "MAX_EMIT_BATCH_SIZE": 0, # NOTE: should not be used
         "MAX_WAIT_TIME": 1,
         "SLO": 0}]},
    {"JOB_TYPE": 1,
     "JOB_NAME": "tts",
     # the minimum amount of time necessary to execute the whole job
     "BEST_EXEC_TIME": 101.4,
     "TASKS": [
        {"MODEL_ID": 4,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [1],
         "INPUT_SIZE": 1000, # kB
         "OUTPUT_SIZE": 2,
         "MAX_EMIT_BATCH_SIZE": 10,
         "MAX_WAIT_TIME": 1,
         "SLO": 0},
        {"MODEL_ID": 5,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [0],
         "NEXT_TASK_INDEX": [2,3],
         "INPUT_SIZE": 2, # kB
         "OUTPUT_SIZE": 2,
         "MAX_EMIT_BATCH_SIZE": 10,
         "MAX_WAIT_TIME": 1,
         "SLO": 0},
        {"MODEL_ID": 6,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [1],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 2, # kB
         "OUTPUT_SIZE": 2,
         "MAX_EMIT_BATCH_SIZE": 10,
         "MAX_WAIT_TIME": 1,
         "SLO": 0},
        {"MODEL_ID": 7,
         "TASK_INDEX": 3,
         "PREV_TASK_INDEX": [1,2],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 4, # kB
         "OUTPUT_SIZE": 4,
         "MAX_EMIT_BATCH_SIZE": 0, # NOTE: should not be used
         "MAX_WAIT_TIME": 1,
         "SLO": 0}]},
]

def get_task_types(job_types: list[int]) -> list[tuple[int,int]]:
    return [(jt, t["TASK_INDEX"]) for jt in job_types for t in WORKFLOW_LIST[jt]["TASKS"]]

def get_model_id_for_task_type(task_type: tuple[int,int]) -> int:
    return WORKFLOW_LIST[task_type[0]]["TASKS"][task_type[1]]["MODEL_ID"]

def get_task_types_for_model(model_id: int) -> list[tuple[int,int]]:
    task_types = []
    for wf in WORKFLOW_LIST:
        for task in wf["TASKS"]:
            if task["MODEL_ID"] == model_id:
                task_types.append((wf["JOB_TYPE"], task["TASK_INDEX"]))
    return task_types