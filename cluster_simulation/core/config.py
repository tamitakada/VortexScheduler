import numpy as np


""" --------      Worker Machine Parameters      -------- """

GPU_MEMORY_SIZE = 24000000  # in KB, 24GB for NVIDIA A30
TOTAL_NUM_OF_NODES = 4
VALID_WORKER_SIZES = [24000000, 12000000, 6000000]


"""  --------       Workload Parameters    --------  """

CLIENT_CONFIGS = [ # in ms
    {0: {"NUM_JOBS": 10000,
         "SEND_RATES": [95],
         "SEND_RATE_CHANGE_INTERVALS": [], 
         "SLO": 111}}
]

WORKLOAD_DISTRIBUTION = "POISSON"  # CONSTANT | POISSON | GAMMA
GAMMA_CV = 10  # Coefficient of variation for gamma distribution


"""  -------        Navigator Parameters  --------- """

LOAD_INFORMATION_STALENESS = 1  # in ms
PLACEMENT_INFORMATION_STALENESS = 1  # in ms
RESCHEDULE_THREASHOLD = 1.5


"""  -------        Shepherd Parameters  --------- """

FLEX_LAMBDA = 3.03
HERD_K = 1.5
HERD_PERIODICITY = 12000    # run HERD every [HERD_PERIODICITY] ms
SHEPHERD_BATCHING_POLICY = "OPTIMAL"


"""  -------        Boost Parameters  --------- """

BOOST_PARAMETER = 0.00104567474


"""  -------        General Scheduling Parameters  --------- """

# OPTIMAL | LARGEST
# [OPTIMAL] Largest batch for which all task SLOs are met
# [LARGEST] Largest batch < model max batch size
BATCH_POLICY = "OPTIMAL"

# CLUSTER_ADMISSION_LIMIT | TASK_ADMISSION_LIMIT | OPTIMAL | LATEST_POSSIBLE | NONE
DROP_POLICY = "OPTIMAL"

SLO_SLACK = 0
SLO_GRANULARITY = "JOB" # TASK | JOB

ENABLE_MULTITHREADING = True # allow multiple models on same partition to run at once
ENABLE_MODEL_PREFETCH = False
ENABLE_DYNAMIC_MODEL_LOADING = False

# use Boost for global queue management (only for centralized schedulers)
USE_BOOST = False

# HERD | VORTEX | CUSTOM
# NOTE: HERD requires ENABLE_DYNAMIC_MODEL_LOADING
ALLOCATION_STRATEGY = "CUSTOM"

# [(partition size in GB, [model ids])]
CUSTOM_ALLOCATION = [(12, [4]), (12, [5,6]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7])]

# ppl1 4 node real alloc:
# [(24, [1]), (24, [1]), (24, [1]), (6, [3]), (6, [3]), (6, [3]), (6, [0, 2])]

# ppl2 4 node real alloc:
# [(12, [4]), (12, [5,6]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7])]