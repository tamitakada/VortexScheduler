import numpy as np

""" --------      Simulation Parameters      -------- """

PRODUCE_EVENT_LOG = True

# Runs a trace of produced event logs to verify simulator actions
ENABLE_VERIFICATION = False

# Print event details for every step of verifier trace
ENABLE_VERIFICATION_DEBUG_LOGGING = True

""" --------      Worker Machine Parameters      -------- """

GPU_MEMORY_SIZE = 24000000  # in KB, 24GB for NVIDIA A30
MIN_NUM_NODES = 1
MAX_NUM_NODES = 8
VALID_WORKER_SIZES = [24000000, 12000000, 6000000]

MAX_NUM_MODELS_PER_NODE = 4

"""  --------       Workload Parameters    --------  """

CLIENT_CONFIGS = [ # in ms
    {0: {"NUM_JOBS": 10000,
         "SEND_RATES": [55, 95],
         "SEND_RATE_CHANGE_INTERVALS": [5000], 
         "SLO": 1014}}, #np.inf}},

    # {1: {"NUM_JOBS": 2000,
    #      "SEND_RATES": [6],
    #      "SEND_RATE_CHANGE_INTERVALS": [], 
    #      "SLO": 507}}, #np.inf}},
    # {4: {"NUM_JOBS": 2000,
    #      "SEND_RATES": [6],
    #      "SEND_RATE_CHANGE_INTERVALS": [], 
    #      "SLO": 1727}},#np.inf}},
    # {5: {"NUM_JOBS": 2000,
    #      "SEND_RATES": [6],
    #      "SEND_RATE_CHANGE_INTERVALS": [], 
    #      "SLO": 768}}#,np.inf}},
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


"""  -------        Boost Parameters  --------- """

BOOST_PARAMETER = 0.00293596042 # 0.00104567474

# JOB_SIZE | REMAINING_JOB_TIME | FCFS | EDF
BOOST_POLICY = "FCFS"

""" -------         Inferline Parameters  -------- """

ESTIMATOR_CLIENT_CONFIGS = [ # in ms
    {0: {"NUM_JOBS": 1000,
         "SEND_RATES": [55, 95],
         "SEND_RATE_CHANGE_INTERVALS": [500], 
         "SLO": 1014}},
]
INFERLINE_TUNING_INTERVAL = 15 * 1000 # ms

ENABLE_ESTIMATOR_LOGGING = True

"""  -------        General Scheduling Parameters  --------- """

# OPTIMAL | LARGEST
# [OPTIMAL] Largest batch for which all task SLOs are met
# [LARGEST] Largest batch <= model max batch size
BATCH_POLICY = "LARGEST"
FALLBACK_TO_LARGEST_BATCH = True

# OPTIMAL | LATEST_POSSIBLE | NONE
DROP_POLICY = "NONE"

SLO_SLACK = 0
SLO_GRANULARITY = "JOB" # TASK | JOB

ENABLE_MULTITHREADING = True # allow multiple models on same partition to run at once

# NONE | INFERLINE
AUTOSCALING_POLICY = "INFERLINE"

# HERD | CUSTOM | INFERLINE
ALLOCATION_STRATEGY = "INFERLINE" #"INFERLINE"

# [(partition size in GB, [model ids])]
CUSTOM_ALLOCATION = [(24, []), (24, []), (24, []), (24, [])]

# 12-node mutlitenant ppl 2 (3 versions) alloc
# [
#  (12, [4]), (6, [5,13]), (6, [5,13]),
#  (12, [6]), (12, [6]), 
#  (12, [7]), (12, [7]), 
#  (12, [7]), (12, [7]),
#  (12, [8]), (12, [8]),
#  (24, [9]), 
#  (24, [9]), 
#  (24, [9]), 
#  (24, [9]),
#  (24, [10]),
#  (24, [10]),
#  (24, [10])]

# ppl1 4 node alloc:
# [(24, [1]), (24, [1]), (24, [1]), (6, [3]), (6, [3]), (6, [3]), (6, [0, 2])]

# ppl2 4 node alloc:
# [(12, [4]), (12, [5,6]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7])]