import scipy

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


srcs = [
    (sys.argv[1], "FCFS"),
    (sys.argv[2], "EDF"),
    (sys.argv[3], "Boost: Job Size"),
    (sys.argv[4], "Boost: Remaining exec time"),
    (sys.argv[5], "Boost: Remaining time to deadline"),
]

slos_by_wf = {1: 345, 4: 1189, 5:502 }#{int(k): v["slo"] for k,v in json.loads(os.path.join(dir, "stats.json"))["clients"][0].values()}


max_res = 0
for (dir, _) in srcs:
    data = pd.read_csv(os.path.join(dir, "job_breakdown.csv"))
    data["slo"] = data["client_id"].map(slos_by_wf)
    # max_res = max(max_res, int((data["response_time"]-data["slo"]).max()) + 1)
    max_res = max(max_res, int(data["response_time"].max()) + 1)

thresholds = np.arange(0, max_res, 1)

plt.figure(figsize=(8, 6))

for (dir, name) in srcs:
    data = pd.read_csv(os.path.join(dir, "job_breakdown.csv"))
    data["slo"] = data["client_id"].map(slos_by_wf)
    # cdf = np.array([(((data["response_time"]-data["slo"])) > t).mean() for t in thresholds])
    cdf = np.array([(data["response_time"] > t).mean() for t in thresholds])

    log_cdf = np.log(cdf)
    log_cdf[cdf == 0] = np.nan

    plt.plot(thresholds, log_cdf, label=name)

plt.xlabel('Response time')
plt.ylabel('log(tail CDF)')
plt.title('Tail CDF comparison')
plt.grid(True)
plt.legend()
plt.show()