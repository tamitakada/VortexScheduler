from ipaddress import summarize_address_range
import gurobipy as gp
from gurobipy import GRB
import csv
import math
import sys
sys.path.append('..')
from dynamic_scheduler import solve_ilp


# read throughput profile
batch_runtimes = {}
with open('../runtimes_by_batch_size.csv', mode='r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        batch_size = int(row['bsize'])
        runtime = float(row['mean_runtime_ms'])
        batch_runtimes[batch_size] = runtime


# Test parameters

B = 16
# d = [-46.877064453132334, -15.977064453130879, -14.77706445313379, 14.322935546864755, 14.622935546867666, 43.122935546867666]
d = [96.77098873671875, 100.38170962949219, 110.54431441152343, 112.37260492285156, 115.04472182402344, 134.83183919746094, 194.36440246933594, 244.07066857832032, 247.64775870917967, 275.8517385871094, 285.1827463472656, 287.62250525703126, 321.1707189373047, 321.9289397513672, 322.29276344843754, 357.95971512929685, 366.0555567246094, 368.5534051322266, 369.7702272451172, 387.68930866132814, 400.7349869419922, 446.240465153711, 454.0825473636719, 467.11905361835943, 488.18413994453124, 502.53224591386726, 553.4400474257812, 567.9287911267578, 577.5869344800782, 594.7569671082031]
# d = d[:10]
# req_deadline = {9: 96.77098873671875, 7: 100.38170962949219, 4: 110.54431441152343, 20: 112.37260492285156, 14: 115.04472182402344, 5: 134.83183919746094, 29: 194.36440246933594, 21: 244.07066857832032, 11: 247.64775870917967, 26: 275.8517385871094, 10: 285.1827463472656, 3: 287.62250525703126, 13: 321.1707189373047, 18: 321.9289397513672, 24: 322.29276344843754, 0: 357.95971512929685, 6: 366.0555567246094, 22: 368.5534051322266, 19: 369.7702272451172, 27: 387.68930866132814, 17: 400.7349869419922, 23: 446.240465153711, 1: 454.0825473636719, 15: 467.11905361835943, 12: 488.18413994453124, 8: 502.53224591386726, 16: 553.4400474257812, 28: 567.9287911267578, 2: 577.5869344800782, 25: 594.7569671082031}

# print(list(req_deadline.values()))

for i in range(len(d)):
    d[i] = max(d[i], 0)
N = len(d)




# Solve the ILP
result = solve_ilp(N, B, d, batch_runtimes)

scheduled = []
if result:
    print("obj:", result["obj"])
    # print("x values:")
    for i in range(N):
        scheduled.append(0)
        for j in range(N):
            # print(f"{abs(result['x'][(i, j)])}", end=" ")
            scheduled[i] += abs(result['x'][(i, j)])
        # print()


    # print the status of each req
    print("id | scheduled | deadline met | t | d")
    for i in range(N):
        print(f"{i} | {scheduled[i]} | {1-result['z'][i]} | {result['t'][i]} | {d[i]}")


    
    # Debug: Count deadline violations
    deadline_violations = sum(result["z"])
    print(f"Deadline violations: {deadline_violations}")
    
    # Print assignments for each j
    print("\nAssignments (x_{i,j} = 1):")
    for j in range(N):
        if result['s'][j] > 0:
            print(f"Position j={j} (batch size s[{j}]={result['s'][j]}):")
            for i in range(N):
                if round(result["x"][(i, j)]) == 1:
                    print(f"  Request {i} assigned to position {j}")



    # print(f"s[0] = {result['s'][0]}")
    # print("x[i,0] values:")
    # for i in range(N):
    #     print(f"  x[{i},0] = {result['x'][(i,0)]}")
    # print(f"sum_i x[i,0] = {sum(result['x'][(i,0)] for i in range(N))}")


else:
    print("No optimal solution found")