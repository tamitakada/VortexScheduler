from ipaddress import summarize_address_range
import gurobipy as gp
from gurobipy import GRB
import csv
import math
import sys
sys.path.append('..')
from dynamic_scheduler import solve_ilp


# Test parameters

B = 16
d = [-46.877064453132334, -15.977064453130879, -14.77706445313379, 14.322935546864755, 14.622935546867666, 43.122935546867666]
for i in range(len(d)):
    d[i] = max(d[i], 0)
N = len(d)




# Solve the ILP
result = solve_ilp(N, B, d)

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



    print(f"s[0] = {result['s'][0]}")
    print("x[i,0] values:")
    for i in range(N):
        print(f"  x[{i},0] = {result['x'][(i,0)]}")
    print(f"sum_i x[i,0] = {sum(result['x'][(i,0)] for i in range(N))}")


else:
    print("No optimal solution found")