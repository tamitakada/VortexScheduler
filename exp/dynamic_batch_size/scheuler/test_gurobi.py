from ipaddress import summarize_address_range
import gurobipy as gp
from gurobipy import GRB
import csv
import math

# Read batch runtime from csv
with open('/home/sl3343/VortexScheduler/exp/dynamic_batch_size/runtimes_by_batch_size.csv', mode='r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    batch_runtimes = {}
    for row in reader:
        batch_size = int(row['bsize'])
        runtime = float(row['mean_runtime_ms'])
        batch_runtimes[batch_size] = runtime

def f(s_k):
    """Function f that maps batch size to runtime"""
    return batch_runtimes.get(s_k, 0)

def solve_ilp(N, B, d):
    """
    Solve the ILP:
    min sum_i Ind{d_i < t_i}
    s.t. sum_j x_{i,j} = 1, forall i
         sum_i x_{i,j} = s_j, forall j  
         s_j <= B, forall j
         e_j = sum_{k=1}^j f(s_k), forall j
         t_i = sum_j x_{i,j} e_j, forall i
    """
    model = gp.Model("IndicatorMinimization")

    # Suppress Gurobi output
    model.setParam('OutputFlag', 0)

    # Variables
    x = model.addVars(N, N, vtype=GRB.BINARY, name="x")  # x_{i,j}
    s = model.addVars(N, vtype=GRB.INTEGER, lb=0, ub=B, name="s")  # s_j
    e = model.addVars(N, vtype=GRB.CONTINUOUS, name="e")  # e_j
    t = model.addVars(N, vtype=GRB.CONTINUOUS, name="t")  # t_i
    z = model.addVars(N, vtype=GRB.BINARY, name="z")  # Indicator: z_i = 1 if d_i < t_i
    position_used = model.addVars(N, vtype=GRB.BINARY, name="position_used")  # position_used[j] = 1 if s[j] > 0

    # Big-M value for indicator constraints
    M = max(d) + sum(f(s_val) for s_val in range(B + 1)) * N

    # Constraint 1: sum_j x_{i,j} <= 1, forall i (allow not scheduling)
    for i in range(N):
        model.addConstr(gp.quicksum(x[i, j] for j in range(N)) <= 1, name=f"assign_row_{i}")

    # Constraint 2: sum_i x_{i,j} = s_j, forall j
    for j in range(N):
        model.addConstr(gp.quicksum(x[i, j] for i in range(N)) == s[j], name=f"assign_col_{j}")

    # Constraint 3: s_j <= B (handled by variable bounds)

    # Constraint 3.5: Sequential batch positions - if s[j] = 0, then s[j'] = 0 for all j' > j
    for j in range(N-1):
        for j_prime in range(j+1, N):
            # If s[j] = 0, then s[j_prime] = 0
            # This is equivalent to: s[j_prime] <= M * s[j] where M is a large constant
            model.addConstr(s[j_prime] <= B * s[j], name=f"sequential_{j}_{j_prime}")

    # Constraint 4: e_j = sum_{k=1}^j f(s_k), forall j
    for j in range(N):
        # Create indicator variables for each possible batch size value for position j
        indicators = {}
        for batch_size in range(B + 1):
            indicators[batch_size] = model.addVar(vtype=GRB.BINARY, name=f"ind_{j}_{batch_size}")
        
        # Constraint: sum of indicators for position j must equal 1
        model.addConstr(gp.quicksum(indicators[batch_size] for batch_size in range(B + 1)) == 1, 
                      name=f"ind_sum_{j}")
        
        # Constraint: s[j] = sum of batch_size * indicator[j, batch_size]
        model.addConstr(s[j] == gp.quicksum(batch_size * indicators[batch_size] for batch_size in range(B + 1)), 
                      name=f"s_value_{j}")
        
        # e_j = sum_{k=1}^j f(s_k) = cumulative sum of batch runtimes
        if j == 0:
            # e[0] = f(s[0])
            model.addConstr(e[j] == gp.quicksum(f(batch_size) * indicators[batch_size] for batch_size in range(B + 1)), 
                        name=f"e_{j}")
        else:
            # e[j] = e[j-1] + f(s[j])
            model.addConstr(e[j] == e[j-1] + gp.quicksum(f(batch_size) * indicators[batch_size] for batch_size in range(B + 1)), 
                        name=f"e_{j}")

    # Constraint 5: t_i = sum_j x_{i,j} e_j, forall i (if scheduled), otherwise t_i = large_value
    for i in range(N):
        # If request i is scheduled, t_i = sum_j x[i,j] * e[j]
        # If request i is not scheduled, t_i = M (large value to ensure z[i] = 1)
        model.addConstr(
            t[i] == gp.quicksum(x[i, j] * e[j] for j in range(N)) + M * (1 - gp.quicksum(x[i, j] for j in range(N))), 
            name=f"t_{i}"
        )

    # Indicator constraint: z_i = 1 if d_i < t_i
    eps = 1e-6  # Small epsilon for strict inequality
    for i in range(N):
        # If t_i > d_i + eps, then z_i = 1
        model.addConstr(t[i] - d[i] + M * (1 - z[i]) >= eps, name=f"ind_lb_{i}")
        # If t_i <= d_i, then z_i = 0
        model.addConstr(t[i] - d[i] <= M * z[i], name=f"ind_ub_{i}")
    
    # Additional constraint: if z[i] = 1 (deadline cannot be met), then don't schedule request i
    for i in range(N):
        # If z[i] = 1, then sum_j x[i,j] = 0 (don't schedule)
        # If z[i] = 0, then sum_j x[i,j] = 1 (must schedule)
        model.addConstr(gp.quicksum(x[i, j] for j in range(N)) <= 1 - z[i], name=f"schedule_if_meet_deadline_{i}")


    # Two-stage optimization: first minimize deadline violations, then minimize positions used
    # Stage 1: Minimize deadline violations
    model.setObjective(gp.quicksum(z[i] for i in range(N)), GRB.MINIMIZE)
    
    # Solve first stage
    model.optimize()
    
    # if model.status == GRB.OPTIMAL:
    #     optimal_deadline_violations = model.ObjVal


    #     # Constraint: position_used[j] = 1 if s[j] > 0
    #     for j in range(N):
    #         # If s[j] > 0, then position_used[j] = 1
    #         model.addConstr(s[j] <= B * position_used[j], name=f"position_used_lb_{j}")
    #         # If s[j] = 0, then position_used[j] = 0
    #         model.addConstr(position_used[j] <= s[j], name=f"position_used_ub_{j}")
        
    #     # Stage 2: Add constraint for optimal deadline violations and minimize positions used
    #     model.addConstr(gp.quicksum(z[i] for i in range(N)) == optimal_deadline_violations, name="optimal_deadline_constraint")
        
    #     # Change objective to minimize positions used
    #     model.setObjective(gp.quicksum(position_used[j] for j in range(N)), GRB.MINIMIZE)
        
    #     # Solve second stage
    #     model.optimize()


    if model.status == GRB.OPTIMAL:
        solution = {
            "x": {(i, j): x[i, j].X for i in range(N) for j in range(N)},
            "s": [s[j].X for j in range(N)],
            "e": [e[j].X for j in range(N)],
            "t": [t[i].X for i in range(N)],
            "z": [z[i].X for i in range(N)],
            "obj": model.ObjVal
        }
        return solution
    else:
        print(f"Model status: {model.status}")
        return None

# Test parameters
N = 19
B = 16
d = [31.28333593753632, 31.28333593753632, 31.28333593753632, 31.58333593752468, 31.883335937542142, 32.1833359375305, 32.78333593753632, 43.58333593752468, 61.28333593753632, 62.1833359375305, 63.08333593752468, 73.88333593754214, 73.88333593754214, 90.08333593752468, 90.38333593754214, 91.88333593754214, 120.08333593752468, 121.88333593754214, 122.48333593754796]

for bsm, runtime in batch_runtimes.items():
    print(f"bsm: {bsm} runtime: {runtime}")

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
                if result["x"][(i, j)] == 1:
                    print(f"  Request {i} assigned to position {j}")
else:
    print("No optimal solution found")