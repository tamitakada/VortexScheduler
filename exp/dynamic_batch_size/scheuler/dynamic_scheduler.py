from collections import deque
# from main import Request
import math
import csv
import os
import gurobipy as gp
from gurobipy import GRB
import logging

# Set Gurobi WLS license credentials
# os.environ['GRB_WLSACCESSID'] = 'a489def3-9272-4267-beba-4ab515eb13a5'
# os.environ['GRB_WLSSECRET'] = '325c2865-d596-42bf-8576-cc8b3698bbe3'


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



class DynamicScheduler:
    max_batch_size: int
    batch_runtimes: dict
    slo: float

    def __init__(self, max_batch_size: int, slo: float, logger=None):
        # Use the logger passed from main, or create one for this module
        if logger is not None:
            self.logger = logger
        else:
            # Use the same logger name as main.py to inherit the configuration
            self.logger = logging.getLogger('__main__')

        self.slo = slo
        self.max_batch_size = max_batch_size

    def preempt(self, current_batch: deque, queue: deque, current_time: float) -> bool:
        return False


    def schedule(self, current_batch: deque, queue: deque, current_time: float) -> float:
        N = len(queue)
        B = self.max_batch_size
        d = [request.arrival_time + self.slo - current_time for request in queue]

        self.logger.info(f"d: {d}")
        self.logger.info(f"base latency: {batch_runtimes}")
        
        assert (len(current_batch) == 0, f"Without preemption, schedule should only happen when the current batch is finished (current_batch_size: {len(current_batch)})")

        if N == 0:
            return math.inf

        result = self.solve_ilp(N, B, d)

        if result is None:
            assert False, "No solution found"
        
        
        # Print assignments for each j
        self.logger.info(f"result: {result}")
        self.logger.info("Assignments (x_{i,j} = 1):")
        for j in range(N):
            self.logger.info(f"Position j={j} (batch size s[{j}]={result['s'][j]}):")
            for i in range(N):
                if result["x"][(i, j)] == 1:
                    self.logger.info(f"  Request {queue[i].id} assigned to position {j}")

        req_ids = set([queue[i].id for i in range(N) if result["x"][(i,0)] == 1])

        remaining_req = deque()
        while len(queue) > 0:
            req = queue.pop()
            if req.id in req_ids:
                current_batch.append(req)
            else:
                remaining_req.append(req)

        while len(remaining_req) > 0:
            req = remaining_req.pop()
            queue.append(req)

        
        

        self.logger.info(f"!!! Current batch: {[req.id for req in current_batch]}")
        self.logger.info(f"!!!Queue: {[req.id for req in queue]}")
        
        assert len(current_batch) == result["s"][0], f"Current batch size is not equal to the scheduled batch size: {len(current_batch)} and {result['s'][0]}"
        assert len(current_batch) <= self.max_batch_size, f"Current batch size is greater than the max batch size: {len(current_batch)} and {self.max_batch_size}"
        assert len(current_batch) > 0, "Current batch size is 0"
        

        return math.inf


    def solve_ilp(self, N, B, d):
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
                "x": {(i, j): round(x[i, j].X) for i in range(N) for j in range(N)},
                "s": [round(s[j].X) for j in range(N)],
                "e": [e[j].X for j in range(N)],
                "t": [t[i].X for i in range(N)],
                "z": [round(z[i].X) for i in range(N)],
                "obj": round(model.ObjVal)
            }
            return solution
        else:
            print(f"Model status: {model.status}")
            return None

