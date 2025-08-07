# Portions adapted from https://github.com/QLM-project/QLM/tree/main/qlm

from core.job import *
from core.task import *
from core.network import *
from core.config import *
from core.events import *
from core.batch import Batch

from schedulers.centralized.scheduler import Scheduler
from schedulers.centralized.qlm.group import Group
from schedulers.centralized.qlm.virtual_queue import VirtualQueue

import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
from gurobipy import LinExpr

from bidict import bidict
from collections import deque

import numpy as np
import copy


class QLMScheduler(Scheduler):

    def __init__(self, simulation, herd_assignment=None):
        super().__init__(simulation, herd_assignment)
        
        self.vqs = [VirtualQueue() for w in simulation.workers]
        self.task_to_group = {}
        self.group_to_vq = {}
        self.vq_worker_bimap = bidict({ self.vqs[i]: simulation.workers[i] for i in range(len(simulation.workers)) })
        self.model_slo_group_bimap = bidict({})

        self.worker_states = { w: [] for w in simulation.workers }

    def worker_rejected_batch(self, worker: Worker, batch: Batch, current_worker_batch: Batch):
        if batch in self.worker_states[worker]:
            self.worker_states[worker].remove(batch)
        
        if current_worker_batch and current_worker_batch not in self.worker_states[worker]:
            self.worker_states[worker].append(current_worker_batch)

    def _get_largest_head_batch(self, vq, current_time, close_group=False) -> list[Task]:
        """
            Returns the largest contiguous set of tasks requiring the same model
            pulled from [vq]. Does not modify [vq].
        """
        if len(vq.groups) == 0:
            return []
        
        head_batch_tasks = copy.copy(vq.groups[0].tasks)

        assert(len(head_batch_tasks) <= head_batch_tasks[0].max_batch_size)
        assert(t.log.task_arrival_at_scheduler_timestamp >= current_time for t in head_batch_tasks)
        assert(len(set(t.job_id for t in head_batch_tasks)) == len(head_batch_tasks))

        if close_group:
            if vq.groups[0] in self.model_slo_group_bimap.inv:
                self.model_slo_group_bimap.inv.pop(vq.groups[0])
        
        return head_batch_tasks
    
    def _is_batch_eq(self, b1: list[Task], b2: list[Task]) -> bool:
        if len(b1) != len(b2):
            return False
        
        b1 = sorted(b1, key=lambda t: t.job_id)
        b2 = sorted(b2, key=lambda t: t.job_id)

        for i in range(len(b1)):
            if b1[i].task_id != b2[i].task_id or b1[i].job_id != b2[i].job_id:
                return False
        
        return True
    
    def _drop_late_jobs(self, current_time):
        """

        """
        for vq in self.vqs:
            for group in list(vq.groups):
                for task in list(group.tasks):
                    deadline = task.job.create_time + task.job.slo * (1 + SLO_SLACK)
                    if current_time >= deadline:
                        group.tasks.remove(task)
                        if not (self.simulation.task_drop_log["job_id"]==task.job.id).any():
                            self.simulation.task_drop_log.loc[len(self.simulation.task_drop_log)] = {
                                "client_id": task.job.client_id,
                                "job_id": task.job_id,
                                "workflow_id": task.task_type[0],
                                "task_id": task.task_type[1],
                                "drop_time": current_time,
                                "arrival_time": task.log.task_arrival_at_scheduler_timestamp,
                                "slo": task.job.slo,
                                "deadline": deadline
                            }
                if len(group.tasks) == 0:
                    vq.groups.remove(group)
                    if group in self.model_slo_group_bimap.inv:
                        self.model_slo_group_bimap.inv.pop(group)

    def schedule_job_on_arrival(self, job, current_time):
        return self.schedule_tasks_on_arrival([t for t in job.tasks if len(t.required_task_ids) == 0], 
                                              current_time)

    def schedule_tasks_on_arrival(self, tasks, current_time):
        if QLM_ENABLE_DROP:
            self._drop_late_jobs(current_time)

        prev_qsize = sum(len(grp.tasks) for vq in self.vqs for grp in vq.groups)
        for task in tasks:
            # Add task to task group
            slo = task.job.slo if SLO_GRANULARITY == "JOB" else task.slo
            if (task.model.model_id, slo) in self.model_slo_group_bimap and \
                len(self.model_slo_group_bimap[(task.model.model_id, slo)].tasks) < task.max_batch_size:
                existing_group = self.model_slo_group_bimap[(task.model.model_id, slo)]
                existing_group.add_task(task)
                self.task_to_group[task] = existing_group
            else:
                new_group = Group(task.model , slo)
                print("Adding new group with model and slo", task.model.model_id, slo)
                new_group.add_task(task)

                self.model_slo_group_bimap[(task.model.model_id, slo)] = new_group
                self.task_to_group[task] = new_group

                # Select a random virtual queue to add the group to
                if ENABLE_DYNAMIC_MODEL_LOADING:
                    # select worker that has space to load model
                    vq_idx = np.random.choice([wid for wid in range(len(self.vqs))
                                                if self.simulation.workers[wid].GPU_state._total_memory >= task.model.model_size])
                else:
                    # select worker that has model
                    vq_idx = np.random.choice([wid for wid in range(len(self.vqs))
                                               if task.model in self.simulation.workers[wid].GPU_state.placed_models(current_time)])     
            
                self.vqs[vq_idx].add_group(new_group)
                self.group_to_vq[new_group] = self.vqs[vq_idx]
                print(f"GROUP {new_group.group_id} ASSIGNED TO W{self.vq_worker_bimap[self.vqs[vq_idx]].worker_id}")

        assert(sum(len(grp.tasks) for vq in self.vqs for grp in vq.groups) == prev_qsize + len(tasks))

        if self._check_violation(self.vqs, current_time):
            return [EventOrders(current_time, ReorderQLMVirtualQueues(self.simulation))]
        else:
            events = []
            workers = set(self.vq_worker_bimap[self.group_to_vq[self.task_to_group[task]]] for task in tasks)
            for worker in workers:
                # TODO: multithreading for QLM
                if all(not s.reserved_batch for s in worker.GPU_state.state_at(current_time)) and \
                    len(self.worker_states[worker]) == 0:
                    events += self.send_head_batch_to_worker(worker, current_time)
        return events
    
    def schedule_on_worker_batch_finish(self, worker, batch, current_time):
        """
            Form the largest batch < max batch size possible based on [worker]'s
            virtual queue ordering and send for execution on the worker.
        """
        if batch in self.worker_states[worker]:
            self.worker_states[worker].remove(batch)
        
        # remove by checking all queues in case of reordering
        for task in batch.tasks:
            if not (self.simulation.task_drop_log["job_id"]==task.job.id).any():
                did_remove = False
                for vq in self.vqs:
                    for group in list(vq.groups):
                        if task in list(group.tasks):
                            group.tasks.remove(task)
                            if len(group.tasks) == 0:
                                vq.groups.remove(group)
                                if group in self.model_slo_group_bimap.inv:
                                    self.model_slo_group_bimap.inv.pop(group)
                            did_remove = True
                            break
                    if did_remove:
                        break
                assert(did_remove)
             
        if QLM_ENABLE_DROP:
            self._drop_late_jobs(current_time)
        
        vq = self.vq_worker_bimap.inv[worker]
        batch_tasks = self._get_largest_head_batch(vq, current_time, close_group=True)
        if batch_tasks:
            new_batch = Batch(batch_tasks)
            self.worker_states[worker].append(new_batch)
            return worker.maybe_start_batch(new_batch, current_time)
        return []
    
    def schedule_tasks_on_queue(self, current_time):
        return super().schedule_tasks_on_queue(current_time)

    def _check_violation(self, vqs, time):
        """
        Checks for SLO violations in the virtual queues.
        :param vqs: The list of virtual queues.
        :return: True if there is a violation, False otherwise.
        """
        for vq in vqs:
            est_time = 0
            curr_model = None

            for group in vq.groups:
                prev_model = curr_model
                curr_model = group.model

                if prev_model != None and prev_model != curr_model:
                    est_time += SameMachineCPUtoGPU_delay(curr_model.model_size)

                # [!] NOTE: Estimation is worst case - no batching
                waiting_time = group.model.batch_exec_times[24][1] * len(group.tasks)
                est_time += waiting_time

                if est_time > group.slo:
                    return True
        return False
    
    def send_head_batch_to_worker(self, worker: Worker, time: float) -> list[EventOrders]:
        events = []
        vq_head_batch_tasks = self._get_largest_head_batch(self.vq_worker_bimap.inv[worker], time, close_group=True)
        new_batch = Batch(vq_head_batch_tasks)
        if len(self.worker_states[worker]) == 0:
            self.worker_states[worker].append(new_batch)
            events.append(EventOrders(
                time + CPU_to_CPU_delay(sum(t.input_size for t in new_batch.tasks)),
                BatchArrivalAtWorker(self.simulation, worker, new_batch)))
        elif not ENABLE_MULTITHREADING:
            old_batch = self.worker_states[worker][0]
            self.worker_states[worker] = [new_batch]
            events.append(EventOrders(
                time + CPU_to_CPU_delay(sum(t.input_size for t in new_batch.tasks)),
                BatchPreemptionAtWorker(self.simulation, worker, new_batch, old_batch.id)))
        else:
            raise NotImplementedError("Multiple abort")
        return events

    def reorder_vqs(self, time) -> list[EventOrders]:
        """
        Reorders the virtual queues based on the scheduling policy.
        :param vqs: The list of virtual queues.
        :return: The reordered list of virtual queues.
        """ 
        prev_qsize = sum(len(grp.tasks) for vq in self.vqs for grp in vq.groups)

        if QLM_REORDER_POLICY == "EDF":
            self._reorder_edf(self.vqs)
        elif QLM_REORDER_POLICY == "LP":
            self._reorder_lp_solver(self.vqs)
        else:
            raise NotImplementedError("Reorder policy not recognized: Use EDF or LP")

        reordered_qsize = sum(len(grp.tasks) for vq in self.vqs for grp in vq.groups)
        assert(prev_qsize == reordered_qsize)

        # update group -> vq map
        for vq in self.vqs:
            for group in vq.groups:
                self.group_to_vq[group] = vq

        events = []
        for vq in self.vqs:
            worker = self.vq_worker_bimap[vq]
            vq_head_batch_tasks = self._get_largest_head_batch(vq, time)
            if not vq_head_batch_tasks:
                continue
            
            worker_same_model_batch = [b for b in self.worker_states[worker]
                                       if b.model == vq_head_batch_tasks[0].model]
            if worker_same_model_batch and \
                self._is_batch_eq(vq_head_batch_tasks, worker_same_model_batch[0].tasks):
                continue # no change in head batch

            # otherwise, head group changed and worker batch should be evicted
            events += self.send_head_batch_to_worker(worker, time)

        return events

    def _reorder_edf(self, vqs):
        """
        Reorders the virtual queues based on the Earliest Deadline First (EDF) policy.
        :param vqs: The list of virtual queues.
        :return: The reordered list of virtual queues.
        """
        for vq in vqs:
            groups = list(vq.groups)
            groups.sort(key=lambda x: x.slo)
            vq.groups = deque(groups)

        return vqs

    def _reorder_lp_solver(self, vqs):
        # TODO: ADDITIONAL STATIC ALLOC CONSTRAINT / TOTAL MEMORY CONSTRAINT

        """
        Reorders the virtual queues based on the Linear Programming (LP) solver.
        :param vqs: The list of virtual queues.
        :return: The reordered list of virtual queues.
        """
        all_groups = [grp for vq in self.vqs for grp in vq.groups]

        G = range(len(self.vqs))                            # set of virtual queues
        I = range(sum(len(vq.groups) for vq in self.vqs))   # set of request groups
        L = len(all_groups)                                 # max length of virtual queue

        print(f"L: {L}")

        models_i = [group.model.model_id for group in all_groups]  # model type for each request group i
        slos_i = [group.slo for group in all_groups]    # SLO value for each request group i
        completion_i = [len(group.tasks) * group.model.batch_exec_times[24][1] for group in all_groups]  # estimated completion time for each request group i
        swap_time = SameMachineCPUtoGPU_delay(max(group.model.model_size for group in all_groups))        # constant swap time for model switching

        # Start model
        m = gp.Model("QLM_GlobalScheduler")
        m.setParam('OutputFlag', 0)

        # Variables
        x = m.addVars(G, I, L, vtype=GRB.BINARY, name="x")  # assignment of i to g at pos j
        t = m.addVars(G, L, vtype=GRB.BINARY, name="t")     # model transition
        wt = m.addVars(G, L, vtype=GRB.CONTINUOUS, name="wt")  # waiting time
        p = m.addVars(G, L, vtype=GRB.CONTINUOUS, name="penalty")  # penalty

        # Custom constraint for MIG and static allocation settings
        if ENABLE_DYNAMIC_MODEL_LOADING:
            for g in G:
                worker_memory = self.vq_worker_bimap[self.vqs[g]].GPU_state._total_memory
                for j in range(L):
                    for i in I:
                        if worker_memory < all_groups[i].model.model_size:
                            m.addConstr(x[g, i, j] == 0)
        else:
            for g in G:
                worker_models = self.vq_worker_bimap[self.vqs[g]].GPU_state.placed_models(0)
                for i in I:
                    if all_groups[i].model not in worker_models:
                        for j in range(L):
                            m.addConstr(x[g, i, j] == 0)

        # Constraint (6): Each request group assigned to exactly one position
        for i in I:
            m.addConstr(quicksum(x[g, i, j] for g in G for j in range(L)) == 1)

        # Constraint: Each position can hold at most one request group
        for g in G:
            for j in range(L):
                m.addConstr(quicksum(x[g, i, j] for i in I) <= 1)

        # Constraints (7) and (8): model and SLO at each position
        m_gj = {}
        slo_gj = {}
        for g in G:
            for j in range(L):
                m_gj[g, j] = m.addVar(vtype=GRB.CONTINUOUS, name=f"model_{g}_{j}")
                slo_gj[g, j] = m.addVar(vtype=GRB.CONTINUOUS, name=f"slo_{g}_{j}")
                m.addConstr(m_gj[g, j] == quicksum(models_i[i] * x[g, i, j] for i in I))
                m.addConstr(slo_gj[g, j] == quicksum(slos_i[i] * x[g, i, j] for i in I))

        # Constraint (9): Model swap indicator
        # Initializing model swap time for first GPU slot to 0
        for g in G:
            m.addConstr(t[g, 0] == 0)

        # Calculating model swap times based on adjacent GPU slots
        num_models = len(models_i)
        for g in G:
            for j in range(1, L):
                m.addConstr(
                    m_gj[g, j] - m_gj[g, j]
                    <= 1 + num_models - num_models * t[g, j]
                )
                m.addConstr(
                    m_gj[g, j] - m_gj[g, j - 1]
                    >= num_models * t[g, j] - num_models - 1
                )
                m.addConstr(
                    m_gj[g, j] - m_gj[g, j - 1]
                    <= num_models * t[g, j] - 1
                )
                m.addConstr(
                    m_gj[g, j] - m_gj[g, j - 1]
                    >= 1 - num_models * t[g, j]
                )

        # Constraint (10): Waiting time
        # Estimating cumulative completion time per GPU slot
        for g in G:
            for j1 in range(L):
                m.addConstr(
                    wt[g, j1]
                    == quicksum(
                        completion_i[i] * x[g, i, j2]
                        for i in I for j2 in range(j1 if j1 == L-1 else (j1+1))
                    )
                )
        
        # Constraint (11): Penalty = wait + swap - SLO
        # Estimating penalty for violating SLOs
        for g in G:
            for j in range(L):
                m.addConstr(
                    p[g, j]
                    == wt[g, j]
                    + swap_time * m_gj[g, j]
                    - slo_gj[g, j]
                )

        # Constraint (12): SLO must be met (penalty <= 0)
        # NOTE: Allows tardy jobs
        # for g in G:
        #     for j in range(L):
        #         m.addConstr(p[g, j] <= 0)

        # Objective: Minimize total penalty
        m.setObjective(quicksum(p[g, j] for g in G for j in range(L)), GRB.MINIMIZE)

        # Solve
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            print("Optimal solution found for LP!")

            for vq in vqs:
                while len(vq.groups) > 0:
                    vq.pop_group()

            for g in G:
                for i in I:
                    for slot in range(L):
                        var_name = f"x[{g},{i},{slot}]"
                        var = m.getVarByName(var_name)
                        if var and abs(var.X) > 0.5:  # Only display non-zero values
                            vqs[g].groups.append(all_groups[i])
            return vqs
        else:
            print("Solver status code:", m.Status)
            print("No optimal solution found for LP, reverting to EDF")

            # debugging
            # m.computeIIS()
            # m.write("model.ilp")
            # for i, grp in enumerate(all_groups):
            #     print(f"{i}: {grp.model.model_id}")

            self._reorder_edf(vqs)