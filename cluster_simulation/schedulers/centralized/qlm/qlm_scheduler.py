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

from bidict import bidict
from collections import deque

import numpy as np
import copy
import time


class QLMScheduler(Scheduler):

    def __init__(self, simulation, herd_assignment=None):
        super().__init__(simulation, herd_assignment)

        self.is_reordering = False
        
        self.vqs = [VirtualQueue() for w in simulation.workers]
        self.task_to_group = {}
        self.group_to_vq = {}
        self.vq_worker_bimap = bidict({ self.vqs[i]: simulation.workers[i] for i in range(len(simulation.workers)) })
        self.model_slo_group_bimap = bidict({})

        self.worker_states = { w: [] for w in simulation.workers }

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
    
    # STATE MANAGEMENT ========================================================================================

    def worker_rejected_batch(self, worker: Worker, batch: Batch, current_worker_batch: Batch):
        """
            Updates scheduler's view of workers upon receiving a batch rejection event.
        """
        if batch in self.worker_states[worker]:
            self.worker_states[worker].remove(batch)
        
        if current_worker_batch and current_worker_batch not in self.worker_states[worker]:
            self.worker_states[worker].append(current_worker_batch)
    
    # VQ REORDERING ========================================================================================

    def _check_violation(self, vqs, current_time):
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

                # NOTE: Estimation is worst case - no batching
                waiting_time = group.model.batch_exec_times[24][1] * len(group.tasks)
                est_time += waiting_time

                if est_time > group.slo:
                    return True
        return False
    
    def update_state_on_reorder(self, current_time):
        """
            Update executing batches on all workers based on current ordering of [self.vqs].
        """
        self.is_reordering = False

        # update group -> vq map
        for vq in self.vqs:
            for group in vq.groups:
                self.group_to_vq[group] = vq

        # given new VQ order, update workers as necessary
        events = []
        for vq in self.vqs:
            worker = self.vq_worker_bimap[vq]
            vq_head_batch_tasks = self._get_largest_head_batch(vq, current_time)
            if not vq_head_batch_tasks and not self.worker_states[worker]:
                continue
            elif not vq_head_batch_tasks:
                # TODO: multithreading
                events.append(EventOrders(current_time, AbortJobsOnWorkerEvent(self.simulation, worker)))
                continue
            
            worker_same_model_batch = [b for b in self.worker_states[worker]
                                       if b.model == vq_head_batch_tasks[0].model]
            if worker_same_model_batch and \
                self._is_batch_eq(vq_head_batch_tasks, worker_same_model_batch[0].tasks):
                continue # no change in head batch

            # otherwise, head group changed and worker batch should be evicted
            events += self.send_head_batch_to_worker(worker, current_time)
        return events

    def reorder_vqs(self, current_time) -> list[EventOrders]:
        """
        Reorders the virtual queues based on the scheduling policy.
        :param vqs: The list of virtual queues.
        :return: The reordered list of virtual queues.
        """ 
        if self.is_reordering:
            return []

        self.is_reordering = True
        if QLM_REORDER_POLICY == "EDF":
            self._reorder_edf(self.vqs)
            return self.update_state_on_reorder(current_time)
        elif QLM_REORDER_POLICY == "LP":
            return self._reorder_lp_solver(self.vqs, current_time)
        else:
            raise NotImplementedError("Reorder policy not recognized: Use EDF or LP")

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

    def _reorder_lp_solver(self, vqs, current_time):
        """
        Reorders the virtual queues based on the Linear Programming (LP) solver.
        :param vqs: The list of virtual queues.
        :return: The reordered list of virtual queues.
        """

        options = {
            "WLSACCESSID": GUROBI_WLSACCESSID,
            "WLSSECRET": GUROBI_WLSSECRET,
            "LICENSEID": GUROBI_LICENSEID,
        }

        groups = [grp for vq in self.vqs for grp in vq.groups]
        group_model_ids = list(set([grp.model.model_id for grp in groups]))
        models = [group_model_ids.index(grp.model.model_id) for grp in groups]
        slos = [grp.slo for grp in groups]

        N = len(groups)  # nunber of request groups
        WORKERs = len(vqs)  # number of GPUs
        SLOTs = len(groups)  # number of slots per GPU
        MODELs = len(set(models))  # number of models being served (assume serially labeled from 0 to MODELs-1)
        MODEL_SWAP_TIME = int(SameMachineCPUtoGPU_delay(max(group.model.model_size for group in groups)))

        solver_start = time.perf_counter()

        with gp.Env(params=options) as env, gp.Model(env=env) as model:
            # Model initialization
            model.setParam("OutputFlag", 0) # silence output

            x = model.addVars(WORKERs, N, N, vtype=GRB.BINARY, name="x")
            completion_slot = model.addVars(WORKERs, N, vtype=GRB.CONTINUOUS, name="ct")
            model_slot = model.addVars(WORKERs, N, vtype=GRB.INTEGER, name="model")
            transition_slot = model.addVars(WORKERs, N, vtype=GRB.BINARY, name="trans")
            slo_slot = model.addVars(WORKERs, N, vtype=GRB.CONTINUOUS, name="slo")
            penalty_slot = model.addVars(
                WORKERs, N, vtype=GRB.CONTINUOUS, name="penalty", lb=-GRB.INFINITY
            )

            # Custom constraint for MIG and static allocation settings
            if ENABLE_DYNAMIC_MODEL_LOADING:
                for g in range(WORKERs):
                    worker_memory = self.vq_worker_bimap[vqs[g]].GPU_state._total_memory
                    for slot in range(SLOTs):
                        for i in range(N):
                            if worker_memory < groups[i].model.model_size:
                                model.addConstr(x[g, i, slot] == 0)
            else:
                for g in range(WORKERs):
                    worker_models = self.vq_worker_bimap[vqs[g]].GPU_state.placed_models(0)
                    for i in range(N):
                        if groups[i].model not in worker_models:
                            for slot in range(SLOTs):
                                model.addConstr(x[g, i, slot] == 0)

            # [From QLM] Each group must be assigned exactly one slot on some worker
            for i in range(N):
                model.addConstr(
                    quicksum(
                        x[g, i, slot] for g in range(WORKERs) for slot in range(SLOTs)
                    )
                    == 1
                )

            # [Mod. from QLM] Each slot on each worker can be assigned up to 1 group
            for g in range(WORKERs):
                for slot in range(SLOTs):
                    # == 1 -> relaxed to <= 1
                    model.addConstr(quicksum(x[g, i, slot] for i in range(N)) <= 1)

            # [From QLM] Calculating model type and SLO for all GPU slots
            for g in range(WORKERs):
                for slot in range(SLOTs):
                    model.addConstr(
                        model_slot[g, slot]
                        == quicksum(models[i] * x[g, i, slot] for i in range(N))
                    )
                    model.addConstr(
                        slo_slot[g, slot]
                        == quicksum(slos[i] * x[g, i, slot] for i in range(N))
                    )

            # model_delta[g, slot] = abs(difference between adjacent slot model IDs)
            diff = model.addVars(WORKERs, SLOTs, lb=-MODELs, ub=MODELs, vtype=GRB.INTEGER, name="diff")
            model_delta = model.addVars(WORKERs, SLOTs, lb=0, ub=MODELs, vtype=GRB.INTEGER, name="model_delta")
            for g in range(WORKERs):
                for slot in range(1, SLOTs):
                    model.addConstr(diff[g, slot] == model_slot[g, slot] - model_slot[g, slot - 1])
                    model.addGenConstrAbs(model_delta[g, slot], diff[g, slot])

            # Initializing model swap time for first GPU slot to 0
            for g in range(WORKERs):
                model.addConstr(transition_slot[g, 0] == 0)
                model.addConstr(model_delta[g, 0] == 0)

            # [Mod. from QLM] Set transition slot based on model ID deltas
            for g in range(WORKERs):
                for slot in range(1, SLOTs):
                    model.addConstr(model_delta[g, slot] >= transition_slot[g, slot])
                    model.addConstr(model_delta[g, slot] <= MODELs * transition_slot[g, slot])

                    # Original QLM:
                    # model.addConstr(
                    #     model_slot[g, slot] - model_slot[g, slot - 1]
                    #     <= 1 + MODELs - MODELs * transition_slot[g, slot]
                    # )
                    # model.addConstr(
                    #     model_slot[g, slot] - model_slot[g, slot - 1]
                    #     >= MODELs * transition_slot[g, slot] - MODELs - 1
                    # )
                    # model.addConstr(
                    #     model_slot[g, slot] - model_slot[g, slot - 1]
                    #     <= MODELs * transition_slot[g, slot] - 1
                    # )
                    # model.addConstr(
                    #     model_slot[g, slot] - model_slot[g, slot - 1]
                    #     >= 1 - MODELs * transition_slot[g, slot]
                    # )

            # Estimating cumulative completion time per GPU slot
            worst_case_group_exec_time = [len(groups[i].tasks) * groups[i].model.batch_exec_times[24][1] for i in range(N)]
            worst_case_vq_exec_time = sum(worst_case_group_exec_time)
            for g in range(WORKERs):
                for slot in range(SLOTs):
                    model.addConstr(
                        completion_slot[g, slot]
                        >= quicksum(
                            worst_case_group_exec_time[i] * x[g, i, j]
                            for i in range(N)
                            for j in range(slot + 1)
                        )
                    )

                    # Prevent a VQ with no groups assigned from being prioritized over
                    # multiple VQs spreading out requests
                    model.addConstr(
                        completion_slot[g, slot]
                        >= worst_case_vq_exec_time * (1 - quicksum(x[g, i, j] for i in range(N) for j in range(slot + 1)))
                    )

            # [Mod. from QLM] Estimating penalty for violating SLOs
            for g in range(WORKERs):
                for slot in range(SLOTs):
                    model.addConstr(
                        penalty_slot[g, slot]
                        == completion_slot[g, slot]
                        + (MODEL_SWAP_TIME * transition_slot[g, slot] if ENABLE_DYNAMIC_MODEL_LOADING else 0)
                        - slo_slot[g, slot]
                    )

            # [Mod. from QLM] Constraining no SLO violation -> Allow tardy
            # for g in range(WORKERs):
            #     for slot in range(SLOTs):
            #         model.addConstr(penalty_slot[g, slot] <= 0)

            # [From QLM] Minimize total penalty
            model.setObjective(
                quicksum(
                    penalty_slot[g, slot]
                    for g in range(WORKERs)
                    for slot in range(SLOTs)
                ),
                GRB.MINIMIZE,
            )

            model.optimize()

            solver_end = time.perf_counter()
            solver_time = solver_end - solver_start

            if model.Status == GRB.OPTIMAL:
                new_vq_map = { vq.vq_id: [] for vq in vqs }
                for g in range(WORKERs):
                    for i in range(N):
                        for slot in range(N):  # same dimension as when creating x
                            var = x[g, i, slot]
                            if var and var.X > 0.5:
                                new_vq_map[vqs[g].vq_id].append(groups[i])
                return [EventOrders(current_time + solver_time,
                                    UpdateQLMVirtualQueueOrdering(self.simulation, new_vq_map))]
            else:
                print("Solver status code:", model.Status)
                print("No optimal solution found for LP, reverting to EDF")
                self._reorder_edf(self.vqs)
                return self.update_state_on_reorder(current_time)

    # BATCHING HELPERS ========================================================================================

    def send_head_batch_to_worker(self, worker: Worker, current_time: float) -> list[EventOrders]:
        """
            Send the current head batch on the virtual queue for [worker] to [worker]
            for execution, preempting any executing batches if necessary.
        """
        events = []
        vq_head_batch_tasks = self._get_largest_head_batch(self.vq_worker_bimap.inv[worker], current_time, close_group=True)
        new_batch = Batch(vq_head_batch_tasks)
        if len(self.worker_states[worker]) == 0:
            self.worker_states[worker].append(new_batch)
            events.append(EventOrders(
                current_time + CPU_to_CPU_delay(sum(t.input_size for t in new_batch.tasks)),
                BatchArrivalAtWorker(self.simulation, worker, new_batch)))
        elif not ENABLE_MULTITHREADING:
            old_batch = self.worker_states[worker][0]
            self.worker_states[worker] = [new_batch]
            events.append(EventOrders(
                current_time + CPU_to_CPU_delay(sum(t.input_size for t in new_batch.tasks)),
                BatchPreemptionAtWorker(self.simulation, worker, new_batch, old_batch.id)))
        else:
            raise NotImplementedError("Multiple abort")
        return events

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