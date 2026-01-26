from core.data_models.model_data import ModelData
import core.configs.gen_config as gcfg

from uuid import uuid4
from copy import deepcopy

class AllocationUpdateStrategy:
    SORT_AND_PACK = 0
    FIRST_VALID_WORKER = 1
    LAST_VALID_WORKER = 2


class ModelAllocation:

    worker_size = max(gcfg.VALID_WORKER_SIZES)

    def __init__(self, simulation, cfgs: dict[str, tuple[int, list[int]]]={}, reset_batch_sizes: bool=True):
        self.simulation = simulation

        self.worker_cfgs: dict[str, tuple[int, list[int]]] = deepcopy(cfgs)
        # [worker ID, create time] sorted by create time
        self.worker_ids_by_create_time: list[tuple[str, float]] = [(k, 0) for k in cfgs.keys()] if cfgs else []

        self.model_ids: list[int] = sum([mids.copy() for (_, mids) in cfgs.values()], []) if cfgs else []
        self.models: dict[int, ModelData] = {
            mid: self.simulation.models[mid].copy() for mid in set(self.model_ids)}
        
        # start at max batch size = 1
        if reset_batch_sizes:
            for mid in self.models.keys():
                self.models[mid].max_batch_size = 1

    def _check_model_count(self, model_id: int):
        assert(self.count(model_id) == 
               sum(mids.count(model_id) for (_, mids) in self.worker_cfgs.values()))
        
        assert(len(mids) < gcfg.MAX_NUM_MODELS_PER_NODE for (_, mids) in self.worker_cfgs.values())

    def add_model(self, time: float, model_id: int, strategy: int, default_bsize_1: bool=False) -> list[tuple[str, tuple[int, list[int]]]]:
        """Attempts to add one copy of a given model to the current
        allocation. If the model cannot be added, returns empty list
        and does not change current allocation.
        
        Args:
            model_id: ID of model to add
            strategy: Strategy for adding model (see AllocationUpdateStrategy)
            default_bsize_1: Behavior when model_id is new to allocation; if True,
            sets model max batch size to 1. Else uses max batch size defined in
            config.

        Returns:
            changed_workers: List of workers (id, worker config) that were modified from
            the current allocation.
        """
        assert(strategy in [AllocationUpdateStrategy.SORT_AND_PACK,
                            AllocationUpdateStrategy.FIRST_VALID_WORKER,
                            AllocationUpdateStrategy.LAST_VALID_WORKER])
        
        if strategy == AllocationUpdateStrategy.SORT_AND_PACK:
            worker_mids = [[]]

            if model_id not in self.models:
                self.models[model_id] = self.simulation.models[model_id].copy()
                if default_bsize_1:
                    self.models[model_id].max_batch_size = 1

            candidate_mids = self.model_ids.copy()
            candidate_mids.append(model_id)
            candidate_mids = sorted(candidate_mids, key=lambda mid: self.models[mid].size)
            for mid in candidate_mids:
                used_mem = sum(self.models[wmid].size for wmid in worker_mids[-1])
                if len(worker_mids[-1]) == 0 or \
                    (used_mem + self.models[mid].size < ModelAllocation.worker_size and len(worker_mids[-1]) < gcfg.MAX_NUM_MODELS_PER_NODE):
                    worker_mids[-1].append(mid)
                elif len(worker_mids) == gcfg.MAX_NUM_NODES:
                    # cannot add more workers, addition is not feasible
                    return []
                else:
                    worker_mids.append([mid])

            delta_cfgs = []
            for i in range(len(worker_mids)):
                if i < len(self.worker_ids_by_create_time) and self.worker_cfgs[self.worker_ids_by_create_time[i][0]] != worker_mids[i]:
                    self.worker_cfgs[self.worker_ids_by_create_time[i][0]] = (ModelAllocation.worker_size // 10**6, worker_mids[i])
                    delta_cfgs.append((self.worker_ids_by_create_time[i][0],
                                       self.worker_cfgs[self.worker_ids_by_create_time[i][0]]))
                    
                elif i >= len(self.worker_ids_by_create_time):
                    new_id = uuid4()
                    self.worker_ids_by_create_time.append((new_id, time))
                    self.worker_cfgs[new_id] = (ModelAllocation.worker_size // 10**6, worker_mids[i])
                    delta_cfgs.append((self.worker_ids_by_create_time[i][0],
                                       self.worker_cfgs[self.worker_ids_by_create_time[i][0]]))

            self.model_ids.append(model_id)
            self._check_model_count(model_id)
            return delta_cfgs
        
        else:
            worker_range = None
            if strategy == AllocationUpdateStrategy.FIRST_VALID_WORKER:
                worker_range = range(0, len(self.worker_ids_by_create_time), 1)
            else:
                worker_range = range(len(self.worker_ids_by_create_time) - 1, -1, -1)
            
            for i in worker_range:
                worker_id = self.worker_ids_by_create_time[i][0]
                worker_cfg = self.worker_cfgs[worker_id]
                used_mem = sum(self.models[wmid].size for wmid in worker_cfg[1])
                if (used_mem + self.models[model_id].size < ModelAllocation.worker_size) and len(worker_cfg[1]) < gcfg.MAX_NUM_MODELS_PER_NODE:
                    self.worker_cfgs[worker_id][1].append(model_id)
                    self.model_ids.append(model_id)
                    self._check_model_count(model_id)
                    return [(worker_id, self.worker_cfgs[worker_id])]
            
            if len(self.worker_ids_by_create_time) < gcfg.MAX_NUM_NODES:
                new_id = uuid4()
                self.worker_ids_by_create_time.append((new_id, time))
                self.worker_cfgs[new_id] = (ModelAllocation.worker_size // 10**6, [model_id])
                self.model_ids.append(model_id)
                self._check_model_count(model_id)
                return [(new_id, self.worker_cfgs[new_id])]
        
        return []
    
    def remove_model(self, model_id: int, strategy: int) -> list[tuple[int, tuple[int, list[int]]]]:
        """Removes one copy of a given model from the current allocation.
        
        Args:
            model_id: ID of model to remove
            strategy: Strategy for adding model (see AllocationUpdateStrategy)

        Returns:
            changed_workers: List of workers (id, worker config) that were 
            modified from the current allocation.
        """
        assert(strategy in [AllocationUpdateStrategy.SORT_AND_PACK,
                            AllocationUpdateStrategy.FIRST_VALID_WORKER,
                            AllocationUpdateStrategy.LAST_VALID_WORKER])
        
        self.model_ids.remove(model_id)

        if strategy == AllocationUpdateStrategy.SORT_AND_PACK:
            worker_mids = [[]]

            for mid in sorted(self.model_ids, key=lambda mid: self.models[mid].size):
                used_mem = sum(self.models[wmid].size for wmid in worker_mids[-1])
                if len(worker_mids[-1]) == 0 or used_mem + self.models[mid].size < ModelAllocation.worker_size:
                    worker_mids[-1].append(mid)
                else:
                    worker_mids.append([mid])

            delta_cfgs = []
            for i in range(len(worker_mids)):
                assert(i < len(self.worker_ids_by_create_time))
                if self.worker_cfgs[self.worker_ids_by_create_time[i][0]] != worker_mids[i]:
                    self.worker_cfgs[self.worker_ids_by_create_time[i][0]] = (ModelAllocation.worker_size // 10**6, worker_mids[i])
                    delta_cfgs.append((self.worker_ids_by_create_time[i][0],
                                       self.worker_cfgs[self.worker_ids_by_create_time[i][0]]))
            
            self._check_model_count(model_id)
            return delta_cfgs
        
        else:
            worker_range = None
            if strategy == AllocationUpdateStrategy.FIRST_VALID_WORKER:
                worker_range = range(0, len(self.worker_ids_by_create_time), 1)
            else:
                worker_range = range(len(self.worker_ids_by_create_time) - 1, -1, -1)

            for i in worker_range:
                worker_id = self.worker_ids_by_create_time[i][0]
                if model_id in self.worker_cfgs[worker_id][1]:
                    self.worker_cfgs[worker_id][1].remove(model_id)
                    
                    # if no models remaining on worker, rm worker
                    if len(self.worker_ids_by_create_time) > gcfg.MIN_NUM_NODES and \
                        len(self.worker_cfgs[worker_id][1]) == 0:

                        for j, (wid, _) in enumerate(self.worker_ids_by_create_time):
                            if wid == worker_id:
                                self.worker_ids_by_create_time.pop(j)

                        self.worker_cfgs.pop(worker_id)
                        self._check_model_count(model_id)
                        return [(worker_id, None)]
                
                    self._check_model_count(model_id)
                    return [(worker_id, self.worker_cfgs[worker_id])]
            
            raise ValueError(f"No worker with model ID={model_id}")

    def count(self, model_id: int) -> int:
        return self.model_ids.count(model_id)
    
    def __repr__(self):
        return str(self.worker_cfgs)