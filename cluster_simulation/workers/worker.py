from core.network import *
from core.data_models.model_data import ModelData
from core.batch import Batch
from core.events.worker_events import *
from core.events.base import *

from workers.gpu_state import ModelState, GPUState


class Worker(object):
    """ Abstract class representing workers. """

    _abandoned_batches = []

    def __init__(self, simulation, id: str, total_memory: int, group_id: int=-1, created_at: float=0):
        assert(total_memory in [24, 12, 6])

        self.create_time = created_at

        self.group_id = group_id
        self.id = id
        self.simulation = simulation
        self.total_memory = total_memory
        
        self.GPU_state = GPUState(total_memory * (10**6))

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return "[Worker_id:{}]".format(self.id)

    def __eq__(self, other):
        if isinstance(other, Worker):
            return self.id == other.id
        return False

    def __ne__(self, other):
        return not (self == other)
    
    def __lt__(self, other):
        return self.create_time < other.create_time
    
    """ ----------  BATCH TRACKING AND MANAGEMENT  ---------- """
    
    def evict_batch(self, batch_id: int, time: float):
        evicted_batch = None
        for s in self.GPU_state.state_at(time):
            if s.reserved_batch and s.reserved_batch.id == batch_id:
                evicted_batch = s.reserved_batch
                break
        self.GPU_state.release_busy_model(batch_id, time)
        Worker._abandoned_batches.append(batch_id)
        return evicted_batch

    def did_abandon_batch(self, batch_id: int):
        return batch_id in Worker._abandoned_batches
    
    """ ----------  LOCAL MEMORY MANAGEMENT  ---------- """

    def evict_model(self, time: float, model_id: int) -> Batch | None:
        """Evict one copy of a model. Prioritizes idle copies.

        Args:
            time: Time to start eviction
            model_id: ID of model to evict (not instance ID)

        Returns:
            evicted_batch: If evicted model was running a batch,
            returns the evicted batch.
        """

        instance_id, evicted_batch = self.GPU_state.evict_any(model_id, time, 0)
        if evicted_batch:
            Worker._abandoned_batches.append(evicted_batch.id)
        
        self.simulation.worker_model_log.loc[len(self.simulation.worker_model_log)] = {
            "start_time": time,
            "end_time": time ,
            "worker_id": self.id,
            "model_id": model_id,
            "instance_id": instance_id,
            "placed_or_evicted": "evicted"
        }
        return evicted_batch
    
    def fetch_model(self, model_data: ModelData | None, batch, current_time: float, exec_time=-1):
        if model_data == None:
            return []
        
        fetch_time = 0
        fetch_time = SameMachineCPUtoGPU_delay(model_data.size)

        reserve_until = -1 if exec_time < 0 else (current_time + fetch_time + exec_time)

        instance_id = self.GPU_state.fetch_model(
            model_data, current_time, fetch_time, 
            reserved_batch=batch, reserve_until=reserve_until)
        
        self.simulation.worker_model_log.loc[len(self.simulation.worker_model_log)] = {
            "start_time": current_time,
            "end_time": current_time + fetch_time,
            "worker_id": self.id,
            "model_id": model_data.id,
            "instance_id": instance_id,
            "placed_or_evicted": "placed"
        }

        return [EventOrders(current_time + fetch_time, 
                            WorkerFinishedModelFetchEvent(self.simulation, self, model_data.id))]
    
    _CAN_RUN_NOW = 0
    _CAN_RUN_ON_EVICT = 1
    _CANNOT_RUN = 2

    def can_run_task(self, current_time: float, model_data: ModelData | None) -> int:
        """
            Returns _CAN_RUN_NOW if model None, or model is on GPU and not currently in use.
            Returns _CAN_RUN_ON_EVICT if model can be loaded onto the GPU upon evicting
            unused models.
            Returns _CANNOT_RUN otherwise.
        """
        if model_data == None or self.GPU_state.does_have_idle_copy(model_data.id, current_time):
            return self._CAN_RUN_NOW
        
        if self.GPU_state.can_fetch_model(model_data, current_time):
            return self._CAN_RUN_NOW
        
        if self.GPU_state.can_fetch_model_on_eviction(model_data, current_time):
            return self._CAN_RUN_ON_EVICT
        
        return self._CANNOT_RUN

    # ------------------------- cached model history update helper functions ---------------
    def get_history(self, history, current_time, info_staleness) -> list:
        delayed_time = current_time - info_staleness
        last_index = len(history) - 1
        while last_index >= 0:
            if history[last_index][0] <= delayed_time:
                return history[last_index][1].copy()
            last_index -= 1  # check the previous one
        return []