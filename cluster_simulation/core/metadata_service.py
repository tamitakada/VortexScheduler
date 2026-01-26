from collections import defaultdict
from core.model import *


# class MetadataService:
#     def __init__(self):
#         """
#         This class is responsible for generation and management of Model location on Workers
#         :param num_machines: number of GPUs available
#         :param num_models:   number of models that exist
#         """
#         # dictionary of {job_type_id1:[model1, model2 ...], job_type_id2:[]}
#         self.job_type_models = parse_models_from_workflows()
#         # keep track of the models' locations at time:  {model1: [ (time1,[w_id1,w_id2,]), (time2,[w_id1,...]),...]
#         self.models_cached_locations = defaultdict(lambda: [])
#         # track of processed intermediate results locations at time : {worker1: [ (time1,[interm1,interm2,]), (time2,[internm1,...]),...]
#         self.interm_locations = defaultdict(lambda: [])



#     def add_model_cached_location(self, model, worker_id, current_time):
#         t_locations = self.models_cached_locations[model]
#         last_index = len(t_locations) - 1
#         # 0. base case
#         if last_index == -1:
#             self.models_cached_locations[model].append(
#                 (current_time, [worker_id]))
#             return
#         # 1. Find the time_stamp place to add this location information
#         while last_index >= 0:
#             if t_locations[last_index][0] == current_time:
#                 cached_workers = t_locations[last_index][1].copy()
#                 if worker_id not in cached_workers:
#                     cached_workers.append(worker_id)
#                     self.models_cached_locations[model][last_index] = (
#                         current_time, cached_workers)
#                 break
#             if t_locations[last_index][0] < current_time:
#                 cached_workers = t_locations[last_index][1].copy()
#                 if worker_id not in cached_workers:
#                     last_index += 1
#                     cached_workers.append(worker_id)
#                     self.models_cached_locations[model].insert(
#                         last_index, (current_time, cached_workers)
#                     )
#                 break
#             # check the previous entry
#             last_index -= 1

#         # 2. added the worker_id to all the subsequent timestamp tuples
#         while last_index < len(t_locations):
#             cur_timestamp = self.models_cached_locations[model][last_index][0]
#             cached_workers = t_locations[last_index][1].copy()
#             if worker_id not in cached_workers:
#                 cached_workers.append(worker_id)
#                 self.models_cached_locations[model][last_index] = (
#                     cur_timestamp, cached_workers)
#             last_index += 1


#     def rm_model_cached_location(self, model, worker_id, current_time):
#         # May be removed in the future: No need to store the location on where models are cached: we can always query the home node to retreive a model
#         t_locations = self.models_cached_locations[model]
#         last_index = len(t_locations) - 1
#         # 0. base case: shouldn't happen
#         if last_index == -1:
#             AssertionError("rm model cached location to an empty list")
#             return
#         # 1. find the place to add this remove_event to the tuple list
#         while last_index >= 0:
#             if t_locations[last_index][0] == current_time:
#                 time_stamp = self.models_cached_locations[model][last_index][0]
#                 cached_workers = t_locations[last_index][1]
#                 if worker_id in cached_workers:
#                     cached_workers.remove(worker_id)
#                     self.models_cached_locations[model][last_index] = (time_stamp, cached_workers,
#                                                                        )
#                 break
#             if t_locations[last_index][0] < current_time:
#                 cached_workers = t_locations[last_index][1].copy()
#                 if worker_id in cached_workers:
#                     cached_workers.remove(worker_id)
#                     last_index = last_index + 1
#                     self.models_cached_locations[model].insert(
#                         last_index, (current_time, cached_workers)
#                     )
#                 break
#             last_index -= 1
#         # 2. remove the worker from all the subsequent tuple
#         while last_index < len(t_locations):
#             cached_workers = t_locations[last_index][1]
#             if worker_id in cached_workers:
#                 time_stamp = self.models_cached_locations[model][last_index][0]
#                 cached_workers.remove(worker_id)
#                 self.models_cached_locations[model][last_index] = (time_stamp, cached_workers
#                                                                    )
#             last_index += 1


#     def get_model_cached_location(self, model, current_time, info_staleness=0) -> list:
#         """
#         return a list of worker ids
#         """
#         delayed_time = current_time - info_staleness
#         t_locations = self.models_cached_locations[model]
#         last_index = len(t_locations) - 1
#         while last_index >= 0:
#             if t_locations[last_index][0] == delayed_time:
#                 return t_locations[last_index][1]
#             if t_locations[last_index][0] < delayed_time:
#                 return t_locations[last_index][1]
#             last_index -= 1
#         return []


#     def print_models_cached_locations(self):
#         ss = ""
#         for model in self.models_cached_locations.keys():
#             ss += model.to_string()
#             model_rec = self.models_cached_locations[model]
#             ss += "\n"
#             for time_rec in model_rec:
#                 ss += "time " + "{:.1f}".format(time_rec[0]) + " : "
#                 ss += ",".join(str(e) for e in time_rec[1])
#                 ss += "\n"
#             ss += "\n"
#         print(ss)
