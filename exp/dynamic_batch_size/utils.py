import bisect

class SortedQueue:
    """
    A queue that maintains requests in ascending order by arrival_time.
    """
    def __init__(self, sort_by="deadline"):
        self.requests = []
        self.sort_by = sort_by

    def append(self, request):
        """
        Insert request at the correct position to maintain ascending order by deadline.
        """
        # Find the insertion point to maintain sorted order
        # Create a list of arrival times for comparison
        if self.sort_by == "deadline":
            deadlines = [req.deadline for req in self.requests]
            insert_pos = bisect.bisect_right(deadlines, request.deadline)
        elif self.sort_by == "arrival_time":
            arrival_times = [req.arrival_time for req in self.requests]
            insert_pos = bisect.bisect_right(arrival_times, request.arrival_time)
        else:
            raise ValueError(f"Invalid sort_by: {self.sort_by}")
        self.requests.insert(insert_pos, request)
    
    # def get_request_by_earliest_arrival_time(self):
    #     """
    #     Get the request with the earliest arrival time.
    #     """
    #     if self.requests:
    #         earliest_arrival_time = float('inf')
    #         earliest_request = None
    #         for req in self.requests:
    #             if req.arrival_time < earliest_arrival_time:
    #                 earliest_arrival_time = req.arrival_time
    #                 earliest_request = req
    #         return earliest_request
    #     return None
    
    def pop(self):
        """
        Remove and return the request with the earliest arrival_time.
        """
        if self.requests:
            return self.requests.pop(0)
        raise IndexError("Queue is empty")
    
    def clear(self):
        """
        Remove all items from the queue.
        """
        self.requests.clear()
    
    def __len__(self):
        return len(self.requests)
    
    def __getitem__(self, index):
        return self.requests[index]

    def get_by_id(self, id):
        """
        Get a request by its id.
        """
        for req in self.requests:
            if req.id == id:
                return req
        return None
        
    def remove(self, request):
        """
        Remove a specific request from the queue.
        """
        self.requests.remove(request)
    
    def copy(self):
        """
        Create a shallow copy of the queue.
        """
        new_queue = SortedQueue()
        new_queue.requests = self.requests.copy()
        return new_queue
    
    def extend(self, items):
        """
        Add multiple items to the queue in sorted order.
        """
        for item in items:
            self.append(item)