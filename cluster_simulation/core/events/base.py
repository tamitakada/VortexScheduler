class Event(object):
    """ Abstract class representing events. """

    def __init__(self):
        raise NotImplementedError("Event is an abstract class and cannot be "
                                  "instantiated directly")

    def run(self, current_time):
        """ Returns any events that should be added to the queue. """
        raise NotImplementedError("The run() method must be implemented by "
                                  "each class subclassing Event")
    
    def should_abandon_event(self, current_time, kwargs: dict):
        return False

    def to_string(self, current_time):
        """ Returns the string describing the event """
        raise NotImplementedError("The to_string() method must be implemented by "
                                  "each class subclassing Event")
    
    def is_worker_event():
        raise NotImplementedError("is_worker_event : () -> bool not implemented")
    

class EventOrders:
    """
    Used so that the Simulation keeps track of the priority queue order
    """

    def __init__(self, current_time, event):
        self.priority = current_time
        self.current_time = current_time
        self.event = event

    def __lt__(self, other):
        return self.priority < other.priority

    def to_string(self):
        return ""+str(self.current_time) + " " + self.event.to_string()