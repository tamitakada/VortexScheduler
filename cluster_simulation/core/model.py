class Model:
    """
    Instantiated Model. Multiple copies can be owned by one worker.
    """

    def __init__(self, id: str, model_data, created_at: float, active_from: float):        
        self.id = id
        self.data = model_data
        self.created_at = created_at
        self.active_from = active_from # time of fetch end, when model can be used

    def __hash__(self):
        return hash(self.id)