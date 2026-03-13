from core.simulation import Simulation

class Verifier:

    def __init__(self, simulation: Simulation):
        self.simulation = simulation

    def run_verifier(self):
        raise NotImplementedError()
    
    def debug_assert(self, predicate: bool, msg: str):
        if not predicate:
            print(f"[ASSERTION FAILED] {msg}")
        
        assert(predicate)