from TrajSeg import TrajSeg
from State import State

class Traj:
    
    D: int = None
    V: int = None
    num_supp_states: int = None
    initial_states: list[State] = None
    segments: list[TrajSeg] = None
    
    def sample(self, t):
        pass
    
    def interpolate(self, dt=0.02):
        pass
    
    def plot(self, dt=0.02):
        pass