from State import State
from Traj import Traj
import Utils
import numpy as np

class TrajBuilder:
    
    # Trajectory Settings
    D: int = Utils.D
    V: int = None
    num_supp_states: int = 3
    initial_states: list[State] = None
    
    # Optimization Settings
    dt: float = 0.02
    Qc: float = 1
    learning_rate: float = 1e-3
    mean_reversion: float = 1e-1
    
    def __init__(self, V=2):
        self.with_V(V)
        self.initial_states = []
    
    def with_V(self, V:int):
        assert V >= 1
        self.V = V
        return self

    def with_num_supp_states(self, num_supp_states:int):
        assert isinstance(num_supp_states, int)
        assert num_supp_states >= 0
        self.num_supp_states = num_supp_states
        return self

    def with_initial_states(self, *states:list[State]):
        assert states is not None and all(isinstance(x, State) and x.D == self.D and x.V == self.V for x in states)
        self.initial_states = sorted(states, key=lambda x: x.time)
        return self
    
    def with_dt(self, dt:float):
        assert dt > 0
        self.dt = dt
        return self
    
    def with_Qc(self, Qc:float):
        assert Qc > 0
        self.Qc = Qc
        return self
    
    def with_learning_rate(self, learning_rate:float):
        assert learning_rate > 0
        self.learning_rate = learning_rate
        return self
    
    def with_mean_reversion(self, mean_reversion:float):
        assert mean_reversion >= 0
        self.mean_reversion = mean_reversion
        return self

    def build(self):
        pass
    
    def isCompatible(self, traj:Traj):
        assert isinstance(traj, Traj)
        if self.D != traj.D: return False
        if self.V != traj.V: return False
        if self.num_supp_states != traj.num_supp_states: return False
        return True
    
    def makeCompatible(self, traj:Traj):
        self.V = traj.V
        self.num_supp_states = traj.num_supp_states
    
# Refactor Seperately
# def optimize(self, traj:Traj, epochs=1000, render=False):
#     pass

if __name__ == "__main__":
    
    D = Utils.D
    V = 2
    
    tBuilder = TrajBuilder(V).with_initial_states(
        State(0, np.zeros((D, V))),
        State(1, np.random.randn(D, V).astype(np.float16)),
        State(2, np.random.randn(D, V).astype(np.float16))
    )
    
    tBuilder.with_num_supp_states(100)
    traj1 = tBuilder.with_Qc(0.5).build()
    # traj2 = tBuilder.with_Qc(1).build()
    # traj3 = tBuilder.with_Qc(5).build()