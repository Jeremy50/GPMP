import Utilities as util
from Traj import Traj
import numpy as np

class TrajBuilder:
    
    def __init__(self, independent_vars=3):
        self.differential_level = util.D
        self.independent_vars = independent_vars
        self.state_means = []
        self.Qc = 1
        self.state_covar0 = None
        self.state_covarN = None

    def addStates(self, state_means=None):
        if state_means is None: state_means = np.random.randn(5, self.differential_level, self.independent_vars)
        if not isinstance(state_means, np.ndarray): raise AttributeError(f"<state means> must be a numpy array of shape (X, {self.differential_level}, {self.independent_vars})")
        if len(state_means.shape) == 2: state_means = np.expand_dims(state_means, 0)
        if len(state_means.shape) != 3 or state_means.shape[-2:] != (self.differential_level, self.independent_vars): raise AttributeError(f"<state means> must have shape (X, {self.differential_level}, {self.independent_vars})")
        self.state_means.append(state_means)
    
    def set_covar0(self, covar0):
        self.state_covar0 = covar0
        # raise NotImplementedError()
    
    def set_covarN(self, covarN):
        self.state_covarN = covarN
        # raise NotImplementedError()
    
    def set_Qc(self, Qc):
        self.Qc = Qc
        
    def buildTraj(self):
        state_means = np.concatenate(self.state_means)
        N = state_means.shape[0]
        if N < 2: raise AttributeError(f"Traj requires at least 2 states, current num states: {N}")
        if self.state_covar0 is None: raise ValueError("Covar0 must set")
        if self.state_covarN is None: raise ValueError("CovarN must set")
        print(f"Creating trajectory of {N} states...")
        return Traj(state_means, self.state_covar0, self.state_covarN, self.Qc)

trajBuilder = TrajBuilder(2)
trajBuilder.addStates()
trajBuilder.set_covar0(np.random.randn(3, 2, 3, 2))#(np.random.randn(3, 3))
trajBuilder.set_covarN(np.random.randn(3, 2, 3, 2))
traj = trajBuilder.buildTraj()
print(traj.at(2.5))
print(traj.optimize(lambda x: np.random.randn(*x.shape)))