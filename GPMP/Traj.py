from State import State
import numpy as np
import Utils

class Traj:
    
    D = Utils.D
    V = None
    num_supp_states = None
    initial_states = None
    
    def __init__(self, V):
        self.V = V
        
    def with_num_supp_states(self, num_supp_states):
        self.num_supp_states = num_supp_states
        return self
    
    def with_initial_states(self, initial_states):
        self.initial_states = initial_states
        return self
    
    def sample(self, t):
        self
    
    def interpolate(self, dt=0.02):
        pass
    
    def plot(self, dt=0.02):
        pass