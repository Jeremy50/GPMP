from State import State
import numpy as np

class TrajSeg:
    
    def __init__(self, sa, sb, ss_dt, ip_dt, Qc):
        pass
    
    def sample(self, t):
        return self.traj(self.mean_prior, t)
    
    def interpolate(self, dt=0.02):
        pass
    
    def plot(self, dt=0.02):
        pass
    
s0 = State(1.2, np.random.randn(3, 5).astype(np.float16))
sN = State(3.5, np.random.randn(3, 5).astype(np.float16))
seg = TrajSeg(s0, sN, 0.05, 0.02, 1)