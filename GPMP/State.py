import numpy as np

class State:
    
    def __init__(self, time, mean, covar=None):
        
        if isinstance(time, int): time = float(time)
        assert isinstance(time, float) and time >= 0
        assert isinstance(mean, np.ndarray) and len(mean.shape) == 2
        self.D, self.V = mean.shape
        if covar is None: covar = np.zeros((self.D, self.V, self.D, self.V))
        assert isinstance(covar, np.ndarray)
        
        self.time = time
        self.mean = mean
        self.covar = covar