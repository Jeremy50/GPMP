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
        if state_means is None: state_means = np.random.randn(3, self.differential_level, self.independent_vars)
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

I = np.eye(3 * 2).reshape(3, 2, 3, 2)
R = np.abs(np.random.randn(3, 2, 3, 2))
Z = np.zeros((3, 2, 3, 2))
trajBuilder.set_covar0(Z)
trajBuilder.set_covarN(Z)
traj = trajBuilder.buildTraj()

r = lambda x: np.random.randn(*x.shape)
o = lambda x: np.ones_like(x)
z = lambda x: np.zeros_like(x)
def g(x):
    x[:, 0, 0] = -np.min([np.zeros_like(x[:, 0, 0]), x[:, 0, 0]], axis=0)
    x[:, 0, 1] = -np.min([np.zeros_like(x[:, 0, 1]), x[:, 0, 1]], axis=0)
    return x
def repulsive_gradient(x, radius=2.0, strength=1.0):
    """Generate gradients that push points away from (0,0).
    
    Args:
        x: States array of shape (N, D, V) where N is number of points,
           D is differential level (usually 3: pos, vel, acc),
           V is number of variables (2 for 2D space)
        radius: Influence radius of the repulsive field
        strength: Magnitude of the repulsion
    
    Returns:
        Gradient array of same shape as input
    """
    # Only modify position coordinates (first differential level)
    pos = x[:, 0, :2]  # Get positions (N, 2)
    
    # Calculate distance from origin for each point
    dist = np.linalg.norm(pos, axis=1)  # Shape: (N,)
    
    # Points within radius get pushed outward
    mask = dist < radius
    grad = np.zeros_like(x)
    
    if np.any(mask):
        # Unit vectors pointing away from origin
        unit_vec = pos[mask] / (dist[mask, None] + 1e-6)
        
        # Strength increases as points get closer (inverse square law)
        magnitude = strength * (1.0 - dist[mask]/radius)**2
        
        # Apply repulsive force to positions only
        grad[mask, 0, :2] = unit_vec * magnitude[:, None]
    
    return -grad

# Test the gradient function
traj.optimize(repulsive_gradient)