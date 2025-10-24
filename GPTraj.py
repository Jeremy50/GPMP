import numpy as np

def state_transition(t, s):
    # (1), (1) => (D, D)
    d = t - s
    return np.array([
        [1, d, 0.5 * d ** 2],
        [0, 1, d],
        [0, 0, 1]
    ])

def gen_mean_prior(state0):
    # (D, D) @ (D, V) => {f((1)) => (D, V)}
    return lambda t: state_transition(t, 0) @ state0
    
def Q(ta, tb, Qc):
    # (1), (1), (1) => (D, D)
    dt = tb - ta
    return Qc / 120 * dt * np.array([
        [ 6 * dt ** 4, 15 * dt ** 3,  20 * dt ** 2],
        [15 * dt ** 3, 40 * dt ** 2,  60 * dt     ],
        [20 * dt ** 2, 60 * dt,       120         ]
    ])

def Qinv(ta, tb, Qc):
    # (1), (1), (1) => (D, D)
    dt = tb - ta
    assert dt != 0 and Qc != 0
    return 3 / Qc / dt ** 5 * np.array([
        [  240,           -120 * dt,       20 * dt ** 2],
        [ -120 * dt,        64 * dt ** 2, -12 * dt ** 3],
        [   20 * dt ** 2,  -12 * dt ** 3,   3 * dt ** 4]
    ])

def beta(t, ti, Qc):
    # (D, D) @ (D, D) @ (D, D) => (D, D)
    return Q(ti, t, Qc) @ state_transition(ti+1, t).T @ Qinv(ti, ti+1, Qc)

def alpha(t, ti, Qc):
    # (D, D) - (D, D) @ (D, D) => (D, D)
    return state_transition(t, ti) - beta(t, ti, Qc) @ state_transition(ti+1, ti)

def theta(t, ti, support_states, Qc, mean_prior = None):
    # (D, D) 
    if mean_prior == None: mean_prior = gen_mean_prior(support_states[0])
    mean = mean_prior(t)
    prev_effect = alpha(t, ti, Qc) @ (support_states[ti] - mean_prior(ti))
    next_effect = beta(t, ti, Qc) @ (support_states[ti + 1] - mean_prior(ti + 1))
    return mean + prev_effect + next_effect


class GPTraj:
    
    D = 3
    V = 3
    
    class State:
        
        def __init__(self):
            self.D = GPTraj.D
            self.V = GPTraj.D
            self.mean = np.zeros((self.D, self.V))
            self.covar = np.zeros((self.D, self.V, self.D, self.V))
        
        def with_X(self, *x):
            assert len(x) == self.D
            return self \
                .with_xD0(x[0]) \
                .with_xD1(x[1]) \
                .with_xD2(x[2])
        def with_xD0(self, xD0):
            self.mean[0, 0] = xD0
            return self
        def with_xD1(self, xD1):
            self.mean[1, 0] = xD1
            return self
        def with_xD2(self, xD2):
            self.mean[2, 0] = xD2
            return self
        
        def with_Y(self, *y):
            assert len(y) == self.D
            return self \
                .with_yD0(y[0]) \
                .with_yD1(y[1]) \
                .with_yD2(y[2])
        def with_yD0(self, yD0):
            self.mean[0, 1] = yD0
            return self
        def with_yD1(self, yD1):
            self.mean[1, 1] = yD1
            return self
        def with_yD2(self, yD2):
            self.mean[2, 1] = yD2
            return self
        
        def with_R(self, *r):
            assert len(r) == self.D
            return self \
                .with_rD0(r[0]) \
                .with_rD1(r[1]) \
                .with_rD2(r[2])
        def with_rD0(self, rD0):
            self.mean[0, 2] = rD0
            return self
        def with_rD1(self, rD1):
            self.mean[1, 2] = rD1
            return self
        def with_rD2(self, rD2):
            self.mean[2, 2] = rD2
            return self
    
    def __init__(self, Qc=1):
        
        self.Qc = Qc
        self.support_states = []
    
    def withStates(self, *states):
        self.support_states = states
        return self
        
    def at(self, t):
        
        N = len(self.support_states) - 1
        assert N >= 1 and 0 <= t and t <= N
        support_states = np.array([x.mean for x in self.support_states])
        
        ti = int(t // 1)
        if ti == N: return support_states[-1]
        return theta(t, ti, support_states, self.Qc)
        

if __name__ == "__main__":
    
    gptraj = GPTraj().withStates(
        GPTraj.State()
            .with_X(0, 1, 0)
            .with_Y(0, -1, 0)
            .with_R(0, 0, 0),
        GPTraj.State()
            .with_X(5, -1, 0)
            .with_Y(-5, 0, 0)
            .with_R(0, 0, 0),
        GPTraj.State()
            .with_X(0, -1, 0)
            .with_Y(-10, -1, 0)
            .with_R(0, 0, 0)
    )
    
    print()
    t = 0
    dt = 0.05
    while round(t, 3) <= 2:
        print(" ".join([f"{t:<10}"] + [f"{round(x, 5):<10}" for x in gptraj.at(t).flatten()]))
        t = round(t + dt, 2)
    print()