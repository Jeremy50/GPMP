import numpy as np

D = 3 # Locked for this implementation

def state_transition(t, s):
    dt = t - s
    return np.array([
        [1, dt, 0.5*dt**2],
        [0,  1, dt       ],
        [0,  0,  1       ]
    ])

def Q(ta, tb, Qc):
    dt = tb - ta
    return Qc / 120 * dt * np.array([
        [ 6 * dt ** 4, 15 * dt ** 3,  20 * dt ** 2],
        [15 * dt ** 3, 40 * dt ** 2,  60 * dt     ],
        [20 * dt ** 2, 60 * dt,       120         ]
    ])

def Qinv(ta, tb, Qc):
    dt = tb - ta
    assert dt != 0 and Qc != 0
    return 3 / Qc / dt ** 5 * np.array([
        [  240,           -120 * dt,       20 * dt ** 2],
        [ -120 * dt,        64 * dt ** 2, -12 * dt ** 3],
        [   20 * dt ** 2,  -12 * dt ** 3,   3 * dt ** 4]
    ])

def gen_traj(states, Qc):
    prior_mean_func = lambda t: state_transition(t, 0) @ states[0]
    def theta_func(t):
        assert t <= len(states) - 1
        ti = int(t)
        if (abs(t - ti) < 1e-5): return states[ti]
        prev2cur = state_transition(t, ti)
        cur2next = state_transition(ti+1, t)
        next_effect = Q(ti, t, Qc) @ cur2next.T @ Qinv(ti, ti+1, Qc)
        prev_effect = prev2cur - next_effect @ cur2next
        prev2cur_delta = states[ti] - prior_mean_func(ti)
        cur2next_delta = states[ti+1] - prior_mean_func(ti+1)
        return prior_mean_func(t) + prev_effect @ prev2cur_delta + next_effect @ cur2next_delta
    return theta_func

if __name__ == "__main__":
    
    test_mode = True
    
    if test_mode:
    
        N = 2
        V = 2
        u = np.random.randn(N+1, D, V)
        K = np.zeros((N+1, N+1, D, V, D, V))
        #print(u[:, :1, :2].reshape(N+1, 2))
    
    else:
        
        D = 3
        V = 3
        u = np.array([
            [[5, 5, 0], [1, 1, 0], [0.5, 0.5, 0]],
            [[-5, -5, 0], [1, 1, 0], [-0.5, -0.5, 0]]
        ])
        
        N = len(u) - 1
        K = np.zeros((N+1, N+1, D, V, D, V))
    
    Qc = 5
    traj = gen_traj(u, Qc)
    for i in range(0, N*100+1):
        print(i/100, traj(i/100))