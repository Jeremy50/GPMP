import matplotlib.pyplot as plt
import numpy as np

D = 3 # Locked for this implementation

# /=====/ Sampling /=====/

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

def gen_mean_func(u0): return lambda t: state_transition(t, 0) @ u0

def gen_state_func(state_means, mean, state2state, mean_func, qinv_static, Qc, precision=1e-5):
    def state_func(t):
        ti = int(t)
        if (abs(t - ti) < precision): return state_means[ti]
        prev2cur = state_transition(t, ti)
        cur2next = state_transition(ti+1, t)
        next_effect = Q(ti, t, Qc) @ cur2next.T @ qinv_static
        prev_effect = prev2cur - next_effect @ state2state
        prev2cur_delta = state_means[ti] - mean[ti]
        cur2next_delta = state_means[ti+1] - mean[ti+1]
        return mean_func(t) + prev_effect @ prev2cur_delta + next_effect @ cur2next_delta
    return state_func

# /=====/ Interpolating /=====/

def gen_covar_func(covar0, Qc):
    def covar_func(a, b):
        ub = min(a, b)
        v00 = lambda s: ((s - a) ** 3 * (6 * s ** 2 + (3 * a - 15 * b) * s + 10 * b ** 2 - 5 * a * b + a ** 2)) / 120
        v01 = lambda s: -((s - a) ** 3 * (3 * s - 4 * b + a)) / 24
        v10 = lambda s: -((s - b) ** 3 * (3 * s + b - 4 * a)) / 24
        F = lambda s: np.array([
            [v00(s), v01(s), (s * a ** 2 - s ** 2 * a + s ** 3 / 3) / 2],
            [v10(s), s * a * b - s ** 2 * (a + b) / 2 + s ** 3 / 3, s * a - s ** 2 / 2],
            [(s * b ** 2 - s ** 2 * b + s ** 3 / 3) / 2, s * b - s ** 2 / 2, s]
        ])
        return state_transition(a, 0) @ covar0 @ state_transition(b, 0).T + Qc * (F(ub) - F(0))
    return covar_func

def gen_M(N, n_ip, Qc):
    I = np.eye(D)
    M = np.zeros((N*n_ip+N+1, N+1, D, D))
    state2state = state_transition(1, 0)
    qinv_static = Qinv(0, 1, Qc)
    for i in range(N):
        M[i*(n_ip+1), i] = I
        block = np.zeros((n_ip, 2, D, D))
        prev_effects = []
        next_effects = []
        for j in range(1, n_ip + 1):
            ti = i + j / (n_ip + 1)
            prev2cur = state_transition(ti, i)
            cur2next = state_transition(i+1, ti)
            next_effects.append(Q(i, ti, Qc) @ cur2next.T @ qinv_static)
            prev_effects.append(prev2cur - next_effects[-1] @ state2state)
        block[:, 0] = prev_effects
        block[:, 1] = next_effects
        M[i*(n_ip+1)+1:(i+1)*(n_ip+1), i:i+2] = block
    M[-1, -1] = I
    return M

def plot_traj(traj, a=0, b=3, dt=0.02, precision=1e-5, show=True):
    
    stateX = []
    stateY = []
    interX = []
    interY = []
    
    for i in range(a, int(b/dt)+1):
        
        t = i * dt
        ti = int(t)
        x, y = traj(i*dt)[0][:2]
        interX.append(x)
        interY.append(y)
        
        if (abs(ti - t) < precision):
            stateX.append(x)
            stateY.append(y)

    plt.title("GPMP Trajectory")
    plt.xlabel("Robot X")
    plt.ylabel("Robot Y")
    plt.plot(stateX, stateY, marker="*", color="green", linestyle="None", markersize=15, label="Support States")
    plt.plot(interX, interY, marker="o", color="black", markersize=3, label="Sampled States")
    if show:
        plt.legend()
        plt.show()

def plot_traj_int(traj, sstates, Qc, a=0, b=3, n_ip=None, dt=0.02, precision=1e-5):
    
    N = len(sstates) - 1
    V = sstates.shape[-1]
    if n_ip == None: n_ip = round(1 / dt) - 1
    M = gen_M(N, n_ip, Qc).transpose([0, 2, 1, 3]).reshape((N*n_ip+N+1)*D, (N+1)*D)
    theta_up = (M @ sstates.reshape((N+1)*D, V)).reshape(N*n_ip+N+1, D, V)
    
    plot_traj(traj, a, b, dt, precision, False)
    plt.plot(theta_up[:, 0, 0], theta_up[:, 0, 1], marker="*", color="red", linestyle="None", markersize=5, label="Interpolated States")
    plt.legend()
    plt.show()