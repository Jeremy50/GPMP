# Import Modules
import numpy as np

# Initial Settings
N = 5
D = 3
V = 2
Qc = 20
n_ip = 25

# Sample Inputs
u0 = np.random.randn(D, V)
uN = np.random.randn(D, V)
u0[0] = [-1, -1]
uN[0] = [1, 1]
# u0[1] = u0[1] / 10 + [1, 1]
# uN[1] = uN[1] / 10 + [1, 1]
# u0[2] = 0
# uN[2] = 0

k0 = np.zeros((D, V, D, V))#np.random.randn(D, V, D, V) * 1e-20
kN = np.zeros((D, V, D, V))#np.random.randn(D, V, D, V) * 1e-20
# k0 *= 0
# kN *= 0

# State Transition
def state_transition(t, s):
    dt = t - s
    return np.array([
        [1, dt, 0.5*dt**2],
        [0,  1, dt       ],
        [0,  0,  1       ]
    ])
    
# Mean Func
def gen_mean_func(u0): return lambda t: state_transition(t, 0) @ u0
mean_func = gen_mean_func(u0)
# print(mean_func(0))
# print(mean_func(N))
# print(uN)
# print("\n"*3)

# Covar Func
def gen_covar_func(k0, Qc):
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
        return np.einsum("aX,XbYd,cY->abcd", state_transition(a, 0), k0, state_transition(b, 0)) + np.einsum("ac,bd->abcd", Qc * (F(ub) - F(0)), np.eye(V))
    return covar_func
covar_func = gen_covar_func(k0, Qc)
# print(covar_func(0, 0).reshape(D*V, D*V))
# print(covar_func(N, N).reshape(D*V, D*V))
# print("\n"*3)

# Mean and Covar Row
mean = []
covar_row = []
covar = []
for i in range(N+1):
    mean.append(mean_func(i))
    covar_row.append(covar_func(N, i))
    covar.append([covar_func(i, j) for j in range(N+1)])
mean = np.array(mean)
covar_row = np.array(covar_row)
covar = np.array(covar).transpose(0, 2, 3, 1, 4, 5)
# print(covar_row)
# print("\n"*3)

# Mean and Covar Prior
inv = np.linalg.pinv((covar_row[N] + kN).reshape(D*V, D*V)).reshape(D, V, D, V)
temp = np.einsum("abcXY, XYde->abcde", covar_row, inv)
mean_prior = mean + np.einsum("abcXY,XY->abc", temp, uN - mean[-1])
covar_prior = covar - np.einsum("abcXY,dXYef->abcdef", temp, covar_row)
#print(covar_prior)
# print(u0)
# print(mean_prior)
# print(uN)
# print("\n"*3)

assert (abs(u0 - mean_prior[0]) <= 1e-3).all()
assert (abs(uN - mean_prior[N]) <= 1e-3).all()

# Noise Covariance
def Q(ta, tb, Qc):
    dt = tb - ta
    return Qc / 120 * dt * np.array([
        [ 6 * dt ** 4, 15 * dt ** 3,  20 * dt ** 2],
        [15 * dt ** 3, 40 * dt ** 2,  60 * dt     ],
        [20 * dt ** 2, 60 * dt,       120         ]
    ])

# Precision
def Qinv(ta, tb, Qc):
    dt = tb - ta
    assert dt != 0 and Qc != 0
    return 3 / Qc / dt ** 5 * np.array([
        [  240,           -120 * dt,       20 * dt ** 2],
        [ -120 * dt,        64 * dt ** 2, -12 * dt ** 3],
        [   20 * dt ** 2,  -12 * dt ** 3,   3 * dt ** 4]
    ])

#
def gen_traj(state_means, Qc, precision=1e-5):
    mean_func = lambda t: state_transition(t, 0) @ state_means[0]
    state2state = state_transition(1, 0)
    qinv_static = Qinv(0, 1, Qc)
    def theta_func(t):
        assert t <= len(state_means) - 1
        ti = int(t)
        if (abs(t - ti) < precision): return state_means[ti]
        prev2cur = state_transition(t, ti)
        cur2next = state_transition(ti+1, t)
        next_effect = Q(ti, t, Qc) @ cur2next.T @ qinv_static
        prev_effect = prev2cur - next_effect @ state2state
        prev2cur_delta = state_means[ti] - mean_func(ti)
        cur2next_delta = state_means[ti+1] - mean_func(ti+1)
        return mean_func(t) + prev_effect @ prev2cur_delta + next_effect @ cur2next_delta
    return theta_func

import matplotlib.pyplot as plt
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
# traj = gen_traj(mean_prior, Qc)
# plot_traj(traj, 0, N)

# Inputs
k0inv = np.linalg.pinv(k0.reshape(D*V, D*V)).reshape(D, V, D, V)
kNinv = np.linalg.pinv(kN.reshape(D*V, D*V)).reshape(D, V, D, V)
state2state = state_transition(1, 0)

# Create QinvMat
QinvMat = np.zeros((N+2, N+2, D, V, D, V))
QinvMat[0, 0] = k0inv
QinvBlock = np.broadcast_to(Qinv(0, 1, Qc), (V, V, D, D)).transpose(2, 0, 3, 1)
for i in range(1, N+1): QinvMat[i, i] = QinvBlock
QinvMat[-1, -1] = kNinv

# Create B
I = np.eye(D)
negstate2state = -state2state
B = np.zeros((N+2, N+1, D, D))
for i in range(N):
    B[i, i] = I
    B[i+1, i] = negstate2state
B[-2:, -1] = I

# Create Kinv
path_string = "WabY,WXYcZf,XdZe->abcdef"
path = np.einsum_path(path_string, B, QinvMat, B, optimize="optimal")[0]
Kinv = np.einsum(path_string, B, QinvMat, B, optimize=path)

# Gen M
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
M = gen_M(N, n_ip, Qc)

obs = [(-0.4, -0.4), (0.4, 0.4)]
r = 0.25
e = 0.25
num_obs = len(obs)

obs_x = []
obs_y = []
for x, y in obs:
    obs_x.append(x)
    obs_y.append(y)
obs_x = np.array(obs_x).reshape(num_obs, 1)
obs_y = np.array(obs_y).reshape(num_obs, 1)

# Optimization
s = np.copy(mean_prior)
for i in range(1000):
    
    inter_states = np.einsum("aXbY,XYc->abc", M, s)
        
    # obs_grads = np.random.randn(N*n_ip+N+1, D, V)
    # obs_grads = np.zeros((N*n_ip+N+1, D, V))
    # obs_grads[:, 0, 1] = 1

    xy = np.broadcast_to(inter_states[:, 0], (num_obs, N*n_ip+N+1, V))
    diff_x = obs_x - xy[:, :, 0]
    diff_y = obs_y - xy[:, :, 1]
    dist = (diff_x ** 2 + diff_y ** 2) ** 0.5
    mask = dist <= r + e
    mask_any = np.any(mask, axis=0)
    
    obs_grads = np.random.randn(N*n_ip+N+1, D, V)
    obs_grads[mask_any == False] *= 1e-5
    r_grads = ((r + e) / dist[mask]) ** 2
    obs_grads[mask_any, 0, 0] = r_grads * diff_x[mask]
    obs_grads[mask_any, 0, 1] = r_grads * diff_y[mask]

    if i % 10 == 0:
        
        traj = gen_traj(s, Qc)
        
        stateX = []
        stateY = []
        interX = []
        interY = []
        
        for j in range(0, int(N/0.02)+1):
            
            t = j * 0.02
            ti = int(t)
            x, y = traj(j*0.02)[0][:2]
            interX.append(x)
            interY.append(y)
            
            if (abs(ti - t) < 1e-5):
                stateX.append(x)
                stateY.append(y)

        ax = plt.gca()
        ax.cla()
        # ax.set_xlim(-1.5, 1.5)
        # ax.set_ylim(-1.5, 1.5)
        
        for xy in obs:
            ax.add_patch(plt.Circle(xy, r+e, color="y"))
            ax.add_patch(plt.Circle(xy, r, color="r"))
        
        plt.title("GPMP Trajectory")
        plt.xlabel("Robot X")
        plt.ylabel("Robot Y")
        
        ax.plot(stateX, stateY, marker="*", color="green", linestyle="None", markersize=15, label="Support States")
        ax.plot(interX, interY, marker="o", color="black", markersize=3, label="Sampled States")
        plt.plot(inter_states[:, 0, 0], inter_states[:, 0, 1], marker="*", color="red", linestyle="None", markersize=5, label="Interpolated States")
        plt.plot(inter_states[mask_any][:, 0, 0], inter_states[mask_any][:, 0, 1], marker="X", color="blue", linestyle="None", markersize=10, label="Obstable Affected States")
        
        plt.legend()
        plt.savefig(f"TrajFrames/Frame{i//5}.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    ss_obs_grads = np.einsum("XaYb,XYc->abc", M, obs_grads)
    dist_grads = np.einsum("abcXYZ,XYZ->abc", Kinv, mean_prior - s)
    grads = np.einsum("abcXYZ,XYZ->abc", covar_prior, 0 * dist_grads + ss_obs_grads)
    s -= 1e-3 * grads

import imageio.v2 as imageio
import cv2
import os
image_files = sorted([os.path.join("TrajFrames", f) for f in os.listdir("TrajFrames")][::-1], key=lambda x:int(x.replace(".png", "").split("/Frame")[-1]))
images = [cv2.resize(imageio.imread(i), (512, 512)) for i in image_files]
imageio.mimsave("Render.gif", images, duration=1/120)