# Import Modules
import numpy as np



# Sample Inputs

D = 3
V = 2
Qc = 20
min_nss_dt = 0.5
min_nip_dt = 0.02

ta = 1.3
ua = np.random.randn(D, V)
ua[0] = [0, 0]
ka = np.random.randn(D, V, D, V) * 1e-8

tb = 2.5
ub = np.random.randn(D, V)
ub[0] = [2, 0]
kb = np.random.randn(D, V, D, V) * 1e-8



#
assert ta < tb
u0, k0 = ua, ka
uN, kN = ub, kb
tN = tb - ta

nss = 0 if tN < min_nss_dt else int(np.ceil(tN / min_nss_dt)) - 1
nip = 0 if tN < min_nip_dt else int(np.ceil(min_nss_dt / min_nip_dt)) - 1
nss_dt = tN / (nss + 1)
nip_dt = nss_dt / (nip + 1)
N = nss + 1

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



# Mean and Covar Row
mean = []
covar = []
for i in range(N+1):
    mean.append(mean_func(i*nss_dt))
    covar.append([covar_func(i*nss_dt, j*nss_dt) for j in range(N+1)])
mean = np.array(mean)
covar_row = np.array(covar[-1])
covar = np.array(covar).transpose(0, 2, 3, 1, 4, 5)



# Mean and Covar Prior
inv = np.linalg.pinv((covar_row[-1] + kN).reshape(D*V, D*V)).reshape(D, V, D, V)
temp = np.einsum("abcXY, XYde->abcde", covar_row, inv)
mean_prior = mean + np.einsum("abcXY,XY->abc", temp, uN - mean[-1])
covar_prior = covar - np.einsum("abcXY,dXYef->abcdef", temp, covar_row)



#
assert (abs(u0 - mean_prior[0]) <= 1e-3).all()
assert (abs(uN - mean_prior[N]) <= 1e-3).all()


# Noise Covariance
def Q(ta, tb, Qc):
    dt = tb - ta
    return Qc * dt * np.array([
        [ dt ** 4 / 20, dt ** 3 / 8, dt ** 2 / 6],
        [ dt ** 3 / 8 , dt ** 2 / 3, dt      / 2],
        [ dt ** 2 / 6 , dt      / 2,           1]
    ])



#
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
    state2state = state_transition(nss_dt, 0)
    qinv_static = Qinv(0, nss_dt, Qc)
    def theta_func(t):
        i = int(t / nss_dt)
        ti = i * nss_dt
        if (abs(t - ti) < precision): return state_means[i]
        prev2cur = state_transition(t, ti)
        cur2next = state_transition(ti+nss_dt, t)
        next_effect = Q(ti, t, Qc) @ cur2next.T @ qinv_static
        prev_effect = prev2cur - next_effect @ state2state
        prev2cur_delta = state_means[i] - mean_func(ti)
        cur2next_delta = state_means[i+1] - mean_func(ti+nss_dt)
        return mean_func(t) + prev_effect @ prev2cur_delta + next_effect @ cur2next_delta
    return theta_func












#
import matplotlib.pyplot as plt
def plot_traj(traj, means, a=0, b=3, dt=0.02):
    
    X = []
    Y = []
    
    for i in range(a, int(b/dt)+1):
        
        t = i * dt
        x, y = traj(t)[0][:2]
        X.append(x)
        Y.append(y)
    ax = plt.gca()
    ax.cla()
    plt.title("GPMP Trajectory")
    plt.xlabel("Robot X")
    plt.ylabel("Robot Y")
    ax.plot(means[0, 0, 0], means[0, 0, 1], marker="p", color="green", linestyle="none", markersize=15, label="Start State")
    ax.plot(means[1:-1, 0, 0], means[1:-1, 0, 1], marker="H", color="orange", linestyle="none", markersize=10, label="Supp States")
    ax.plot(means[-1, 0, 0], means[-1, 0, 1], marker="p", color="red", linestyle="none", markersize=15, label="End State")
    ax.plot(X, Y, marker=".", color="black", markersize=3, label="Sampled States")
    plt.legend()
    plt.savefig("Sampled.png", dpi=512)
    plt.show()
traj = gen_traj(mean_prior, Qc)
plot_traj(traj, mean_prior, 0, tN, 0.005)








































# Inputs
k0inv = np.linalg.pinv(k0.reshape(D*V, D*V)).reshape(D, V, D, V)
kNinv = np.linalg.pinv(kN.reshape(D*V, D*V)).reshape(D, V, D, V)
state2state = state_transition(nss_dt, 0)



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
def gen_M(N, nip, Qc):
    I = np.eye(D)
    M = np.zeros((N*nip+N+1, N+1, D, D))
    state2state = state_transition(nss_dt, 0)
    qinv_static = Qinv(0, nss_dt, Qc)
    for i in range(N):
        M[i*(nip+1), i] = I
        block = np.zeros((nip, 2, D, D))
        prev_effects = []
        next_effects = []
        ti = i * nss_dt
        for j in range(1, nip + 1):
            t = ti + j * nss_dt / (nip + 1)
            prev2cur = state_transition(t, ti)
            cur2next = state_transition(ti+nss_dt, t)
            next_effects.append(Q(ti, t, Qc) @ cur2next.T @ qinv_static)
            prev_effects.append(prev2cur - next_effects[-1] @ state2state)
        block[:, 0] = prev_effects
        block[:, 1] = next_effects
        M[i*(nip+1)+1:(i+1)*(nip+1), i:i+2] = block
    M[-1, -1] = I
    return M
M = gen_M(N, nip, Qc)



#
print()
print(f"Total Duration: {tN}")
print(f"Nss: {nss} & Nip: {nip}")
print(f"Start Position: {u0[0, :2].round(1).tolist()} - Speed: {np.sqrt(np.sum(u0[1, :2]**2)).tolist()}")
print(f"End Position: {uN[0, :2].round(1).tolist()} - Speed: {np.sqrt(np.sum(uN[1, :2]**2)).tolist()}")
print()

inter_states = np.einsum("aXbY,XYc->abc", M, mean_prior)
ax = plt.gca()
ax.cla()
plt.title("GPMP Trajectory")
plt.xlabel("Robot X")
plt.ylabel("Robot Y")
ax.plot(u0[0, 0], u0[0, 1], marker="p", color="green", linestyle="none", markersize=15, label="Start State")
ax.plot(uN[0, 0], uN[0, 1], marker="p", color="red", linestyle="none", markersize=15, label="End State")
ax.plot(mean_prior[1:-1, 0, 0], mean_prior[1:-1, 0, 1], marker="h", color="orange", linestyle="none", markersize=10, label="Support States")
ax.plot(inter_states[:, 0, 0], inter_states[:, 0, 1], color="black", linestyle="-")
ax.plot(inter_states[:, 0, 0], inter_states[:, 0, 1], marker=".", color="grey", linestyle="none", markersize=5, label="Interpolated States")
plt.legend()
plt.savefig("Interpolated.png", dpi=512)
plt.show()








    


ax = plt.gca()
ax.cla()
plt.title("GPMP Trajectory")
plt.xlabel("Robot X")
plt.ylabel("Robot Y")
ax.plot(u0[0, 0], u0[0, 1], marker="p", color="green", linestyle="none", markersize=15, label="Start State")
ax.plot(uN[0, 0], uN[0, 1], marker="p", color="red", linestyle="none", markersize=15, label="End State")

dts = [0.05, 0.02, 0.01, 0.005]
clrs = ["blue", "lime", "magenta", "black"]
for trial_i in range(len(dts)):
    X = []
    Y = []
    dt = dts[trial_i]
    x = np.copy(u0)
    X.append(x[0, 0])
    Y.append(x[0, 1])
    for i in range(int(tN/dt)+1):
        t = i * dt
        x[2] = traj(t)[2]
        x = state_transition(dt, 0) @ x
        X.append(x[0, 0])
        Y.append(x[0, 1])
    ldt = tN - int(tN/dt)*dt
    x[2] = traj(tN)[2][:2]
    x = state_transition(ldt, 0) @ x
    X.append(x[0, 0])
    Y.append(x[0, 1])
    ax.plot(X, Y, marker=".", color=clrs[trial_i], linestyle="--", markersize=5, label=f"States[dt={dts[trial_i]}]")
plt.legend()
plt.savefig("Simulated.png", dpi=512)
plt.show()

exit()
















#
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

    xy = np.broadcast_to(inter_states[:, 0], (num_obs, N*nip+N+1, V))
    diff_x = obs_x - xy[:, :, 0]
    diff_y = obs_y - xy[:, :, 1]
    dist = (diff_x ** 2 + diff_y ** 2) ** 0.5
    mask = dist <= r + e
    mask_any = np.any(mask, axis=0)
    
    obs_grads = np.random.randn(N*nip+N+1, D, V)
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