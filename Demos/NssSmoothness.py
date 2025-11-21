# Import Modules
import matplotlib.pyplot as plt
import numpy as np
import cv2



# Sample Inputs
D = 3
V = 2
Qc = 100
min_ss_dt = 0.02
min_ip_dt = 0.01
verbose = False
show_all_states = True
block = False
time_scale_interpolate = 0.5
time_scale_sample = 0.5
corner = 0.1
win_size = [512, 512]
v_scale = 5
seed = None

ta = 2.382
ua = np.random.randn(D, V)
ua[0] = [-2.5, 0]
ua[1] = abs(ua[1]) / 10
ua[2] = [0, 0]
ka = np.random.randn(D, V, D, V) * 1e-5

tb = 8.382
ub = np.random.randn(D, V)
ub[0] = [2.5, 0]
ub[1] = -abs(ub[1]) / 10
ub[2] = [0, 0]
kb = np.random.randn(D, V, D, V) * 1e-5



# Generate Inputs
if seed is not None: np.random.seed(seed)

ta = 1.3
ua = np.random.randn(D, V)
ua[0] = [-2.5, 0]
ua[1] = abs(ua[1]) / 10
ua[2] = [0, 0]
ka = np.random.randn(D, V, D, V) * 1e-8 * 0

tb = 2.5
ub = np.random.randn(D, V)
ub[0] = [2.5, 0]
ub[1] = -abs(ub[1]) / 10
ub[2] = [0, 0]
kb = np.random.randn(D, V, D, V) * 1e-8 * 0



# Transform
tN = tb - ta
u0, k0 = ua, ka
uN, kN = ub, kb



# State Transition
def state_transition(tb, ta):
    dt = tb - ta
    return np.array([
        [1, dt, 0.5*dt**2],
        [0,  1, dt       ],
        [0,  0,  1       ]
    ])



# Mean and Covar Func Generators
def gen_mean_func(u0): return lambda t: state_transition(t, 0) @ u0
def gen_covar_func(k0, V, Qc):
    def covar_func(ta, tb):
        ub = min(ta, tb)
        v00 = lambda s: ((s - ta) ** 3 * (6 * s ** 2 + (3 * ta - 15 * tb) * s + 10 * tb ** 2 - 5 * ta * tb + ta ** 2)) / 120
        v01 = lambda s: -((s - ta) ** 3 * (3 * s - 4 * tb + ta)) / 24
        v10 = lambda s: -((s - tb) ** 3 * (3 * s + tb - 4 * ta)) / 24
        F = lambda s: np.array([
            [v00(s), v01(s), (s * ta ** 2 - s ** 2 * ta + s ** 3 / 3) / 2],
            [v10(s), s * ta * tb - s ** 2 * (ta + tb) / 2 + s ** 3 / 3, s * ta - s ** 2 / 2],
            [(s * tb ** 2 - s ** 2 * tb + s ** 3 / 3) / 2, s * tb - s ** 2 / 2, s]
        ])
        return np.einsum("aX,XbYd,cY->abcd", state_transition(ta, 0), k0, state_transition(tb, 0)) + np.einsum("ac,bd->abcd", Qc * (F(ub) - F(0)), np.eye(V))
    return covar_func



# Mean and Covar Funcs
mean_func = gen_mean_func(u0)
covar_func = gen_covar_func(k0, V, Qc)



# Q
def Q(ta, tb, Qc):
    dt = tb - ta
    return Qc * dt * np.array([
        [ dt ** 4 / 20, dt ** 3 / 8, dt ** 2 / 6],
        [ dt ** 3 / 8 , dt ** 2 / 3, dt      / 2],
        [ dt ** 2 / 6 , dt      / 2,           1]
    ])



# Qinv
def Qinv(ta, tb, Qc):
    dt = tb - ta
    assert dt != 0 and Qc != 0
    return 3 / Qc / dt ** 5 * np.array([
        [  240,           -120 * dt,       20 * dt ** 2],
        [ -120 * dt,        64 * dt ** 2, -12 * dt ** 3],
        [   20 * dt ** 2,  -12 * dt ** 3,   3 * dt ** 4]
    ])



# Trajectory Generator
def gen_traj(state_means, ss_dt, Qc, precision=1e-5):
    state2state = state_transition(ss_dt, 0)
    qinv_static = Qinv(0, ss_dt, Qc)
    def theta_func(t, verbose=False):
        i = int(t / ss_dt)
        ti = i * ss_dt
        if (abs(t - ti) < precision): return state_means[i]
        if verbose: print(ti, t, ti+ss_dt)
        prev2cur = state_transition(t, ti)
        cur2next = state_transition(ti+ss_dt, t)
        next_effect = Q(ti, t, Qc) @ cur2next.T @ qinv_static
        prev_effect = prev2cur - next_effect @ state2state
        prev2cur_delta = state_means[i] - mean_func(ti)
        cur2next_delta = state_means[i+1] - mean_func(ti+ss_dt)
        return mean_func(t) + prev_effect @ prev2cur_delta + next_effect @ cur2next_delta
    return theta_func



# Main Loop
Nmax = int(np.ceil(tN / min_ss_dt))
nss = Nmax - 1

for N in range(1, Nmax + 1):
    
    ss_dt = tN / N
    
    if verbose:
        print(f"N: {N}")
        print(f"Nss: {N - 1}")
    
    mean = []
    covar = []
    for i in range(N + 1):
        if verbose: print(f"  {i * ss_dt}")
        mean.append(mean_func(i * ss_dt))
        covar.append([covar_func(i * ss_dt, j * ss_dt) for j in range(N+1)])
    mean = np.array(mean)
    covar_row = np.array(covar[-1])
    covar = np.array(covar).transpose(0, 2, 3, 1, 4, 5)
    
    inv = np.linalg.inv((covar_row[-1] + kN).reshape(D*V, D*V)).reshape(D, V, D, V)
    temp = np.einsum("abcXY, XYde->abcde", covar_row, inv)
    mean_prior = mean + np.einsum("abcXY,XY->abc", temp, uN - mean[-1])
    covar_prior = covar - np.einsum("abcXY,dXYef->abcdef", temp, covar_row)
    
    if verbose:
        if N == 1:
            print(f"Expected u0: {u0[0, :2]}")
            print(f"Expected uN: {uN[0, :2]}")
        print(mean_prior[:, 0, :2])
        print()
    
    ax = plt.gca()
    ax.cla()
    plt.title("GPMP Trajectory")
    plt.xlabel("Robot X")
    plt.ylabel("Robot Y")
    ax.plot(u0[0, 0], u0[0, 1], marker="p", color="green", linestyle="none", markersize=15, label="Start State")
    ax.plot(uN[0, 0], uN[0, 1], marker="p", color="red", linestyle="none", markersize=15, label="End State")
    ax.plot(mean_prior[1:-1, 0, 0], mean_prior[1:-1, 0, 1], marker="h", color="orange", linestyle="none", markersize=10, label="Support States")
    ax.plot(mean_prior[:, 0, 0], mean_prior[:, 0, 1], color="black", linestyle="-")
    if show_all_states or N == Nmax:
        plt.legend()
        plt.show(block=block)



    #
    Nip = int(np.ceil(ss_dt / min_ip_dt))
    ip_dt = ss_dt / Nip
    nip = Nip - 1

    if verbose:
        print(nss)
        print(nip)
        print(ss_dt)
        print(ip_dt)

    X = []
    Y = []
    ip_pts = []
    traj = gen_traj(mean_prior, ss_dt, Qc)
    for i in range(N*Nip+1):
        
        ti = (i // Nip) * ss_dt + (i % Nip) * ip_dt
        if verbose:
            if i % Nip == 0: print(f"{i // Nip}: {ti}")
            else: print(f"  {ti}")
        
        point = traj(ti, verbose)
        ip_pts.append(point)
        X.append(point[0, 0])
        Y.append(point[0, 1])



    # Trajectory Render

    ip_pts = np.array(ip_pts)
    x_min, y_min = np.min(ip_pts[:, 0], 0).tolist()
    x_max, y_max = np.max(ip_pts[:, 0], 0).tolist()
    cx = win_size[1] * corner
    cy = win_size[0] * corner
    rx = (win_size[1] * (1 - corner * 2)) / (x_max - x_min)
    ry = (win_size[0] * (1 - corner * 2)) / (y_max - y_min)
    def rescale(x, y):
        x = rx * (x - x_min) + cx
        y = ry * (y - y_min) + cy
        return round(x), win_size[0] - round(y)



    # Support State View
    ss_pts = []
    ss_v = []
    traj = gen_traj(mean_prior, tN / N, Qc)
    for i in range(int((tb - ta) / ss_dt) + 1):
        
        ti = ta + i * ss_dt
        
        ti = ti - ta
        res = traj(ti, False)
        x, y = res[0, :2].tolist()
        
        pt = rescale(x, y)
        img = np.zeros(win_size+[3]).astype(np.uint8)
        ss_pts.append(pt)
        cv2.circle(img, pt, 3, (255, 255, 255))
        
        ss_v.append(rescale(*(res[0, :2]+res[1, :2]*ss_dt).tolist()))
        
        cv2.imshow("Test", img)
        cv2.waitKey(int(ss_dt*1e+3/time_scale_interpolate))



    # Interpolated Point View
    X = np.copy(u0)
    pts = [traj(i * ip_dt) for i in range(int((tb - ta) / ip_dt) + 1)]
    ip_pts = [rescale(*p[0, :2].tolist()) for p in pts]
    for i in range(int((tb - ta) / ip_dt) + 1):
        
        ti = ta + i * ip_dt
        
        ti = ti - ta
        X = state_transition(ip_dt, 0) @ X
        res = traj(ti, False)
        X[2] = res[2]
        x, y = res[0, :2].tolist()
        vx, vy = res[1, :2].tolist()
        x2 = x + v_scale * vx * ip_dt
        y2 = y + v_scale * vy * ip_dt
        pt2 = rescale(x2, y2)
        
        pt = rescale(x, y)
        img = np.zeros(win_size+[3]).astype(np.uint8)
        for i in range(len(ss_pts)-1): cv2.line(img, ss_pts[i], ss_pts[i+1], (0, 255, 0), 1)
        for i in range(len(ip_pts)-1): cv2.line(img, ip_pts[i], ip_pts[i+1], (0, 0, 255), 1)
        for i in range(len(ss_v)-1): cv2.line(img, ss_pts[i], ss_v[i], (255, 0, 0), 1)
        cv2.circle(img, pt, 3, (255, 255, 255))
        cv2.line(img, pt, pt2, (255, 0, 0), 1)
        
        cv2.imshow("Test", img)
        cv2.waitKey(int(ip_dt*1e+3/time_scale_sample))
    
    print(uN[0, :2], X[0, :2], uN[0, :2] - X[0, :2], np.sqrt(np.sum(np.square(uN[0, :2] - X[0, :2]))))