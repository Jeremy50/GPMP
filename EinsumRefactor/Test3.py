# Import Modules
import matplotlib.pyplot as plt
import numpy as np



# Sample Inputs
D = 3
V = 2
Qc = 1

t0 = 2
u0 = np.random.randn(D, V)
u0[0] = [0, 0]
u0[1, 0] = abs(u0[1, 0])
k0 = np.random.randn(D, V, D, V) * 1e-8 * 0

tN = 2.5
uN = np.random.randn(D, V)
uN[0] = [2, 0]
uN[1, 0] = -abs(uN[1, 0])
kN = np.random.randn(D, V, D, V) * 1e-8 * 0



# State Transition
def state_transition(t, s):
    dt = t - s
    return np.array([
        [1, dt, 0.5*dt**2],
        [0,  1, dt       ],
        [0,  0,  1       ]
    ])



#
def gen_mean_func(t0, u0): return lambda t: state_transition(t, t0) @ u0
def gen_covar_func(t0, k0, V, Qc):
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
        return np.einsum("aX,XbYd,cY->abcd", state_transition(a, t0), k0, state_transition(b, t0)) + np.einsum("ac,bd->abcd", Qc * (F(ub) - F(t0)), np.eye(V))
    return covar_func



#
mean_func = gen_mean_func(t0, u0)
covar_func = gen_covar_func(t0, k0, V, Qc)



#
Nmax = 50
for N in range(1, Nmax + 1):
    
    print(f"N: {N}")
    print(f"Nss: {N - 1}")
    
    mean = []
    covar = []
    for i in range(N + 1):
        print(f"  {t0 + i * (tN - t0) / N}")
        mean.append(mean_func(t0 + i * (tN - t0) / N))
        covar.append([covar_func(t0 + i * (tN - t0) / N, t0 + j * (tN - t0) / N) for j in range(N+1)])
    mean = np.array(mean)
    covar_row = np.array(covar[-1])
    covar = np.array(covar).transpose(0, 2, 3, 1, 4, 5)
    
    
    inv = np.linalg.pinv((covar_row[-1] + kN).reshape(D*V, D*V)).reshape(D, V, D, V)
    temp = np.einsum("abcXY, XYde->abcde", covar_row, inv)
    mean_prior = mean + np.einsum("abcXY,XY->abc", temp, uN - mean[-1])
    covar_prior = covar - np.einsum("abcXY,dXYef->abcdef", temp, covar_row)
    if N == 1:
        print(f"Expected u0: {u0[0, :2]}")
        print(f"Expected uN: {uN[0, :2]}")
    print(mean_prior[:, 0, :2])
    
    ax = plt.gca()
    ax.cla()
    plt.title("GPMP Trajectory")
    plt.xlabel("Robot X")
    plt.ylabel("Robot Y")
    ax.plot(u0[0, 0], u0[0, 1], marker="p", color="green", linestyle="none", markersize=15, label="Start State")
    ax.plot(uN[0, 0], uN[0, 1], marker="p", color="red", linestyle="none", markersize=15, label="End State")
    ax.plot(mean_prior[1:-1, 0, 0], mean_prior[1:-1, 0, 1], marker="h", color="orange", linestyle="none", markersize=10, label="Support States")
    ax.plot(mean_prior[:, 0, 0], mean_prior[:, 0, 1], color="black", linestyle="-")
    plt.legend()
    plt.show()

    print()