import numpy as np
print("\n" * 3)

# Presets for the code (Don't Modify)
n = 2 # States
V = 2 # Variables
D = 3 # Differential

support_states = np.array(
    [
        np.array([
            [0, 0],
            [1, -1],
            [0, 0]
        ]),
        np.array([
            [5, -5],
            [-1, 0],
            [0, 0]
        ])
    ]
, dtype=np.float64).reshape(n, V * D)
print(support_states)
print()

state_transition_matrix = lambda t, s: np.array([
    [1, (t - s), 0.5 * (t - s) ** 2],
    [0, 1, (t - s)],
    [0, 0, 1],
]).reshape(D, D)
A = state_transition_matrix(5, 0)
B = support_states[0].reshape(D, -1)
C = A @ B
print(A.shape, B.shape, C.shape)
print()

mean_prior = lambda t: (state_transition_matrix(t, 0) @ support_states[0].reshape(D, -1)).flatten()
print(mean_prior(0)[:2])
print(mean_prior(0.5)[:2])
print(mean_prior(1)[:2])
print()

Qc = 1.5
Q = lambda a, b: Qc / 120 * (b - a) * np.array([
    [ 6 * (b - a) ** 4, 15 * (b - a) ** 3,  20 * (b - a) ** 2],
    [15 * (b - a) ** 3, 40 * (b - a) ** 2,  60 * (b - a)     ],
    [20 * (b - a) ** 2, 60 * (b - a),       120              ]
])
print(Q(3, 5).shape)
print(Q(3, 5))
print()

Qinv = lambda a, b: None if (Qc == 0 or b == a) else 3 / Qc / (b - a) ** 5 * np.array([
    [  240,                -120 * (b - a),        20 * (b - a) ** 2],
    [ -120 * (b - a),        64 * (b - a) ** 2,  -12 * (b - a) ** 3],
    [   20 * (b - a) ** 2,  -12 * (b - a) ** 3,    3 * (b - a) ** 4]
])
print("Det", np.linalg.det(Q(3, 5)))
print("Det", Qc**3 / 8640 * (5-3)**9)
print("Inv", np.linalg.inv(Q(3, 6)))
print("Inv", Qinv(3, 6))
print("Inv", (abs(np.linalg.inv(Q(3, 6)) - Qinv(3, 6)) < 1e-10).all())
print()

beta = lambda t, ti: Q(ti, t) @ state_transition_matrix(ti+1, t).T @ Qinv(ti, ti+1)
alpha = lambda t, ti: state_transition_matrix(t, ti) - beta(t, ti) @ state_transition_matrix(ti+1, ti)
print(beta(0.5, 0))
print(alpha(0.5, 0))
print()

theta = lambda t, ti: mean_prior(t) + (alpha(t, ti) @ (support_states[ti] - mean_prior(ti)).reshape(D, -1)).flatten() + (beta(t, ti) @ (support_states[ti+1] - mean_prior(ti+1)).reshape(D, -1)).flatten()
print(0.001, theta(0.001, 0).astype(np.float32).reshape(D, -1))
print(0.01, theta(0.01, 0).astype(np.float32)[:2])
print(0.05, theta(0.05, 0).astype(np.float32)[:2])
print(0.1, theta(0.1, 0).astype(np.float32)[:2])
print(0.5, theta(0.5, 0).astype(np.float32)[:2])
print(0.5, theta(0.5, 0).astype(np.float32))
print(0.9, theta(0.9, 0).astype(np.float32)[:2])
print(0.95, theta(0.95, 0).astype(np.float32)[:2])
print(0.99, theta(0.99, 0).astype(np.float32)[:2])
print(0.999, theta(0.999, 0).astype(np.float32).reshape(D, -1))
print(0.999999, theta(0.999999, 0).astype(np.float32).reshape(D, -1))
print("\n" * 3)

N = 2
n = N + 1
K = np.zeros((D, D, n, n))
B = np.zeros((D, D, n+1, n))
for i in range(n): B[:, :, i, i] = np.eye(D)
for i in range(n-1): B[:, :, i+1, i] = -state_transition_matrix(i+1, i)
B[:, :, n, n-1] = np.eye(D)
#print(B.transpose(2, 3, 0, 1))

QinvMat = np.zeros((D, D, n+1, n+1))
QinvMat[:, :, 0, 0] = K[:, :, 0, 0]
for i in range(1, n): QinvMat[:, :, i, i] = Qinv(i-1, i)
QinvMat[:, :, n, n] = K[:, :, n-1, n-1]
#print(QinvMat.transpose(2, 3, 0, 1))

KinvMat = B.transpose([0, 1, 3, 2]) @ QinvMat @ B
#print(QinvMat.transpose(2, 3, 0, 1))

lr = 1e-3
print(KinvMat.shape)
print(support_states.reshape(D, V, -1).shape)
print((KinvMat @ support_states.reshape(D, V, -1).transpose([2, 0, 1])).shape)
support_states -= lr * K @ (KinvMat @ (support_states.reshape(D, V, -1)))

# Issues:
#   We need to constantly transpose with this structure
#   No safety for functions, Functions may be impure