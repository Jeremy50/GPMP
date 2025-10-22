import numpy as np
print("\n" * 3)

# Presets for the code (Don't Modify)
N = 2 # States
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
            [10, -10],
            [-1, -1],
            [0, 0]
        ])
    ]
).reshape(N, V * D)
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

Qc = 1
Q = lambda a, b: Qc * np.array([
    [(b - a) ** 5 / 20, (b - a) ** 4 / 8, (b - a) ** 3 / 6],
    [(b - a) ** 4 / 8, (b - a) ** 3 / 3, (b - a) ** 2 / 2],
    [(b - a) ** 3 / 6, (b - a) ** 2 / 2, (b - a)]
])
print(Q(3, 5).shape)
print(Q(3, 5))
print()

beta = lambda t, ti: Q(ti, t) @ state_transition_matrix(ti+1, t).T @ np.linalg.inv(Q(ti, ti+1))
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
print(0.9, theta(0.9, 0).astype(np.float32)[:2])
print(0.95, theta(0.95, 0).astype(np.float32)[:2])
print(0.99, theta(0.99, 0).astype(np.float32)[:2])
print(0.999, theta(0.999, 0).astype(np.float32).reshape(D, -1))
print("\n" * 3)