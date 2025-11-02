# Imports
import numpy as np

# Settings
N = 3
D = 2
V = 1

# Inputs
k0inv = np.random.randn(D, V, D, V)
kNinv = np.random.randn(D, V, D, V)
Qinv = np.random.randn(D, D)
state2state = np.random.randn(D, D)

# Create QinvMat
QinvMat = np.zeros((N+2, N+2, D, V, D, V))
QinvMat[0, 0] = k0inv
QinvBlock = np.broadcast_to(Qinv, (V, V, D, D)).transpose(2, 0, 3, 1)
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
for iteration in range(10): Kinv = np.einsum(path_string, B, QinvMat, B, optimize=path)