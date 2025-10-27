import numpy as np

N = 5
V = 2
n_ip = 3

M = np.zeros((N*(n_ip+1)+1, N+1))
for i in range(N):
    M[i*(n_ip+1), i] = 1
    M[i*(n_ip+1)+1:(i+1)*(n_ip+1), i:i+2] = 3 * np.ones((n_ip, 2))
M[-1, -1] = 1
print(M)

D = 3
I = np.eye(D)
M = np.zeros((N*(n_ip+1)+1, N+1, D, D))
for i in range(N):
    M[i*(n_ip+1), i] = I
    M[i*(n_ip+1)+1:(i+1)*(n_ip+1), i:i+2] = 3 * np.ones((n_ip, 2, D, D))
M[-1, -1] = I
print(M)