import numpy as np

N = 5
V = 2
D = 3
Nip = 50

KinvMat = np.random.randn(N+1, N+1, D, V, D, V).transpose([0, 2, 3, 1, 4, 5]).reshape((N+1)*D*V, (N+1)*D*V)
diff = np.random.randn((N+1), D, V).reshape((N+1)*D*V, 1)
Mt = np.random.randn(N+1, N*(Nip+1))
Gup = np.random.randn(N*(Nip+1), D*V)

print((KinvMat @ diff).reshape(N+1, D, V) + (Mt @ Gup).reshape(N+1, D, V))