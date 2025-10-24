import numpy as np

N = 5
D = 3
V = 2

QinvMat = np.random.randn(N+2, N+2, D, D)
B = np.random.randn(N+2, N+1, D, D)
Bt = B.transpose([1, 0, 2, 3])

a = Bt.transpose([2, 3, 0, 1])
b = QinvMat.transpose([2, 3, 0, 1])
c = B.transpose([2, 3, 0, 1])

print(a.shape)
print(b.shape)
print(c.shape)

KinvMat = a @ b @ c
print(KinvMat.shape)

KinvMat = KinvMat.transpose([2, 3, 0, 1])
print(KinvMat.shape)

I = np.eye(V)
KinvMat = np.kron(KinvMat, I).reshape(N+1, N+1, D, V, D, V)
print(KinvMat.shape)

K = np.random.randn(N+1, N+1, D, V, D, V)
a = K.transpose([2, 3, 4, 5, 0, 1])
b = KinvMat.transpose([2, 3, 4, 5, 0, 1])
delta = (a @ b).transpose([4, 5, 0, 1, 2, 3])
print(delta.shape)