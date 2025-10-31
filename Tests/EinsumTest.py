import numpy as np

D = 3
V = 2

A = np.random.randn(D, D)
B = np.random.randn(D, V, D, V)
C = np.random.randn(D, D)

res1 = np.einsum("ab,bXcY,cd->aXdY", A, B, C)
res2 = (A @ (B.transpose([1, 3, 0, 2])) @ C).transpose([2, 0, 3, 1])

print(res1.shape, res2.shape)
print((abs(res1-res2)<1e-5).all())

X = np.random.randn(3, 5)
Y = np.random.randn(5, 3)
res1 = X @ Y
res2 = np.einsum("ab,bc->ac", X, Y)
print((abs(res1-res2)<1e-5).all())