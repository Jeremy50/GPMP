import numpy as np

A = np.random.randn(3, 5, 6, 7)
B = np.eye(6*7, 6*7)

print((A == (A.reshape(3*5, 6*7) @ B).reshape(3, 5, 6, 7)).all())