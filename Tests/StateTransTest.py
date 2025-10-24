import numpy as np

D = 3
V = 2

u0 = np.random.randn(D, V)
K0 = np.random.randn(D, V, D, V)

def gen_state_transition(X):
    
    # Works for D = 3
    shape = X.shape
    assert shape[0] % 3 == 0
    
    def state_transition(t, s):
        dt = t - s
        return np.array([
            [1, dt, 0.5*dt**2],
            [0,  1, dt       ],
            [0,  0,  1       ]
        ])
        
    return state_transition, lambda x: x.reshape(3, -1), lambda x: x.reshape(*shape)

a, b, c = gen_state_transition(u0)
print(c(a(3,0)@b(u0)))
print((c(a(0,0)@b(u0))==u0).all())

a, b, c = gen_state_transition(K0)
print(c(a(3,0)@b(K0)))
print((c(a(0,0)@b(K0))==K0).all())

# This works well, but the requirement to return back to original shape may cause excess operations