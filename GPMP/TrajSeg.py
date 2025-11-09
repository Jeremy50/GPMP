from State import State
import matplotlib.pyplot as plt
import numpy as np
        
def state_transition(tb, ta):
    dt = tb - ta
    return np.array([
        [1, dt, 0.5*dt**2],
        [0,  1, dt       ],
        [0,  0,  1       ]
    ])
    
def Q(ta, tb, Qc):
    dt = tb - ta
    return Qc / 120 * dt * np.array([
        [ 6 * dt ** 4, 15 * dt ** 3,  20 * dt ** 2],
        [15 * dt ** 3, 40 * dt ** 2,  60 * dt     ],
        [20 * dt ** 2, 60 * dt,       120         ]
    ])

def Qinv(ta, tb, Qc):
    dt = tb - ta
    assert dt != 0 and Qc != 0
    return 3 / Qc / dt ** 5 * np.array([
        [  240,           -120 * dt,       20 * dt ** 2],
        [ -120 * dt,        64 * dt ** 2, -12 * dt ** 3],
        [   20 * dt ** 2,  -12 * dt ** 3,   3 * dt ** 4]
    ])

class TrajSeg:
    
    def __init__(self, sa, sb, ss_dt, ip_dt, Qc):
        
        D = 3
        V = sa.V
        ta, u0, k0 = sa.time, sa.mean, sa.covar
        tb, uN, kN = sb.time, sb.mean, sb.covar
        tN = tb - ta
        nss = int(np.ceil(tN / ss_dt - 1))
        nip = int(np.ceil(ss_dt / ip_dt - 1))
        N = nss + 1
        
        mean_func = lambda t: state_transition(t, 0) @ u0
        def covar_func(ta, tb):
            ub = min(ta, tb)
            v00 = lambda s: ((s - ta) ** 3 * (6 * s ** 2 + (3 * ta - 15 * tb) * s + 10 * tb ** 2 - 5 * ta * tb + ta ** 2)) / 120
            v01 = lambda s: -((s - ta) ** 3 * (3 * s - 4 * tb + ta)) / 24
            v10 = lambda s: -((s - tb) ** 3 * (3 * s + tb - 4 * ta)) / 24
            F = lambda s: np.array([
                [v00(s), v01(s), (s * ta ** 2 - s ** 2 * ta + s ** 3 / 3) / 2],
                [v10(s), s * ta * tb - s ** 2 * (ta + tb) / 2 + s ** 3 / 3, s * ta - s ** 2 / 2],
                [(s * tb ** 2 - s ** 2 * tb + s ** 3 / 3) / 2, s * tb - s ** 2 / 2, s]
            ])
            return np.einsum("aX,XbYd,cY->abcd", state_transition(ta, 0), k0, state_transition(tb, 0)) + np.einsum("ac,bd->abcd", Qc * (F(ub) - F(0)), np.eye(V))
        
        mean = []
        covar = []
        dt = tN / (nss + 1)
        for i in range(N+1):
            mean.append(mean_func(i * dt))
            covar.append([covar_func(i * dt, j * dt) for j in range(N+1)])
        mean = np.array(mean)
        covar_row = np.array(covar[-1])
        covar = np.array(covar).transpose(0, 2, 3, 1, 4, 5)

        inv = np.linalg.pinv((covar_row[-1] + kN).reshape(D*V, D*V)).reshape(D, V, D, V)
        temp = np.einsum("abcXY, XYde->abcde", covar_row, inv)
        self.mean = mean + np.einsum("abcXY,XY->abc", temp, uN - mean[-1])
        self.covar = covar - np.einsum("abcXY,dXYef->abcdef", temp, covar_row)
        
        state2state = state_transition(dt, 0)
        qinv_static = Qinv(0, dt, Qc)
        def traj(mean_prior, t):
            assert 0 <= t and t <= tN
            i = int(t / dt)
            ti = i * dt
            if ti == tN: return mean_prior[i]
            prev2cur = state_transition(t, ti)
            cur2next = state_transition(ti+dt, t)
            next_effect = Q(ti, t, Qc) @ cur2next.T @ qinv_static
            prev_effect = prev2cur - next_effect @ state2state
            prev2cur_delta = mean_prior[i] - mean_func(ti)
            cur2next_delta = mean_prior[i+1] - mean_func(ti+dt)
            return mean_func(t) + prev_effect @ prev2cur_delta + next_effect @ cur2next_delta
        self.traj = traj
    
    def sample(self, t):
        return self.traj(self.mean_prior, t)
    
    def interpolate(self, dt=0.02):
        pass
    
    def plot(self, dt=0.02):
        pass
    

s0 = State(0.5, np.random.randn(3, 5).astype(np.float16))
sN = State(3.5, np.random.randn(3, 5).astype(np.float16))
seg = TrajSeg(s0, sN, 0.05, 0.02, 1)