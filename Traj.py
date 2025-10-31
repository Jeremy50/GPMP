import Utilities as util
import numpy as np
#import cv2

class Traj:
    
    def __init__(self, u, k0, kN, Qc):
        
        self.N = u.shape[0] - 1
        self.D = util.D
        self.V = u.shape[-1]
        self.X = self.D * self.V
        
        self.u = u # (N, D, V)
        self.k0 = k0.transpose(1, 3, 0, 2) # (D, V, D, V) -> (V, V, D, D)
        self.kN = kN.transpose(1, 3, 0, 2) # (D, V, D, V) -> (V, V, D, D)
        
        self.Qc = Qc
        self.sstates = np.copy(self.u)
        
        self.STATE2STATE = util.state_transition(1, 0)
        self.MEAN_FUNC = util.gen_mean_func(self.u[0])
        self.covar_func = util.gen_covar_func(self.k0, self.Qc)
        
        self.B = np.zeros((self.N+2, self.N+1, self.D, self.D))
        self.B[:self.N+1, :self.N+1] = np.eye((self.N+1)*self.D).reshape(self.N+1, self.D, self.N+1, self.D).transpose(0, 2, 1, 3)
        for i in range(self.N+1): self.B[i+1, i] = -self.STATE2STATE
        self.B[-1, -1] = np.eye(self.D)
        self.B = self.B.transpose(0, 2, 1, 3).reshape((self.N+2) * self.D, (self.N+1) * self.D)
        self.B = np.kron(self.B, np.eye(self.V))
    
        self.MEAN = []
        covar_row = []
        for t in range(self.N+1):
            self.MEAN.append(self.MEAN_FUNC(t))
            covar_row.append(self.covar_func(self.N, t))
        self.MEAN = np.array(self.MEAN)
        covar_row = np.array(covar_row).transpose([0, 1, 3, 2, 4]).reshape(self.N+1, self.X, self.X)
        
        temp = np.linalg.inv(self.covar_func(self.N, self.N) + self.kN).reshape(self.X, self.X)
        temp = covar_row.reshape((self.N+1) * self.X, self.X) @ temp @ covar_row.transpose(1, 0, 2).reshape(self.X, (self.N+1) * self.X)
        temp = temp.reshape(self.N+1, self.D, self.V, self.N+1, self.D, self.V)
        
        self.covar = np.zeros((self.N+1, self.D, self.V, self.N+1, self.D, self.V)) - temp
        for i in range(self.N+1):
            for j in range(self.N+1):
                self.covar[i, :, :, j] += self.covar_func(i, j).transpose(2, 0, 3, 1)
        self.covar = self.covar.reshape((self.N+1) * self.X, (self.N+1) * self.X)
        
        self.qinv = util.Qinv(0, 1, self.Qc)
        qinv2 = np.kron(self.qinv, np.eye(self.V)).reshape(self.D, self.V, self.D, self.V).transpose(1, 3, 0, 2)
        qinvtemp = [np.linalg.inv(self.k0)] + [qinv2 for i in range(self.N+1)] + [np.linalg.inv(self.kN)]
        self.qinvmat = np.zeros((self.N+2, self.N+2, self.V, self.V, self.D, self.D))
        for i in range(self.N+2): self.qinvmat[i, i] = qinvtemp[i]
        self.qinvmat = self.qinvmat.transpose(0, 4, 2, 1, 5, 3).reshape((self.N+2) * self.X, (self.N+2) * self.X)        
        self.Kinv = self.B.T @ self.qinvmat @ self.B
        self.state_func = util.gen_state_func(self.u, self.MEAN, self.STATE2STATE, self.MEAN_FUNC, self.qinv, self.Qc)
        
        self.dt = None
        self.n_ip = None
        self.M = None
        self.Mt = None
    
    def setQc(self, Qc):
        
        self.Qc = Qc
        self.covar_func = util.gen_covar_func(self.k0, self.Qc)
        
        covar_row = np.array([self.covar_func(self.N, t) for t in range(self.N+1)])
        covar_row = covar_row.transpose([0, 1, 3, 2, 4]).reshape(self.N+1, self.X, self.X)
        
        temp = np.linalg.inv(self.covar_func(self.N, self.N) + self.kN).reshape(self.X, self.X)
        temp = covar_row.reshape((self.N+1) * self.X, self.X) @ temp @ covar_row.transpose(1, 0, 2).reshape(self.X, (self.N+1) * self.X)
        temp = temp.reshape(self.N+1, self.D, self.V, self.N+1, self.D, self.V)
        
        self.covar = np.zeros((self.N+1, self.D, self.V, self.N+1, self.D, self.V)) - temp
        for i in range(self.N+1):
            for j in range(self.N+1):
                self.covar[i, :, :, j] += self.covar_func(i, j).transpose(2, 0, 3, 1)
        self.covar = self.covar.reshape((self.N+1) * self.X, (self.N+1) * self.X)

        self.qinv = util.Qinv(0, 1, self.Qc)
        self.qinvmat = np.diag([np.linalg(self.k0)] + [self.qinv for i in range(self.N+1)] + [np.linalg(self.kN)])
        self.state_func = util.gen_state_func(self.u, self.MEAN, self.STATE2STATE, self.MEAN_FUNC, self.qinv, self.Qc)
        
        self.dt = None
        self.n_ip = None
        self.M = None
        self.Mt = None
    
    def at(self, t):
        assert self.N >= 1 and 0 <= t and t <= self.N
        return self.state_func(t)
        
    def optimize(self, grads_func, epochs=1000, lr=1e-7, tradeoff=0.1, dt=0.02):

        if self.dt != dt:
            self.dt = dt
            self.n_ip = round(1.0 / dt) - 1
            self.M = util.gen_M(self.N, self.n_ip, self.Qc).transpose([0, 2, 1, 3]).reshape((self.N * self.n_ip + self.N + 1) * self.D, (self.N+1) * self.D)
            self.Mt = self.M.T
        
        for epoch in range(epochs):
            
            if epoch % 5 == 0: util.plot_traj_int(self.at, self.sstates, self.Qc, 0, self.N)

            istates = (self.M @ self.u.reshape((self.N+1) * self.D, self.V)).reshape(self.N * self.n_ip + self.N + 1, self.D, self.V)
            g_up = grads_func(istates).reshape((self.N * self.n_ip + self.N + 1) * self.D, self.V)
            grads = (self.Mt @ g_up).reshape((self.N+1) * self.X, 1)
            self.sstates -= (lr * self.covar @ (tradeoff*self.Kinv@(self.sstates-self.MEAN).reshape(*grads.shape) + grads)).reshape(*self.sstates.shape)