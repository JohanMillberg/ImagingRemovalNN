import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.signal import convolve2d

"""
Parameters for real implementation
N_s = 50
N_x = N_y = 512
c_0 = 1000 m/s
offsets o_x = 25, o_y = 81
Nim_x = 175
Nim_y = 350
spatial grid step delta_x = 0.0063

Timestepping params 
tau = 3.0303*10^-5
delta_t = tau/20
N_t = 70
"""

class ForwardSolver:

    def __init__(self, 
                N_x: int = 512,
                N_y: int = 512,
                N_s: int = 5,
                delta_x: float = 0.0063,
                tau: float = 3.0303*10**(-5),
                N_t: int = 70):

        self.N = N_x
        self.N_s = N_s
        self.N_x = N_x
        self.N_y = N_y
        self.delta_x = delta_x

        self.tau = tau
        self.N_t = N_t
        self.delta_t = tau/20

        self.wavelength = np.ones(self.N**2)

    def init_simulation(self):
        # Calculate operators
        I_k = sparse.identity(self.N)
        D_k = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(self.N, self.N))
        L = (1/self.delta_x**2)*(sparse.kron(D_k, I_k) + sparse.kron(I_k, D_k)) 
        C = 1000*sparse.diags(self.wavelength, 0) 

        A = - C @ L @ C

        u = np.zeros((3, self.N_x * self.N_y, self.N_s)) # Stores past, current and future instances
        u[:, int(self.N_x * self.N_y / 2), :] = 1
        
        #Below is for testing purposes
        u = u.reshape(3, self.N_x, self.N_y, self.N_s) 
        blur_kernel = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]])
        for i in range(3):
            for j in range(self.N_s):
                u[i, :, :, j] = convolve2d(u[i, :, :, j],
                                           blur_kernel,
                                           'same',
                                           'symm') 

        u = u.reshape(3, self.N_x*self.N_y, self.N_s) 

        return u, A      

    def forward_solver(self):

        # Discretize time
        T = self.N_t * self.delta_t
        time = np.linspace(0, T, num=self.N_t)
        u, A = self.init_simulation()

        for i in range(1,len(time)-1):
            u[0] = (-self.delta_t**2 * A) @ u[1] - u[2] + 2*u[1]
            u[2] = u[1] 
            u[1] = u[0]
            if i % 100 == 0:
                plt.gray()
                plt.imshow(u[0, :, 0].reshape(self.N, self.N)) 
                plt.show()

def main():
    
    solver = ForwardSolver(N_t=1000)
    solver.forward_solver()

if __name__ == "__main__":
    main()
