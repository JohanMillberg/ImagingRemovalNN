import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import math
import pickle

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
                N_s: int = 50,
                delta_x: float = 0.0063,
                tau: float = 3.0303*10**(-5),
                N_t: int = 70,
                Bsrc_file: str = "Bsrc_T.txt"):

        self.N = N_x
        self.N_s = N_s
        self.N_x = N_x
        self.N_y = N_y
        self.delta_x = delta_x

        self.tau = tau
        self.N_t = N_t
        self.delta_t = tau/20

        self.wavelength = np.ones(self.N**2)
        self.Bsrc_file = Bsrc_file 

    def import_sources(self):
        
        b = np.loadtxt(self.Bsrc_file, delimiter =',')
        np.reshape(b, (self.N_x * self.N_y, self.N_s))

        return b
        
    def init_simulation(self):
        # Calculate operators
        I_k = sparse.identity(self.N)
        D_k = sparse.diags([1,-2,1],[-1,0,1], shape=(self.N,self.N))
        L = (1/self.delta_x**2)*(sparse.kron(D_k, I_k) + sparse.kron(I_k, D_k)) 
        C = 1000*sparse.diags(self.wavelength) 

        A = - C @ L @ C

        u = np.zeros((3, self.N_x * self.N_y, self.N_s)) # Stores past, current and future instances

        b = self.import_sources()
        u[1] = b
        u[0] = (-0.5* self.delta_t**2 * A) @ b + b

        D = np.zeros((2*self.N_t, self.N_s, self.N_s))
        D[0] = np.transpose(b) @ u[1]

        return u, A, D, b 

    def forward_solver(self):

        # Discretize time
        T = self.N_t * self.delta_t
        time = np.linspace(0, T, num=2*self.N_t)
        u, A, D, b = self.init_simulation()

        for i in range(1,len(time)-1):
            u[2] = u[1] 
            u[1] = u[0] 
            u[0] = (-self.delta_t**2 * A) @ u[1] - u[2] + 2*u[1]

            # Update data
            D[i] = np.transpose(b) @ u[1]

            # For displaying images
            """if i % 5 == 0:
                plt.gray()
                plt.title('FD solution at t = %f' %time[i])
                plt.imshow(D[i])
                plt.show()"""

        plt.imshow(D[0,:,:].reshape(self.N_s, self.N_s))
        plt.show()

def main():
    
    solver = ForwardSolver(N_t=1000)
    solver.forward_solver()
    

if __name__ == "__main__":
    main()
