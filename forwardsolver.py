import numpy as np
import matplotlib.pyplot as plt

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
                N_x: int = 10,
                N_y: int = 10,
                N_s: int = 5,
                delta_x: float = 630,
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
        I_k = np.eye(self.N)
        D_k = np.diag(np.full(self.N, -2)) + np.diag(np.ones(self.N-1),1) + np.diag(np.ones(self.N-1),-1)
        L = (1/self.delta_x**2)*(np.kron(D_k, I_k) + np.kron(I_k, D_k)) 
        C = np.diag(self.wavelength) 
        A = - C @ L @ C

        u = self.tau*np.ones((3, self.N_x * self.N_y)) # Stores past, current and future instances
        print(np.shape(A))
        return u, A      

    def forward_solver(self):

        # Discretize time
        T = self.N_t * self.delta_t
        time = np.linspace(0, T, num=self.N_t)
        u, A = self.init_simulation()

        for i in range(1,len(time)-1):
            u[2] = u[1]
            u[1] = u[0]

            u[0] = (2 - self.delta_t**2 * A) @ u[1] + u[2]
            print(u[0])

def main():
    
    solver = ForwardSolver()
    solver.forward_solver()

if __name__ == "__main__":
    main()
