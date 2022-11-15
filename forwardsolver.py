import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import math
from cholesky import mblockchol
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
                N_x: int = 160,
                N_y: int = 160,
                N_s: int = 50,
                delta_x: float = 0.0063,
                tau: float = 3.0303*10**(-5),
                N_t: int = 70,
                Bsrc_file: str = "Bsrc_T_small.txt"):

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
        b = np.genfromtxt(self.Bsrc_file, delimiter=',')

        # filename = 'Bsrc_T'
        # with open(filename, 'rb') as f:
        #     b = pickle.load(f)

        # b = pickle.load(open('Bsrc_T', 'rb'))

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

        return u, A      

    def forward_solver(self):

        # Discretize time
        T = self.N_t * self.delta_t
        time = np.linspace(0, T, num=self.N_t)
        u, A = self.init_simulation()

        for i in range(1,len(time)-1):
            u[2] = u[1] 
            u[1] = u[0]
            u[0] = (-self.delta_t**2 * A) @ u[1] - u[2] + 2*u[1]

            # For displaying images
            # if i % 5 == 0:
            #     plt.gray()
            #     plt.title('FD solution at t = %f' %time[i])
            #     plt.imshow(u[0,:,0].reshape(self.N,self.N))
            #     plt.show()
            
        return u

    def imaging_function(self):    
        # Get the values of u

        # Which of the u's should one use? Thinking of when creating "U" from all "u(tau)"
        # u, A = self.init_simulation()
        u = self.forward_solver()

        # Create matrices which are needed: I, U, M (via U), R (via M), V (via R and U)
        I = np.zeros((self.N_y, self.N_x))
        M = np.zeros((self.N_s * self.N_t, self.N_s*self.N_t))

        for k in range(0, 3):
            for j in range(0, self.N_t):
                for i in range(0, self.N_t):
                    # Create the desired indexes to use for insertion
                    i_ind = [int(x) for x in np.linspace(0, self.N_s, self.N_s) + i*self.N_s]
                    j_ind = [int(y) for y in np.linspace(0, self.N_s, self.N_s) + j*self.N_s]
                    
                    # Compute middle solution of equation (2.2) 
                    # M[j+k*self.N_s][i+k*self.N_s] = u[k][i].T @ u[k][j]
                    M[j][i] = u[k][i].T @ u[k][j]
                    print(u[k][i].T @ u[k][j])
                    
        #print(M)

        print(f"Dimensions of M  = {np.shape(M)}\n")
        print(M)

        print("Working with U_0")
        print(np.shape(u))
        for k in range(0, 3):
            U_0 = u[k, :, :].reshape((self.N_x*self.N_y, self.N_t*self.N_s))

        M_0 = U_0.T @ U_0

        R_0 = mblockchol(M_0, self.N_s, self.N_t)

        # Should be BIG "U" here and not "u"
        V_0 = np.multiply(U_0, np.invert(R_0))
        print("Finished calculating V_0")


        # Problem right now = M is a singular matrix and we cannot do Cholesky Factorization
        R = mblockchol(M, self.N_s, self.N_t)

        # Need to implement V_0
        for i in range(0, self.N**2):
            I[i] = np.linalg.norm(np.multiply(V_0[i], R))**2

        return I, V_0, R_0

def main():
    
    solver = ForwardSolver()
    # solver.forward_solver()
    solver.imaging_function()


if __name__ == "__main__":
    main()
