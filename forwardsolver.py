import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from cholesky import mblockchol


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

    def index(self,j):
        ind_t = np.linspace(1, self.N_s, self.N_s) + self.N_s*j - 1
        ind_list = [int(x) for x in ind_t]
        return ind_list

    def forward_solver(self):

        # Discretize time
        T = self.N_t * self.delta_t
        time = np.linspace(0, T, num=2*self.N_t)
        u, A, D, b = self.init_simulation()

        # counter for data D
        c = 0
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

        return D

    def mass_matrix(self):
        D = self.forward_solver()
        M = np.zeros((self.N_s*self.N_t, self.N_s*self.N_t))

        for i in range(self.N_t-1):
            for j in range(self.N_t-1):
                ind_i = self.index(i)
                ind_j = self.index(j)

                M[ind_i[0]:ind_i[-1]+1,ind_j[0]:ind_j[-1]+1] = 0.5 * (D[abs(i-j)] + D[abs(i+j)])

        R = mblockchol(M, self.N_s, self.N_t)

        print(R)

def main():
    solver = ForwardSolver()
    solver.mass_matrix()
    
    
if __name__ == "__main__":
    main()
