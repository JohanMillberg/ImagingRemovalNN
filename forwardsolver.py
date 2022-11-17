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
                background_velocity_value: float = 1000,
                Bsrc_file: str = "Bsrc_T.txt",
                N_x_im: int = 175,
                N_y_im: int = 350,
                O_x: int = 25,
                O_y: int = 81):

        self.N = N_x
        self.N_s = N_s
        self.N_x = N_x
        self.N_y = N_y
        self.delta_x = delta_x
        self.N_x_im = N_x_im
        self.N_y_im = N_y_im
        self.O_x = O_x
        self.O_y = O_y

        self.tau = tau
        self.N_t = N_t
        self.delta_t = tau/20
        self.background_velocity_value = background_velocity_value

        self.background_velocity = np.full(self.N**2,
                                           background_velocity_value,
                                           dtype=np.float64)
        self.Bsrc_file = Bsrc_file 

    def import_sources(self):
        
        b = np.loadtxt(self.Bsrc_file, delimiter =',', dtype=np.float64)
        np.reshape(b, (self.N_x * self.N_y, self.N_s))

        return b
        
    def init_simulation(self):
        # Calculate operators
        I_k = sparse.identity(self.N)
        D_k = (1/self.delta_x**2)*sparse.diags([1,-2,1],[-1,0,1], shape=(self.N,self.N), dtype=np.float64)
        D_k = sparse.csr_matrix(D_k)
        D_k[0, 0] = -1 * (1/self.delta_x**2)

        L = sparse.kron(D_k, I_k) + sparse.kron(I_k, D_k)
        C = sparse.diags(self.background_velocity, 0, dtype=np.float64)

        A = (- C @ L @ C)

        u = np.zeros((3, self.N_x * self.N_y, self.N_s), dtype=np.float64) # Stores past, current and future instances

        b = self.import_sources()

        u[1] = b
        u[0] = (-0.5* self.delta_t**2 * A) @ b + b

        D = np.zeros((2*self.N_t, self.N_s, self.N_s), dtype=np.float64)
        D[0] = np.transpose(b) @ u[1]

        return u, A, D, b 

    def index(self,j):
        ind_t = np.linspace(0, self.N_s, self.N_s) + self.N_s*j 
        ind_list = [int(x) for x in ind_t]
        return ind_list

    def forward_solver(self):
        # Discretize time
        T = self.N_t * self.delta_t
        nts = 20
        time = np.linspace(0, T, num=2*self.N_t)
        u, A, D, b = self.init_simulation()

        for i in range(1,len(time)):
            u[2] = u[1] 
            u[1] = u[0] 
            u[0] = (-self.delta_t**2 * A) @ u[1] - u[2] + 2*u[1]

            if (i % nts) == 0:
                D[i] = np.transpose(b) @ u[1]
                D[i] = 0.5*(D[i].T + D[i])

        return D

    def mass_matrix(self):
        D = self.forward_solver()
        M = np.zeros((self.N_s*self.N_t, self.N_s*self.N_t), dtype=np.float64)

        for i in range(self.N_t):
            for j in range(self.N_t):
                ind_i = self.index(i)
                ind_j = self.index(j)

                M[ind_i[0]:ind_i[-1],ind_j[0]:ind_j[-1]] = 0.5 * (D[abs(i-j)] + D[abs(i+j)])

        R = mblockchol(M, self.N_s, self.N_t)
        print(R)
        eigs = np.linalg.eigvals(M)
        print(np.max(eigs))
        print(np.min(eigs))

        return M, D, R

    def background_snapshots(self):
        """
        Function to calculate the orthogonalized background snapshots V_0
        - size of V_0 = (N_x_im*N_y_im, N_s*N_t)
        """
       
        U_0 = np.full((self.N_x_im * self.N_y_im, self.N_t * self.N_s),
                       self.background_velocity_value,
                       dtype=np.float64)

        M, D, R = self.mass_matrix()

        V_0 = U_0 @ np.linalg.inv(R)

        print(np.shape(V_0))
        print(V_0)
        
        I = self.imaging_func(V_0, R)
        print(I)
        np.savetxt("I_result.txt", I)


    def imaging_func(self, V_0, R):
        """
        Imaging function at a point.
        Find good way to use R and V_0! 
        Then only np.linalg.norm()**2 for each i in 1, .., N_x_im*N_y_im
        """
        I = np.zeros((self.N_y_im * self.N_x_im), dtype = np.float64)

        print(f"Shape of input to norm: {np.shape(V_0[1, :] @ R)}")

        for i in range(self.N_x_im*self.N_y_im):
            I[i] = np.linalg.norm(V_0[i, :] @ R, 2)**2

        return I

def main():
    solver = ForwardSolver()
    solver.background_snapshots()
    
if __name__ == "__main__":
    main()