import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from cholesky import mblockchol
import scipy as sp
from scipy import ndimage
from os.path import exists
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
                O_x: int = 150,
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
        self.Bsrc_file = Bsrc_file 

        self.imaging_region_indices = self.get_imaging_region_indices()

        self.background_velocity_value = background_velocity_value
        self.background_velocity = np.full(self.N**2,
                                           background_velocity_value,
                                           dtype=np.float64)

        self.tau = tau
        self.N_t = N_t
        self.delta_t = tau/20
        
        if exists("./V0.npy"):
            self.V_0 = np.load("./V0.npy")
        
        else:
            self.V_0 = self.calculate_V0()
            np.save("./V0.npy", self.V_0)

    def import_sources(self):
        
        b = np.loadtxt(self.Bsrc_file, delimiter =',', dtype=np.float64)
        np.reshape(b, (self.N_x * self.N_y, self.N_s))

        return b
    
    def get_imaging_region_indices(self):
        im_y_indices = range(self.O_y, self.O_y+self.N_y_im)
        im_x_indices = range(self.O_x, self.O_x+self.N_x_im)
        indices = [y*self.N_x + x for y in im_y_indices for x in im_x_indices] 

        return indices
        
    def init_simulation(self, c: np.array):
        # Calculate operators
        I_k = sparse.identity(self.N)
        D_k = (1/self.delta_x**2)*sparse.diags([1,-2,1],[-1,0,1], shape=(self.N,self.N), dtype=np.float64)
        D_k = sparse.csr_matrix(D_k)
        D_k[0, 0] = -1 * (1/self.delta_x**2)

        L = sparse.kron(D_k, I_k) + sparse.kron(I_k, D_k)
        C = sparse.diags(c, 0, dtype=np.float64)

        A = (- C @ L @ C)

        u = np.zeros((3, self.N_x * self.N_y, self.N_s), dtype=np.float64) # Stores past, current and future instances

        b = self.import_sources()

        u[1] = b
        u[0] = (-0.5* self.delta_t**2 * A) @ b + b

        D = np.zeros((2*self.N_t, self.N_s, self.N_s), dtype=np.float64)
        D[0] = np.transpose(b) @ u[1]

        return u, A, D, b 

    def calculate_V0(self):
        u_init, A, D_init, b = self.init_simulation(self.background_velocity)
        D, U_0 = self.calculate_u(u_init, A, D_init, b) 
        R = self.calculate_mass_matrix(D)
        V_0 = U_0 @ np.linalg.inv(R)

        return V_0

    def find_indices(self,j):
        ind_t = np.linspace(0, self.N_s, self.N_s) + self.N_s*j 
        ind_list = [int(x) for x in ind_t]
        return ind_list

    def calculate_u(self, u, A, D, b):
        # Discretize time
        nts = 20
        T = (self.N_t * 2 - 1) * self.delta_t * nts
        time = np.linspace(0, T, num=2*self.N_t*nts)

        U_0 = np.zeros((self.N_x_im*self.N_y_im, self.N_s, self.N_t))
        # print(u[1][self.imaging_region_indices].shape) 
        U_0[:,:,0] = u[1][self.imaging_region_indices]
        
        count_storage_D = 0
        count_storage_U_0 = 0
        for i in range(1,len(time)):
            u[2] = u[1] 
            u[1] = u[0] 
            u[0] = (-self.delta_t**2 * A) @ u[1] - u[2] + 2*u[1]

            if (i % nts) == 0:
                index = int(i/nts)
                D[index] = np.transpose(b) @ u[1]
                D[index] = 0.5*(D[index].T + D[index])

                count_storage_D += 1
                print(count_storage_D)

                if i <= self.N_t*nts-1:
                    U_0[:,:,index] = u[1][self.imaging_region_indices]

                    count_storage_U_0 += 1

        U_0 = np.reshape(U_0, (self.N_x_im * self.N_y_im, self.N_s * self.N_t))

        print(f"Count D = {count_storage_D}")
        print(f"Count stage U_0 = {count_storage_U_0}")

        return D, U_0


    def calculate_intensity(self, C: np.array):
        u_init, A_init, D_init, b = self.init_simulation(C)
        D, U_0 = self.calculate_u(u_init, A_init, D_init, b)
        R = self.calculate_mass_matrix(D)
        # V_0 = self.calculate_background_snapshots(U_0, R)
        # V_0 = U_0 @ np.linalg.inv(R) # or just have the "calculate_background_snapshots" here directly instead
        I = self.calculate_imaging_func(R)

        return I

    def calculate_mass_matrix(self, D):
        M = np.zeros((self.N_s*self.N_t, self.N_s*self.N_t), dtype=np.float64)

        for i in range(self.N_t):
            for j in range(self.N_t):
                ind_i = self.find_indices(i)
                ind_j = self.find_indices(j)

                M[ind_i[0]:ind_i[-1],ind_j[0]:ind_j[-1]] = 0.5 * (D[abs(i-j)] + D[abs(i+j)])

        R = mblockchol(M, self.N_s, self.N_t)

        return R
    
    def calculate_imaging_func(self, R):
        """
        # for i in range(self.N_x_im*self.N_y_im):
        #     I[i] = np.linalg.norm(V_0[i, :] @ R, 2)**2
        """
        I = self.V_0 @ R
        I = np.square(I)

        I = I.sum(axis=1)
        
        return I
    
##### New by 2022-11-18 : try to plot to see if reasonable or not
    def plot_intensity(self, I):
        data_temp = np.reshape(I, (self.N_y_im, self.N_x_im))

        self.plot_result_matrix(data_temp, 'I', np.shape(data_temp)[1], np.shape(data_temp)[0])

    def calculate_I_matrices(self, n_images: int, plot: bool, output_file: str = ""):
        for i in range(n_images):
            c = np.load(f"./images/fractured/im{i}.npy")
            I = self.calculate_intensity(c)

            if plot:
                self.plot_intensity(I)

            if output_file:
                np.save(output_file, I)


    def plot_result_matrix(self,
                        matrix_results,
                        matrix_name: str='matrix_results',
                        x_dim: int= 175,
                        y_dim: int= 150):

        fig, ax = plt.subplots()
        im = ax.imshow(ndimage.rotate(np.squeeze(matrix_results), -90), aspect = 'equal', cmap='Greys') 
        plt.title(f"Colormap of matrix {matrix_name}")
        plt.xlabel("Coordinate in y-dim")
        plt.ylabel("Coordinate in x-dim")
        plt.xlim(0, y_dim)
        plt.ylim(0, x_dim)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.invert_yaxis()

        # Continue to look at getting axis equal!!!
        
        plt.show()


        # From the plot, get the axis, and set equal
        # Try to transpose the image to get the width at the x-axis
        # something that scales the axis according to the units (plt.axis = equal)

        # fracture at 100 - 150 and depth of 70
        # plot the V's and send to Jörn
        # Can already store the V's to a file and don't have to calculate them

        # Once V's stored => Have a flag to not have to store them!


def main():
    solver = ForwardSolver( 
                N_x = 512,
                N_y = 512,
                N_s = 50,
                delta_x = 0.0063,
                tau = 3.0303*10**(-5),
                N_t = 70,
                background_velocity_value = 1000,
                Bsrc_file = "Bsrc_T.txt",
                N_x_im = 175,
                N_y_im = 350,
                O_x = 150,
                O_y = 81
    )

    solver.calculate_I_matrices(10, True, False)

    # solver.calculate_intensities()
    # solver.plot_intensity_I("./I_result.npy")

if __name__ == "__main__":
    main()