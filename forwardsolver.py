import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from cholesky import mblockchol

"""
Code with Moa's U_0 code and Johan's indexing-function merged here together.
Also plots a figure of I as it is right now!

Runs without error messages but could be wrong.. :(
"""

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

        self.imaging_region_indices = self.get_imaging_region_indices()

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
    
    def get_imaging_region_indices(self):
        im_y_indices = range(self.O_y, self.O_y+self.N_y_im)
        im_x_indices = range(self.O_x, self.O_x+self.N_x_im)
        indices = [y*self.N_x + x for y in im_y_indices for x in im_x_indices] 

        return indices
        
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
        nts = 20
        #T = self.N_t * self.delta_t * nts
        T = (self.N_t * 2 - 1) * self.delta_t * nts
        time = np.linspace(0, T, num=2*self.N_t*nts)
        # when finding u0 (from Moa's smart brain at Jörn's office)
        #time = np.linspace(0, T, num=self.N_t)

        u, A, D, b = self.init_simulation()

        # U_0 = np.zeros((self.N_x*self.N_y, self.N_s, self.N_t))
        # U_0[:,:,0] = u[1]

        # New
        U_0 = np.zeros((self.N_x_im*self.N_y_im, self.N_s, self.N_t))
        print(u[1][self.imaging_region_indices].shape) 
        U_0[:,:,0] = u[1][self.imaging_region_indices]
        

        ### Continue to look here how many times the values are being stored in D-matrix and U_0-matrix
        # Still only 6 and 3 times in the end... Maybe not that good...? Maybe this is  the error?
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
                    #U_0[:,:,index] = u[1]
                    # New
                    U_0[:,:,index] = u[1][self.imaging_region_indices]

                    count_storage_U_0 += 1

        U_0 = np.reshape(U_0, (self.N_x_im * self.N_y_im, self.N_s * self.N_t))

        print(f"Count D = {count_storage_D}")
        print(f"Count stage U_0 = {count_storage_U_0}")

        return D, U_0

    def mass_matrix(self):
        D, U_0 = self.forward_solver()
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

        return M, R

    def background_snapshots(self):
        """
        Function to calculate the orthogonalized background snapshots V_0
        - size of V_0 = (N_x_im*N_y_im, N_s*N_t)
        """

        # Import U_0
        D, U_0 = self.forward_solver()

        # Only take the part of U_0 which is in the imaging region
        # U_0 = U_0_temp[self.imaging_region_indices]

        # Since only background velocity as it is right now, we have R = R_0
        M, R = self.mass_matrix()

        V_0 = U_0 @ np.linalg.inv(R)
        print(np.shape(V_0))
        
        I = self.imaging_func(V_0, R)
        print(np.shape(I))
        np.save("./I_result.npy", I)

        #### This step does not work... :( Look at later!
        self.plot_result_matrix(V_0, 'V_0', np.shape(V_0)[1], np.shape(V_0)[0])
        # Pick a few dimension and reshape to 2-dim and plot these instead
        # one of the first, one around 500, one around
        # one around each interval 0-50


    def imaging_func(self, V_0, R):
        """
        Imaging function at a point.
        Find good way to use R and V_0! 
        Then only np.linalg.norm()**2 for each i in 1, .., N_x_im*N_y_im
        """
        I = np.zeros((self.N_y_im * self.N_x_im), dtype = np.float64)
        # print(f"Shape of input to norm: {np.shape(V_0[1, :] @ R)}")

        for i in range(self.N_x_im*self.N_y_im):
            I[i] = np.linalg.norm(V_0[i, :] @ R, 2)**2
        # look at V_0 times R, square all entries and then sum them into the 2nd dimensions

        return I
    
##### New by 2022-11-18 : try to plot to see if reasonable or not
    def plot_intensity_I(self):
        """
        Function to plot a colormap of the values stored in I.
        First, store as a matrix over the grid.
        Second, call the plot function to see how the results looks.
        """
        I = np.load("I_result.npy")
        #data_temp = np.zeros((self.N_y_im, self.N_x_im), dtype=np.float64)
        data_temp = np.reshape(I, (self.N_y_im, self.N_x_im))

        # for j in range(self.N_x_im):
        #     for i in range(self.N_y_im):
        #         data_temp[i, j] = I[j + i]
        
        #self.plot_result_matrix(data_temp, 'I', self.N_x_im, self.N_y_im)
        self.plot_result_matrix(data_temp, 'I', np.shape(data_temp)[1], np.shape(data_temp)[0])


    
    def plot_result_matrix(self,
                        matrix_results,
                        matrix_name: str='matrix_results',
                        x_dim: int= 175,
                        y_dim: int= 150):
        """
        A function which plots the results in a matrix as a colormap!

        Choose a gradient color map
        Try to insert a variation in C (fraction) and see if you can see it! 
        See if algorithm finds the fractions.

        Store the V's and send to Jörn!
        """
        #plt.style.use('seaborn-white')
        plt.gray()
        x = np.linspace(0, x_dim, x_dim)
        y = np.linspace(0, y_dim, y_dim)

        X, Y = np.meshgrid(x, y)
        #plt.contourf(X, Y, matrix_results, cmap='RdGy')
        plt.imshow(np.squeeze(matrix_results)) 
        plt.colorbar()
        plt.title(f"Colormap of matrix {matrix_name}")
        plt.xlabel("x-coordinate/pixel")
        plt.ylabel("y-coordinate/pixel")
        plt.show()

        # From the plot, get the axis, and set equal
        # Try to transpose the image to get the width at the x-axis
        # something that scales the axis according to the units (plt.axis = equal)

        # fracture at 100 - 150 and depth of 70
        # plot the V's and send to Jörn
        # Can already store the V's to a file and don't have to calculate them

        # Once V's stored => Have a flag to not have to store them!


def main():
    solver = ForwardSolver()
    solver.background_snapshots()
    solver.plot_intensity_I()
    
if __name__ == "__main__":
    main()


### Question 1
# Team padding VS Team shrinking?!?!?!?!?
# What to do with elements/values not stored? What should we do there?
## Answer: Do nothing. Everything is fixed thanks to delta_t = tau/20, and sample each 20:th time step
# Everythingg gets correct dimensions
#  Using the Nyqvist theorem/formula/sampling technique

# If to shrink, how to cope with dimensions?
## Answer: Solved above

# If padding: 0 or background_velocity?
## Answer: Solved above

#### Question 2
# Only get indexes from R to get R_0?
## Answer: Before adding in any velocity c from the fractures, are R are the used R_0

### Explaination from Jörn (with Lollos words) of how to use the background velocity when having fractures
# Think like this: Use R_0 and U_0_im to find V_0, store R_0 and V_0
# Then add the fractions into the program
# Do the calculations with the mass matrix again to find the "real" R
# Use V_0 and R to  calculate the true I


##### Diskutera kod och hur den funkar
# Skickar in U_0 i background_snapshots på något sätt
# Plocka ur värden ur ett bra skapat U_0 med index funktion i Johans branch "indexing" 
# (förutsatt att bilden är en lång vektor där elementen kommer radvis)

# Ta hänsyn till dimensioner när vi räknar ut V_0
# Ta in ett U_0 och använd index_funktion 

# Indexera även R-matrisen så den blir R_0

# Kolla U_0 värden efter Moas beräkningar!

# Jörn = away until next wednesdqy