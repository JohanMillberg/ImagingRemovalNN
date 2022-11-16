"""
Python translation of the Block-Cholesky factorization given as Matlab-code!
"""

# Try to translate the matlab code into python
# Original from Alexander Mamonov, 2015
import numpy as np
from scipy.linalg import sqrtm

def ind(Ns, j):
    """ Help function for the mblockchol-function 
        Returns: a list of the "active indexes" of size Ns
        Corresponding to "ind = @(j) in (1:Ns) + Ns*(j-1)" in Matlab code
        Added a -1 since k in range(1, Nt) and we want to have starting index 0
    """
    ind_t = np.linspace(0, Ns, Ns) + Ns*(j-1)
    ind_list = [int(x) for x in ind_t]
    return ind_list


def mblockchol(M, Ns, Nt):
    """ The function for the Block-Cholesky factorization
        mblockchol: Block Cholesky M = R' * R
        Ns = size of each block
        Nt = number of blocks
    """

    L = np.zeros([Ns*Nt, Ns*Nt])

    for k in range(1, Nt+1):
        msum = np.zeros((Ns, Ns), dtype=np.float64)
        for j in range(1, k):
            msum = np.add(msum, \
                np.matmul(L[ind(Ns, k)[0]:ind(Ns, k)[-1], ind(Ns, j)[0]:ind(Ns, j)[-1]],\
                          (L[ind(Ns, k)[0]:ind(Ns, k)[-1], ind(Ns, j)[0]:ind(Ns, j)[-1]]).transpose()))

        # Update L-matrix
        L[ind(Ns, k)[0]:ind(Ns, k)[-1],ind(Ns, k)[0]:ind(Ns, k)[-1]] = \
             sqrtm(np.subtract(M[ind(Ns, k)[0]:ind(Ns, k)[-1], ind(Ns, k)[0]:ind(Ns, k)[-1]],msum))

        for i in range(k, Nt+1):
            msum = np.zeros((Ns, Ns), dtype=np.float64)
            for j in range(1, k):
                msum = np.add(msum, \
                    np.matmul(L[ind(Ns, i)[0]:ind(Ns, i)[-1], ind(Ns, j)[0]:ind(Ns, j)[-1]],\
                              L[ind(Ns, k)[0]:ind(Ns, k)[-1], ind(Ns, j)[0]:ind(Ns, j)[-1]].transpose()))

            # Update L-matrix for the "inner"-matrices
            M_new = np.subtract(M[ind(Ns, i)[0]:ind(Ns, i)[-1], ind(Ns, k)[0]:ind(Ns, k)[-1]], msum)
            norm_frac = L[ind(Ns, k)[0]:ind(Ns, k)[-1], ind(Ns, k)[0]:ind(Ns, k)[-1]]

            # print("\n\n")
            # print(f"i = {i}")
            # print(f"k = {k}")
            # print("\nCholesky: norm_frac")
            # print(norm_frac)

            # print("\nCholesky: np.linalg.inv(norm_frac)")
            # print(np.linalg.inv(norm_frac))
            
            L[ind(Ns, i)[0]:ind(Ns, i)[-1], ind(Ns, k)[0]:ind(Ns, k)[-1]] =\
                 np.matmul(M_new, np.linalg.inv(norm_frac))

    # Convert to final resulting matrix
    R = L.transpose()
    return R


if __name__ == '__main__':
    # Given input data
    Ns = 2
    Nt = 3
    M = np.array([[6, 5, 4, 3, 2, 1], 
                  [5, 6, 5, 4, 3, 2],
                  [4, 5, 6, 5, 4, 3],
                  [3, 4, 5, 6, 5, 4],
                  [2, 3, 4, 5, 6, 5],
                  [1, 2, 3, 4, 5, 6]], dtype=np.float64)
    #print(M)

    # Test cholesky function
    R = mblockchol(M, Ns, Nt)
    print("Result of function:")
    print(R)

    print("\nCheck R.T * R to see if correct (--> should be M exactly):")
    print(np.matmul(R.T,R))