"""
Still some problems but a good foundation.
Think there's some troubles with the indexing etc..
"""

# Try to translate the matlab code into python
# Original from Alexander Mamonov, 2015
import numpy as np
import scipy as sp
from scipy.linalg import sqrtm

def ind(Ns, j):
    # Indexing into blocks of size Ns
    # Return a list of the "active" indexes
    # Corresponding to "ind = @(j) in (1:Ns) + Ns*(j-1)" in Matlab code
    # Added a -1 since k in range(1, Nt) and we want to have starting index 0
    ind_t = np.linspace(1, Ns, Ns) + Ns*(j-1) - 1
    ind_list = [int(x) for x in ind_t]
    return ind_list


def mblockchol(M, Ns, Nt):
    # mblockchol: Block Cholesky M = R' * R
    # Ns = size of each block
    # Nt = number of blocks

    L = np.zeros([Ns*Nt, Ns*Nt])

    for k in range(1, Nt+1):
        print("\nCheck list of indexes for k = ", k)
        print(ind(Ns, k))
        print("---")
        msum = np.zeros([Ns, Ns])
        for j in range(1, k):
            msum = np.add(msum, \
                np.matmul(L[ind(Ns, k)[0]:ind(Ns, k)[-1]+1, ind(Ns, j)[0]:ind(Ns, j)[-1]+1],\
                          (L[ind(Ns, k)[0]+1:ind(Ns, k)[-1]+2, ind(Ns, j)[0]+1:ind(Ns, j)[-1]+2]).transpose()))

        # print("--- Check index and which values of M that are 'active' ---")
        # print("Active indexes: ")
        # print(ind(Ns, k))
        # print("Values of M that are in consideration:")
        # print(M[ind(Ns, k)[0]:ind(Ns, k)[-1]+1, ind(Ns, k)[0]:ind(Ns, k)[-1]+1])
        # print("\n")

        L[ind(Ns, k)[0]:ind(Ns, k)[-1]+1,ind(Ns, k)[0]:ind(Ns, k)[-1]+1] = \
             sqrtm(np.subtract(M[ind(Ns, k)[0]:ind(Ns, k)[-1]+1, ind(Ns, k)[0]:ind(Ns, k)[-1]+1],msum))

        # print("\n--- Updated L matrix ---")
        # print(L)

        for i in range(k+1, Nt+1):
            msum = np.zeros([Ns, Ns])
            for j in range(1, k):
                msum = np.add(msum, \
                    np.matmul(L[ind(Ns, i)[0]:ind(Ns, i)[-1]+1, ind(Ns, j)[0]+1:ind(Ns, j)[-1]+2],\
                              L[ind(Ns, k)[0]:ind(Ns, k)[-1]+1, ind(Ns, j)[0]+1:ind(Ns, j)[-1]+2].transpose()))

            M_new = np.subtract(M[ind(Ns, i)[0]:ind(Ns, i)[-1]+1, ind(Ns, k)[0]:ind(Ns, k)[-1]+1], msum)
            norm_frac = L[ind(Ns, k)[0]:ind(Ns, k)[-1]+1, ind(Ns, k)[0]:ind(Ns, k)[-1]+1]
            L[ind(Ns, i)[0]:ind(Ns, i)[-1]+1, ind(Ns, k)[0]:ind(Ns, k)[-1]+1] =\
                 np.matmul(M_new, np.linalg.inv(norm_frac))

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
                  [1, 2, 3, 4, 5, 6]])
    #print(M)

    # Test cholesky function
    R = mblockchol(M, Ns, Nt)
    print("Result of function:")
    print(R)

    print("\nCheck R.T * R to see if correct (--> should be M exactly):")
    print(np.matmul(R.T,R))