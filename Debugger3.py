import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import time
plt.rcParams.update({'font.size': 14})

from utilities import convert2D_to_1D, error, prepare_phi_and_S, convert1D_to_2D, plot_phi,residual, matrixA


def NZA(A):
    NZA = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                NZA[i, j] = 1

    return NZA


def ILU0(NZA, A):
    L = np.zeros(NZA.shape)
    U = np.zeros(NZA.shape)
    K = NZA.shape[0]

    for k in tqdm(range(K - 1)):
        for i in range(k, K):

            if NZA[i, k] == 1:
                A[i, k] = A[i, k] / A[k, k]

            for j in range(k, K):
                if NZA[i, j] == 1:
                    A[i, j] = A[i, j] + A[i, k] * A[k, j]

    for i in range(K):
        for j in range(i, K):
            U[i, j] = A[i, j]

        for j in range(i - 1):
            L[i, j] = A[i, j]

    return L, U


def matmul_between_transpose_and_normal(mat1, mat2, N, M):
    # mat1 is the matrix that will be transposed
    # mat2 is normal matrix
    # this function computes [mat1].T @ [mat2]
    # return a scalar
    output = 0
    for i in range(1, N - 1):
        for j in range(1, M - 1):
            k = (j - 1) * N + i
            output = output + mat1[k] * mat2[k]

    return output


def Compute_Rm(L, U, Nx, Ny, R):
    '''
    [ M ] [ Rm ] = [ R ]
    [ M ] = [ L ] [ U ]
    [ L ] [ U ] [ Rm ] = [ R ]
    [ U ] [ Rm ] = [ Y ]
    [ L ] [ Y ] = [ R ]
    '''
    # Compute above steps from last eqn to top eqn without using matrix

    Y = matmul_between_transpose_and_normal(L, R, Nx, Ny)

    Rm = matmul_between_transpose_and_normal(U, Y, Nx, Ny)
    return Rm

tart = time.time()
Nx = 21
Ny = 21
Length = 1  # length
Height = 1
phi = np.zeros((Nx*Ny))
S = np.zeros((Nx*Ny))

dx = Length / (Nx - 1)  # Grid size
dy = Height / (Ny - 1)  # Grid size

tol = 1e-6

aE = 1 / dx ** 2
aW = 1 / dx ** 2
aN = 1 / dy ** 2
aS = 1 / dy ** 2
a0 = -(2 / dx ** 2 + 2 / dy ** 2)

phi, S = prepare_phi_and_S(Nx,Ny,phi,Length, Height,convert_to_K=False)

A = matrixA(Nx,Ny,dx,dy)
nza = NZA(A)
L, U = ILU0(nza, A)
# L = convert2D_to_1D(L, Nx,Ny)
# U = convert2D_to_1D(U, Nx,Ny)

# Initial residual
R = S - A @ phi
R2sum_old = 0
for i in range(1,Nx-1):

    for j in range(1,Ny-1):
        k = (j - 1) * Nx + i
        R2sum_old = R2sum_old + R[k] ** 2

R2_old = np.sqrt(R2sum_old)

# Modified residual Rm
Rm = Compute_Rm(L, U, Nx, Ny, R)

# Step 3: Set the initial search direction vector equal to the residual vector
D = Rm

for _ in tqdm(range(100000)):

    # Compute new alpha

    alpha_new = R.T @ R / (D.T @ A @ D)

    phi = phi + alpha_new * D

    # Compute residual
    R2sum_new, R_new = residual(Nx, Ny, phi, S, aE, aW, aN, aS, a0)

    R2_new = np.sqrt(R2sum_new)

    # Compute new modified residual
    Rm_new = Compute_Rm(L, U, Nx, Ny, R)

    # Compute Beta, beta = [ { R_new.T } @ {Rm_new} ]/ [ { R_old.T } @ {Rm_old} ]
    num = matmul_between_transpose_and_normal(R_new, Rm_new, Nx, Ny)
    den = matmul_between_transpose_and_normal(R, Rm, Nx, Ny)

    beta = num / den

    # Update search direction vector
    D = Rm_new + beta * D

    # Update old residual vector
    R2_old = R2_new
    R = R_new
    Rm = Rm_new

    if _ % 100 == 0:
        clear_output(True)
        print(f'alpha: {beta}')
        print("Residual: ", R2_new)

    if R2_new < tol:
        print('Converged! Residual: ', R2_new, 'Time elapsed: ', time.time() - start)
        break