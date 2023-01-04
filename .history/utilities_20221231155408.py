import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import time
from scipy import interpolate
plt.rcParams.update({'font.size': 14})


def error(phi_a, phi_n):
    # compute error between analytical and numerical numbers

    return (phi_a - phi_n) / phi_a * 100


def convert1D_to_2D(A, N, M):
    # Convert phi1D to 2D
    A2D = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            k = (j - 1) * N + i
            A2D[i, j] = A[k]

    return A2D


def convert2D_to_1D(A, N, M):
    # Convert phi1D to 2D
    A1D = np.zeros((N * M))
    for i in range(N):
        for j in range(M):
            k = (j - 1) * N + i
            A1D[k] = A[i, j]

    return A1D


def prepare_phi_and_S(Nx, Ny, phi, L, H, convert_to_K=False):
    # Generate RHS source terms matrix and unknowns 'phi' with Dirichlet BCs
    if convert_to_K:
        S = np.zeros((Nx * Ny))
        phi = np.zeros((Nx * Ny))
    else:
        S = np.zeros((Nx, Ny))
        phi = np.zeros((Nx , Ny))
    dx = L / (Nx - 1)  # Grid size
    dy = H / (Ny - 1)  # Grid size
    # RHS source terms
    for i in range(Nx):
        for j in range(Ny):
            x = i * dx
            y = j * dy

            source = 2 * np.sinh(10 * (x - 0.5)) + 40 * (x - 0.5) * np.cosh(10 * (x - 0.5)) + 100 * (
                    x - 0.5) ** 2 * np.sinh(10 * (x - 0.5)) + 2 * np.sinh(10 * (y - 0.5)) + 40 * (
                             y - 0.5) * np.cosh(10 * (y - 0.5)) + 100 * (y - 0.5) ** 2 * np.sinh(
                10 * (y - 0.5)) + 4 * (x ** 2 + y ** 2) * np.exp(2 * x * y)

            if convert_to_K:
                k = (j - 1) * Nx + i
                S[k] = source
            else:
                S[i, j] = source

    ## Right BC
    i = Nx - 1
    for j in range(1, Ny - 1):

        x = i * dx
        y = j * dy
        phiR = 0.25 * np.sinh(5) + (y - 0.5) ** 2 * np.sinh(10 * (y - 0.5)) + np.exp(2 * y)

        if convert_to_K:
            k = (j - 1) * Nx + i
            phi[k] = phiR
            S[k] = phiR
        else:
            phi[i, j] = phiR
            S[i, j] = phiR

    ## left BC
    i = 0
    for j in range(1, Ny - 1):

        x = i * dx
        y = j * dy

        phiL = 0.25 * np.sinh(-5) + (y - 0.5) ** 2 * np.sinh(10 * (y - 0.5)) + 1

        if convert_to_K:
            k = (j - 1) * Nx + i
            phi[k] = phiL
            S[k] = phiL
        else:
            phi[i, j] = phiL
            S[i, j] = phiL

            ## Bottom BC
    j = 0
    for i in range(Nx):

        x = i * dx
        y = j * dy

        phiB = 0.25 * np.sinh(-5) + (x - 0.5) ** 2 * np.sinh(10 * (x - 0.5)) + 1
        if convert_to_K:
            k = (j - 1) * Nx + i
            phi[k] = phiB
            S[k] = phiB
        else:
            phi[i, j] = phiB
            S[i, j] = phiB

    ## Top BC
    j = Ny - 1
    for i in range(Nx):

        x = i * dx
        y = j * dy

        phiT = 0.25 * np.sinh(5) + (x - 0.5) ** 2 * np.sinh(10 * (x - 0.5)) + np.exp(2 * x)

        if convert_to_K:
            k = (j - 1) * Nx + i
            phi[k] = phiT
            S[k] = phiT
        else:
            phi[i, j] = phiT
            S[i, j] = phiT

    return phi, S


def plot_phi(phi, phi_A, Nx, Ny, method_name, convert=False):
    figsize = ((10, 6))
    # If need convert phi from phi[K] to phi[i,j], aka, phi1D to phi2D
    if convert:
        # Analytical solution
        phi2D = np.zeros((Nx, Ny))
        # Convert phi1D to 2D
        for i in range(Nx):
            for j in range(Ny):
                k = (j - 1) * Nx + i
                phi2D[i, j] = phi[k]
        phi = phi2D

    # Plot        
    x, y = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny), indexing='ij')
    fig, ax = plt.subplots(figsize=figsize)
    CS = ax.contour(x, y, phi, levels=np.arange(-30, 30, 5))
    ax.clabel(CS, inline=True, fontsize=10)
    CB = fig.colorbar(CS)
    ax.set_xlabel('Distance, x')
    ax.set_ylabel('Distance, y')

    ax.set_title('Numerical solution by ' + method_name + ' iterative solver, code by Congcong Fan')
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('Analytical solution, code by Congcong Fan')

    CS = ax.contour(x, y, phi_A, levels=np.arange(-30, 30, 5))
    ax.clabel(CS, inline=True, fontsize=10)
    CB = fig.colorbar(CS)
    ax.set_xlabel('Distance, x')
    ax.set_ylabel('Distance, y')
    fig.tight_layout()

    # # Error
    # e = error(phi_A,phi)
    # fig, ax = plt.subplots(figsize=(14,8))
    # CS = ax.contour(e)
    # ax.clabel(CS, inline=True, fontsize=10)
    # ax.set_title('Errors, code by Congcong Fan')
    # # make a colorbar for the contour lines
    # CB = fig.colorbar(CS)
    # ax.set_xlabel('Distance, x')
    # ax.set_ylabel('Distance, y')
    # fig.tight_layout()


def residual(Nx, Ny, phi, S, aE, aW, aN, aS, a0, convert=False):
    """

    :param Nx: grid points in X
    :param Ny: grid points in Y
    :param phi: soln
    :param S: RHS source terms matrix
    :param aE: East coefficient
    :param aW: West coefficient
    :param aN: North coefficient
    :param aS: South coefficient
    :param a0: Diagonal term
    :param convert: Boolean for whether converting from i,j to k
    :return: Residual between [S] - [A][phi]
    """
    if convert:
        # In [k] index
        R_vector = np.zeros(Nx * Ny)
        Rsum = 0
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                k = (j - 1) * Nx + i
                R_vector[k] = S[k] - aE * phi[k + 1] - aW * phi[k - 1] - aN * phi[k + Nx] - aS * phi[k - Nx] - a0 * phi[
                    k]

                Rsum = Rsum + R_vector[k] ** 2
        R2 = np.sqrt(Rsum)

    else:
        # In [i, j] index
        Rsum = 0
        R_vector = np.zeros((Nx, Ny))

        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                R_vector[i, j] = S[i, j] - aE * phi[i + 1, j] - aW * phi[i - 1, j] - aN * phi[i, j + 1] - aS * phi[
                    i, j - 1] - a0 * phi[i, j]
                Rsum = Rsum + R_vector[i, j] ** 2
        R2 = np.sqrt(Rsum)
    return R2, Rsum, R_vector


def matrixA(Nx, Ny, dx, dy):
    A = np.zeros((Nx ** 2, Ny ** 2))
    ## Right BC
    i = Nx-1
    for j in range(Ny):
        k = (j - 1) * Nx + i
        A[k, k] = 1
        
    ## left BC
    i = 0
    for j in range(Ny):
        k = (j - 1) * Nx + i
        A[k, k] = 1
        ## Bottom BC
    j = 0
    for i in range(Nx):
        k = (j - 1) * Nx + i
        A[k, k] = 1
        
    ## Top BC
    j = Ny - 1
    for i in range(Nx):
        k = (j - 1) * Nx + i
        A[k, k] = 1
        
    for i in range(1, Nx-1):

        for j in range(1, Ny-1):
            k = (j -1) * Nx + i
            if k<0:
                print(i,j,k)
            A[k, k] = -2 / dx ** 2 - 2 / dy ** 2
            A[k, k - 1] = 1 / dx ** 2
            A[k, k + 1] = 1 / dx ** 2

            
            if k>Nx:
                A[k, k - Nx] = 1 / dy ** 2
            A[k, k + Nx] = 1 / dy ** 2
    # plt.spy(A)
    return A


def backward_substitution(A, b):
    '''
    In linear algebra, backward substitution is a method for solving a system of linear equations in the form Ax = b,
    where A is a lower triangular matrix. The goal is to find the solution x for the system of equations. :param A:

    '''
    # Solve the system of equations using backward substitution
    n = len(b)
    x = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x


def forward_substitution(A, b):
    """
    n linear algebra, forward substitution is a method for solving a system of linear equations in the form Ax = b,
    where A is an upper triangular matrix. The goal is to find the solution x for the system of equations.

    :param A:
    :param b:
    :return:
    """
    # Solve the system of equations using forward substitution
    n = len(b)
    x = np.zeros_like(b)
    for i in range(n):
        x[i] = (b[i] - np.dot(A[i, :i], x[:i])) 

    return x

def GS(Nx,Ny,phi,S,aE,aW,aN,aS,a0):
    for i in range(1,Nx-1):
            for j in range(1,Ny-1):

                # Gauss-Siedel Update
                phi[i,j] = (S[i,j] - aE*phi[i+1,j] - aW*phi[i-1,j] - aN*phi[i,j+1] - aS*phi[i,j-1]) / a0
    return phi

def smoothing(Nxf, Nyf, phif, Sf, aEf, aWf, aNf, aSf, a0f, xf_list, yf_list, xc_list,yc_list):

    phif = GS(Nxf, Nyf, phif, Sf, aEf, aWf, aNf, aSf, a0f)

    R2f, Rsumf, Rf_new = residual(Nxf, Nyf, phif, Sf, aEf, aWf, aNf, aSf, a0f, convert=False)

    # Transfer Residual to corse mesh
    # Since in current 2 mesh size, there is always a corse mesh sitting on the top of fine mesh
    f = interpolate.RectBivariateSpline(xf_list, yf_list, Rf_new)
    Rc_new = f(xc_list, yc_list)

    return Rc_new

def restriction():
phic = np.zeros((Nxc, Nyc))
    
    # Smoothing the errors on the coarse mesh
    # Use the correction form [A'][phi'] = [R] to calculate the correction form of phi
 
    phic = GS(Nxc, Nyc, phic, Rc_new, aEc, aWc, aNc, aSc, a0c)
    # phic += corrector
    R2c, _, _ = residual(Nxc, Nyc, phic, Rc_new, aEc, aWc, aNc, aSc, a0c, convert=False)

    # Transfer the correction form of phi at coarse mesh to finer mesh

    f = interpolate.RectBivariateSpline(xc_list, yc_list, phic)
    phif_corrector = f(xf_list, yf_list)
    return