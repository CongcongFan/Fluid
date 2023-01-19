import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import time
from utilities import plot_phi
plt.rcParams.update({'font.size': 14})

def prepare_phi_and_S(Nx, Ny, phi, L, H, convert_to_K=False):
    # Generate RHS source terms matrix and unknowns 'phi' with Dirichlet BCs
    if convert_to_K:
        S = np.zeros((Nx * Ny))
        phi = np.zeros((Nx * Ny))
    else:
        S = np.zeros((Nx, Ny))
        phi = np.zeros((Nx, Ny))
    dx = L / (Nx - 1)  # Grid size
    dy = H / (Ny - 1)  # Grid size
    # RHS source terms
    for i in range(Nx):
        for j in range(Ny):
            x = i * dx
            y = j * dy

            source = 1000*(2*np.sinh(x-0.5) + 4*(x-0.5)*np.cosh(x-0.5) + (x-0.5)**2 * np.sinh(x-0.5))
            + 1000*(2*np.sinh(y-0.5) + 4*(y-0.5)*np.cosh(y-0.5) + (y-0.5)**2 * np.sinh(y-0.5))

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
        phiR = 1000 * (0.25*np.sinh(0.5) + (y-0.5)**2*np.sinh(y-0.5))

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

        phiL = 1000 * (0.25*np.sinh(-0.5) + (y-0.5)**2*np.sinh(y-0.5))

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

        phiB = 1000 * (0.25*np.sinh(-0.5) + (x-0.5)**2*np.sinh(x-0.5))
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

        phiT = 1000 * (0.25*np.sinh(0.5) + (x-0.5)**2*np.sinh(x-0.5))

        if convert_to_K:
            k = (j - 1) * Nx + i
            phi[k] = phiT
            S[k] = phiT
        else:
            phi[i, j] = phiT
            S[i, j] = phiT

    return phi, S


def applyBC(phi, Nx,Ny,S,L,H,convert_to_K = False):
    ## Right BC
    dx = L / (Nx - 1)  # Grid size
    dy = H / (Ny - 1)  # Grid size

    i = Nx - 1
    for j in range(1, Ny - 1):

        x = i * dx
        y = j * dy
        phiR = 1000 * (0.25*np.sinh(0.5) + (y-0.5)**2*np.sinh(y-0.5))

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

        phiL = 1000 * (0.25*np.sinh(-0.5) + (y-0.5)**2*np.sinh(y-0.5))

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

        phiB = 1000 * (0.25*np.sinh(-0.5) + (x-0.5)**2*np.sinh(x-0.5))
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

        phiT = 1000 * (0.25*np.sinh(0.5) + (x-0.5)**2*np.sinh(x-0.5))

        if convert_to_K:
            k = (j - 1) * Nx + i
            phi[k] = phiT
            S[k] = phiT
        else:
            phi[i, j] = phiT
            S[i, j] = phiT

    return phi

def residual(dt, Nx, Ny, phi, S, aE, aW, aN, aS, a0,):
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

    # In [i, j] index
    Rsum = 0
    R_vector = np.zeros((Nx, Ny))

    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            R_vector[i, j] = phi[i,j]/dt + S[i, j] - aE * phi[i + 1, j] - aW * phi[i - 1, j] - aN * phi[i, j + 1] - aS * phi[
                i, j - 1] - a0 * phi[i, j]
            Rsum = Rsum + R_vector[i, j] ** 2
    R2 = np.sqrt(Rsum)
    return R2, Rsum, R_vector


# numbering scheme used is k = (j-1)*N + i
start = time.time()
Nx = 41
Ny = 41
L = 1  # length
H = 1  # length

phi = np.zeros((Nx, Ny))

dx = L / (Nx - 1)  # Grid size
dy = L / (Ny - 1)  # Grid size

t = 0
alpha = 1
tol = 1e-6
dt = 10 * (0.5 / alpha) / (1 / dx ** 2 + 1 / dy ** 2)

aE = -alpha / dx ** 2
aW = -alpha / dx ** 2
aN = -alpha / dy ** 2
aS = -alpha / dy ** 2
a0 = (1 / dt + 2 * alpha / dx ** 2 + 2 * alpha / dy ** 2)

phi, S = prepare_phi_and_S(Nx, Ny, phi, L, H)


R2_old, _, _ = residual(dt, Nx, Ny, phi, S, aE, aW, aN, aS, a0)
for n in range(10):
    phi_next = np.zeros(phi.shape)
    phi_next = applyBC(phi_next, Nx, Ny, S, L, H)

    for _ in tqdm(range(10000)):

        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                # Gauss-Siedel Update
                phi[i, j] = (phi[i, j] / dt + S[i, j] - aE * phi[i + 1, j] - aW * phi[i - 1, j] - aN * phi[
                    i, j + 1] - aS * phi[i, j - 1]) / a0


        # Calculate residual
        R2, Rsum, R = residual(dt, Nx, Ny, phi, S, aE, aW, aN, aS, a0)

        R2 = np.sqrt(R2)
        if _ % 100 == 0:
            clear_output(True)
            print('Time: ', dt * n, 'Residual:', R2)

        if R2_old / R2 > 9:
            R2_old = R2
            print('Converged! Residual: ', R2, 'Time elapsed: ', time.time() - start)
            break


