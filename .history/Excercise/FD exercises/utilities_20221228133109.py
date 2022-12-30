import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import time
plt.rcParams.update({'font.size': 14})
def error(phi_a,phi_n):
    # compute error between analytical and numerical numbers

    return (phi_a-phi_n)/phi_a*100

def convert1D_to_2D(A,N,M):
# Convert phi1D to 2D
    A2D = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            k = (j-1)*N + i
            A2D[i,j]=A[k]

    return A2D

def prepare_phi_and_S(Nx,Ny,phi,L, H, convert_to_K = False):

    # Generate RHS source terms matrix and unknowns 'phi' with Dirichlet BCs
    if convert_to_K:
        S = np.zeros((Nx*Ny))
    else:   
        S = np.zeros((Nx,Ny))
    dx = L/(Nx-1)    # Grid size
    dy = H/(Ny-1)    # Grid size
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
                S[i,j] = source

    
    ## Right BC
    i = Nx-1 
    for j in range(1,Ny-1):

        x = i*dx
        y = j*dy
        phiR = 0.25*np.sinh(5)+(y-0.5)**2*np.sinh(10*(y-0.5))+np.exp(2*y)
        
        if convert_to_K:
            k = (j - 1) * Nx + i
            phi[k] = phiR
            S[k] = phiR
        else:
            phi[i,j] = phiR
            S[i,j] = phiR
        
    ## left BC
    i = 0
    for j in range(1,Ny-1):
        
        x = i*dx
        y = j*dy
        
        phiL = 0.25*np.sinh(-5) + (y-0.5)**2*np.sinh(10*(y-0.5))+1

        if convert_to_K:
            k = (j - 1) * Nx + i
            phi[k] = phiL
            S[k] = phiL
        else:
            phi[i,j] = phiL
            S[i,j] = phiL 

    ## Bottom BC
    j=0
    for i in range(Nx):
        
        x = i*dx
        y = j*dy
        
        phiB = 0.25*np.sinh(-5) + (x-0.5)**2*np.sinh(10*(x-0.5))+1
        if convert_to_K:
            k = (j - 1) * Nx + i
            phi[k] = phiB
            S[k] = phiB
        else:
            phi[i,j] = phiB
            S[i,j] = phiB
        

    ## Top BC
    j=Ny-1
    for i in range(Nx):
        
        x = i*dx
        y = j*dy
        
        phiT = 0.25*np.sinh(5)+(x-0.5)**2*np.sinh(10*(x-0.5))+np.exp(2*x)
        
        if convert_to_K:
            k = (j - 1) * Nx + i
            phi[k] = phiT
            S[k] = phiT
        else:
            phi[i,j] = phiT
            S[i,j] = phiT

    return phi, S


def plot_phi(phi,phi_A,Nx,Ny,method_name,convert=False):

    figsize = ((10,6))
    # If need convert phi from phi[K] to phi[i,j], aka, phi1D to phi2D
    if convert:
        # Analytical solution
        phi2D = np.zeros((Nx,Ny))
        # Convert phi1D to 2D
        for i in range(Nx):
            for j in range(Ny):
                k = (j-1)*Nx+ i
                phi2D[i,j]=phi[k]
        phi = phi2D

    # Plot        
    x,y = np.meshgrid(np.linspace(0,1,Nx),np.linspace(0,1,Ny), indexing='ij')
    fig, ax = plt.subplots(figsize=figsize)
    CS = ax.contour(x,y,phi, levels=np.arange(-30,30,5))
    ax.clabel(CS, inline=True, fontsize=10)
    CB = fig.colorbar(CS)
    ax.set_xlabel('Distance, x')
    ax.set_ylabel('Distance, y')

    ax.set_title('Numerical solution by '+method_name+' iterative solver, code by Congcong Fan')
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('Analytical solution, code by Congcong Fan')

    CS = ax.contour(x,y,phi_A, levels=np.arange(-30,30,5))
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

def residual(Nx,Ny,phi,S,aE,aW,aN,aS,a0):
    
    R = np.zeros(Nx*Ny)
    R2sum = 0
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):

            k = (j - 1) * Nx + i
            R[k] = S[k] - aE * phi[k + 1] - aW * phi[k - 1] - aN * phi[k + Nx] - aS * phi[k - Nx] - a0 * phi[k]

            R2sum = R2sum + R[k] ** 2
    
    return R2sum, R


def matrixA(Nx,Ny):
    A = np.zeros((Nx ** 2, Ny ** 2))

    ## Right BC
    i = Nx - 1
    for j in range(Ny):
        x = i * dx
        y = j * dy

        k = (j - 1) * Ny + i
        A[k, k] = 1

    ## left BC
    i = 0
    for j in range(Ny):
        x = i * dx
        y = j * dy
        k = (j - 1) * Ny + i
        A[k, k] = 1
    ## Bottom BC
    j = 0
    for i in range(Nx):
        x = i * dx
        y = j * dy
        k = (j - 1) * Ny + i
        A[k, k] = 1
    ## Top BC
    j = Ny - 1
    for i in range(Nx):
        x = i * dx
        y = j * dy
        k = (j - 1) * Ny + i
        A[k, k] = 1

    for i in range(1, Nx - 1):

        for j in range(1, Ny - 1):
            k = (j - 1) * Ny + i
            A[k, k] = -2 / dx ** 2 - 2 / dy ** 2
            A[k, k - 1] = 1 / dx ** 2
            A[k, k + 1] = 1 / dx ** 2

            A[k, k - Nx] = 1 / dy ** 2
            A[k, k + N] = 1 / dy ** 2

    return A