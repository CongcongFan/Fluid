import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import time
plt.rcParams.update({'font.size': 14})

from utilities import error, prepare_phi_and_S, convert1D_to_2D, plot_phi,residual


def PEN(N, e, a, d, c, f, Q):
    '''
    ┌                                        ┐
    │  d1  c1  f1 ...                       0│
    │  a1  d2  c2  f2 ...                   0│
    │  e1         .                          │
    │                .                       │
    │                    .                   │┌     ┐  ┌     ┐
    │...         ei-2   ai-1  di  ci fi  ... ││Phi_x│= | Q_x |
    │                          .             │└     ┘  └     ┘
    │                            .           │
    │                               .    fN-2│
    │0     ... 0              aN-2  dN-1 cN-1│
    │0                    ... eN-2   aN-1  dN│
    └                                        ┘
    '''
    # Tridiagonal matrix solver
    phi = np.zeros(N)

    # Forward Elemination
    for i in range(2, N - 1):
        const1 = a[i - 1] / d[i - 1]
        d[i] = d[i] - const1 * c[i - 1]
        c[i] = c[i] - const1 * f[i - 1]
        Q[i] = Q[i] - const1 * Q[i - 1]

        const2 = e[i - 1] / d[i - 1]
        a[i] = a[i] - const2 * c[i - 1]
        d[i + 1] = d[i + 1] - const2 * f[i - 1]
        Q[i + 1] = Q[i + 1] - const2 * d[i - 1]
    # Solve last equation
    const3 = a[N - 2] / d[N - 2]
    d[N - 1] = d[N - 1] - const3 * c[N - 2]
    phi[N - 1] = (Q[N - 1] - const3 * Q[N - 2]) / d[N - 1]
    phi[N - 2] = [Q[N - 2] - c[N - 2] * Q[N - 1]] / d[N - 2]

    # Backward Elemination
    for i in range(N - 3, -1, -1):
        phi[i] = (Q[i] - c[i] * phi[i + 1] - f[i] * phi[i + 2]) / d[i]

    return phi


# numbering scheme used is k = (j-1)*N + i
start = time.time()
Nx = 41
Ny = 41
L, H = 1, 1  # Computational domain
phi = np.zeros((Nx, Ny))
S = np.zeros((Nx, Ny))

dx = L / (Nx - 1)  # Grid size
dy = H / (Ny - 1)  # Grid size

tol = 1e-6

phi, S = prepare_phi_and_S(Nx, Ny, phi, L, H)

high_acc = True
for _ in tqdm(range(100000)):
    '''
    [A] [Phi]= [Q]
    d, a, c are diagonal terms for PDMA solver
    For row-wise sweep, the PDMA will take current j-th row in [A], then assemble the coefficients.
    Since start with 2nd row, the BCs for LHS and RHS is always 'felt' by PDMA. Thus, d[0]=1, Q_x[0]=Phi_left, d[Nx-1]=1. Q_x[Nx-1]=Phi_right
    The every row, solve the matrix showed below. And update new Phi for that j-th row. Phi[:,j] = Phi_x
    ┌                                        ┐
    │  d1  c1  0 ...                        0│
    │  a1  d2  c2  0 ...                    0│
    │           .                            │
    │                .                       │
    │                    .                   │┌     ┐  ┌     ┐
    │...         0   ai-2  di-1  ci-1 0  ... ││Phi_x│= | Q_x |
    │                          .             │└     ┘  └     ┘
    │                            .           │
    │                               .        │
    │0     ... 0              aN-2  dN-1 cN-1│
    │0                       ... 0   aN-1  dN│ 
    └                                        ┘
    '''

    # Row-wise sweep
    # Start with initialize the coefficients for PDMA
    c = np.zeros(Nx - 1)
    a = np.zeros(Nx - 1)
    f = np.zeros(Nx - 2)
    e = np.zeros(Nx - 2)
    d = np.zeros(Nx)
    Q_x = np.zeros(Nx)
    for j in range(1, Ny - 1):

        # For Dirichlet BC
        d[0] = 1

        # Left BC
        Q_x[0] = phi[0, j]

        for i in range(1, Nx - 1):

            # If need 4th-order acc
            if high_acc:

                # Since higher order approximation is special at one node after BC, as shown below with square dot.
                # The pratical way to deal with that inner node is back to 2nd order approximation as before
                '''
                ┌                                  ┐
                │ . .  .  .  .  .  .  .  .  .  .  .│
                │ . ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  .│
                │ . ■                          ■  .│
                │ . ■                          ■  .│
                │ . ■                          ■  .│
                │ . ■                          ■  .│
                │ . ■                          ■  .│    
                │ . ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  .│
                │ . .  .  .  .  .  .  .  .  .  .  .│
                └                                  ┘
                '''

                # If at one node after BC node
                if i == 1 or i == Nx - 2 or j == 1 or j == Ny - 2:

                    # PDMA Coefficients
                    d[i] = -(2 / dx ** 2 + 2 / dy ** 2)
                    a[i - 1] = 1 / dx ** 2
                    c[i] = 1 / dx ** 2

                    # RHS for PDMA matrix
                    Q_x[i] = S[i, j] - phi[i, j + 1] * (1 / dy ** 2) - phi[i, j - 1] * (1 / dy ** 2)

                else:
                    # PDMA Coefficients with higher order approximation
                    d[i] = -(5 / (2 * dx ** 2) + 5 / (2 * dy ** 2))
                    a[i - 1] = 4 / (3 * dx ** 2)
                    c[i] = 4 / (3 * dx ** 2)
                    f[i] = -1 / (12 * dx ** 2)
                    e[i - 2] = -1 / (12 * dx ** 2)
                    # RHS for PDMA matrix with higher order approximation
                    Q_x[i] = S[i, j] + ((phi[i, j + 2] - 4 * phi[i, j + 1] - 4 * phi[i, j - 1] + phi[i, j - 2])) / (
                                12 * dy ** 2) - (phi[i, j + 1] + phi[i, j - 1]) / dy ** 2

            # Normal 2nd order acc
            else:
                d[i] = -(2 / dx ** 2 + 2 / dy ** 2)
                a[i - 1] = 1 / dx ** 2
                c[i] = 1 / dx ** 2

                # RHS for Tri matrix
                Q_x[i] = S[i, j] - phi[i, j + 1] * (1 / dx ** 2) - phi[i, j - 1] * (1 / dx ** 2)

        # For Dirichlet BC
        d[Nx - 1] = 1

        # Right BC
        Q_x[Nx - 1] = phi[-1, j]

        # Solve for current row
        phix = PEN(Nx, e, a, d, c, f, Q_x)

        # Update current new row solution
        phi[:, j] = phix

    # Calculate residual
    R2 = 0
    R = np.zeros((Nx, Ny))
    R_TRI_logger = []
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            if high_acc:
                if i == 1 or i == Nx - 2 or j == 1 or j == Ny - 2:
                    aE = 1 / dx ** 2
                    aW = 1 / dx ** 2
                    aN = 1 / dy ** 2
                    aS = 1 / dy ** 2
                    a0 = -(2 / dx ** 2 + 2 / dy ** 2)
                    R[i, j] = S[i, j] - aE * phi[i + 1, j] - aW * phi[i - 1, j] - aN * phi[i, j + 1] - aS * phi[
                        i, j - 1] - a0 * phi[i, j]
                    R2 = R2 + R[i, j] ** 2

                else:
                    # aE = 4/(3*dx**2)
                    # aEE = 1/(12*dx**2)

                    # aW = 4/(3*dx**2)
                    # aWW = 1/(12*dx**2)

                    # aN = -4/(12*dy**2) - 1/dy**2
                    # aNN = 1/(12*dy**2)

                    # aS = -4/(12*dy**2) - 1/dy**2
                    # aSS = 1/(12*dy**2)

                    a0 = -(5 / (2 * dx ** 2) + 5 / (2 * dy ** 2))
                    R[i, j] = S[i, j] + (1 / (12 * dy ** 2)) * (
                                phi[i, j + 2] - 4 * phi[i, j + 1] - 4 * phi[i, j - 1] + phi[i, j - 2]) - (
                                          phi[i, j + 1] + phi[i, j - 1]) / dy ** 2 + (phi[i + 2, j] + phi[i - 2, j]) / (
                                          12 * dx ** 2) - a0 * phi[i, j] - 4 / (3 * dx ** 2) * phi[i + 1, j] - 4 / (
                                          3 * dx ** 2) * phi[i - 1, j]
                    R2 = R2 + R[i, j] ** 2

            else:
                aE = 1 / dx ** 2
                aW = 1 / dx ** 2
                aN = 1 / dy ** 2
                aS = 1 / dy ** 2
                a0 = -(2 / dx ** 2 + 2 / dy ** 2)
                R[i, j] = S[i, j] - aE * phi[i + 1, j] - aW * phi[i - 1, j] - aN * phi[i, j + 1] - aS * phi[
                    i, j - 1] - a0 * phi[i, j]
                R2 = R2 + R[i, j] ** 2
    R2 = np.sqrt(R2)
    R_TRI_logger.append(R2)
    if _ % 50 == 0:
        clear_output(True)
        print("Residual: ", R2)

    if R2 < tol:
        print('Converged! Residual: ', R2, 'Time elapsed: ', time.time() - start)
        break

