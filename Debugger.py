import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import time
plt.rcParams.update({'font.size': 16})
def error(phi_a,phi_n):
    # compute error between analytical and numerical numbers

    return (phi_a-phi_n)/phi_a*100

def stonesmethod(N, K, B, D, E, F, H):
    d, e, c, b, f = np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K)
    # Bbar,Cbar,Dbar,Ebar,Fbar,Gbar,Hbar = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    alpha = 1

    # Step 3
    d[0] = E[0]
    e[0] = F[0] / d[0]
    f[0] = H[0] / d[0]

    for k in range(1, K):
        if k > N:
            b[k] = B[k] / (1 + alpha * e[k - N])
        # print(k, K, c.shape)
        c[k] = D[k] / (1 + alpha * f[k - 1])

        if k > N:
            d[k] = E[k] + alpha * (b[k] * e[k - N] + c[k] * f[k - 1]) - b[k] * f[k - N] - c[k] * e[k - 1]
        else:
            d[k] = E[k] + alpha * (c[k] * f[k - 1]) - c[k] * e[k - 1]

        f[k] = (H[k] - alpha * c[k] * f[k - 1]) / d[k]

        if k > N:
            e[k] = (F[k] - alpha * b[k] * e[k - N]) / d[k]

        else:
            e[k] = F[k] / d[k]

    return b, c, d, e, f

def matrixA(N):
    A = np.zeros((N ** 2, N ** 2))

    ## Right BC
    i = N - 1
    for j in range(N):
        x = i * dx
        y = j * dy

        k = (j - 1) * N + i
        A[k, k] = 1

    ## left BC
    i = 0
    for j in range(N):
        x = i * dx
        y = j * dy
        k = (j - 1) * N + i
        A[k, k] = 1
    ## Bottom BC
    j = 0
    for i in range(N):
        x = i * dx
        y = j * dy
        k = (j - 1) * N + i
        A[k, k] = 1
    ## Top BC
    j = N - 1
    for i in range(N):
        x = i * dx
        y = j * dy
        k = (j - 1) * N + i
        A[k, k] = 1

    for i in range(1, N - 1):

        for j in range(1, N - 1):
            k = (j - 1) * N + i
            A[k, k] = -2 / dx ** 2 - 2 / dy ** 2
            A[k, k - 1] = 1 / dx ** 2
            A[k, k + 1] = 1 / dx ** 2

            A[k, k - N] = 1 / dy ** 2
            A[k, k + N] = 1 / dy ** 2

    return A

start = time.time()
N = 41
L = 1  # length
phi = np.zeros((N ** 2))
Q = np.zeros((N ** 2))

dx = L / (N - 1)  # Grid size
dy = L / (N - 1)  # Grid size

tol = 1e-3

aE = 1 / dx ** 2
aW = 1 / dx ** 2
aN = 1 / dy ** 2
aS = 1 / dy ** 2
a0 = -(2 / dx ** 2 + 2 / dy ** 2)

L = np.zeros((N ** 2, N ** 2))
U = np.zeros((N ** 2, N ** 2))

# RHS source terms
S = np.zeros((N, N))
for i in range(N):

    for j in range(N):
        x = i * dx
        y = j * dy

        S[i, j] = 2 * np.sinh(10 * (x - 0.5)) + 40 * (x - 0.5) * np.cosh(10 * (x - 0.5)) + 100 * (
                x - 0.5) ** 2 * np.sinh(10 * (x - 0.5)) + 2 * np.sinh(10 * (y - 0.5)) + 40 * (
                          y - 0.5) * np.cosh(10 * (y - 0.5)) + 100 * (y - 0.5) ** 2 * np.sinh(
            10 * (y - 0.5)) + 4 * (x ** 2 + y ** 2) * np.exp(2 * x * y)
K = N**2
B = np.full(K, aS)
D = np.full(K, aW)
E = np.full(K, a0)
F = np.full(K, aE)
H = np.full(K, aN)
## Right BC
i = N - 1
for j in range(1,N-1):
    x = i * dx
    y = j * dy

    k = (j - 1) * N + i
    phi[k] = 0.25 * np.sinh(5) + (y - 0.5) ** 2 * np.sinh(10 * (y - 0.5)) + np.exp(2 * y)
    Q[k] = phi[k]
    L[k, k] = 1
    U[k, k] = 1
    B[k] = 0  # South
    D[k] = 0  # West
    E[k] = 1  # O
    F[k] = 0  # East
    H[k] = 0  # North

## left BC
i = 0
for j in range(1,N-1):
    x = i * dx
    y = j * dy
    k = (j - 1) * N + i
    phi[k] = 0.25 * np.sinh(-5) + (y - 0.5) ** 2 * np.sinh(10 * (y - 0.5)) + 1
    Q[k] = phi[k]
    L[k, k] = 1
    U[k, k] = 1
    B[k] = 0  # South
    D[k] = 0  # West
    E[k] = 1  # O
    F[k] = 0  # East
    H[k] = 0  # North
## Bottom BC
j = 0
for i in range(N):
    x = i * dx
    y = j * dy
    k = (j - 1) * N + i
    phi[k] = 0.25 * np.sinh(-5) + (x - 0.5) ** 2 * np.sinh(10 * (x - 0.5)) + 1
    Q[k] = phi[k]
    L[k, k] = 1
    U[k, k] = 1
    B[k] = 0  # South
    D[k] = 0  # West
    E[k] = 1  # O
    F[k] = 0  # East
    H[k] = 0  # North
## Top BC
j = N - 1
for i in range(N):
    x = i * dx
    y = j * dy
    k = (j - 1) * N + i
    phi[k] = 0.25 * np.sinh(5) + (x - 0.5) ** 2 * np.sinh(10 * (x - 0.5)) + np.exp(2 * x)
    Q[k] = phi[k]
    L[k, k] = 1
    U[k, k] = 1
    B[k] = 0  # South
    D[k] = 0  # West
    E[k] = 1  # O
    F[k] = 0  # East
    H[k] = 0  # North

aE = 1 / dx ** 2
aW = 1 / dx ** 2
aN = 1 / dy ** 2
aS = 1 / dy ** 2
a0 = -(2 / dx ** 2 + 2 / dy ** 2)


# Step
for i in range(1, N - 1):
    for j in range(1, N - 1):
        k = (j - 1) * N + i


        b, c, d, e, f = stonesmethod(N, k, B, D, E, F, H)

        # print(k)

        L[k, k] = d[k-1]
        L[k, k - 1] = c[k-1]
        L[k, k - N] = b[k-1]

        U[k, k] = 1
        U[k, k + 1] = e[k-1]
        U[k, k + N] = f[k-1]

R2 = 0
R = np.zeros(N ** 2)

A = matrixA(N)
# for _ in tqdm(range(100000)):
#     # Step 4
#     # Calculate residual
#     R2 = 0
#     for i in range(1, N - 1):

#         for j in range(1, N - 1):
#             x = i * dx
#             y = j * dy
#             k = (j - 1) * N + i
#             R[k] = Q[k] - aE * phi[k + 1] - aW * phi[k - 1] - aN * phi[k + N] - aS * phi[k - N] - a0 * phi[k]

#             R2 = R2 + R[k] ** 2
for _ in tqdm(range(100000)):
    # Step 4
    # Calculate residual
    R2 = 0
    R = Q-np.matmul(A,phi)
    for i in range(1, N - 1):

        for j in range(1, N - 1):
            x = i * dx
            y = j * dy
            k = (j - 1) * N + i


            R2 = R2 + R[k] ** 2
        # Step 5
    Y = np.matmul(np.linalg.inv(L),R)

    # Step 6
    delta = np.matmul(np.linalg.inv(U) , Y)

    phi += delta
    # print(delta)
    R2 = np.sqrt(R2)
    if _ % 1 == 0:
        clear_output(True)
        print("Residual: ", R2)

    if R2 < tol:
        print('Converged! Residual: ',R2, 'Time elapsed: ', time.time()-start)
        break