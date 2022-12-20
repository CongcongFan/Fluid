import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import time
plt.rcParams.update({'font.size': 16})
def error(phi_a,phi_n):
    # compute error between analytical and numerical numbers

    return (phi_a-phi_n)/phi_a*100
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
M = 41
L = 1  # length
phi = np.zeros((N * M))
S = np.zeros((N * M))

dx = L / (N - 1)  # Grid size
dy = L / (M - 1)  # Grid size

tol = 1e-6

aE = 1 / dx ** 2
aW = 1 / dx ** 2
aN = 1 / dy ** 2
aS = 1 / dy ** 2
a0 = -(2 / dx ** 2 + 2 / dy ** 2)

# RHS source terms

for i in range(N):

    for j in range(M):
        x = i * dx
        y = j * dy
        k = (j - 1) * N + i
        S[k] = 2 * np.sinh(10 * (x - 0.5)) + 40 * (x - 0.5) * np.cosh(10 * (x - 0.5)) + 100 * (
                x - 0.5) ** 2 * np.sinh(10 * (x - 0.5)) + 2 * np.sinh(10 * (y - 0.5)) + 40 * (
                       y - 0.5) * np.cosh(10 * (y - 0.5)) + 100 * (y - 0.5) ** 2 * np.sinh(
            10 * (y - 0.5)) + 4 * (x ** 2 + y ** 2) * np.exp(2 * x * y)

## Right BC
i = N - 1
for j in range(1, M - 1):
    x = i * dx
    y = j * dy

    k = (j - 1) * N + i
    phi[k] = 0.25 * np.sinh(5) + (y - 0.5) ** 2 * np.sinh(10 * (y - 0.5)) + np.exp(2 * y)
    S[k] = phi[k]

## left BC
i = 0
for j in range(1, M - 1):
    x = i * dx
    y = j * dy
    k = (j - 1) * N + i
    phi[k] = 0.25 * np.sinh(-5) + (y - 0.5) ** 2 * np.sinh(10 * (y - 0.5)) + 1
    S[k] = phi[k]

## Bottom BC
j = 0
for i in range(N):
    x = i * dx
    y = j * dy
    k = (j - 1) * N + i
    phi[k] = 0.25 * np.sinh(-5) + (x - 0.5) ** 2 * np.sinh(10 * (x - 0.5)) + 1
    S[k] = phi[k]

## Top BC
j = N - 1
for i in range(N):
    x = i * dx
    y = j * dy
    k = (j - 1) * N + i
    phi[k] = 0.25 * np.sinh(5) + (x - 0.5) ** 2 * np.sinh(10 * (x - 0.5)) + np.exp(2 * x)
    S[k] = phi[k]

aE = 1 / dx ** 2
aW = 1 / dx ** 2
aN = 1 / dy ** 2
aS = 1 / dy ** 2
a0 = -(2 / dx ** 2 + 2 / dy ** 2)

A = matrixA(N)

# Initial residual
R = S - A @ phi
R2sum_old = 0
for i in range(N):

    for j in range(M):
        k = (j - 1) * N + i
        R2sum_old = R2sum_old + R[k] ** 2

R2_old = np.sqrt(R2sum_old)
# Step 3: Set the initial search direction vector equal to the residual vector
D = R

for _ in tqdm(range(100000)):

    # Compute new alpha

    alpha_new = R.T @ R / (D.T @ A @ D)

    phi = phi + alpha_new * D

    # Compute residual
    R = S - A @ phi

    R2sum_new = 0
    for i in range(N):

        for j in range(M):
            k = (j - 1) * N + i
            R2sum_new = R2sum_new + R[k] ** 2

    R2_new = np.sqrt(R2sum_new)

    beta = R2_new ** 2 / R2_old ** 2

    # Update search direction vector
    D = R + beta * D

    # Update old residual vector
    R2_old = R2_new

    if _ % 100 == 0:
        clear_output(True)
        print(f'alpha: {alpha}')
        print("Residual: ", R2)

    if R2 < tol:
        print('Converged! Residual: ', R2, 'Time elapsed: ', time.time() - start)
        break