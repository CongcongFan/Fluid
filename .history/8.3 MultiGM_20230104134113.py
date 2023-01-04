import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import time
from scipy import interpolate

plt.rcParams.update({'font.size': 14})

from utilities import error, prepare_phi_and_S, convert1D_to_2D, plot_phi, residual


def GS(Nx, Ny, phi, S, aE, aW, aN, aS, a0):
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            # Gauss-Siedel Update
            phi[i, j] = (S[i, j] - aE * phi[i + 1, j] - aW * phi[i - 1, j] - aN * phi[i, j + 1] - aS * phi[
                i, j - 1]) / a0
    return phi



# The name of variable ends with c and f are corse and fine mesh
start = time.time()
Nx1 = 81
Ny1 = 81
Nx2 = 41
Ny2 = 41
Nx3 = 21
Ny3 = 21
L = 1  # length
H = 1  # length

phi1 = np.zeros((Nx1, Ny1))
phi2 = np.zeros((Nx2, Ny2))
phi3 = np.zeros((Nx3, Ny3))

dx1 = L / (Nx1 - 1)  # Grid size
dy1 = L / (Ny1 - 1)  # Grid size
x1_list = np.arange(0, 1 + dx1, dx1)
y1_list = np.arange(0, 1 + dy1, dy1)
x1, y1 = np.meshgrid(x1_list, y1_list, indexing='ij')

dx2 = L / (Nx2 - 1)  # Grid size
dy2 = L / (Ny2 - 1)  # Grid size
x2_list = np.arange(0, 1 + dx2, dx2)
y2_list = np.arange(0, 1 + dy2, dy2)
x2, y2 = np.meshgrid(x2_list, y2_list, indexing='ij')

dx3 = L / (Nx3 - 1)  # Grid size
dy3 = L / (Ny3 - 1)  # Grid size
x3_list = np.arange(0, 1 + dx3, dx3)
y3_list = np.arange(0, 1 + dy3, dy3)
x3, y3 = np.meshgrid(x3_list, y3_list, indexing='ij')

tolf = 1e-6
tolc = 1e-1

aE1 = 1 / dx1 ** 2
aW1 = 1 / dx1 ** 2
aN1 = 1 / dy1 ** 2
aS1 = 1 / dy1 ** 2
a01 = -(2 / dx1 ** 2 + 2 / dy1 ** 2)

aE2 = 1 / dx2 ** 2
aW2 = 1 / dx2 ** 2
aN2 = 1 / dy2 ** 2
aS2 = 1 / dy2 ** 2
a02 = -(2 / dx2 ** 2 + 2 / dy2 ** 2)

aE3 = 1 / dx3 ** 2
aW3 = 1 / dx3 ** 2
aN3 = 1 / dy3 ** 2
aS3 = 1 / dy3 ** 2
a03 = -(2 / dx3 ** 2 + 2 / dy3 ** 2)

phi1, S1 = prepare_phi_and_S(Nx1, Ny1, phi1, L, H)
R2_1_old, _, _ = residual(Nx1, Ny1, phi1, S1, aE1, aW1, aN1, aS1, a01, convert=False)

"""
Grid1 --> Grid2 --> Grid3 ---> Grid 2 correction by using Grid3 ---> Grid 1 correction
"""
for l in tqdm(range(10000)):

    # Partial Solve [A1][𝜙1] = [Q]
    phi1 = GS(Nx1, Ny1, phi1, S1, aE1, aW1, aN1, aS1, a01)

    # Compute [R_1] = [Q] - [A1][𝜙1]
    R2_1, _, R_1_new = residual(Nx1, Ny1, phi1, S1, aE1, aW1, aN1, aS1, a01, convert=False)

    # Transfer Residual to 2nd grid
    # Since in current 2 mesh size, there is always a corse mesh sitting on the top of fine mesh
    # But can use interpolation

    # [R_1] ==> [R_2<-1]
    f = interpolate.RectBivariateSpline(x1_list, y1_list, R_1_new)
    R_2_new = f(x2_list, y2_list)

    # Smoothing the errors on the 2nd grid
    # Use the correction form [A2][𝜙2'] = [R_2<-1] to calculate the correction form of 𝜙 for 2nd grid

    # Again, use GS to partial solve [A2][𝜙2'] = [R_2<-1]
    phi2 = np.zeros((Nx2, Ny2))
    phi2 = GS(Nx2, Ny2, phi2, R_2_new, aE2, aW2, aN2, aS2, a02)

    # Compute [R_2] = [R_2<-1] - [A2][𝜙'2]
    R2_2, _, R_2_new = residual(Nx2, Ny2, phi2, R_2_new, aE2, aW2, aN2, aS2, a02, convert=False)

    # Transfer Residual to 3rd grid
    # [R_3] ==> [R_3<-2]
    f = interpolate.RectBivariateSpline(x2_list, y2_list, R_2_new)
    R_3_new = f(x3_list, y3_list)

    # Smoothing the errors on the 3rd grid
    # Use the correction form [A3][𝜙3'] = [R_3<-2] to calculate the correction form of 𝜙 for 3rd grid

    # Again, use GS to partial solve [A3][𝜙3'] = [R_3<-2]
    phi3 = np.zeros((Nx3, Ny3))
    phi3 = GS(Nx3, Ny3, phi3, R_3_new, aE3, aW3, aE3, aS3, a03)

    # Compute [R_3] = [R_3<-2] - [A3][𝜙'3]
    R2_3, _, R_3_new = residual(Nx3, Ny3, phi3, R_3_new, aE3, aW3, aN3, aS3, a03, convert=False)

    # Up stage in a V-cycle
    f = interpolate.RectBivariateSpline(x3_list, y3_list, phi3)
    phi2_corrector = f(x2_list, y2_list)
    phi2 = phi2 + phi2_corrector

    f = interpolate.RectBivariateSpline(x2_list, y2_list, phi2)
    phi1_corrector = f(x1_list, y1_list)
    phi1 = phi1 + phi1_corrector

    if l % 20 == 0:
        clear_output(True)
        print('Residual on Fine mesh:', R2_1, 'Residual on Coarse mesh:', R2_2)

    if R2_1 < tolf:
        print('Converged! Residual: ', R2_1, 'Time elapsed: ', time.time() - start)
        break
