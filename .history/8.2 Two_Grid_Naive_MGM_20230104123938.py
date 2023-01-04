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


# numbering scheme used is k = (j-1)*N + i
# The name of variable ends with c and f are corse and fine mesh
start = time.time()
Nxf = 81
Nyf = 81
Nxc = 41
Nyc = 41
L = 1  # length
H = 1  # length

phif = np.zeros((Nxf, Nyf))
phic = np.zeros((Nxc, Nyc))

dxf = L / (Nxf - 1)  # Grid size
dyf = L / (Nyf - 1)  # Grid size
xf_list = np.arange(0, 1 + dxf, dxf)
yf_list = np.arange(0, 1 + dxf, dyf)
xf, yf = np.meshgrid(xf_list, yf_list, indexing='ij')

dxc = L / (Nxc - 1)  # Grid size
dyc = L / (Nyc - 1)  # Grid size
xc_list = np.arange(0, 1 + dxc, dxc)
yc_list = np.arange(0, 1 + dxc, dyc)
xc, yc = np.meshgrid(xc_list, yc_list, indexing='ij')

tolf = 1e-6
tolc = 1e-1

aEf = 1 / dxf ** 2
aWf = 1 / dxf ** 2
aNf = 1 / dyf ** 2
aSf = 1 / dyf ** 2
a0f = -(2 / dxf ** 2 + 2 / dyf ** 2)

aEc = 1 / dxc ** 2
aWc = 1 / dxc ** 2
aNc = 1 / dyc ** 2
aSc = 1 / dyc ** 2
a0c = -(2 / dxc ** 2 + 2 / dyc ** 2)

phif, Sf = prepare_phi_and_S(Nxf, Nyf, phif, L, H)
R2f_old, _, _ = residual(Nxf, Nyf, phif, Sf, aEf, aWf, aNf, aSf, a0f, convert=False)
R2f = 1e10
for l in tqdm(range(10000)):
    R2c = 1e10

    while R2f / R2f_old > 0.5:
        # One GS sweep on fine mesh

        phif = GS(Nxf, Nyf, phif, Sf, aEf, aWf, aNf, aSf, a0f)

        R2f, Rsumf, Rf_new = residual(Nxf, Nyf, phif, Sf, aEf, aWf, aNf, aSf, a0f, convert=False)
    R2f_old = R2f

    # Transfer Residual to corse mesh
    # Since in current 2 mesh size, there is always a corse mesh sitting on the top of fine mesh
    f = interpolate.RectBivariateSpline(xf_list, yf_list, Rf_new)
    Rc_new = f(xc_list, yc_list)

    phic = np.zeros((Nxc, Nyc))
    # Smoothing the errors on the coarse mesh
    # Use the correction form [A'][phi'] = [R] to calculate the correction form of phi
    for _ in range(5):
        phic = GS(Nxc, Nyc, phic, Rc_new, aEc, aWc, aNc, aSc, a0c)
        # phic += corrector
        R2c, _, _ = residual(Nxc, Nyc, phic, Rc_new, aEc, aWc, aNc, aSc, a0c, convert=False)

    # Transfer the correction form of phi at coarse mesh to finer mesh

    f = interpolate.RectBivariateSpline(xc_list, yc_list, phic)
    phif_corrector = f(xf_list, yf_list)

    phif = phif + phif_corrector

    if l % 20 == 0:
        clear_output(True)
        print('Residual on Fine mesh:', R2f, 'Residual on Coarse mesh:', R2c)

    if R2f < tolf:
        print('Converged! Residual: ', R2f, 'Time elapsed: ', time.time() - start)
        break
