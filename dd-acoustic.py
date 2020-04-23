# Collocation ("discrete dipole") approximation of the volume integral 
# equation for 3D acoustic scattering
import numpy as np
from geometry import shape

geom = 'sphere'

aspectRatio = 1/10  # ratio of column's height to it's width
sizeParam = 20      # size parameter
nPerLam = 10        # number of voxels per interior wavelength

# Refractive index of scatterer (real and imaginary parts)
refRe = 1.2
refIm = 0.0
refInd = refRe + 1j*refIm

# Polyhedral geometries
r, idx, res, P, lambda_ext, lambda_int = shape(geom, refInd, sizeParam,
                                               nPerLam, aspectRatio)


dx = r[1, 0, 0, 0] - r[0, 0, 0, 0]
vol = (dx)**3  # voxel volume
a = (3/4 * vol / np.pi)**(1/3)  # radius of sphere of same volume


(L, M, N, _) = r.shape  # number of voxels in x-, y-, z-directions
#  Arrays containing the x, y, z coordinates, these are handy for plotting the geometry
xd = r[:, :, :, 0]
yd = r[:, :, :, 1]
zd = r[:, :, :, 2]

from fields_acoustic import PlaneWave
Uo = 1       # amplitude
dInc = np.array((1, 0, 0))   # direction
ko = 2*np.pi/lambda_ext
Uinc = PlaneWave(Uo, ko, dInc, r)

# Voxel permittivities
Mr = np.zeros((L, M, N), dtype=np.complex128)
Mr[idx] = refInd**2 - 1

# Create Toeplitz operator
toep = np.zeros((L, M, N), dtype=np.complex128)
R0 = r[0, 0, 0, :]

# self = np.pi * 1j / 2 * (np.pi / 2 + ko * a)
# self = (np.exp(1j*ko*a)-1)/(1j*ko)
self = (1/ko**2 - 1j*a/ko) * np.exp(1j*ko*a) - 1/ko**2

# self = np.pi * 1j / 2 * (-np.pi / 2 + ko * a - 1/18*(ko * a)**3)
nearby_quad = 'off'
n_quad = 10
xG, wG = np.polynomial.legendre.leggauss(n_quad)
XG, YG, ZG = np.meshgrid(xG, xG, xG)
XW, YW, ZW = np.meshgrid(wG*0.5, wG*0.5, wG*0.5)

for i in range(0, L):
        for j in range(0, M):
            for k in range(0, N):
                R1 = r[i,j,k,:]
                rk_to_rj = R1-R0
                rjk = np.linalg.norm(rk_to_rj)
                if nearby_quad in 'on':
                    if rjk < 5 * dx and rjk > 1e-15:
                        x_grid = R1[0] + dx/2 * XG
                        y_grid = R1[1] + dx/2 * YG
                        z_grid = R1[2] + dx/2 * ZG

                        temp = 0.0+0.0j
                        for iQ in range(0, n_quad):
                            for jQ in range(0, n_quad):
                                for kQ in range(0, n_quad):
                                    RQ = np.array([x_grid[iQ, jQ, kQ],
                                         y_grid[iQ, jQ, kQ],z_grid[iQ, jQ, kQ]])

                                    rk_to_rj = RQ - R0

                                    rjk = np.linalg.norm(rk_to_rj)
                                    rjk_hat = rk_to_rj / rjk
                                    rjkrjk = np.outer(rjk_hat, rjk_hat)

                                    Ajk = np.exp(1j * ko * rjk) / (4 * np.pi * rjk) * dx**3
                                         # Draine & Flatau
                                    temp = temp + Ajk * XW[iQ, jQ, kQ] * YW[iQ, jQ, kQ] * ZW[iQ, jQ, kQ]
                        # from IPython import embed; embed()
                        toep[i, j, k] = temp
                    else:
                        if np.abs(rjk) > 1e-15:
                            toep[i, j, k] = \
                                np.exp(1j * ko * rjk) / (4 * np.pi * rjk) * dx**3
                        else:
                            toep[i, j, k] = self
                else:
                    if np.abs(rjk) > 1e-15:
                        toep[i, j, k] = \
                            np.exp(1j * ko * rjk) / (4 * np.pi * rjk) * dx**3
                    else:
                        toep[i, j, k] = self

# # Dense matrix assembly
# mat = np.zeros((L * M * N, L * M * N), dtype=np.complex128)
# r_ro = r.reshape(L*M*N, 3, order='F')
# for i in range(0, L*M*N):
#     R0 = r_ro[i, :]
#     for j in range(0, L*M*N):
#         R1 = r_ro[j, :]
#         rk_to_rj = R1-R0
#         rjk = np.linalg.norm(rk_to_rj)
#         if np.abs(rjk) > 1e-15:
#             mat[i, j] = \
#                 np.exp(1j * ko * rjk) / (4 * np.pi * rjk)
#         else:
#             mat[i, j] = self

toep = ko**2 * toep
# Circulant embedding
circ = np.zeros((2 * L, 2 * M, 2 * N), dtype=np.complex128)

circ[0:L, 0:M, 0:N] = toep
circ[0:L, 0:M, N+1:2*N] = toep[0:L, 0:M, -1:0:-1]
circ[0:L, M+1:2*M, 0:N] = toep[0:L, -1:0:-1, 0:N]
circ[0:L, M+1:2*M, N+1:2*N] = toep[0:L, -1:0:-1, -1:0:-1]
circ[L+1:2*L, 0:M, 0:N] = toep[-1:0:-1, 0:M, 0:N]
circ[L+1:2*L, 0:M, N+1:2*N] = toep[-1:0:-1, 0:M, -1:0:-1]
circ[L+1:2*L, M+1:2*M, 0:N] = toep[-1:0:-1, -1:0:-1, 0:N]
circ[L+1:2*L, M+1:2*M, N+1:2*N] = toep[-1:0:-1, -1:0:-1, -1:0:-1]

# FFT of circulant operator
circ_op = np.fft.fftn(circ)

# Matrix-vector product with Toeplitz operator
def mvp_vec(xIn, circ_op, idx, Mr):
    (L, M, N) = Mr.shape
    xInRO = xIn.reshape(L, M, N, order = 'F') 
    xInRO[np.invert(idx)] = 0.0

    xOut = np.zeros((L, M, N), dtype=np.complex128)
    xOutVec = np.zeros((L * M * N, 1), dtype=np.complex128)

    xFFT = np.fft.fftn(xInRO, [2 * L, 2 * M, 2 * N])
    Y = np.fft.ifftn(circ_op * xFFT)
    xPerm = Mr * Y[0:L, 0:M, 0:N]
    xOut = xInRO - xPerm
    xOut[np.invert(idx)] = 0.0
    xOutVec = xOut.reshape(L * M * N, 1, order='F')
    return xOutVec


xIn = np.zeros((L, M, N), dtype=np.complex128)
xIn[idx] = Uinc[idx]
xInVec = xIn.reshape((L*M*N, 1), order='F')

mvp = lambda x: mvp_vec(x, circ_op, idx, Mr)


from scipy.sparse.linalg import LinearOperator, gmres
import time

A = LinearOperator((L*M*N, L*M*N), matvec=mvp)

# A_dense = np.identity(L*M*N) - \
#     ko**2 * np.matmul(np.diag(Mr.reshape(L*M*N, 1, order='F')[:, 0]), mat)

it_count = 0


def iteration_counter(x):
    global it_count
    it_count += 1

start = time.time()
sol, info = gmres(A, xInVec, tol=1e-4, callback=iteration_counter)
print("The linear system was solved in {0} iterations".format(it_count))
end = time.time()
print('Solve time = ', end-start,'s')
print('Relative residual = ',
      np.linalg.norm(mvp(sol)-xInVec)/np.linalg.norm(xInVec))

J = sol.reshape(L, M, N, order='F')

# Evaluate solution in domain - requires on MVP
def mvp_domain(xIn, circ_op, idx, Mr):
    (L, M, N) = Mr.shape
    xInRO = xIn.reshape(L, M, N, order = 'F') 
    xInRO[np.invert(idx)] = 0.0

    xOut = np.zeros((L, M, N), dtype=np.complex128)
    xOutVec = np.zeros((L * M * N, 1), dtype=np.complex128)

    xFFT = np.fft.fftn(Mr * xInRO, [2 * L, 2 * M, 2 * N])
    Y = np.fft.ifftn(circ_op * xFFT)
    xPerm = Y[0:L, 0:M, 0:N]
    xOut = xInRO - xPerm
    xOut[np.invert(idx)] = 0.0
    xOutVec = xOut.reshape(L * M * N, 1, order='F')
    return xOutVec


idx_n = np.ones((L, M, N), dtype=bool)
mvp_all = lambda x:mvp_domain(x, circ_op, idx_n, Mr)

temp = mvp_all(sol)

Utemp = temp.reshape(L, M, N, order='F')
U = Uinc - Utemp + J
# U= Utemp
U_centre = U[:, :, np.int(np.round(N/2))]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 22})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig = plt.figure(figsize=(10, 8))
ax = fig.gca()

plt.imshow(np.real(U_centre).T,
           extent=[min(xd[:,0,0]) , max(xd[:,0,0]),
           min(yd[0,:,0]), max(yd[0,:,0])],
           cmap=plt.cm.get_cmap('RdBu_r'), interpolation='spline16')

circle2 = plt.Circle((0, 0), sizeParam/ko, color='black', fill=False)
ax.add_artist(circle2)

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
cbar = plt.colorbar()
# cbar.ax.set_ylabel('Pressure (MPa)')
fig.savefig('out.png')
plt.close()

# matplotlib.rcParams.update({'font.size': 22})
# fig = plt.figure(figsize=(10, 8))
# ax = fig.gca()

# plt.imshow(np.real(A_dense).T,
#            cmap=plt.cm.get_cmap('RdBu_r'), interpolation='spline16')
# plt.xlabel(r'$x$')
# plt.ylabel(r'$y$')
# cbar = plt.colorbar()
# # cbar.ax.set_ylabel('Pressure (MPa)')
# fig.savefig('mat.png')


from mie_series_function import mie_function
P = mie_function(sizeParam, refInd, L)

fig = plt.figure(figsize=(10, 8))
ax = fig.gca()

plt.imshow(np.abs(U_centre-np.conj(P)).T,
           cmap=plt.cm.get_cmap('RdBu_r'), interpolation='spline16')


plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
cbar = plt.colorbar()
# cbar.ax.set_ylabel('Pressure (MPa)')
fig.savefig('diff.png')
plt.close()
error = np.linalg.norm(U_centre-np.conj(P)) / np.linalg.norm(P)
print('Error = ', error)

# from IPython import embed; embed()