# Collocation ("discrete dipole") approximation of the volume integral 
# equation for 3D acoustic scattering
import numpy as np
from geometry import shape
import time

geom = 'sphere'

aspectRatio = 1/10  # ratio of column's height to it's width
sizeParam = 25      # size parameter
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

from numba import jit, njit, prange
@njit(parallel=True)
def potential_fast(ko):
    toep = np.zeros((L, M, N), dtype=np.complex128)
    for i in prange(0, L):
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
    return toep

start = time.time()
toep = potential_fast(ko)
end = time.time()
print('Operator assembly time:', end-start)

toep = ko**2 * toep

start = time.time()
# Circulant embedding
from operators import circulant_embed
circ_op = circulant_embed(toep, L, M, N)
end = time.time()
print('Time for circulant embedding and FFT:', end-start)

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

# Solving the linear system
from scipy.sparse.linalg import LinearOperator, gmres
import time

A = LinearOperator((L*M*N, L*M*N), matvec=mvp)

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


start = time.time()
# 2-level circulant preconditioner for acoustic problem
from circulant_acoustic import circ_1_level_acoustic, circ_2_level_acoustic
_, circ_L_opToep = circ_1_level_acoustic(toep, L, M, N, 'off')
circ2, circ_M_opToep = circ_2_level_acoustic(circ_L_opToep, L, M, N)
end = time.time()
print('Circulant preconditioner construction (s):', end-start)

start = time.time()
# Invert preconditioner
# from numba import jit, prange

# @autojit
# def parallel_sum(A):
#     sum = 0.0
#     for i in prange(A.shape[0]):
#         sum += A[i]

#     return sum

# @jit(parallel=True, nopython=True)
# def do_inv_parallel(circ2):
#     circ2_inv = np.zeros_like(circ2)
#     for i in prange(0, L):
#         for j in range(0, M):
#             circ2_inv[i, j, :, :] = np.linalg.inv(np.identity(N) - (refInd**2 - 1)
#                                                 * circ2[i, j, :, :])
#     return circ2_inv

# circ2_inv = do_inv_parallel(circ2)

def processInput(i):
    inverse_blocks = np.zeros((M, N, N), dtype=np.complex128)
    for j in range(0, M):
            inverse_blocks[j, :, :] = np.linalg.inv(np.identity(N) - (refInd**2 - 1)
                                              * circ2[i, j, :, :])
    return inverse_blocks

from joblib import Parallel, delayed  
import multiprocessing
inputs = range(L)

num_cores = multiprocessing.cpu_count()

print("numCores = " + str(num_cores))

results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs) 
circ2_inv = np.asarray(results)

# circ2_inv = np.zeros_like(circ2)
# for i in range(0, L):
#     for j in range(0, M):
#         circ2_inv[i, j, :, :] = np.linalg.inv(np.identity(N) - (refInd**2 - 1)
#                                               * circ2[i, j, :, :])

end = time.time()
print('Preconditioner inversion (s):', end-start)

from circulant_acoustic import mvp_circ2_acoustic
mvp_prec = lambda x: mvp_circ2_acoustic(x, circ2_inv, L, M, N, idx)
prec = LinearOperator((L*M*N, L*M*N), matvec=mvp_prec)

#  Callback function to output the residual
resvec = []
def residual_vector(rk):
    global resvec
    resvec.append(rk)


start = time.time()
resvec = []
sol1, info1 = gmres(A, xInVec, M=prec, tol=1e-4, restart=500, callback=residual_vector)
end = time.time()
print("The linear system was solved in {0} iterations".format(len(resvec)))
print('Solve time = ', end-start,'s')
print('Relative residual = ',
      np.linalg.norm(mvp(sol1)-xInVec)/np.linalg.norm(xInVec))

# from mie_series_function import mie_function
# P = mie_function(sizeParam, refInd, L)

# idx_n = np.ones((L, M, N), dtype=bool)
# mvp_all = lambda x:mvp_domain(x, circ_op, idx_n, Mr)

# temp = mvp_all(sol1)

# Utemp = temp.reshape(L, M, N, order='F')
# U = Uinc - Utemp + J
# # U= Utemp
# U_centre = U[:, :, np.int(np.round(N/2))]

# error = np.linalg.norm(U_centre-np.conj(P)) / np.linalg.norm(P)
# print('Error = ', error)
