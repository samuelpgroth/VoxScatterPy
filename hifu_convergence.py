# Collocation ("discrete dipole") approximation of the volume integral 
# equation for 3D acoustic scattering
import numpy as np
from geometry import generatedomain
from IPython import embed;

k1 = 4225.410428500058 + 0.02498j
n_elements = 2**11
radius = 0.03  # 7.5cm
aperture_radius = 0.00 # 2cm
focal_length = 0.05  # 13cm
focus = [0., 0., 0.]
lam = 2 * np.pi / np.real(k1)

c0 = 1487.0              # wavespeed
# p0 = 1.35e4             # initial pressure amplitude
# p0 = 4.41e6
beta = 3.5e0
rho = 998

def bowl_transducer(k, focal_length, focus, radius,
                    n_elements, aperture_radius, points,
                    axis):
    import numpy as np
    theta1 = np.arcsin(aperture_radius / focal_length)
    theta2 = np.arcsin(radius / focal_length)

    r = 1.0  # radius of the sphere
    n_count = 0
    a = 2 * np.pi * r**2 * (np.cos(theta1) - np.cos(theta2)) / n_elements
    d = np.sqrt(a)
    M_theta = np.int(np.round((theta2 - theta1) / d))
    d_theta = (theta2 - theta1) / M_theta
    d_phi = a / d_theta
    x = []
    y = []
    z = []
    for m in range(0, M_theta):
        theta = (theta2 - theta1) * (m + 0.5) / M_theta + theta1
        M_phi = np.int(np.round(2 * np.pi * np.sin(theta) / d_phi))
        for n in range(0, M_phi):
            phi = 2 * np.pi * n / M_phi
            x.append(focal_length * np.sin(theta) * np.cos(phi))
            y.append(focal_length * np.sin(theta) * np.sin(phi))
            z.append(focal_length * np.cos(theta))
            n_count += 1

    if axis in 'z':
        x = np.array(x)
        y = np.array(y)
        z = focus[2] - np.array(z)
    elif axis in 'x':
        x_t = np.array(x)
        y_t = np.array(y)
        z_t = focus[2] - np.array(z)
        z = -y_t
        y = -x_t
        x = z_t

    def eval_source_vec(x, y, z, points, k):
        # p = np.zeros_like(points[0])
        p = 0.0
        for i in range(x.shape[0]):
            dist = np.sqrt((points[0] - x[i])**2 + (points[1] - y[i])**2 +
                           (points[2] - z[i])**2)
            p += np.exp(1j * k * dist) / (4 * np.pi * dist)
        return p

    from numba import jit, njit, prange
    @njit(parallel=True)
    def eval_source(x, y, z, points, k):
        p = np.zeros_like(points[0], dtype=np.complex128)
        # from IPython import embed; embed()
        for i in prange(points.shape[1]):
            temp = 0.0
            for j in range(x.shape[0]):
                dist = np.sqrt((points[0, i] - x[j])**2 +
                               (points[1, i] - y[j])**2 +
                               (points[2, i] - z[j])**2)
                temp += np.exp(1j * k * dist) / (4 * np.pi * dist)
            p[i] = temp
        return p


    p = eval_source(x, y, z, points, k)
    # p = eval_source_vec(x, y, z, points, k)

    return x, y, z, p*a

nPerLam = 5        # number of voxels per interior wavelength

res = lam / nPerLam

# Dimension of computation domain
x_start = -0.98 * np.sqrt(focal_length**2 - radius**2) # just to the right of bowl
x_end = 0.02
wx = x_end - x_start
wy = 2 * 0.881 * radius # slightly wider than bowl
wz = wy

# # Dimensions of domain (cuboid)
# wx = 40 * lam
# wy = 30.1 * lam
# wz = 30.1 * lam

import time
start = time.time()
r, L, M, N = generatedomain(res, wx, wy, wz)
# Adjust r
r[:, :, :, 0]  = r[:, :, :, 0] - r[0, 0, 0, 0] + x_start
end = time.time()
print('Mesh generation time:', end-start)
# embed()
points = r.reshape(L*M*N, 3, order='F')


start = time.time()
x, y, z, p = bowl_transducer(k1, focal_length, focus, radius,
                    n_elements, aperture_radius, points.T,
                    'x')
end = time.time()

# Normalise incident field to achieve desired total acoutic power
n_quad = 500
r_quad_dim = radius * 1.0
r_quad = np.linspace(0, r_quad_dim, n_quad)
x_location_disk = x_start
points_quad = np.vstack((x_location_disk * np.ones(n_quad),
                         np.zeros(n_quad),
                         r_quad))

_, _, _, p_quad = bowl_transducer(np.real(k1), focal_length, focus, radius,
                    n_elements, aperture_radius, points_quad,'x')
integral = 2*np.pi*np.sum(np.abs(p_quad)**2 * r_quad)*r_quad_dim/n_quad 
p0 = np.sqrt(2*rho*c0*50/integral)
# Normalise field
p *= p0
P = p.reshape(L, M, N, order='F')
print('Incident field evaluation time (s):', end-start)



import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
matplotlib.rcParams.update({'font.size': 22})
# plt.rc('font', family='serif')
fig = plt.figure(figsize=(10, 10))
ax = fig.gca()
plt.imshow(np.abs(P[:, :, np.int(np.floor(N/2))].T),
           cmap=plt.cm.get_cmap('RdBu_r'), interpolation='spline16')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.colorbar()
fig.savefig('images/VIE_bowl_transducer.png')
plt.close()
# embed()
# Generate volume potential operator for 2xk
dx = r[1, 0, 0, 0] - r[0, 0, 0, 0]
vol = (dx)**3  # voxel volume
a = (3/4 * vol / np.pi)**(1/3)  # radius of sphere of same volume

ko = 2 * k1
# Create Toeplitz operator
R0 = r[0, 0, 0, :]

self = (1/ko**2 - 1j*a/ko) * np.exp(1j*ko*a) - 1/ko**2

nearby_quad = 'off'
n_quad = 10
xG, wG = np.polynomial.legendre.leggauss(n_quad)
XG, YG, ZG = np.meshgrid(xG, xG, xG)
XW, YW, ZW = np.meshgrid(wG*0.5, wG*0.5, wG*0.5)

# @njit(parallel=True)
# def eval_source(x, y, z, points, k):
#     p = np.zeros_like(points[0], dtype=np.complex128)
#     # from IPython import embed; embed()
#     for i in prange(points.shape[1]):
#         temp = 0.0
#         for j in range(x.shape[0]):
#             dist = np.sqrt((points[0, i] - x[j])**2 +
#                             (points[1, i] - y[j])**2 +
#                             (points[2, i] - z[j])**2)
#             temp += np.exp(1j * k * dist) / (4 * np.pi * dist)
#         p[i] = temp
#     return p

from numba import jit, njit, prange
@njit(parallel=True)
def potential_fast():
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
    return toep

def potential():
    toep = np.zeros((L, M, N), dtype=np.complex128)
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
    return toep


start = time.time()
toep = potential_fast()

# for i in range(0, L):
#         for j in range(0, M):
#             for k in range(0, N):
#                 R1 = r[i,j,k,:]
#                 rk_to_rj = R1-R0
#                 rjk = np.linalg.norm(rk_to_rj)
#                 if nearby_quad in 'on':
#                     if rjk < 5 * dx and rjk > 1e-15:
#                         x_grid = R1[0] + dx/2 * XG
#                         y_grid = R1[1] + dx/2 * YG
#                         z_grid = R1[2] + dx/2 * ZG

#                         temp = 0.0+0.0j
#                         for iQ in range(0, n_quad):
#                             for jQ in range(0, n_quad):
#                                 for kQ in range(0, n_quad):
#                                     RQ = np.array([x_grid[iQ, jQ, kQ],
#                                          y_grid[iQ, jQ, kQ],z_grid[iQ, jQ, kQ]])

#                                     rk_to_rj = RQ - R0

#                                     rjk = np.linalg.norm(rk_to_rj)
#                                     rjk_hat = rk_to_rj / rjk
#                                     rjkrjk = np.outer(rjk_hat, rjk_hat)

#                                     Ajk = np.exp(1j * ko * rjk) / (4 * np.pi * rjk) * dx**3
#                                          # Draine & Flatau
#                                     temp = temp + Ajk * XW[iQ, jQ, kQ] * YW[iQ, jQ, kQ] * ZW[iQ, jQ, kQ]
#                         # from IPython import embed; embed()
#                         toep[i, j, k] = temp
#                     else:
#                         if np.abs(rjk) > 1e-15:
#                             toep[i, j, k] = \
#                                 np.exp(1j * ko * rjk) / (4 * np.pi * rjk) * dx**3
#                         else:
#                             toep[i, j, k] = self
#                 else:
#                     if np.abs(rjk) > 1e-15:
#                         toep[i, j, k] = \
#                             np.exp(1j * ko * rjk) / (4 * np.pi * rjk) * dx**3
#                     else:
#                         toep[i, j, k] = self
end = time.time()
print('Operator assembly time:', end-start)
# toep = ko**2 * toep
toep = toep

start = time.time()
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
    xOut = xPerm
    xOut[np.invert(idx)] = 0.0
    xOutVec = xOut.reshape(L * M * N, 1, order='F')
    return xOutVec


xIn = np.zeros((L, M, N), dtype=np.complex128)
ny_centre = np.int(np.floor(M/2))
nz_centre = np.int(np.floor(N/2)) 
print(ny_centre, nz_centre)

P_vec = P.reshape(L * M * N, 1)
# rel_p = np.log10(np.abs(P_vec)/np.max(np.abs(P_vec)))
rel_p = np.log10(np.abs(P)/np.max(np.abs(P)))


# TOL = np.array([-0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75, -2])
# line_harmonic = np.zeros((TOL.shape[0], L), dtype=np.complex128) 

# for i_tol in range(TOL.shape[0]):
#     # idx_set_to_zero = np.abs(r[:, :, :, 0]) >= wx/2 - x_subtract[i_x] * lam
#     where_bigger = np.argwhere(rel_p > TOL[i_tol])
#     min_x_idx = np.min(where_bigger[:, 0])
#     max_x_idx = np.max(where_bigger[:, 0])
#     min_y_idx = np.min(where_bigger[:, 1])
#     max_y_idx = np.max(where_bigger[:, 1])
#     min_z_idx = np.min(where_bigger[:, 2])
#     max_z_idx = np.max(where_bigger[:, 2])

#     P_trim = np.zeros((L, M, N), dtype=np.complex128)
#     P_trim[min_x_idx:max_x_idx, min_y_idx:max_y_idx, min_z_idx:max_z_idx] = \
#         P[min_x_idx:max_x_idx, min_y_idx:max_y_idx, min_z_idx:max_z_idx]
    
#     xIn = P_trim
#     xInVec = xIn.reshape((L*M*N, 1), order='F')

#     idx = np.ones((L, M, N), dtype=bool)
#     mvp = lambda x: mvp_vec(x, circ_op, idx, Mr)

#     # Voxel permittivities
#     Mr = np.ones((L, M, N), dtype=np.complex128)

#     start = time.time()
#     xOut = mvp(2 * beta * k1**2 / (rho * c0**2) * xInVec * xInVec)
#     end = time.time()
#     print('Time for MVP:', end-start)
#     # from IPython import embed; embed()
#     field = xOut.reshape(L, M, N, order='F')
#     line = field[:, ny_centre, nz_centre]
#     line_harmonic[i_tol, :] = line

# # filename = 'results/axial_x60lam_yz20lam_nPerLam10.npy'
# # filename = 'results/axial_x40lam_vary_yz30lam_nPerLam20.npy'
# filename = 'results/axial_vary_domain_size_log.npy'
# np.save(filename, line_harmonic)



X_SUB = np.array([0.01])#, 0.02, 0.03, 0.04, 0.05, 0.06])
line_harmonic = np.zeros((X_SUB.shape[0], L), dtype=np.complex128) 

for i_tol in range(X_SUB.shape[0]):
    # idx_set_to_zero = np.abs(r[:, :, :, 0]) >= wx/2 - x_subtract[i_x] * lam
    where_trim = np.min(np.argwhere(r[:,0,0,0]>X_SUB[i_tol])) 

    P_trim = np.zeros((L, M, N), dtype=np.complex128)
    P_trim[0:where_trim, :, :] = \
        P[0:where_trim, :, :]
    print(where_trim)
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.gca()
    # plt.imshow(np.abs(P_trim[:, :, np.int(np.floor(N/2))].T),
    #         cmap=plt.cm.get_cmap('RdBu_r'), interpolation='spline16')
    # plt.xlabel(r'$x$')
    # plt.ylabel(r'$y$')
    # plt.colorbar()
    # fig.savefig('images/VIE_bowl_transducer_trim.png')
    # plt.close()
    # embed()
    xIn = P_trim
    xInVec = xIn.reshape((L*M*N, 1), order='F')

    idx = np.ones((L, M, N), dtype=bool)
    mvp = lambda x: mvp_vec(x, circ_op, idx, Mr)

    # Voxel permittivities
    Mr = np.ones((L, M, N), dtype=np.complex128)

    start = time.time()
    xOut = mvp(2 * beta * k1**2 / (rho * c0**2) * xInVec * xInVec)
    end = time.time()
    print('Time for MVP:', end-start)
    # from IPython import embed; embed()
    field = xOut.reshape(L, M, N, order='F')
    line = field[:, ny_centre, nz_centre]
    line_harmonic[i_tol, :] = line

# filename = 'results/axial_x60lam_yz20lam_nPerLam10.npy'
# filename = 'results/axial_x40lam_vary_yz30lam_nPerLam20.npy'
filename = 'results/axial_vary_domain_length.npy'
np.save(filename, line_harmonic)


# embed()
# n_line = 6
# x_subtract = (np.arange(n_line)) * 2.5
# line_harmonic = np.zeros((n_line, L), dtype=np.complex128) 
# for i_x in range(n_line):
#     # idx_set_to_zero = np.abs(r[:, :, :, 0]) >= wx/2 - x_subtract[i_x] * lam
#     idx_set_to_zero = np.logical_and(np.abs(r[:, :, :, 1]) >= wy/2 - x_subtract[i_x] * lam,
#                                      np.abs(r[:, :, :, 1]) >= wz/2 - x_subtract[i_x] * lam)
#     P[idx_set_to_zero] = 0.0
    
#     xIn = P
#     xInVec = xIn.reshape((L*M*N, 1), order='F')

#     idx = np.ones((L, M, N), dtype=bool)
#     mvp = lambda x: mvp_vec(x, circ_op, idx, Mr)

#     # Voxel permittivities
#     Mr = np.ones((L, M, N), dtype=np.complex128)

#     start = time.time()
#     xOut = mvp(2 * beta * k1**2 / (rho * c0**2) * xInVec * xInVec)
#     end = time.time()
#     print('Time for MVP:', end-start)
#     # from IPython import embed; embed()
#     field = xOut.reshape(L, M, N, order='F')
#     line = field[:, ny_centre, nz_centre]
#     line_harmonic[i_x, :] = line

# # filename = 'results/axial_x60lam_yz20lam_nPerLam10.npy'
# filename = 'results/axial_x40lam_vary_yz30lam_nPerLam20.npy'
# np.save(filename, line_harmonic)

# # Figure
# fig = plt.figure(figsize=(10, 10))
# ax = fig.gca()
# plt.imshow(np.abs(field[:, :, np.int(np.floor(N/2))].T),
#            cmap=plt.cm.get_cmap('RdBu_r'), interpolation='spline16')
# plt.xlabel(r'$x$')
# plt.ylabel(r'$y$')
# plt.colorbar()
# fig.savefig('images/VIE_next_harm.png')
# plt.close()

# total = P - field

# # Plot harmonics along central axis
# fig = plt.figure(figsize=(10, 10))
# ax = fig.gca()
# ny_centre = np.int(np.floor(M/2))
# nz_centre = np.int(np.floor(N/2)) 
# plt.plot(np.abs(P[:, ny_centre, nz_centre]))
# plt.plot(np.abs(field[:, ny_centre, nz_centre]))
# plt.plot(np.real(total[:, ny_centre, nz_centre]))
# plt.grid(True)
# fig.savefig('images/VIE_central_axis0_40lam_20lam.png')
# plt.close()

# line = field[:, ny_centre, nz_centre]
# np.save('axial_field_40lam_20lam_nPerLam_20.npy', line)