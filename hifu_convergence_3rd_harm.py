# Collocation ("discrete dipole") approximation of the volume integral 
# equation for 3D acoustic scattering
import numpy as np
from geometry import generatedomain
from IPython import embed;

k1 = 4225.410428500058 + 0.02498j
n_elements = 2**12
radius = 0.015  # 7.5cm
aperture_radius = 0.00 # 2cm
focal_length = 0.05  # 13cm
focus = [0., 0., 0.]
lam = 2 * np.pi / np.real(k1)

c0 = 1487.0              # wavespeed
# p0 = 1.35e4             # initial pressure amplitude
# p0 = 4.41e6
beta = 3.5e0
rho = 998

nPerLam = 20        # number of voxels per interior wavelength

res = lam / nPerLam

# Dimension of computation domain
x_start = -0.98 * np.sqrt(focal_length**2 - radius**2) # just to the right of bowl
x_end = 0.02
wx = x_end - x_start
wy = 2 * 0.881 * radius # slightly wider than bowl
wz = wy

import time
start = time.time()
r, L, M, N = generatedomain(res, wx, wy, wz)
# Adjust r
r[:, :, :, 0]  = r[:, :, :, 0] - r[0, 0, 0, 0] + x_start
end = time.time()
print('Mesh generation time:', end-start)
# embed()
points = r.reshape(L*M*N, 3, order='F')

from fields_acoustic import bowl_transducer
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

# Array to be populated with different harmonics evaluated on central axis
ny_centre = np.int(np.floor(M/2))
nz_centre = np.int(np.floor(N/2)) 
harmonics = np.zeros((4, L), dtype=np.complex128)

# First harmonic (i.e., incident field)
harmonics[0, :] = P[:, ny_centre, nz_centre]

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
fig.savefig('images/VIE_bowl_transducer_1p5cm.png')
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

# from operators import potential_fast
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
toep = potential_fast(ko)
end = time.time()
print('Operator assembly time:', end-start)
# toep = ko**2 * toep
toep = toep

start = time.time()
from operators import circulant_embed
circ_op = circulant_embed(toep, L, M, N)
end = time.time()
print('Time for circulant embedding and FFT:', end-start)

from matvecs_acoustic import mvp_vec


xIn = np.zeros((L, M, N), dtype=np.complex128)

P_vec = P.reshape(L * M * N, 1)
# rel_p = np.log10(np.abs(P_vec)/np.max(np.abs(P_vec)))
rel_p = np.log10(np.abs(P)/np.max(np.abs(P)))

xIn = P
xInVec = xIn.reshape((L*M*N, 1), order='F')

idx = np.ones((L, M, N), dtype=bool)
mvp = lambda x: mvp_vec(x, circ_op, idx, Mr)

# Voxel permittivities
Mr = np.ones((L, M, N), dtype=np.complex128)

start = time.time()
xOut = mvp(2 * beta * np.real(k1)**2 / (rho * c0**2) * xInVec * xInVec)
end = time.time()
print('Time for MVP:', end-start)
# from IPython import embed; embed()
field = xOut.reshape(L, M, N, order='F')

# Second harmonic
harmonics[1, :] = field[:, ny_centre, nz_centre]

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
matplotlib.rcParams.update({'font.size': 22})
# plt.rc('font', family='serif')
fig = plt.figure(figsize=(10, 10))
ax = fig.gca()
plt.imshow(np.abs(field[:, :, np.int(np.floor(N/2))].T),
           cmap=plt.cm.get_cmap('RdBu_r'), interpolation='spline16')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.colorbar()
fig.savefig('images/VIE_bowl_transducer_harm2_1p5cm.png')
plt.close()

#-------------- Next harmonic ----------------#
# Create volume potential to evaluate next harmonic
ko = 3 * k1
start = time.time()
toep = potential_fast(ko)
end = time.time()
print('Operator assembly time:', end-start)

start = time.time()
from operators import circulant_embed
circ_op = circulant_embed(toep, L, M, N)
end = time.time()
print('Time for circulant embedding and FFT:', end-start)

xIn = np.zeros((L, M, N), dtype=np.complex128)

xIn = field
xInVec2 = xIn.reshape((L*M*N, 1), order='F')

start = time.time()
xOut = mvp(9 * beta * np.real(k1)**2 / (rho * c0**2) * xInVec * xInVec2)
end = time.time()
print('Time for MVP:', end-start)
# from IPython import embed; embed()
field3 = xOut.reshape(L, M, N, order='F')

# Third harmonic
harmonics[2, :] = field3[:, ny_centre, nz_centre]

fig = plt.figure(figsize=(10, 10))
ax = fig.gca()
plt.imshow(np.abs(field3[:, :, np.int(np.floor(N/2))].T),
           cmap=plt.cm.get_cmap('RdBu_r'), interpolation='spline16')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.colorbar()
fig.savefig('images/VIE_bowl_transducer_harm3_1p5cm.png')
plt.close()


#-------------- Next harmonic ----------------#
# Create volume potential to evaluate next harmonic
ko = 4 * k1
start = time.time()
toep = potential_fast(ko)
end = time.time()
print('Operator assembly time:', end-start)

start = time.time()
from operators import circulant_embed
circ_op = circulant_embed(toep, L, M, N)
end = time.time()
print('Time for circulant embedding and FFT:', end-start)

xIn = np.zeros((L, M, N), dtype=np.complex128)

xIn = field3
xInVec3 = xIn.reshape((L*M*N, 1), order='F')

start = time.time()
xOut = mvp(8 * beta * np.real(k1)**2 / (rho * c0**2) * \
        (xInVec2 * xInVec2 + 2 * xInVec * xInVec3))
end = time.time()
print('Time for MVP:', end-start)
# from IPython import embed; embed()
field4 = xOut.reshape(L, M, N, order='F')

# Third harmonic
harmonics[3, :] = field4[:, ny_centre, nz_centre]

fig = plt.figure(figsize=(10, 10))
ax = fig.gca()
plt.imshow(np.abs(field4[:, :, np.int(np.floor(N/2))].T),
           cmap=plt.cm.get_cmap('RdBu_r'), interpolation='spline16')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.colorbar()
fig.savefig('images/VIE_bowl_transducer_harm4_1p5cm.png')
plt.close()

# Plot harmonics along central axis
x_line = (r[:, ny_centre, nz_centre, 0] + 0.05) * 100
fig = plt.figure(figsize=(14, 8))
ax = fig.gca()
plt.plot(x_line, np.abs(harmonics[0,:])/1e6,'k-', linewidth=2)
plt.plot(x_line, np.abs(harmonics[1,:])/1e6,'r-', linewidth=2)
plt.plot(x_line, np.abs(harmonics[2,:])/1e6,'b-', linewidth=2)
plt.plot(x_line, np.abs(harmonics[3,:])/1e6,'g-', linewidth=2)
plt.grid(True)
plt.xlim([1, 7])
plt.ylim([0, 9])
plt.xlabel(r'Axial distance (cm)')
plt.ylabel(r'Pressure (MPa)')
fig.savefig('images/VIE_harms_axis.png')
plt.close()

# np.savez('results/VIE_harms_1p5cm.npy', harms=harmonics, x_line=x_line)
np.save('results/VIE_harms_1p5cm.npy', np.array([harmonics, x_line]))


embed()