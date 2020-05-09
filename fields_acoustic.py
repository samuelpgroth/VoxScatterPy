def PlaneWave(Uo, k, dInc, r):
    import numpy as np

    (L, M, N, _) = r.shape

    krx = k * dInc[0] * r[:, :, :, 0]
    kry = k * dInc[1] * r[:, :, :, 1]
    krz = k * dInc[2] * r[:, :, :, 2]

    kr = krx + kry + krz

    expKr = np.exp(1j * kr)

    Uinc = np.zeros((L, M, N), dtype=np.complex128)
    Uinc = Uo * expKr

    return Uinc


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
