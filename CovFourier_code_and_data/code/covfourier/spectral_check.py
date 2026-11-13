"""Independent spectral-route evaluation of covariance inner products.

<f,g>_{H} = c_H int_R  F(J f)(xi) conj(F(J g)(xi)) |xi|^(1-2H) d xi,
c_H = Gamma(2H+1) sin(pi H), unitary Fourier transform, J = zero extension
(Theorem 'whole-line spectral realization' of the paper and the zero-extension identity for the Gram).

The transforms of zero-extended cosines and monomials on [0,T] are elementary,
so this route shares no formula with the Kummer-function Gram.  It is used
only for validation (double precision, panel Gauss-Legendre with a substitution
removing the |xi|^(1-2H) singularity at 0 and a finite cutoff Xi).
"""
import math
import numpy as np
from numpy.polynomial.legendre import leggauss


def _E(z, T):
    """int_0^T exp(i z t) dt, stable at z = 0."""
    return T * np.exp(0.5j * z * T) * np.sinc(z * T / (2 * np.pi))


def ft_cos(k, xi, T):
    """Unitary FT of zero-extended h_k at frequency xi (vector)."""
    if k == 0:
        return _E(-xi, T) / math.sqrt(T) / math.sqrt(2 * math.pi)
    w = k * math.pi / T
    return math.sqrt(2.0 / T) * 0.5 * (_E(w - xi, T) + _E(-w - xi, T)) / math.sqrt(2 * math.pi)


def ft_mono(j, xi, T):
    """Unitary FT of zero-extended t^j on [0,T], j = 0,1,2 (closed form; series near 0)."""
    xi = np.asarray(xi, dtype=float)
    z = -xi
    out = np.empty(xi.shape, dtype=complex)
    small = np.abs(z * T) < 1e-2
    zs = z[small]
    # series: int_0^T t^j e^{izt} dt = sum_n (iz)^n T^(n+j+1)/(n!(n+j+1))
    acc = np.zeros(zs.shape, dtype=complex)
    term = np.ones(zs.shape, dtype=complex)
    for n in range(0, 30):
        if n > 0:
            term = term * (1j * zs) / n
        acc += term * T ** (n + j + 1) / (n + j + 1)
    out[small] = acc
    zl = z[~small]
    e = np.exp(1j * zl * T)
    iz = 1j * zl
    if j == 0:
        val = (e - 1) / iz
    elif j == 1:
        val = e * T / iz - (e - 1) / iz ** 2
    elif j == 2:
        val = e * T ** 2 / iz - 2 * e * T / iz ** 2 + 2 * (e - 1) / iz ** 3
    else:
        raise ValueError('j <= 2')
    out[~small] = val
    return out / math.sqrt(2 * math.pi)


def spectral_ip(fa, fb, H, T, Xi=2.0e5, panel=None, nodes=24, ends=None):
    """2 c_H Re int_0^inf fa(xi) conj(fb(xi)) xi^(1-2H) d xi (fa, fb: vectorized FTs).

    ends = (fa(0), fa(T), fb(0), fb(T)) adds the analytic leading-order tail beyond
    Xi: integration by parts gives  sqrt(2 pi) F(Jf)(xi) = (f(0) - f(T)e^{-i xi T})/(i xi)
    + O(xi^-2), hence Re[F(Jf) conj F(Jg)] = [f0 g0 + fT gT - (f0 gT + fT g0) cos(xi T)]
    /(2 pi xi^2) + O(xi^-3).  The omitted remainder is O(Xi^(-1-2H))."""
    cH = math.gamma(2 * H + 1) * math.sin(math.pi * H)
    tail = 0.0
    if ends is not None:
        f0, fT, g0, gT = ends
        alpha = f0 * g0 + fT * gT
        gam = f0 * gT + fT * g0
        t1 = Xi ** (-2 * H) / (2 * H)
        # int_Xi^inf x^(-1-2H) cos(xT) dx, by parts (two terms)
        t2 = -math.sin(Xi * T) * Xi ** (-1 - 2 * H) / T + (1 + 2 * H) * math.cos(Xi * T) * Xi ** (-2 - 2 * H) / T ** 2
        tail = 2 * cH * (alpha * t1 - gam * t2) / (2 * math.pi)
    x, w = leggauss(nodes)
    total = 0.0
    # [0,1]: xi = u^(1/(2-2H)) removes the weight  (xi^(1-2H) dxi = du/(2-2H))
    p = 1.0 / (2.0 - 2.0 * H)
    u = 0.5 * (x + 1); wu = 0.5 * w
    # graded panels in u to handle the u^(p-1) behaviour of xi(u)
    edges = np.r_[0.0, 2.0 ** -np.arange(30, 0, -1), 1.0]
    for a0, b0 in zip(edges[:-1], edges[1:]):
        uu = a0 + (b0 - a0) * u
        xi = uu ** p
        total += (b0 - a0) * np.sum(wu * (fa(xi) * np.conj(fb(xi))).real) / (2 - 2 * H)
    # [1, Xi]: panels of width panel (default pi/T / 2)
    if panel is None:
        panel = math.pi / T / 2
    n = int(math.ceil((Xi - 1.0) / panel))
    starts = 1.0 + panel * np.arange(n)
    for i in range(0, n, 4000):
        s = starts[i:i + 4000]
        xi = (s[:, None] + panel * u[None, :]).ravel()
        val = (fa(xi) * np.conj(fb(xi))).real * xi ** (1 - 2 * H)
        total += panel * np.sum(val.reshape(len(s), -1) * wu[None, :])
    return 2 * cH * total + tail
