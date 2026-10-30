"""Competing methods and exact references for the application benchmarks.

All implementations are written independently for this work.  None selects a
method by the name of an input; references are used only to measure errors.
"""
import math
import time
import numpy as np
import mpmath as mp
from scipy.special import roots_jacobi
from scipy.linalg import matmul_toeplitz


# ----------------------------------------------------------------- exact reference
class ExpFamily:
    """Integrands f_i(s) = Re( c_i exp(a_i (s - s0)) ) on [0,1], complex a_i, c_i (vectorized callables).

    s0 = 1 avoids overflow for stiff memory kernels k(1-s) = exp(-lam (1-s))."""
    def __init__(self, a, c, s0=0.0):
        self.a = np.asarray(a, dtype=complex)
        self.c = np.asarray(c, dtype=complex)
        self.s0 = float(s0)

    def __len__(self):
        return len(self.a)

    def subset(self, idx):
        return ExpFamily(self.a[idx], self.c[idx], self.s0)

    def values(self, s):
        s = np.asarray(s, dtype=float)
        return np.real(self.c[:, None] * np.exp(np.outer(self.a, s.ravel() - self.s0))).reshape((len(self.a),) + s.shape)

    def callables(self):
        return [lambda s, a=a, c=c: np.real(c * np.exp(a * (np.asarray(s, dtype=float) - self.s0)))
                for a, c in zip(self.a, self.c)]

    def exact_covariance(self, H, dps=30):
        """K_ij = <f_i, f_j>_H via the bilinear exponential moment (Supplementary Material), T = 1."""
        q = len(self.a)
        K = np.zeros((q, q))
        with mp.workdps(dps):
            Hm = mp.mpf(repr(float(H))); b = 2 * Hm - 1
            cache = {}
            def F(z):
                key = (z.real, z.imag)
                if key not in cache:
                    cache[key] = mp.hyp1f1(b, b + 1, z) / b if b != 0 else None
                return cache[key]
            def Jab(x, y):
                s = x + y
                if b == 0:
                    return (mp.e ** s - 1) / s
                return Hm * b / s * (mp.e ** s * (F(-x) + F(-y)) - F(x) - F(y))
            A = [mp.mpc(complex(v)) for v in self.a]
            s0 = mp.mpf(repr(self.s0))
            C = [mp.mpc(complex(v)) * mp.e ** (-mp.mpc(complex(av)) * s0) for v, av in zip(self.c, self.a)]
            for i in range(q):
                for j in range(i + 1):
                    # Re(c_i e^{a_i s}) = (c_i e^{a_i s} + conj)/2
                    v = (C[i] * C[j] * Jab(A[i], A[j]) + C[i] * mp.conj(C[j]) * Jab(A[i], mp.conj(A[j])))
                    v = mp.re(v) / 2
                    K[i, j] = K[j, i] = float(v)
        return K


# ----------------------------------------------------------------- comparator 1
def _jacobi01(n, alpha, beta):
    x, w = roots_jacobi(n, alpha, beta)
    return (x + 1) / 2, w / 2 ** (alpha + beta + 1)


def triangle_gj_covariance(values, H, n):
    """Pairwise covariance by the two-triangle form with Gauss-Jacobi rules:
    <f,g> = H b int_0^1 x^b int_0^1 (1-y)^(b-1) [f(x) g(xy) + f(xy) g(x)] dy dx  (T = 1).
    values(s) -> array (q, ...) of all integrands.  Cost O(q n^2 + q^2 n)."""
    t0 = time.perf_counter()
    if H == 0.5:
        x, w = _jacobi01(n, 0.0, 0.0)
        V = values(x)
        K = (V * w) @ V.T
        return K, time.perf_counter() - t0
    b = 2 * H - 1
    x, wx = _jacobi01(n, 0.0, b)                # outer weight x^b (benign exponent)
    # inner weight (1-y)^(b-1): substitution v = (1-y)^b removes it (Gauss-Legendre in v);
    # avoids scipy Jacobi nodes with exponent close to -1 (inaccurate for small b).
    if b >= 0.5:                                 # mild exponent: Gauss-Jacobi is accurate
        y, wv = _jacobi01(n, b - 1.0, 0.0)
    else:
        v, wv = np.polynomial.legendre.leggauss(n)
        v = (v + 1) / 2; wv = wv / 2 / b
        y = 1.0 - v ** (1.0 / b)
    A = values(x)                               # (q, n)
    XY = np.outer(x, y)
    rows = max(1, int(4_000_000 // (n * A.shape[0])))    # chunk over outer nodes to bound memory
    Bm = np.empty_like(A)
    for i0 in range(0, n, rows):
        Bm[:, i0:i0 + rows] = values(XY[i0:i0 + rows]) @ wv          # inner averages
    half = (A * wx) @ Bm.T
    K = H * b * (half + half.T)
    return 0.5 * (K + K.T), time.perf_counter() - t0


# ----------------------------------------------------------------- comparator 2
def increment_covariance(H, N):
    """Covariance of unit-width cell indicators: gamma_k = ((k+1)^2H - 2k^2H + (k-1)^2H)/2."""
    k = np.arange(N, dtype=float)
    p = 2 * H
    g = np.empty(N)
    g[0] = 1.0
    if N > 1:
        kk = k[1:]
        # stable form for large k: binomial series (Section 'Whole-line realization' of the paper), direct for small k
        small = kk < 8
        g[1:][small] = 0.5 * ((kk[small] + 1) ** p - 2 * kk[small] ** p + (kk[small] - 1) ** p)
        kl = kk[~small]
        acc = np.zeros_like(kl)
        from scipy.special import binom
        for nn in range(1, 13):
            acc += binom(p, 2 * nn) * kl ** (-2 * nn)
        g[1:][~small] = kl ** p * acc
    return g


def cell_toeplitz_covariance(values, H, N, order=8):
    """Cell averages on N uniform cells, exact increment covariance (Toeplitz, FFT)."""
    t0 = time.perf_counter()
    xg, wg = np.polynomial.legendre.leggauss(order)
    xg = (xg + 1) / 2; wg = wg / 2
    t = (np.arange(N)[:, None] + xg[None, :]) / N
    q = values(t[:1]).shape[0]
    rows = max(1, int(4_000_000 // (order * q)))
    means = np.concatenate([values(t[i0:i0 + rows]) @ wg for i0 in range(0, N, rows)], axis=1)   # (q, N)
    gam = increment_covariance(H, N) / N ** (2 * H)
    GM = matmul_toeplitz((gam, gam), means.T, check_finite=False)
    K = means @ GM
    return 0.5 * (K + K.T), time.perf_counter() - t0


# ----------------------------------------------------------------- exact grid simulation
class DaviesHarte:
    """Exact simulation of fBm increments on a uniform grid of [0,1] (circulant embedding)."""
    def __init__(self, H, N):
        self.H, self.N = H, N
        g = increment_covariance(H, N + 1)[:N + 1] / N ** (2 * H)
        row = np.r_[g[:N], g[N], g[1:N][::-1]]              # standard embedding, length 2N
        lam = np.fft.fft(row).real
        if lam.min() < -1e-12 * lam.max():
            raise ArithmeticError('circulant embedding not nonnegative')
        self.sq = np.sqrt(np.clip(lam, 0, None) / (2 * N))

    def increments(self, M, rng):
        N = self.N
        Z = rng.standard_normal((M, 2 * N)) + 1j * rng.standard_normal((M, 2 * N))
        W = np.fft.fft(self.sq * Z, axis=1)
        return W[:, :N].real
