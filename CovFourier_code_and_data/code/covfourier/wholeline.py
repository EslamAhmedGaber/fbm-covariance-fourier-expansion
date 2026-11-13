"""Whole-line realization of the fBm covariance norm (1/2 <= H < 1).

Implements the fixed expanding-window cell rule
    Q_m f = sum_j c_{m,j} 1_{I_{m,j}},   I_{m,j} cells of width 2^-m on [-2^m, 2^m),
the exact covariance norm of a step function through the Toeplitz increment
sequence of fractional Gaussian noise (one FFT autocorrelation), and closed
reference norms used for validation.

Conventions: unitary Fourier transform (2 pi)^{-1/2} int f(t) e^{-i t xi} dt,
c_H = Gamma(2H+1) sin(pi H),  ||f||^2 = c_H int |f^(xi)|^2 |xi|^{1-2H} d xi.
"""
import numpy as np
from scipy.special import binom, gamma, roots_legendre
import mpmath as mp

__all__ = ['c_H', 'increments', 'step_norm2', 'cell_grid', 'cell_averages_gl',
           'upsample', 'packet_norm2_closed', 'packet_norm2_spectral']


def c_H(H):
    return float(gamma(2 * H + 1) * np.sin(np.pi * H))


def increments(H, K, n_terms=12, k_series=8, dps=40):
    """rho_k = ((k+1)^p - 2 k^p + (k-1)^p)/2 for k = 0..K-1 (rho_0 = 1), p = 2H.

    Large lags use the positive binomial series k^p sum_{n>=1} binom(p,2n) k^{-2n}
    (no cancellation); lags below `k_series` use `dps`-digit arithmetic.
    Returns (rho, tail_bound) where tail_bound bounds the omitted series terms
    (exact arithmetic) relative to rho_k, maximised over the series lags.
    """
    p = 2.0 * H
    rho = np.empty(K)
    rho[0] = 1.0
    if abs(p - 1.0) < 1e-15:
        rho[1:] = 0.0
        return rho, 0.0
    mp.mp.dps = dps
    for k in range(1, min(K, k_series)):
        kk = mp.mpf(k)
        rho[k] = float(((kk + 1) ** p - 2 * kk ** p + (kk - 1) ** p) / 2)
    if K > k_series:
        k = np.arange(k_series, K, dtype=float)
        inv2 = k ** -2.0
        s = np.zeros_like(k)
        powk = np.ones_like(k)
        for n in range(1, n_terms + 1):
            powk = powk * inv2
            s += binom(p, 2 * n) * powk
        rho[k_series:] = k ** p * s
        # omitted terms are positive with ratio < k^-2:
        b_next = abs(binom(p, 2 * n_terms + 2))
        first = abs(binom(p, 2))
        kmin = float(k_series)
        tail_rel = (b_next * kmin ** (-2 * n_terms - 2) / (1 - kmin ** -2)) / (first * kmin ** -2)
    else:
        tail_rel = 0.0
    return rho, tail_rel


def step_norm2(c, H, delta, rho=None):
    """Exact ||sum_j c_j 1_{[a+j delta, a+(j+1) delta)}||_{H,R}^2 (up to rounding)."""
    c = np.asarray(c, float)
    n = c.size
    if rho is None:
        rho, _ = increments(H, n)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(c, nfft)
    r = np.fft.irfft(F * np.conj(F), nfft)[:n]
    return delta ** (2 * H) * (rho[0] * r[0] + 2.0 * np.dot(rho[1:n], r[1:n]))


def cell_grid(m):
    """Level-m window [-2^m, 2^m), cell width 2^-m, N = 2^(2m+1) cells."""
    L = 2.0 ** m
    delta = 2.0 ** (-m)
    N = 2 ** (2 * m + 1)
    a = -L + delta * np.arange(N)
    return a, a + delta, delta


def cell_averages_gl(f, m, q=8, chunk=1 << 16):
    """q-point Gauss-Legendre cell averages of a vectorized callable f."""
    a, b, delta = cell_grid(m)
    x, w = roots_legendre(q)
    out = np.empty(a.size)
    for s in range(0, a.size, chunk):
        aa = a[s:s + chunk]
        mid = aa + delta / 2
        t = mid[:, None] + (delta / 2) * x[None, :]
        out[s:s + chunk] = 0.5 * (f(t) @ w)
    return out


def upsample(c_coarse):
    """Represent a level-m cell vector on the level-(m+1) grid (window doubled)."""
    n = c_coarse.size
    fine = np.zeros(4 * n)          # level m+1 has 4x as many cells (window x2, width /2)
    fine[n:3 * n] = np.repeat(c_coarse, 2)
    return fine


def packet_norm2_closed(H, a, omega, dps=40):
    """||exp(-a t^2 + i omega t)||_{H,R}^2 from the physical kernel (closed form)."""
    mp.mp.dps = dps
    H = mp.mpf(H); a = mp.mpf(a); omega = mp.mpf(omega)
    if H == mp.mpf('0.5'):
        return mp.sqrt(mp.pi / (2 * a))
    beta = 2 * H - 1
    return (H * beta * mp.sqrt(mp.pi / (2 * a)) * (a / 2) ** (-beta / 2) * mp.gamma(beta / 2)
            * mp.hyp1f1(beta / 2, mp.mpf('0.5'), -omega ** 2 / (2 * a)))


def packet_norm2_spectral(H, a, omega, dps=40):
    """Same norm from the weighted spectral integral c_H int |f^|^2 |xi|^{1-2H}."""
    mp.mp.dps = dps
    H = mp.mpf(H); a = mp.mpf(a); omega = mp.mpf(omega)
    cH = mp.gamma(2 * H + 1) * mp.sin(mp.pi * H)
    s = mp.sqrt(2 * a)
    gam = 2 - 2 * H                       # xi = u^(1/gam) removes |xi|^(1-2H) exactly on [0, 1]

    def half(sign):                       # int_0^inf xi^(1-2H) |f^(sign xi)|^2 d xi
        g = lambda x: mp.exp(-(sign * x - omega) ** 2 / (2 * a)) / (2 * a)
        near = mp.quad(lambda u: g(u ** (1 / gam)), [0, 1]) / gam
        c = sign * omega
        pts = sorted(set([mp.mpf(1)] + [p for p in (c - 12 * s, c, c + 12 * s) if p > 1] + [mp.inf]))
        far = mp.quad(lambda x: x ** (1 - 2 * H) * g(x), pts)
        return near + far

    return cH * (half(1) + half(-1))
