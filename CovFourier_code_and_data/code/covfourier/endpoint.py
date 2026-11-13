"""Fourier stage for point-evaluable inputs with algebraic endpoint singularities (general, no closed forms).

Input assumption (stated in the paper): f can be evaluated at any point of (0, T); f is smooth in (0, T);
near each endpoint, |f^(m)(t)| <= C dist(t, {0, T})^(-alpha - m) for m = 0, 1, 2 with alpha < 1
(for f in L^{1/H} one needs alpha < H).  Interior jumps are allowed (the uniform part handles them).

Rule: classical cosine coefficients a_k = int_0^T f h_k (k < N) from the uniform midpoint DCT on N cells, in which
the contributions of the M cells next to each endpoint are replaced by a geometrically graded Gauss-Legendre rule
(levels subintervals with ratio sigma, q points each).  The graded rule integrates t^(-alpha) cos(omega t) with
error decaying geometrically in the number of levels; the remaining uniform part sees only cells at distance
>= M h from the endpoint, where the midpoint error is O(h^2 |f''|).
"""
import math

import numpy as np
from scipy.fft import dct


def _graded_nodes(h, M, levels, sigma, q):
    """Nodes/weights on [0, M h]: geometric grading towards 0 inside the first cell [0, h], and a q-point
    Gauss-Legendre rule on each further cell [j h, (j+1) h], j = 1..M-1.  Every subinterval is at most one cell
    wide, so h_k (k < N) has at most half a period on it."""
    x0, w0 = np.polynomial.legendre.leggauss(q)
    b = h * sigma ** np.arange(levels + 1)                       # h, h sigma, ..., h sigma^levels
    edges = [(lo, hi) for lo, hi in zip(np.r_[b[1:], 0.0], b)] + [(j * h, (j + 1) * h) for j in range(1, M)]
    xs, ws = [], []
    for lo, hi in edges:
        xs.append(0.5 * (hi - lo) * x0 + 0.5 * (hi + lo)); ws.append(0.5 * (hi - lo) * w0)
    return np.concatenate(xs), np.concatenate(ws)


def graded_cosine_coefficients(f, N, T=1.0, M=16, levels=40, sigma=0.15, q=12, chunk=8192, over=1,
                               f_left_local=None, f_right_local=None):
    """a_k, k = 0..N-1, of a point-evaluable f with possible algebraic endpoint singularities.

    over > 1: evaluate on over*N cells and keep the first N coefficients (reduces aliasing for k near N).
    f_left_local(s) = f(s), f_right_local(s) = f(T - s): optional evaluations in endpoint-local coordinates
    (distance s to the endpoint).  Without them the grading depth near T is limited by the floating-point
    spacing of T - s; the levels are then truncated so that every right-end node satisfies T - s < T."""
    Nout = N
    N = over * N
    h = T / N
    tm = (np.arange(N) + 0.5) * h
    F = np.asarray(f(tm), dtype=float)
    y = dct(F, type=2)                                           # y_k = 2 sum_j F_j cos(pi k (2j+1) / (2N))
    a = h * math.sqrt(2 / T) * y / 2
    a[0] = h * T ** -0.5 * y[0] / 2
    xg, wg = _graded_nodes(h, M, levels, sigma, q)
    fl = np.asarray((f_left_local or f)(xg), dtype=float)
    if f_right_local is not None:
        sr, wr_ = xg, wg
        fr = np.asarray(f_right_local(sr), dtype=float)
    else:
        keep = (T - xg) < T                                            # representable distance to T
        sr, wr_ = xg[keep], wg[keep]
        fr = np.asarray(f(T - sr), dtype=float)
    jl = np.arange(M); jr = np.arange(N - M, N)
    a = a[:Nout]
    for k0 in range(0, Nout, chunk):
        k = np.arange(k0, min(Nout, k0 + chunk), dtype=float)
        om = k * np.pi / T
        hk = lambda x: np.cos(np.outer(x, om))
        sg = (-1.0) ** k                                                # cos(om (T - s)) = (-1)^k cos(om s)
        corr = (wg * fl) @ hk(xg) + sg * ((wr_ * fr) @ hk(sr)) - h * (F[jl] @ hk(tm[jl]) + sg * (F[jr] @ hk(T - tm[jr])))
        scale = np.full(k.size, math.sqrt(2 / T)); scale[k == 0] = T ** -0.5
        a[k0:k0 + k.size] += scale * corr
    return a


def power_coefficient_bound(p, k, T=1.0):
    """Rigorous |a_k(t^p)| bound on [0, 1] (T = 1), -1 < p < 0, k >= 1:
    |a_k - sqrt2 Gamma(1+p) cos(pi(1+p)/2) (k pi)^(-1-p)| <= 2 sqrt2 |p| (k pi)^(-2)."""
    assert T == 1.0
    om = np.pi * np.asarray(k, dtype=float)
    return math.sqrt(2) * (math.gamma(1 + p) * abs(math.cos(math.pi * (1 + p) / 2)) * om ** (-1 - p) + 2 * abs(p) * om ** -2)


def power_tail_bound(p, H, N, K=None):
    """tau_N^2 >= ||sum_{k>=N} a_k(t^p) h_k||_H^2 on [0,1] via the upper weighted cosine bound (Theorem 'two-sided bounds' of the paper):
    sum_{k>=N} w_k A_k^2, summed exactly to K and bounded by an integral beyond (decreasing summand)."""
    cH = math.gamma(2 * H + 1) * math.sin(math.pi * H)
    K = K or 64 * N
    k = np.arange(N, K, dtype=float)
    g = lambda kk: cH * (np.pi * kk) ** (1 - 2 * H) * power_coefficient_bound(p, kk) ** 2
    s = float(np.sum(g(k)))
    # integral of g from K-1 to infinity: g(k) <= cH pi^(1-2H) k^(1-2H) * 2 (A1^2 k^(-2-2p) + A2^2 k^(-4)), with
    A1 = math.sqrt(2) * math.gamma(1 + p) * abs(math.cos(math.pi * (1 + p) / 2)) * np.pi ** (-1 - p)
    A2 = math.sqrt(2) * 2 * abs(p) * np.pi ** -2
    k0 = K - 1.0
    e1 = 1 - 2 * H - 2 - 2 * p; e2 = 1 - 2 * H - 4
    tail = cH * np.pi ** (1 - 2 * H) * 2 * (A1 ** 2 * k0 ** (e1 + 1) / -(e1 + 1) + A2 ** 2 * k0 ** (e2 + 1) / -(e2 + 1))
    return s + float(tail)
