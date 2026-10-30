"""S3: endpoint-singular inputs given only by point evaluations; a general remedy for the sampled route.

Inputs (T = 1): t^-0.25, t^-0.4, (1-t)^-0.3, t^-0.25 + 1_[0,0.3).  Three routes into the same projector
(LargeProjector, explicit auxiliary residual, no endpoint polynomial):
  U  uniform midpoint DCT on N cells (the S2 sampled route);
  G  graded end-cell Gauss-Legendre + 8x oversampling (covfourier.endpoint; point evaluations only);
  X  exact classical coefficients (closed form; reference only).
Exact-arithmetic statements for the reference:
  |e_D(f) - e_D(f_N)| <= tau_N,  tau_N^2 >= ||sum_{k>=N} a_k h_k||_H^2 (upper weighted inequality, rigorous |a_k| bounds),
  |e_D(f_N) - e_D(f~_N)| <= |d_0| + (sum_k w_k d_k^2)^(1/2)  for coefficient errors d = a~ - a.
Floating point is assessed, not bounded.  A-posteriori estimate for G without the reference: |G(8x) - G(4x)|.
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.fft import dct

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from covfourier.large import LargeProjector             # noqa: E402
from covfourier import endpoint as EP                   # noqa: E402
from covfourier.moments import Power                    # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')


def c_H(H):
    return math.gamma(2 * H + 1) * math.sin(math.pi * H)


def inputs(N):
    k = np.arange(N, dtype=float)
    def ind_coef(tau):
        a = np.empty(N); a[0] = tau; a[1:] = math.sqrt(2) * np.sin(k[1:] * np.pi * tau) / (k[1:] * np.pi); return a
    sgn = (-1.0) ** k
    return [
        ('t^-0.25', lambda t: np.asarray(t, float) ** -0.25, lambda: Power(-0.25).cosine(N), [(-0.25, 1.0)], 0.0, {}),
        ('t^-0.4', lambda t: np.asarray(t, float) ** -0.4, lambda: Power(-0.4).cosine(N), [(-0.4, 1.0)], 0.0, {}),
        ('(1-t)^-0.3', lambda t: (1 - np.asarray(t, float)) ** -0.3, lambda: Power(-0.3).cosine(N) * sgn, [(-0.3, 1.0)], 0.0,
         {'f_right_local': lambda s: np.asarray(s, float) ** -0.3}),
        ('t^-0.25+1[0,0.3)', lambda t: np.asarray(t, float) ** -0.25 + (np.asarray(t, float) < 0.3), lambda: Power(-0.25).cosine(N) + ind_coef(0.3), [(-0.25, 1.0)], 1.0, {}),
    ]


def tail_bound(H, N, powers, ind_amp):
    """Upper bound tau_N^2 for the H-norm of the tail beyond N (sum of rigorous coefficient bounds, triangle in l2_w)."""
    cH = c_H(H); K = 64 * N
    k = np.arange(N, K, dtype=float)
    A = np.zeros_like(k)
    for p, amp in powers:
        A += amp * EP.power_coefficient_bound(p, k)
    A += ind_amp * math.sqrt(2) / (k * np.pi)
    s = float(np.sum(cH * (k * np.pi) ** (1 - 2 * H) * A ** 2))
    # beyond K: each |a_k| <= B k^(-g) with the slowest decay g; integral bound with a factor 2 for the cross terms
    gs = [1 + p for p, _ in powers] + ([1.0] if ind_amp else [])
    g = min(gs)
    B = float(A[-1] * (K - 1) ** g) * 1.0001
    e = 1 - 2 * H - 2 * g
    s += cH * np.pi ** (1 - 2 * H) * B ** 2 * (K - 1.0) ** (e + 1) / -(e + 1)
    return s


def coef_error_bound(d, H):
    cH = c_H(H); k = np.arange(1, d.size, dtype=float)
    return abs(d[0]) + math.sqrt(float(np.sum(cH * (k * np.pi) ** (1 - 2 * H) * d[1:] ** 2)))


def uniform_coefficients(f, N):
    tm = (np.arange(N) + 0.5) / N
    y = dct(np.asarray(f(tm), float), type=2)
    a = (1 / N) * math.sqrt(2) * y / 2; a[0] = (1 / N) * y[0] / 2
    return a


def run(Hs=(0.55, 0.75, 0.95), n=1023, Ns=(32768, 262144), out='S3_endpoint_remedy.json'):
    data = {'note': __doc__, 'rows': []}
    for H in Hs:
        for N in Ns:
            t0 = time.perf_counter(); lp = LargeProjector(H, n, N); ts = time.perf_counter() - t0
            for name, f, exact, powers, ind_amp, kw in inputs(N):
                aX = exact()
                t0 = time.perf_counter(); aU = uniform_coefficients(f, N); tU = time.perf_counter() - t0
                t0 = time.perf_counter(); aG = EP.graded_cosine_coefficients(f, N, M=64, q=8, over=8, **kw); tG = time.perf_counter() - t0
                aG4 = EP.graded_cosine_coefficients(f, N, M=64, q=8, over=4, **kw)
                A = np.stack([aU, aG, aG4, aX], axis=1); c0 = np.zeros((2, 4))
                r = lp.project(A, c0, explicit=True)
                E2 = r['E2_aux_explicit']
                tau2 = tail_bound(H, N, powers, ind_amp)
                eX = math.sqrt(max(E2[3], 0.0)); tau = math.sqrt(tau2)
                row = {'H': H, 'n': n, 'D': n + 2, 'N': N, 'input': name, 'setup_s': ts, 'uniform_s': tU, 'graded_s': tG,
                       'E2_uniform': float(E2[0]), 'E2_graded_8x': float(E2[1]), 'E2_graded_4x': float(E2[2]), 'E2_exact_coef': float(E2[3]),
                       'coef_err_bound_uniform': coef_error_bound(aU - aX, H), 'coef_err_bound_graded': coef_error_bound(aG - aX, H),
                       'tau_N_bound': tau, 'eD_true_bracket': [max(eX - tau, 0.0) ** 2, (eX + tau) ** 2],
                       'interface': 'f + right-local' if kw else 'f',
                       'aposteriori_graded_estimate': abs(math.sqrt(max(E2[1], 0)) - math.sqrt(max(E2[2], 0)))}
                data['rows'].append(row)
                print('H %.2f N %6d %-17s E2: U %.4e  G %.4e  X %.4e | G/X-1 %+.1e  U/X-1 %+.2f | true e_D^2 in [%.4e, %.4e] | coef-err bound G %.1e'
                      % (H, N, name, E2[0], E2[1], E2[3], E2[1] / E2[3] - 1, E2[0] / E2[3] - 1, *row['eD_true_bracket'], row['coef_err_bound_graded']), flush=True)
                with open(os.path.join(RES, out), 'w') as fh:
                    json.dump(data, fh, indent=1)
    return data


if __name__ == '__main__':
    run()
