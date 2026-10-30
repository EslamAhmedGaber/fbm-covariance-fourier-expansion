"""A5: weighted quadratic functionals of fBm through the general procedure.

Q_w = int_0^1 w(t) B_H(t)^2 dt,  K_w = sqrt(w) K sqrt(w) on L^2(0,1),  L_w(a) = E exp(-a Q_w) = prod (1 + 2 a lambda_n)^(-1/2).
Galerkin in the L^2 cosine basis h_0..h_{J-1}:  A_ij = E[(int sqrt(w) h_i B)(int sqrt(w) h_j B)] = <F_i, F_j>_H,
F_i(s) = int_s^1 sqrt(w(t)) h_i(t) dt.  The J integrands are handed to the one general procedure as samples
(LargeProjector: endpoint-corrected DCT, cosine block with Schur complement, Chebyshev white coordinates, two
residual directions), giving white coordinates Z (J x D).  Then A = Z Z^T + R with R the Gram matrix of the
residuals (positive semidefinite), so in exact arithmetic for the computed coordinates:
  * mu_n(Z Z^T) <= mu_n(A) <= lambda_n            (Weyl monotonicity; Rayleigh-Ritz),
  * sum_n (lambda_n - mu_n(Z Z^T)) = tr K_w - ||Z||_F^2 =: delta'   (tr K_w = int w(t) t^{2H} dt, exact),
  * B(a) exp(-a delta') <= L_w(a) <= B(a),  B(a) = prod_n (1 + 2 a mu_n(Z Z^T))^(-1/2).
Floating point is assessed, not bounded: coordinates have relative errors ~1e-15 and the eigenvalues are computed
by a dense SVD.  Weights: w = 1 (validation against the closed-form Galerkin matrix of A4) and w = 1/t
(no closed form used; F_i through Fresnel integrals, exact samples).  Checks for w = 1/t: Monte Carlo with exact
Davies-Harte paths, and the Weyl-type law lambda_n ~ c_H (n pi / kappa_w)^(-1-2H), kappa_w = int w^(1/(1+2H))
(a heuristic asymptotic, used only as a consistency check).
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.special import fresnel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from covfourier.large import LargeProjector            # noqa: E402
from covfourier.comparators import DaviesHarte         # noqa: E402
import app4_kl_spectrum as A4                          # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
A_VALUES = (0.5, 1.0, 2.0, 5.0)


def c_H(H):
    return math.gamma(2 * H + 1) * math.sin(math.pi * H)


def F_samples(weight, J, N):
    t = (np.arange(N) + 0.5) / N
    F = np.empty((J, N))
    if weight == 'one':
        F[0] = 1 - t
        i = np.arange(1, J, dtype=float)[:, None]
        F[1:] = -math.sqrt(2) * np.sin(i * np.pi * t[None, :]) / (i * np.pi)
    elif weight == 'inv_t':                                  # sqrt(w) = t^(-1/2)
        F[0] = 2 * (1 - np.sqrt(t))
        for i0 in range(1, J, 128):
            i = np.arange(i0, min(J, i0 + 128), dtype=float)[:, None]
            cc = i * np.pi
            _, C1 = fresnel(np.sqrt(2 * cc / np.pi))
            _, Ct = fresnel(np.sqrt(2 * cc * t[None, :] / np.pi))
            F[i0:i0 + i.shape[0]] = math.sqrt(2) * np.sqrt(2 * np.pi / cc) * (C1 - Ct)
    else:
        raise ValueError(weight)
    return F


def trace_K(weight, H):
    return 1 / (2 * H + 1) if weight == 'one' else 1 / (2 * H)


def run_case(H, weight, J, n=2047, N=16384, chunk=256):
    t0 = time.perf_counter()
    lp = LargeProjector(H, n, N)
    t_setup = time.perf_counter() - t0
    t0 = time.perf_counter()
    F = F_samples(weight, J, N)
    t_samples = time.perf_counter() - t0
    t0 = time.perf_counter()
    Zs, e2, eta, nrm = [], [], [], []
    for i0 in range(0, J, chunk):
        a_g, c = lp.fourier_stage(F[i0:i0 + chunk])
        r = lp.project(a_g, c)
        Zs.append(r['coords'].T); e2.append(r['error2']); eta.append(r['eta']); nrm.append(r['norm2'])
    Z = np.vstack(Zs); e2 = np.concatenate(e2); eta = np.concatenate(eta); nrm = np.concatenate(nrm)
    t_proj = time.perf_counter() - t0
    t0 = time.perf_counter()
    mu = np.linalg.svd(Z, compute_uv=False) ** 2
    t_eig = time.perf_counter() - t0
    trK = trace_K(weight, H)
    frob = float(np.sum(Z ** 2))
    delta = trK - frob
    out = {'H': H, 'weight': weight, 'J': J, 'n': n, 'D': lp.D if hasattr(lp, 'D') and not callable(lp.D) else n + 3, 'N': N,
           'setup_s': t_setup, 'samples_s': t_samples, 'project_s': t_proj, 'eig_s': t_eig,
           'trace_K': trK, 'frob_Z2': frob, 'trace_gap_delta': delta, 'sum_residual_E2': float(np.sum(np.clip(e2, 0, None))),
           'n_unresolved_inputs': int(np.sum(e2 <= eta)), 'orthogonality_defect': float(lp.orthogonality_defect),
           'mu_first50': mu[:50].tolist(), 'mu_n100_200_400': [float(mu[k - 1]) for k in (100, 200, 400) if k <= J]}
    lap = {}
    for a in A_VALUES:
        logB = -0.5 * np.sum(np.log1p(2 * a * mu))
        lap[str(a)] = [float(math.exp(logB - a * delta)), float(math.exp(logB))]
    out['laplace_bracket'] = lap
    if weight == 'one':                                     # validation against the closed-form Galerkin (A4)
        A = A4.galerkin_matrix(H, J)
        mu_ex = np.sort(np.linalg.eigvalsh(A))[::-1]
        d = mu_ex - mu[:J]
        out['validation_closed_form'] = {'trace_A_minus_frobZ': float(np.trace(A) - frob),
                                         'max_mu_exact_minus_mu_Z': float(d.max()), 'min_mu_exact_minus_mu_Z': float(d.min()),
                                         'max_rel_first100': float(np.max(np.abs(d[:100]) / mu_ex[:100]))}
    else:
        kappa = (1 + 2 * H) / (2 * H)                       # int_0^1 t^(-1/(1+2H)) dt
        k = np.arange(1, mu.size + 1)
        out['weyl_ratio'] = {str(kk): float(mu[kk - 1] / (c_H(H) * (kk * np.pi / kappa) ** (-1 - 2 * H))) for kk in (10, 50, 100, 200, 400) if kk <= J}
    print(json.dumps({k: out[k] for k in ('H', 'weight', 'J', 'trace_gap_delta', 'sum_residual_E2', 'n_unresolved_inputs', 'project_s')}),
          '| L(1) in', lap['1.0'], '| val' if weight == 'one' else '| weyl', out.get('validation_closed_form', out.get('weyl_ratio')), flush=True)
    return out


def monte_carlo(H, a_values=A_VALUES, M=20000, N=4096, seed=11):
    dh = DaviesHarte(H, N)
    rng = np.random.default_rng(seed)
    t = np.arange(1, N + 1) / N
    acc = {str(a): [] for a in a_values}
    t0 = time.perf_counter()
    for m0 in range(0, M, 1000):
        B = np.cumsum(dh.increments(1000, rng), axis=1)
        Q = np.sum(B ** 2 / t[None, :], axis=1) / N          # right-endpoint rule on the exact path grid
        for a in a_values:
            acc[str(a)].append(np.exp(-a * Q))
    out = {}
    for a in a_values:
        v = np.concatenate(acc[str(a)])
        out[str(a)] = [float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size))]
    return {'M': M, 'N': N, 'seconds': time.perf_counter() - t0, 'mean_se': out}


def run(Hs=(0.55, 0.75, 0.95), Js=(256, 512, 1024), out='A5_weighted_kl.json'):
    data = {'note': __doc__, 'cases': [], 'monte_carlo_inv_t': {}}
    for H in Hs:
        for weight in ('one', 'inv_t'):
            for J in Js:
                data['cases'].append(run_case(H, weight, J))
                with open(os.path.join(RES, out), 'w') as fh:
                    json.dump(data, fh, indent=1)
        data['monte_carlo_inv_t'][str(H)] = monte_carlo(H)
        print('MC', H, data['monte_carlo_inv_t'][str(H)]['mean_se'], flush=True)
        with open(os.path.join(RES, out), 'w') as fh:
            json.dump(data, fh, indent=1)
    return data


if __name__ == '__main__':
    run()
