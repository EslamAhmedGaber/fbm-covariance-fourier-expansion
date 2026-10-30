"""A6.1: borehole heat kernels and climate targets through the unchanged general procedure, against an independent
quadrature reference.

Integrands on the window [1500, 2030] (kappa = 1e-6 m^2/s): subsurface temperature at depths 20-600 m logged in 1985
and 2030 (20 kernels), epoch means over 1850-1900, 1961-1990, 1991-2020, and ground heat content in 1960 and 2020.
Expansion: CovarianceFourier + BatchProjector (endpoint-corrected DCT), n = 32..1024, N = max(8192, 16 n).
Reference: covfourier.khref.KHQuad (composite graded quadrature of the singular double integral; features declared),
at two resolutions to show its own convergence.  Metrics: covariance error relative to sqrt(K_ii K_jj); per-entry
Cauchy-Schwarz bound |K_ij - (Gamma Gamma^T)_ij| <= e_i e_j; relative residual energies; timings.
Output: results/A6_heat_kernel_check.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from covfourier.engine import CovarianceFourier          # noqa: E402
from covfourier.batch import BatchProjector              # noqa: E402
from covfourier.khref import KHQuad                      # noqa: E402
from covfourier import heat                              # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
WIN = heat.Window(1500.0, 2030.0)
DEPTHS = np.array([20., 30., 50., 75., 100., 150., 200., 300., 400., 600.])
TLOGS = (1985.0, 2030.0)
EPOCHS = ((1850.0, 1900.0), (1961.0, 1990.0), (1991.0, 2020.0))
HEAT_T = (1960.0, 2020.0)


def integrand_values(u):
    u = np.asarray(u, dtype=float)
    rows = [heat.borehole_values(DEPTHS, t, u, WIN) for t in TLOGS]
    rows += [heat.epoch_mean_values(A, B, u, WIN)[None, :] for A, B in EPOCHS]
    rows += [heat.heat_content_values(t, u, WIN)[None, :] / 1e8 for t in HEAT_T]     # 1e8 J/m^2 units
    return np.vstack(rows)


def labels():
    lab = [f'T(z={int(z)} m, {int(t)})' for t in TLOGS for z in DEPTHS]
    lab += [f'mean dT {int(A)}-{int(B)}' for A, B in EPOCHS] + [f'heat content {int(t)} (1e8 J/m2)' for t in HEAT_T]
    return lab


def main(Hs=(0.6, 0.75, 0.9), ns=(32, 64, 128, 256, 512, 1024)):
    feats = heat.features(WIN, TLOGS, EPOCHS, HEAT_T)
    out = {'config': {'window': [WIN.t0, WIN.t1], 'kappa_m2_per_yr': heat.KAPPA_M2_PER_YR, 'depths_m': DEPTHS.tolist(),
                      't_logs': list(TLOGS), 'epochs': [list(e) for e in EPOCHS], 'heat_times': list(HEAT_T),
                      'ns': list(ns), 'labels': labels()}, 'cases': []}
    for H in Hs:
        t0 = time.perf_counter()
        Kref = {}
        for hm in (1 / 128, 1 / 256):
            Q = KHQuad(H, features=feats, h_max=hm)
            Kref[hm] = Q.gram(integrand_values)
            print(H, 'reference h_max', hm, 'outer nodes', Q.n_outer, '%.1fs' % (time.perf_counter() - t0), flush=True)
        K = Kref[1 / 256]
        dg = np.sqrt(np.outer(np.diag(K), np.diag(K)))
        ref_self = float(np.max(np.abs(Kref[1 / 128] - K) / dg))
        t_ref = time.perf_counter() - t0
        rows = []
        for n in ns:
            N = max(8192, 16 * n)
            t1 = time.perf_counter()
            cf = CovarianceFourier(H, n, enrich=True, Naux=N)
            bp = BatchProjector(cf)
            t_prep = time.perf_counter() - t1
            t2 = time.perf_counter()
            Z, E2, r = heat.project_samples(bp, integrand_values(heat.midpoints(N)))
            t_proj = time.perf_counter() - t2
            Kc = Z @ Z.T
            err = np.abs(K - Kc)
            e = np.sqrt(E2)
            bound = np.outer(e, e)
            margin = float(np.max((err - bound) / dg))                       # <= 0 means the bound holds everywhere
            rows.append({'n': n, 'D': n + 2, 'N': N, 'max_rel_cov_err': float(np.max(err / dg)),
                         'rel_frobenius': float(np.linalg.norm(K - Kc) / np.linalg.norm(K)),
                         'cs_bound_margin_rel': margin, 'cs_bound_holds': bool(margin <= 1e-12),
                         'rel_residual_energy': (E2 / np.diag(K)).tolist(),
                         'max_rel_residual_borehole': float(np.max((E2 / np.diag(K))[:2 * len(DEPTHS)])),
                         'max_rel_residual_targets': float(np.max((E2 / np.diag(K))[2 * len(DEPTHS):])),
                         'endpoint_correction_applied': bp.last_corrected.tolist(),
                         'prep_s': t_prep, 'project_s': t_proj})
            print(H, n, 'cov err %.2e' % rows[-1]['max_rel_cov_err'], 'CS margin %.1e' % margin,
                  'res(bh) %.1e res(tg) %.1e' % (rows[-1]['max_rel_residual_borehole'], rows[-1]['max_rel_residual_targets']),
                  'prep %.2fs proj %.2fs' % (t_prep, t_proj), flush=True)
        out['cases'].append({'H': H, 'reference_self_consistency_rel': ref_self, 'reference_seconds': t_ref,
                             'K_diag': np.diag(K).tolist(), 'rows': rows})
    with open(os.path.join(RES, 'A6_heat_kernel_check.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
