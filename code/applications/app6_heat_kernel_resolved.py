"""A6.1b: the refinement rule closes the Fourier-stage floor of A6.1; comparison with product quadrature.

Same 25 integrands as A6.1 (app6_heat_kernel_check.py).  Readings are sampled with N_cand in {8192, 32768}, the five
targets (kinks and interior square-root cusps) with N_targ = 2^18, all through ONE prepared basis (two projectors on
the same CovarianceFourier).  Reference: covfourier.khref.KHQuad (h_max = 1/128).  Competitor: two-triangle
Gauss-Jacobi product quadrature (A1) with n_q in {400, 800, 1600, 3200} nodes.  Errors are relative to
sqrt(K_ii K_jj), reported by block (reading-reading, reading-target, target-target), with the Cauchy-Schwarz margin.
Output: results/A6_heat_kernel_resolved.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from covfourier.khref import KHQuad                      # noqa: E402
from covfourier.engine import CovarianceFourier          # noqa: E402
from covfourier.batch import BatchProjector              # noqa: E402
from covfourier.comparators import triangle_gj_covariance   # noqa: E402
from covfourier import heat                              # noqa: E402
import app6_heat_kernel_check as E1                      # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
NK = 2 * len(E1.DEPTHS)


def blocks(err):
    return {'kk': float(err[:NK, :NK].max()), 'kt': float(err[:NK, NK:].max()), 'tt': float(err[NK:, NK:].max())}


def main(Hs=(0.6, 0.75, 0.9), ns=(128, 256, 512), Ncs=(8192, 32768), Nt=1 << 18, nqs=(400, 800, 1600, 3200)):
    feats = heat.features(E1.WIN, E1.TLOGS, E1.EPOCHS, E1.HEAT_T)
    out = {'config': {'ns': list(ns), 'N_cand': list(Ncs), 'N_targ': Nt, 'n_q': list(nqs), 'labels': E1.labels()}, 'cases': []}
    ker = lambda u: np.vstack([heat.borehole_values(E1.DEPTHS, t, u, E1.WIN) for t in E1.TLOGS])
    tgt = lambda u: E1.integrand_values(u)[NK:]
    for H in Hs:
        t0 = time.perf_counter()
        K = KHQuad(H, features=feats, h_max=1 / 128).gram(E1.integrand_values)
        t_ref = time.perf_counter() - t0
        dg = np.sqrt(np.outer(np.diag(K), np.diag(K)))
        rows = []
        for n in ns:
            cf = CovarianceFourier(H, n, enrich=True, Naux=Nt)
            bpt = BatchProjector(cf, N=Nt, dense_rows_limit=2e6)
            Zt, Et, _ = heat.project_samples(bpt, tgt(heat.midpoints(Nt)))
            for Nc in Ncs:
                bpc = BatchProjector(cf, N=Nc)
                Zk, Ek, _ = heat.project_samples(bpc, ker(heat.midpoints(Nc)))
                Z = np.vstack([Zk, Zt]); E = np.r_[Ek, Et]
                d = np.abs(K - Z @ Z.T)
                rows.append({'n': n, 'D': n + 2, 'N_cand': Nc, 'err': blocks(d / dg),
                             'cs_margin_rel': float(np.max((d - np.sqrt(np.outer(E, E))) / dg))})
                print(H, n, Nc, rows[-1], flush=True)
        quad = []
        for nq in nqs:
            Kq, ts = triangle_gj_covariance(lambda s: E1.integrand_values(np.asarray(s).ravel()).reshape((-1,) + np.shape(s)), H, nq)
            quad.append({'n_q': nq, 'err': blocks(np.abs(Kq - K) / dg), 'seconds': ts})
            print(H, 'quad', nq, quad[-1], flush=True)
        out['cases'].append({'H': H, 'reference_s': t_ref, 'expansion': rows, 'product_quadrature': quad})
        with open(os.path.join(RES, 'A6_heat_kernel_resolved.json'), 'w') as fh:
            json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
