"""E8 real_boreholes: the single-history borehole model of the CovFourier package v1.3.0 with the evidence bracket added.

The v1.3.0 application code (read-only imports from borehole_model/repo_v1.3.0/code/applications) gives the 690 selected
profiles, the train/test split and the CovFourier coordinates (n = 256, N = 8192, endpoint-corrected; computed here
in memory, nothing is written to that folder).  Per site:
  y = T0 + Gamma z + sigma_T I_H(f) + sigma_g W(z) + eps,  eps ~ N(0, sigma_e^2 I),  (T0, Gamma) flat (REML).
Working model (v1.3.0): K_H ~ Z Z^T + diag(E2).  Bracket: C = sigma_T^2 Z Z^T + sigma_g^2 min(z_i, z_j) + sigma_e^2 I,
residual sigma_T^2 R (diag sigma_T^2 E2):  Ld = REML(C) - 1/2 sum log(1 + sigma_T^2 E2 / sigma_e^2),
U1 = -1/2 logdet-part(C) - 1/2 (y^T P y - sigma_T^2 (|P y|^T e)^2).  Hyperparameters on the v1.3.0 grids; sigma_e
profiled per site; L*(H), U*(H) = grid maxima of the sums over training sites.

Usage: python e8_real_boreholes.py  ->  results/E8_real_boreholes.json
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [os.environ.get('BOREHOLE_APP', ''),
               os.path.normpath(os.path.join(HERE, '..', '..', 'borehole_model', 'repo_v1.3.0', 'code', 'applications')),
               ]
APPDIR = next(p for p in _CANDIDATES if p and os.path.isdir(p))   # env BOREHOLE_APP or the bundled copy
sys.path.insert(0, HERE)
from common import save_json, Log                                           # noqa: E402
sys.path.insert(0, os.path.dirname(APPDIR)); sys.path.insert(0, APPDIR)
import numpy as np                                                          # noqa: E402
from scipy.linalg import cho_factor, cho_solve                              # noqa: E402
import borehole_problem as BP                                               # noqa: E402
import app7_borehole_realdata as V1                                         # noqa: E402
import app7_borehole_realdata_v2 as V2                                      # noqa: E402
import app7_model_selection as MS                                           # noqa: E402
from covfourier import heat as _heat                                        # noqa: E402

SIGMA_T = MS.SIGMA_T_EXT
SIGMA_G = V2.SIGMA_G_GRID
SIGMA_E = V2.SIGMA_E_GRID
L2PI = math.log(2 * math.pi)


def site_terms(y, z, ZZ, E2, sT, sG, sE, M, Hn):
    """(working REML, Ld, U1) for one site and one hyperparameter triple."""
    m = len(y)
    out = []
    for working in (True, False):
        K = sT ** 2 * ZZ + sG ** 2 * M
        if working:
            K[np.diag_indices_from(K)] += sT ** 2 * E2
        K[np.diag_indices_from(K)] += sE ** 2
        cf = cho_factor(K, lower=True, check_finite=False)
        Ki_y = cho_solve(cf, y, check_finite=False); Ki_H = cho_solve(cf, Hn, check_finite=False)
        A = Hn.T @ Ki_H
        cA = cho_factor(A, lower=True, check_finite=False)
        beta = cho_solve(cA, Hn.T @ Ki_y, check_finite=False)
        Py = Ki_y - Ki_H @ beta
        q = float(y @ Py)
        ld = 2 * float(np.sum(np.log(np.diag(cf[0])))) + 2 * float(np.sum(np.log(np.diag(cA[0]))))
        base = -0.5 * (ld + (m - 2) * L2PI)
        if working:
            out.append(base - 0.5 * q)
        else:
            Ld = base - 0.5 * q - 0.5 * float(np.sum(np.log1p(sT ** 2 * E2 / sE ** 2)))
            U1 = base - 0.5 * (q - sT ** 2 * float(np.abs(Py) @ np.sqrt(E2)) ** 2)
            out += [Ld, U1]
    return out


def main():
    log = Log('E8.log')
    cru = V1.Crutem()
    sites, reasons, n_files = V1.select_sites(cru)
    v2 = json.load(open(os.path.join(V1.RES, 'A7_borehole_realdata_v2.json')))
    is_train = np.array(v2['is_train'])
    log('sites %d (train %d), readings %d' % (len(sites), int(is_train.sum()), sum(len(s['z']) for s in sites)))
    msref = json.load(open(os.path.join(V1.RES, 'A7_model_selection.json')))['single_history']
    res = {'experiment': 'E8', 'n_sites': len(sites), 'n_train': int(is_train.sum()),
           'n_readings': int(sum(len(s['z']) for s in sites)), 'by_H': {}}
    tr = np.where(is_train)[0]; te = np.where(~is_train)[0]
    HS = tuple(float(h) for h in os.environ['E8_HS'].split(',')) if os.environ.get('E8_HS') else V2.HS
    OUTNAME = os.environ.get('E8_OUT', 'E8_real_boreholes.json')
    for H in HS:
        t0 = time.perf_counter()
        prep = BP.Prepared(H, 256, N_cand=8192, N_targ=1 << 16, win=_heat.Window(1500.0, 2030.0))
        Z, E2, Zt, Et, off, tim = V1.coordinates_for_sites(prep, sites)
        t_coord = time.perf_counter() - t0
        nS = len(sites)
        S = []
        for i, s in enumerate(sites):
            Zi = Z[off[i]:off[i + 1]]; e2 = np.maximum(E2[off[i]:off[i + 1]], 0.0)
            z = np.asarray(s['z'], float)
            S.append((np.asarray(s['T'], float), z, Zi @ Zi.T, e2, np.minimum.outer(z, z), np.stack([np.ones(len(z)), z], 1)))
        nT, nG, nE = len(SIGMA_T), len(SIGMA_G), len(SIGMA_E)
        W = np.empty((nT, nG, nS)); Ld = np.empty_like(W); U = np.empty_like(W)
        for a_, sT in enumerate(SIGMA_T):
            for b_, sG in enumerate(SIGMA_G):
                for i, (y, z, ZZ, e2, M, Hn) in enumerate(S):
                    vals = np.array([site_terms(y, z, ZZ, e2, sT, sG, sE, M, Hn) for sE in SIGMA_E])   # (nE, 3)
                    W[a_, b_, i], Ld[a_, b_, i], U[a_, b_, i] = vals.max(axis=0)
        Wtr, Ltr, Utr = W[:, :, tr].sum(axis=2), Ld[:, :, tr].sum(axis=2), U[:, :, tr].sum(axis=2)
        ia, ib = np.unravel_index(np.argmax(Wtr), Wtr.shape)
        rec = {'sigma_T_hat': SIGMA_T[ia], 'sigma_g_hat': SIGMA_G[ib], 'W_train': float(Wtr[ia, ib]),
               'W_test_at_hat': float(W[ia, ib, te].sum()), 'Ld_train_max': float(Ltr.max()), 'U_train_max': float(Utr.max()),
               'Ld_train_at_hat': float(Ltr[ia, ib]), 'U_train_at_hat': float(Utr[ia, ib]),
               'Ld_test_at_hat': float(Ld[ia, ib, te].sum()), 'U_test_at_hat': float(U[ia, ib, te].sum()),
               'W_inside_bracket_all_grid': bool(np.all(Ltr <= Wtr + 1e-6 * np.abs(Wtr)) and np.all(Utr >= Wtr - 1e-6 * np.abs(Wtr))),
               'ref_ll_train': msref[str(H)]['ll_train'], 'ref_ll_test': msref[str(H)]['ll_test'],
               'ref_sigma_T': msref[str(H)]['sigma_T'], 'ref_sigma_g': msref[str(H)]['sigma_g'],
               'max_E2_over_var': float(np.max(E2) / np.mean([np.var(s['T']) for s in sites])),
               'seconds_coords': t_coord, 'seconds_total': time.perf_counter() - t0}
        res['by_H'][str(H)] = rec
        log('H %.2f: W train %.1f (ref %.1f) | Ld* %.1f U* %.1f | test W %.1f Ld %.1f U %.1f | sT %.1f sg %.3f (%.0fs)' % (
            H, rec['W_train'], rec['ref_ll_train'], rec['Ld_train_max'], rec['U_train_max'], rec['W_test_at_hat'],
            rec['Ld_test_at_hat'], rec['U_test_at_hat'], rec['sigma_T_hat'], rec['sigma_g_hat'], rec['seconds_total']))
        save_json(res, OUTNAME)
    Ls = {H: r['Ld_train_max'] for H, r in res['by_H'].items()}
    Us = {H: r['U_train_max'] for H, r in res['by_H'].items()}
    thr = max(Ls.values())
    res['enclosure_H_grid'] = [float(H) for H in Us if Us[H] >= thr]
    res['decision'] = {'H_best_lower': max(Ls, key=Ls.get), 'excluded': [float(H) for H in Us if Us[H] < thr]}
    save_json(res, OUTNAME)
    log('enclosure', res['enclosure_H_grid'], 'decision', res['decision'])


if __name__ == '__main__':
    main()
