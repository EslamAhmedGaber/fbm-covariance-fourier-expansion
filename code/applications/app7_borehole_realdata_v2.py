"""A7.1 v2: real borehole profiles with a geological-noise component; validated against CRUTEM5.

Change against v1 (app7_borehole_realdata.py, kept as a documented step): v1 put every non-climatic deviation into
white noise, and its pooled sigma_T sat at the upper end of the grid (2.8 K) with the likelihood preferring the
roughest prior H = 1/2.  Conductivity layering perturbs the steady geotherm by q int_0^z delta(1/k): an integrated,
random-walk-like deviation in depth.  v2 adds this term with covariance sigma_g^2 min(z_i, z_j) (pooled sigma_g;
its constant and linear parts are absorbed by the flat (T0, Gamma) prior), widens the sigma_T grid and caches the
coordinates.  Everything else (selection, target, REML, split, CRUTEM5 only for validation) is as in v1.
Model per site: y = T0 + Gamma z + sigma_T I_H(f) + sigma_g W(z) + eps,  eps ~ N(0, sigma_eps_s^2 I).
Output: results/A7_borehole_realdata_v2.json; cache results/cache/A7_coords_H*.npz
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import borehole_problem as BP                            # noqa: E402
import app7_borehole_realdata as V1                      # noqa: E402

RES = V1.RES
CACHE = os.path.join(RES, 'cache')
HS = (0.5, 0.55, 0.65, 0.75, 0.85, 0.95)
SIGMA_T_GRID = (0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6)
SIGMA_G_GRID = (0.0, 0.002, 0.004, 0.008, 0.016, 0.032)          # K / sqrt(m)
SIGMA_E_GRID = np.exp(np.linspace(np.log(0.003), np.log(0.3), 13))


def cached_coordinates(H, n, sites, t0w=1500.0):
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.sha256(json.dumps([[s['file'], s['t_log'], len(s['z'])] for s in sites]).encode()).hexdigest()[:12]
    tag = '' if t0w == 1500.0 else f'_w{int(t0w)}'
    path = os.path.join(CACHE, f'A7_coords_H{H}_n{n}{tag}_{key}.npz')
    if os.path.exists(path):
        f = np.load(path)
        return f['Z'], f['E2'], f['Zt'], f['Et'], f['off'], json.loads(str(f['timing']))
    from covfourier import heat as _heat
    prep = BP.Prepared(H, n, N_cand=8192, N_targ=1 << 16, win=_heat.Window(t0w, 2030.0))
    Z, E2, Zt, Et, off, tim = V1.coordinates_for_sites(prep, sites)
    tim['prep_s'] = prep.prep_s
    np.savez(path, Z=Z, E2=E2, Zt=Zt, Et=Et, off=off, timing=json.dumps(tim))
    return Z, E2, Zt, Et, off, tim


class Site:
    """Per-site matrices for fast REML over (sigma_T, sigma_g, sigma_eps)."""

    def __init__(self, s, Z, E2, g, e2g):
        self.y, self.z = s['T'], s['z']
        self.m = len(self.y)
        self.G = Z @ Z.T + np.diag(E2)
        self.M = np.minimum.outer(self.z, self.z)
        self.Hn = np.stack([np.ones(self.m), self.z], axis=1)
        self.k1 = Z @ g
        self.v1 = float(g @ g + e2g)

    def terms(self, sT, sG, sE, predict=False):
        K = sT ** 2 * self.G + sG ** 2 * self.M
        K[np.diag_indices_from(K)] += sE ** 2
        cf = cho_factor(K, lower=True, check_finite=False)
        Ki_y = cho_solve(cf, self.y, check_finite=False); Ki_H = cho_solve(cf, self.Hn, check_finite=False)
        A = self.Hn.T @ Ki_H
        cA = cho_factor(A, lower=True, check_finite=False)
        beta = cho_solve(cA, self.Hn.T @ Ki_y, check_finite=False)
        r = self.y - self.Hn @ beta
        Ki_r = cho_solve(cf, r, check_finite=False)
        ll = -0.5 * (r @ Ki_r + 2 * np.sum(np.log(np.diag(cf[0]))) + 2 * np.sum(np.log(np.diag(cA[0])))
                     + (self.m - 2) * math.log(2 * math.pi))
        if not predict:
            return ll
        k = sT ** 2 * self.k1
        mu = float(k @ Ki_r)
        Ki_k = cho_solve(cf, k, check_finite=False)
        R = self.Hn.T @ Ki_k
        var = float(sT ** 2 * self.v1 - k @ Ki_k + R @ cho_solve(cA, R, check_finite=False))
        # standardized residuals of the profile (leave-nothing-out; for a model check only)
        Lr = np.linalg.solve(np.tril(cf[0]), r)
        return ll, mu, var, beta, Lr

    def best_e(self, sT, sG):
        lls = [self.terms(sT, sG, sE) for sE in SIGMA_E_GRID]
        i = int(np.argmax(lls))
        return SIGMA_E_GRID[i], lls[i]


def main(n=256, seed=3, t0w=1500.0, Hs=HS, out_name='A7_borehole_realdata_v2.json'):
    cru = V1.Crutem()
    sites, reasons, n_files = V1.select_sites(cru)
    rng = np.random.default_rng(seed)
    train = rng.permutation(len(sites))[: len(sites) // 2]
    is_train = np.zeros(len(sites), bool); is_train[train] = True
    out = {'config': {'n': n, 'Hs': list(HS), 'sigma_T_grid': list(SIGMA_T_GRID), 'sigma_g_grid': list(SIGMA_G_GRID),
                      'sigma_e_grid': SIGMA_E_GRID.tolist(), 'early_window': V1.EARLY, 'recent_window': '[t_log-30, t_log-1]',
                      'window': [t0w, 2030], 'seed': seed, 'selection_rejections': reasons, 'n_files': n_files,
                      'n_selected': len(sites), 'model': 'T0 + Gamma z + sigma_T I_H(f) + sigma_g W(z) + eps'},
           'sites': [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items() if k not in ('z', 'T')}
                     | {'n_readings': int(len(s['z']))} for s in sites],
           'is_train': is_train.tolist(), 'by_H': {}}
    for H in Hs:
        t0 = time.perf_counter()
        Z, E2, Zt, Et, off, tim = cached_coordinates(H, n, sites, t0w)
        t_coord = time.perf_counter() - t0
        S = [Site(s, Z[off[i]:off[i + 1]], E2[off[i]:off[i + 1]], Zt[i], Et[i]) for i, s in enumerate(sites)]
        prof = {}
        for sT in SIGMA_T_GRID:
            for sG in SIGMA_G_GRID:
                prof[(sT, sG)] = sum(S[i].best_e(sT, sG)[1] for i in np.where(is_train)[0])
        sT, sG = max(prof, key=prof.get)
        per = []
        for i, st in enumerate(S):
            sE, _ = st.best_e(sT, sG)
            ll, mu, var, beta, Lr = st.terms(sT, sG, sE, predict=True)
            per.append({'sigma_e': float(sE), 'll': float(ll), 'mu': mu, 'sd': math.sqrt(max(var, 0.0)),
                        'T0': float(beta[0]), 'Gamma_K_per_km': float(1e3 * beta[1]),
                        'std_resid_sd': float(np.std(Lr)), 'std_resid_lag1': float(np.corrcoef(Lr[:-1], Lr[1:])[0, 1]) if len(Lr) > 3 else None})
        rec = {'sigma_T': sT, 'sigma_g': sG, 'll_train': float(prof[(sT, sG)]),
               'll_test': float(sum(p['ll'] for p, tr in zip(per, is_train) if not tr)),
               'profile_ll_train': {f'{a}|{b}': v for (a, b), v in prof.items()},
               'timing': tim | {'coords_or_cache_s': t_coord}, 'site': per, 'seconds': time.perf_counter() - t0}
        out['by_H'][str(H)] = rec
        print('H', H, 'sigma_T', sT, 'sigma_g', sG, 'll_train %.1f ll_test %.1f' % (rec['ll_train'], rec['ll_test']),
              '%.1fs' % rec['seconds'], flush=True)
        with open(os.path.join(RES, out_name), 'w') as fh:
            json.dump(out, fh, indent=0)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'window':
        # window sensitivity: history from t0w (argv[2]) with n = argv[3], H values argv[4:]
        main(n=int(sys.argv[3]), t0w=float(sys.argv[2]), Hs=tuple(float(h) for h in sys.argv[4:]),
             out_name=f'A7_borehole_realdata_v2_window{int(float(sys.argv[2]))}.json')
    else:
        main()
