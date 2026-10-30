"""A7.7: regional model with groundwater advection in the steady geotherm (sensitivity of A7.6).

Vertical groundwater flow bends the steady geotherm (Bredehoeft & Papadopulos 1965):
    T(z) = T0 + (T_L - T0) (exp(beta z / L) - 1) / (exp(beta) - 1),  beta the Peclet number,
which to first order in beta adds a quadratic term to T0 + Gamma z.  It enters each site as a random effect
    a_s (z/100 m)^2,  a_s ~ N(0, s_q^2)  (pooled s_q, fitted by restricted likelihood; s_q = 0 is the model of A7.6),
so that the likelihoods of the two models are directly comparable.  Everything else as in app7_borehole_regional.py.
Output: results/A7_borehole_regional_adv.json and A7_regional_history_adv_<country>.json
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import app7_borehole_realdata as V1                      # noqa: E402
import app7_borehole_realdata_v2 as V2                   # noqa: E402
import app7_borehole_regional as R1                      # noqa: E402

RES = V1.RES
SC_GRID = (1.0, 1.4, 2.0, 2.8, 4.0)
SL_GRID = (1.0, 1.4, 2.0, 2.8, 4.0)
SG_GRID = (0.002, 0.004, 0.008)
SQ_GRID = (0.0, 0.01, 0.03, 0.1, 0.3)          # K per (100 m)^2


class SiteA(R1.SiteR):
    def __init__(self, s, Z, E2, g, e2g, sE):
        super().__init__(s, Z, E2, g, e2g, sE)
        q = (self.z / 100.0) ** 2
        self.QQ = np.outer(q, q)
        self.sQ = 0.0

    def local(self, sL, sG):
        K = sL ** 2 * self.GG + sG ** 2 * self.M + self.sQ ** 2 * self.QQ
        K[np.diag_indices_from(K)] += self.sE ** 2
        cf = cho_factor(K, lower=True, check_finite=False)
        KiH = cho_solve(cf, self.Hn, check_finite=False)
        A = self.Hn.T @ KiH
        cA = cho_factor(A, lower=True, check_finite=False)
        def Pi(V):
            KiV = cho_solve(cf, V, check_finite=False)
            return KiV - KiH @ cho_solve(cA, self.Hn.T @ KiV, check_finite=False)
        ld = 2 * np.sum(np.log(np.diag(cf[0]))) + 2 * np.sum(np.log(np.diag(cA[0])))
        return Pi, ld


def set_sq(S, sQ):
    for lst in S.values():
        for st in lst:
            st.sQ = sQ


def main(H='0.5', n=256, src='A7_borehole_realdata_v2.json'):
    fit = json.load(open(os.path.join(RES, src)))
    cru = V1.Crutem()
    sites, _, _ = V1.select_sites(cru)
    countries = {}
    for i, s in enumerate(sites):
        countries.setdefault(s['country'], []).append(i)
    countries = {c: v for c, v in countries.items() if len(v) >= R1.MIN_SITES}
    t0 = time.perf_counter()
    Z, E2, Zt, Et, off, _ = V2.cached_coordinates(float(H), n, sites)
    sE = [p['sigma_e'] for p in fit['by_H'][str(H)]['site']]
    S = {c: [SiteA(sites[i], Z[off[i]:off[i + 1]], E2[off[i]:off[i + 1]], Zt[i], Et[i], sE[i]) for i in idx]
         for c, idx in countries.items()}
    prof = {}
    for sQ in SQ_GRID:
        set_sq(S, sQ)
        for sC in SC_GRID:
            for sL in SL_GRID:
                for sG in SG_GRID:
                    prof[(sC, sL, sG, sQ)] = sum(R1.country_fit(S[c], sC, sL, sG) for c in S)
        best = max((k for k in prof if k[3] == sQ), key=prof.get)
        print('s_q %.2f best %s ll %.1f' % (sQ, best[:3], prof[best]), flush=True)
    sC, sL, sG, sQ = max(prof, key=prof.get)
    set_sq(S, sQ)
    rec = {'sigma_C': sC, 'sigma_L': sL, 'sigma_g': sG, 'sigma_q': sQ, 'll': float(prof[(sC, sL, sG, sQ)]),
           'll_best_by_sq': {str(q): float(max(v for k, v in prof.items() if k[3] == q)) for q in SQ_GRID},
           'countries': {}, 'sites': {}}
    for c, idx in countries.items():
        _, res, reg = R1.country_fit(S[c], sC, sL, sG, predict=True)
        W = [sites[i]['inst']['W'] for i in idx]; sdi = [sites[i]['inst']['sd'] for i in idx]
        rec['countries'][c] = reg | {'n': len(idx), 'W_inst': float(np.mean(W)),
                                     'sd_inst': float(math.sqrt(np.sum(np.square(sdi))) / len(idx))}
        for i, r in zip(idx, res):
            rec['sites'][sites[i]['file']] = r | {'W_inst': sites[i]['inst']['W'], 'sd_inst': sites[i]['inst']['sd'], 'country': c}
    rec['seconds'] = time.perf_counter() - t0
    out = {'config': {'H': H, 'n': n, 'sC_grid': SC_GRID, 'sL_grid': SL_GRID, 'sG_grid': SG_GRID, 'sQ_grid': SQ_GRID,
                      'model': 'A7.6 + advection random effect a_s (z/100 m)^2, a_s ~ N(0, s_q^2)'}, 'by_H': {H: rec}}
    with open(os.path.join(RES, 'A7_borehole_regional_adv.json'), 'w') as fh:
        json.dump(out, fh, indent=0)
    print('selected sigma_C %.2f sigma_L %.2f sigma_g %.3f sigma_q %.2f ll %.1f (%.0fs)' % (sC, sL, sG, sQ, rec['ll'], rec['seconds']))
    for c in sorted(rec['countries'], key=lambda c: rec['countries'][c]['W_inst']):
        v = rec['countries'][c]
        print('  %-26s inst %.2f  borehole %.2f +- %.2f' % (c, v['W_inst'], v['mu'], v['sd']))


def by_country(H='0.5', n=256, grid=(0.0, 0.01, 0.03, 0.1, 0.3, 1.0), src='A7_borehole_realdata_v2.json'):
    """Country-specific hydrogeology: profile s_q separately for each country (its likelihood term is separable),
    with (sigma_C, sigma_L, sigma_g) fixed at the pooled fit; reports the country mean at every s_q."""
    fit = json.load(open(os.path.join(RES, src)))
    A = json.load(open(os.path.join(RES, 'A7_borehole_regional_adv.json')))['by_H'][H]
    sC, sL, sG = A['sigma_C'], A['sigma_L'], A['sigma_g']
    cru = V1.Crutem()
    sites, _, _ = V1.select_sites(cru)
    countries = {}
    for i, s in enumerate(sites):
        countries.setdefault(s['country'], []).append(i)
    countries = {c: v for c, v in countries.items() if len(v) >= R1.MIN_SITES}
    Z, E2, Zt, Et, off, _ = V2.cached_coordinates(float(H), n, sites)
    sE = [p['sigma_e'] for p in fit['by_H'][str(H)]['site']]
    out = {'config': {'H': H, 'n': n, 'grid': grid, 'sigma_C': sC, 'sigma_L': sL, 'sigma_g': sG}, 'countries': {}}
    for c, idx in countries.items():
        S = [SiteA(sites[i], Z[off[i]:off[i + 1]], E2[off[i]:off[i + 1]], Zt[i], Et[i], sE[i]) for i in idx]
        rows = []
        for sQ in grid:
            for st in S:
                st.sQ = sQ
            ll, _, reg = R1.country_fit(S, sC, sL, sG, predict=True)
            rows.append({'sigma_q': sQ, 'll': float(ll), 'mu': reg['mu'], 'sd': reg['sd']})
        best = max(rows, key=lambda r: r['ll'])
        out['countries'][c] = {'n': len(idx), 'profile': rows, 'best': best,
                               'max_depth_median': float(np.median([sites[i]['z'].max() for i in idx]))}
        print('%-26s n %3d  depth %4.0f m  best s_q %.2f  mean %.2f +- %.2f  (s_q=0: %.2f; ll gain %.1f)' % (
            c, len(idx), out['countries'][c]['max_depth_median'], best['sigma_q'], best['mu'], best['sd'], rows[0]['mu'],
            best['ll'] - rows[0]['ll']), flush=True)
    with open(os.path.join(RES, 'A7_borehole_regional_adv_by_country.json'), 'w') as fh:
        json.dump(out, fh, indent=0)


if __name__ == '__main__':
    if sys.argv[1:2] == ['by_country']:
        by_country(*(sys.argv[2:3] or ['0.5']))
    else:
        main(*(sys.argv[1:2] or ['0.5']))
