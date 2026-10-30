"""A7.4: posterior surface-temperature histories for example sites (decadal means), with the CRUTEM5 cell series.

Sites: the two selected sites with the most readings (deterministic rule), and the second log of each repeat pair.
Targets: decadal means of dT from 1500 to the logging decade, each relative to the 1901-1930 mean, as one batch of
integrands through the same basis; joint posterior mean and marginal sd (REML hyperparameters of the v2 fit).
Output: results/A7_site_history.json
"""
import json
import math
import os
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from covfourier import heat                              # noqa: E402
import borehole_problem as BP                            # noqa: E402
import app7_borehole_realdata as V1                      # noqa: E402
import app7_borehole_realdata_v2 as V2                   # noqa: E402

RES = V1.RES


def posterior_history(prep, s, sT, sG, H):
    z, y, tl = s['z'], s['T'], s['t_log']
    Z, E2 = prep.readings(z, np.full(len(z), tl))
    decs = np.arange(1500, int(tl // 10) * 10, 10)
    u = heat.midpoints(prep.N_targ)
    F = np.vstack([heat.epoch_mean_values(d, d + 10, u, prep.win) - heat.epoch_mean_values(1901, 1931, u, prep.win) for d in decs])
    Zt, Et = [], []
    for i0 in range(0, len(F), 20):
        a, b, _ = heat.project_samples(prep.bp_t, F[i0:i0 + 20]); Zt.append(a); Et.append(b)
    Zt = np.vstack(Zt); Et = np.concatenate(Et)
    st = V2.Site(s, Z, E2, Zt[0], Et[0])
    sE, _ = st.best_e(sT, sG)
    m = len(y)
    K = sT ** 2 * st.G + sG ** 2 * st.M; K[np.diag_indices_from(K)] += sE ** 2
    cf = cho_factor(K, lower=True)
    Hn = st.Hn
    Ki_H = cho_solve(cf, Hn); A = Hn.T @ Ki_H
    beta = np.linalg.solve(A, Hn.T @ cho_solve(cf, y))
    r = y - Hn @ beta
    Kt = sT ** 2 * (Z @ Zt.T)                                     # (m, n_dec)
    mu = Kt.T @ cho_solve(cf, r)
    Ki_Kt = cho_solve(cf, Kt)
    R = Hn.T @ Ki_Kt
    Vt = sT ** 2 * (Zt @ Zt.T + np.diag(Et)) - Kt.T @ Ki_Kt + R.T @ np.linalg.solve(A, R)
    return {'decades': decs.tolist(), 'mu': mu.tolist(), 'sd': np.sqrt(np.clip(np.diag(Vt), 0, None)).tolist(),
            'sigma_e': float(sE), 'T0': float(beta[0]), 'Gamma_K_per_km': float(1e3 * beta[1]),
            'residual_profile': {'z': z.tolist(), 'reduced_T': (y - Hn @ beta).tolist()}}


def main(n=256, src='A7_borehole_realdata_v2.json'):
    fit = json.load(open(os.path.join(RES, src)))
    byH = fit['by_H']
    H = max((float(h) for h in byH), key=lambda h: byH[str(h)]['ll_train'])
    sT, sG = byH[str(H)]['sigma_T'], byH[str(H)].get('sigma_g', 0.0) or 0.0
    cru = V1.Crutem()
    sites, _, _ = V1.select_sites(cru)
    order = sorted(range(len(sites)), key=lambda i: -len(sites[i]['z']))
    pick = order[:2]
    prep = BP.Prepared(H, n, N_cand=8192, N_targ=1 << 16)
    out = {'H': H, 'sigma_T': sT, 'sigma_g': sG, 'sites': []}
    for i in pick:
        s = sites[i]
        rec = posterior_history(prep, s, sT, sG, H)
        a = cru.annual(s['lat'], s['lon'])
        yrs = sorted(a)
        ref = np.mean([a[y] for y in range(1901, 1931) if y in a])
        rec.update({'site': s['site'], 'file': s['file'], 'country': s['country'], 't_log': s['t_log'], 'lat': s['lat'], 'lon': s['lon'],
                    'crutem_years': yrs, 'crutem_anom_rel_1901_1930': [a[y] - ref for y in yrs]})
        out['sites'].append(rec)
        print(s['site'], s['country'], len(s['z']), 'readings; last decade mu %.2f sd %.2f' % (rec['mu'][-1], rec['sd'][-1]), flush=True)
    with open(os.path.join(RES, 'A7_site_history.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
