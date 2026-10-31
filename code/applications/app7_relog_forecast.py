"""A7.2: forecast test with repeated loggings of the same borehole (real data, held-out future log).

Pairs (same file stem and coordinates, similar deep gradients): US-nha 1964 -> 1992, US-nos 1965 -> 1992,
US-sar 1964 -> 1992, US-HAY 1980.5 -> 1995.5, CA-442-18 1986.8 -> CA-JM-b 1999.6.  Excluded on data grounds before
any fit: AU-10MMDD003 1970 -> 2011 (9 K offset) and CZ-srs-4 -> CZ-v454-1 (2.9 K offset): different holes.
Joint model: one surface-temperature history for both logs; shared (T0, Gamma) and a calibration offset delta for the
second log, all with flat priors; white noise sigma_eps (REML on log 1 only).  (H, sigma_T) from the pooled fit of
A7.1 (training half).  Forecast: predictive law of log 2 given log 1 alone, evaluated in the contrasts orthogonal to
the offset direction (the offset of log 2 is not predictable without log-2 data).  Compared priors: fitted H, H = 1/2,
and 'no change' (log 2 = log 1 interpolated, noise only).  Score: log predictive density of the contrasts, the
whitened chi^2 and the RMSE of the predicted change profile.
Output: results/A7_relog_forecast.json
"""
import json
import math
import os
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve, null_space

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from covfourier import heat                              # noqa: E402
import borehole_problem as BP                            # noqa: E402
import borehole_data as BD                               # noqa: E402
import app7_borehole_realdata as A7                      # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
PAIRS = [('US-nha64', 'US-nha92'), ('US-nos65', 'US-nos92'), ('US-sar64', 'US-sar92'), ('US-HAYSPR', 'US-HAY955'),
         ('CA-442-18', 'CA-JM-b')]


def load(stem):
    m, bl = BD.parse(os.path.join(BD.DATA, 'huang2016-' + stem + '.txt'))
    idx, _ = BD.attribute(m, bl)
    d = bl[idx][np.argsort(bl[idx][:, 0])]
    ok = (d[:, 0] >= 20.0) & (d[:, 0] <= 600.0)
    return float(m['log_year']), d[ok, 0], d[ok, 1]


def forecast(prep, sT, t1, z1, y1, t2, z2, y2, sE, sG=0.0):
    Z1, E1 = prep.readings(z1, np.full(len(z1), t1)); Z2, E2 = prep.readings(z2, np.full(len(z2), t2))
    Z = np.vstack([Z1, Z2]); E = np.r_[E1, E2]
    m1, m2 = len(z1), len(z2)
    zz = np.r_[z1, z2]
    K = sT ** 2 * (Z @ Z.T + np.diag(E)) + sG ** 2 * np.minimum.outer(zz, zz)   # same hole: same geology
    K[np.diag_indices_from(K)] += sE ** 2
    Hn = np.zeros((m1 + m2, 3)); Hn[:, 0] = 1.0; Hn[:, 1] = np.r_[z1, z2]; Hn[m1:, 2] = 1.0   # T0, Gamma, offset
    K11, K12, K22 = K[:m1, :m1], K[:m1, m1:], K[m1:, m1:]
    H1, H2 = Hn[:m1, :2], Hn[m1:, :2]                      # offset not identifiable from log 1
    cf = cho_factor(K11, lower=True)
    Ki_y = cho_solve(cf, y1); Ki_H = cho_solve(cf, H1); Ki_K12 = cho_solve(cf, K12)
    A = H1.T @ Ki_H
    beta = np.linalg.solve(A, H1.T @ Ki_y)
    mean = H2 @ beta + K12.T @ cho_solve(cf, y1 - H1 @ beta)
    R = H2.T - H1.T @ Ki_K12
    cov = K22 - K12.T @ Ki_K12 + R.T @ np.linalg.solve(A, R)
    # contrasts orthogonal to the offset direction (constant vector on log 2)
    Q = null_space(np.ones((1, m2)))
    r = Q.T @ (y2 - mean); C = Q.T @ cov @ Q
    cC = cho_factor(C, lower=True)
    w = np.linalg.solve(np.tril(cC[0]), r)
    chi2 = float(w @ w)
    lpd = float(-0.5 * (chi2 + 2 * np.sum(np.log(np.diag(cC[0]))) + len(r) * math.log(2 * math.pi)))
    # predicted change profile (log 2 minus log 1 interpolated), offset removed by centring both
    y1i = np.interp(z2, z1, y1)
    obs_change = y2 - y1i; pred_change = mean - y1i
    obs_c = obs_change - obs_change.mean(); pred_c = pred_change - pred_change.mean()
    return {'lpd_contrast': lpd, 'chi2': chi2, 'dof': int(len(r)), 'rmse_change_centred': float(np.sqrt(np.mean((obs_c - pred_c) ** 2))),
            'pred_change': pred_change.tolist(), 'obs_change': obs_change.tolist(),
            'pred_sd': np.sqrt(np.clip(np.diag(cov), 0, None)).tolist(), 'z2': z2.tolist()}


def no_change(z1, y1, z2, y2, sE):
    """Naive forecast: log 2 = log 1 interpolated; predictive covariance from noise of both logs (interpolation ignored)."""
    m2 = len(z2)
    y1i = np.interp(z2, z1, y1)
    Q = null_space(np.ones((1, m2)))
    r = Q.T @ (y2 - y1i); C = (2 * sE ** 2) * (Q.T @ Q)
    cC = cho_factor(C, lower=True)
    w = np.linalg.solve(np.tril(cC[0]), r)
    chi2 = float(w @ w)
    lpd = float(-0.5 * (chi2 + 2 * np.sum(np.log(np.diag(cC[0]))) + len(r) * math.log(2 * math.pi)))
    oc = (y2 - y1i) - np.mean(y2 - y1i)
    return {'lpd_contrast': lpd, 'chi2': chi2, 'dof': int(len(r)), 'rmse_change_centred': float(np.sqrt(np.mean(oc ** 2)))}


def main(n=256, H_fit=None, src='A7_borehole_realdata_v2.json'):
    fit = json.load(open(os.path.join(RES, src)))
    byH = fit['by_H']
    if H_fit is None:
        H_fit = max((float(h) for h in byH if float(h) != 0.5), key=lambda h: byH[str(h)]['ll_train'])
    priors = {'fitted H=%.2f' % H_fit: (H_fit, byH[str(H_fit)]['sigma_T'], byH[str(H_fit)].get('sigma_g', 0.0) or 0.0),
              'H=0.50': (0.5, byH['0.5']['sigma_T'], byH['0.5'].get('sigma_g', 0.0) or 0.0)}
    out = {'config': {'n': n, 'pairs': PAIRS, 'fit_source': src, 'priors': {k: list(v) for k, v in priors.items()}}, 'pairs': []}
    preps = {k: BP.Prepared(H, n, N_cand=8192, N_targ=1 << 14) for k, (H, _, _) in priors.items()}
    for a, b in PAIRS:
        t1, z1, y1 = load(a); t2, z2, y2 = load(b)
        rec = {'pair': [a, b], 't1': t1, 't2': t2, 'n1': len(z1), 'n2': len(z2), 'by_prior': {}}
        for name, (H, sT, sG) in priors.items():
            prep = preps[name]
            # sigma_eps by REML on log 1 alone (no target needed: use a dummy zero target)
            Z1, E1 = prep.readings(z1, np.full(len(z1), t1))
            g0 = np.zeros(prep.D)
            import app7_borehole_realdata_v2 as A72
            st = A72.Site({'T': y1, 'z': z1}, Z1, E1, g0, 0.0)
            sE, _ = st.best_e(sT, sG)
            rec['by_prior'][name] = forecast(prep, sT, t1, z1, y1, t2, z2, y2, sE, sG) | {'sigma_e': float(sE), 'sigma_T': sT, 'sigma_g': sG, 'H': H}
        sE0 = rec['by_prior'][list(priors)[0]]['sigma_e']
        rec['no_change'] = no_change(z1, y1, z2, y2, sE0)
        out['pairs'].append(rec)
        print(a, '->', b, ' '.join('%s: lpd %.1f chi2/dof %.2f rmse %.3f' % (k, v['lpd_contrast'], v['chi2'] / v['dof'], v['rmse_change_centred'])
                                   for k, v in list(rec['by_prior'].items()) + [('no change', rec['no_change'])]), flush=True)
    with open(os.path.join(RES, 'A7_relog_forecast.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    if sys.argv[1:2] == ['ms']:
        # hyperparameters from the model selection with the extended sigma_T grid (A7.11); H = 0.75 as the alternative
        main(H_fit=0.75, src='A7_borehole_realdata_ms.json')
    else:
        main()
