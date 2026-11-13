"""A7.11: model selection for the real borehole profiles: the role of H, fit against prediction.

In v1.1.0 the single-history model (A7.1 v2) was compared across H with sigma_T on a grid ending at 5.6 K, and for
H >= 0.65 the optimum sat at that edge, so those likelihoods were understated.  Here the grid is extended until the
optimum is interior, and three questions are answered separately for every H:
  fit                  restricted log-likelihood of the training half (345 profiles), the criterion that selects H;
  new profiles         log-likelihood of the held-out half at the training hyperparameters (prediction of unseen data);
  independent target   the CRUTEM5 window change W at the held-out sites, scored by coverage, CRPS and squared error,
                       with the interannual noise of the cell and with the target error of A7.9 (station archive).
The same comparison is made for the 500-2030 window, and the regional model (A7.6) is refitted at H = 0.75.
Output: results/A7_model_selection.json (regional H = 0.75 also in results/A7_borehole_regional_H075.json)
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import app7_borehole_realdata as V1                      # noqa: E402
import app7_borehole_realdata_v2 as V2                   # noqa: E402
import app7_borehole_regional as R1                      # noqa: E402
import app7_target_error as TE                           # noqa: E402
import app7_station_validation as SV                     # noqa: E402

RES = V1.RES
SIGMA_T_EXT = tuple(V2.SIGMA_T_GRID) + (8.0, 11.3, 16.0, 22.6, 32.0)


def crps(x, mu, sd):
    z = (x - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))


def scores(W, mu, sd, tgt):
    z = (W - mu) / np.sqrt(sd ** 2 + tgt ** 2)
    return {'cover90': float(np.mean(np.abs(z) <= 1.645)), 'cover50': float(np.mean(np.abs(z) <= 0.6745)),
            'z_sd': float(np.std(z, ddof=1)), 'crps': float(np.mean(crps(W, mu, np.sqrt(sd ** 2 + tgt ** 2)))),
            'mse': float(np.mean((W - mu) ** 2)), 'mse_prior_only': float(np.mean(W ** 2)),
            'corr': float(np.corrcoef(mu, W)[0, 1])}


def single_history(sites, is_train, H, n, t0w, sd_target):
    t0 = time.perf_counter()
    Z, E2, Zt, Et, off, _ = V2.cached_coordinates(H, n, sites, t0w)
    S = [V2.Site(s, Z[off[i]:off[i + 1]], E2[off[i]:off[i + 1]], Zt[i], Et[i]) for i, s in enumerate(sites)]
    tr = np.where(is_train)[0]
    prof = {(sT, sG): sum(S[i].best_e(sT, sG)[1] for i in tr) for sT in SIGMA_T_EXT for sG in V2.SIGMA_G_GRID}
    sT, sG = max(prof, key=prof.get)
    mu, sd, ll = np.zeros(len(S)), np.zeros(len(S)), np.zeros(len(S))
    for i, st in enumerate(S):
        sE, _ = st.best_e(sT, sG)
        ll[i], mu[i], var, _, _ = st.terms(sT, sG, sE, predict=True)
        sd[i] = math.sqrt(max(var, 0.0))
    te = ~is_train
    W = np.array([s['inst']['W'] for s in sites]); sdi = np.array([s['inst']['sd'] for s in sites])
    best_old = max((v, k) for k, v in prof.items() if k[0] <= 5.6)
    return {'H': H, 'n': n, 't0': t0w, 'sigma_T': sT, 'sigma_g': sG,
            'sigma_T_at_edge': bool(sT == SIGMA_T_EXT[-1]), 'sigma_g_at_edge': bool(sG == V2.SIGMA_G_GRID[-1]),
            'll_train': float(prof[(sT, sG)]), 'll_test': float(ll[te].sum()),
            'll_train_old_grid': float(best_old[0]), 'sigma_T_old_grid': best_old[1][0],
            'heldout_interannual': scores(W[te], mu[te], sd[te], sdi[te]),
            'heldout_target_error': scores(W[te], mu[te], sd[te], sd_target[te]),
            'mean_mu_all': float(mu.mean()), 'mean_W_all': float(W.mean()), 'seconds': time.perf_counter() - t0}


def main():
    v2 = json.load(open(os.path.join(RES, 'A7_borehole_realdata_v2.json')))
    is_train = np.array(v2['is_train'])
    cru = V1.Crutem()
    sites, _, _ = V1.select_sites(cru)
    st = SV.station_index()
    vg = json.load(open(os.path.join(RES, 'A7_target_error.json')))['variogram']
    sd_target = np.array([math.sqrt(TE.target_variance(st, vg, [s], [TE.cell_stations(st, cru, s)[0]])) for s in sites])
    out = {'config': {'sigma_T_grid': SIGMA_T_EXT, 'sigma_g_grid': list(V2.SIGMA_G_GRID),
                      'n_train': int(is_train.sum()), 'n_test': int((~is_train).sum()),
                      'target_error': 'site cell value as an estimate of the regional change, A7.9 variogram'},
           'single_history': {}, 'single_history_window500': {}, 'regional': {}}
    for H in V2.HS:
        r = single_history(sites, is_train, H, 256, 1500.0, sd_target)
        out['single_history'][str(H)] = r
        print('H %.2f sigma_T %5.1f%s sigma_g %.3f | fit %.1f (old grid %.1f at %.1f) | new profiles %.1f | '
              'CRUTEM5 cover90 %.2f / %.2f crps %.3f mse %.3f corr %.2f (%.0fs)' % (
                  H, r['sigma_T'], '*' if r['sigma_T_at_edge'] else ' ', r['sigma_g'], r['ll_train'], r['ll_train_old_grid'],
                  r['sigma_T_old_grid'], r['ll_test'], r['heldout_interannual']['cover90'], r['heldout_target_error']['cover90'],
                  r['heldout_target_error']['crps'], r['heldout_target_error']['mse'], r['heldout_target_error']['corr'],
                  r['seconds']), flush=True)
        with open(os.path.join(RES, 'A7_model_selection.json'), 'w') as fh:
            json.dump(out, fh, indent=1)
    for H in (0.5, 0.75):
        r = single_history(sites, is_train, H, 1024, 500.0, sd_target)
        out['single_history_window500'][str(H)] = r
        print('window 500: H %.2f sigma_T %.1f fit %.1f new profiles %.1f' % (H, r['sigma_T'], r['ll_train'], r['ll_test']), flush=True)
    regional(out)


def regional(out):
    """Regional model at H = 0.75 next to the stored H = 1/2 fit.  Smoother priors need larger amplitudes, so the grids
    are extended upwards until the optimum is interior (a first run with sigma_C <= 8 ended at that edge)."""
    grid_C = tuple(R1.SC_GRID) + (5.6, 8.0, 11.3, 16.0)
    grid_L = tuple(R1.SL_GRID) + (5.6, 8.0, 11.3)
    R1.SC_GRID, R1.SL_GRID = grid_C, grid_L
    R1.main(H_list=['0.75'], out_name='A7_borehole_regional_H075.json')
    tgt = json.load(open(os.path.join(RES, 'A7_target_error.json')))['countries_target']
    grids = {'0.5': (json.load(open(os.path.join(RES, 'A7_borehole_regional.json')))['config']['sC_grid'],
                     json.load(open(os.path.join(RES, 'A7_borehole_regional.json')))['config']['sL_grid']),
             '0.75': (grid_C, grid_L)}
    for H, f in (('0.5', 'A7_borehole_regional.json'), ('0.75', 'A7_borehole_regional_H075.json')):
        rr = json.load(open(os.path.join(RES, f)))['by_H'][H]
        C = rr['countries']
        z = {c: (v['mu'] - v['W_inst']) / math.hypot(v['sd'], tgt[c]['sd_target']) for c, v in C.items()}
        gC, gL = grids[H]
        out['regional'][H] = {'sigma_C': rr['sigma_C'], 'sigma_L': rr['sigma_L'], 'sigma_g': rr['sigma_g'], 'll': rr['ll'],
                              'sigma_C_at_edge': rr['sigma_C'] == max(gC), 'sigma_L_at_edge': rr['sigma_L'] == max(gL),
                              'inside95_target_error': int(sum(abs(v) <= 1.96 for v in z.values())), 'n_countries': len(z),
                              'rmse_country': float(np.sqrt(np.mean([(v['mu'] - v['W_inst']) ** 2 for v in C.values()]))),
                              'z': z}
        print('regional H %s: sigma_C %.1f sigma_L %.1f sigma_g %.3f ll %.1f, countries inside 95%% %d/%d, rmse %.2f K' % (
            H, rr['sigma_C'], rr['sigma_L'], rr['sigma_g'], rr['ll'], out['regional'][H]['inside95_target_error'],
            len(z), out['regional'][H]['rmse_country']), flush=True)
    with open(os.path.join(RES, 'A7_model_selection.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    if sys.argv[1:2] == ['regional']:
        regional(json.load(open(os.path.join(RES, 'A7_model_selection.json'))))
    else:
        main()
