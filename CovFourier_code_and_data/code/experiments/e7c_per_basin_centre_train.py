"""E7c: per-basin estimates, decisions and enclosures for the CAMELS analysis with training-period preprocessing
(climatology, centring and scaling from the training years: E7 configuration 'centre_train'), on the full H grid
0.50:0.05:0.90 and 0.91:0.01:0.99, with the bracket.  Writes results/E7c_per_basin_centre_train.json."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import np, save_json, Log                  # noqa: E402
import e7_real_rivers as E7                             # noqa: E402


def main():
    log = Log('E7c.log')
    bas, excl, files = E7.load_basins()
    cfg = dict(L=730, clim='train', centre='train', mean='const', noise='free')
    Ys, trs, owners = [], [], []
    for g in sorted(bas):
        Yb, it, n_int = E7.make_blocks(bas[g], cfg['L'], cfg['clim'], cfg['centre'])
        Ys.append(Yb); trs.append(it); owners += [g] * Yb.shape[1]
    Y = np.hstack(Ys); is_tr = np.concatenate(trs); owners = np.array(owners)
    basins = {g: np.where(owners == g)[0] for g in sorted(bas)}
    grid = tuple(np.round(np.r_[np.arange(0.50, 0.905, 0.05), np.arange(0.91, 0.995, 0.01)], 2))
    t0 = time.perf_counter()
    rows = E7.analyse(Y, is_tr, n_int, grid, mean='const', noise='free', log=log, with_bracket=True, basins=basins)
    per = {}
    for g in sorted(bas):
        exb = {r['H']: r['exact_basin_gridmax'][g] for r in rows}
        Lb = {r['H']: r['Ld_basin_gridmax'][g] for r in rows}
        Ub = {r['H']: r['U_basin_gridmax'][g] for r in rows}
        Hhat = max(exb, key=exb.get)
        thr = max(Lb.values())
        inside = sorted(H for H in Ub if Ub[H] >= thr)
        per[g] = {'H_hat_exact': Hhat, 'delta_vs_half_exact': exb[Hhat] - exb[0.5], 'decided_vs_half': bool(thr > Ub[0.5]),
                  'margin_vs_half': thr - Ub[0.5], 'enclosure': [inside[0], inside[-1]], 'enclosure_set': inside}
        log('%s H %.2f d %.1f margin %.1f enc [%.2f, %.2f]' % (g, Hhat, per[g]['delta_vs_half_exact'], per[g]['margin_vs_half'], inside[0], inside[-1]))
    best = max(rows, key=lambda r: r['exact_train']); r05 = [r for r in rows if abs(r['H'] - 0.5) < 1e-9][0]
    thr = max(r['Ld_train_gridmax'] for r in rows)
    summ = {'H_hat_train': best['H'], 'delta_train_vs_half': best['exact_train'] - r05['exact_train'],
            'delta_test_vs_half': best['exact_test'] - r05['exact_test'],
            'margin_L_hat_minus_U_half': best['Ld_train_gridmax'] - r05['U_train_gridmax'],
            'enclosure_grid': [min(r['H'] for r in rows if r['U_train_gridmax'] >= thr), max(r['H'] for r in rows if r['U_train_gridmax'] >= thr)]}
    log('summary', summ)
    save_json({'experiment': 'E7c', 'config': cfg, 'H_grid': list(grid), 'rows': rows, 'summary': summ, 'per_basin': per,
               'seconds': time.perf_counter() - t0}, 'E7c_per_basin_centre_train.json')


if __name__ == '__main__':
    main()
