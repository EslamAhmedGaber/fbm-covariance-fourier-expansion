"""E7b: per-basin estimates, decisions and enclosures of E7 (main configuration) on the full H grid
(0.50, 0.55, ..., 0.90 from E7 plus 0.91, ..., 0.99 computed here), so that no basin estimate sits at a grid edge
by construction.  Writes results/E7b_per_basin.json."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import np, save_json, Log, RESULTS          # noqa: E402
import e7_real_rivers as E7                             # noqa: E402


def main():
    log = Log('E7b.log')
    bas, excl, files = E7.load_basins()
    cfg = dict(L=730, clim='all', centre='block', mean='const', noise='free')
    Ys, trs, owners = [], [], []
    for g in sorted(bas):
        Yb, it, n_int = E7.make_blocks(bas[g], cfg['L'], cfg['clim'], cfg['centre'])
        Ys.append(Yb); trs.append(it); owners += [g] * Yb.shape[1]
    Y = np.hstack(Ys); is_tr = np.concatenate(trs); owners = np.array(owners)
    basins = {g: np.where(owners == g)[0] for g in sorted(bas)}
    fine = tuple(np.round(np.arange(0.91, 0.995, 0.01), 2))
    fine = tuple(h for h in fine if abs(h - 0.95) > 1e-9)
    t0 = time.perf_counter()
    rows_f = E7.analyse(Y, is_tr, n_int, fine, mean='const', noise='free', log=log, with_bracket=True, basins=basins)
    prev = json.load(open(os.path.join(RESULTS, 'E7_real_rivers.json')))['runs']['main']['rows']
    rows = sorted([r for r in prev if 'exact_basin_gridmax' in r] + rows_f, key=lambda r: r['H'])
    per = {}
    for g in sorted(bas):
        exb = {r['H']: r['exact_basin_gridmax'][g] for r in rows}
        Lb = {r['H']: r['Ld_basin_gridmax'][g] for r in rows}
        Ub = {r['H']: r['U_basin_gridmax'][g] for r in rows}
        Hhat = max(exb, key=exb.get)
        thr = max(Lb.values())
        inside = sorted(H for H in Ub if Ub[H] >= thr)
        holds = all(Lb[H] <= exb[H] + 1e-6 + 1e-9 * abs(exb[H]) and Ub[H] >= exb[H] - 1e-6 - 1e-9 * abs(exb[H]) for H in exb)
        per[g] = {'H_hat_exact': Hhat, 'delta_vs_half_exact': exb[Hhat] - exb[0.5], 'decided_vs_half': bool(thr > Ub[0.5]),
                  'margin_vs_half': thr - Ub[0.5], 'enclosure': [inside[0], inside[-1]], 'enclosure_set': inside,
                  'bracket_holds_on_grid': bool(holds), 'n_blocks': int(len(basins[g])),
                  'n_train': int(is_tr[basins[g]].sum())}
        log('%s H %.2f d %.1f margin %.1f enc [%.2f, %.2f] holds %s' % (g, Hhat, per[g]['delta_vs_half_exact'], per[g]['margin_vs_half'],
                                                                 inside[0], inside[-1], holds))
    save_json({'experiment': 'E7b', 'H_grid': [r['H'] for r in rows], 'per_basin': per, 'seconds': time.perf_counter() - t0},
              'E7b_per_basin.json')


if __name__ == '__main__':
    main()
