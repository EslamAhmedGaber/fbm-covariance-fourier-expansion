"""E7 real_rivers: CAMELS-US storage anomalies, exact restricted likelihood and the CovFourier bracket per H.

Data preparation with the controls of experiments/PROTOCOL.md:
  clim   'all' (day-of-year means over all years) | 'train' (training years only)
  centre 'block' (each block demeaned and standardised, S(T) = 0) | 'train' (basin training mean and sd)
  mean   'const' | 'trend'      noise 'free' (kappa estimated) | 'fixed' (kappa = 1e-6)
  L      365 | 730 | 1460 days (10-day windows)
Model per block b: y_b = beta_b h + sigma_b (window averages of B_H) + tau_b eps, kappa = tau_b^2/sigma_b^2 shared,
sigma_b profiled, beta_b restricted (flat).  H is selected on the training blocks only.

Usage: python e7_real_rivers.py  ->  results/E7_real_rivers.json
"""
import glob
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import C, np, save_json, Log                                    # noqa: E402
import pandas as pd                                                         # noqa: E402
from scipy.linalg import eigh, cholesky, cho_solve                          # noqa: E402

def _data_root():
    """04_DATA folder: env CAMELS_ROOT (folder containing 04_DATA), else the first ancestor that contains 04_DATA."""
    if os.environ.get('CAMELS_ROOT'):
        return os.path.join(os.environ['CAMELS_ROOT'], '04_DATA')
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(d, '04_DATA')):
            return os.path.join(d, '04_DATA')
        up = os.path.dirname(d)
        if up == d:
            raise FileNotFoundError('04_DATA not found; set CAMELS_ROOT')
        d = up


DATA = _data_root()
HGRID_COARSE = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
LK = np.linspace(-9.0, 1.0, 201)
KAP = 10.0 ** LK
W_DAYS = 10
NS = (128, 256, 512)


def load_basins():
    files = sorted(glob.glob(os.path.join(DATA, '**', '*_streamflow_qc.txt'), recursive=True))
    out, excluded = {}, {}
    for fp in files:
        df = pd.read_csv(fp, sep=r'\s+', header=None, names=['gauge', 'year', 'month', 'day', 'q', 'qc'], dtype={'gauge': str})
        gid = str(df['gauge'].iloc[0]).zfill(8)
        df = df[(df['year'] >= 1981) & (df['year'] <= 2014)].copy()
        q = df['q'].values.astype(float)
        q[q < 0] = np.nan
        frac = float(np.isnan(q).mean())
        if frac > 0.05:
            excluded[gid] = frac
            continue
        q = pd.Series(q).interpolate().bfill().ffill().values
        doy = pd.to_datetime(df[['year', 'month', 'day']]).dt.dayofyear.values
        out[gid] = {'logq': np.log1p(np.maximum(q, 0.0)), 'doy': doy, 'missing_fraction': frac}
    return out, excluded, [os.path.basename(f) for f in files]


def make_blocks(b, L, clim='all', centre='block'):
    x = b['logq'].copy(); doy = b['doy']
    nb = len(x) // L
    ntr = int(0.6 * nb)
    train_days = ntr * L
    for d in range(1, 367):
        msk = doy == d
        if msk.sum() == 0:
            continue
        if clim == 'all':
            x[msk] -= x[msk].mean()
        else:
            mtr = msk.copy(); mtr[train_days:] = False
            x[msk] -= x[mtr].mean() if mtr.sum() else x[msk].mean()
    mu_tr, sd_tr = x[:train_days].mean(), x[:train_days].std()
    n_int = L // W_DAYS
    Y, is_tr = [], []
    for k in range(nb):
        seg = x[k * L:(k + 1) * L]
        if centre == 'block':
            seg = seg - seg.mean(); s = seg.std()
            if s < 1e-8:
                continue
            seg = seg / s
        else:
            seg = (seg - mu_tr) / sd_tr
        S = np.cumsum(seg) / math.sqrt(L)
        Y.append(S[:n_int * W_DAYS].reshape(n_int, W_DAYS).mean(axis=1))
        is_tr.append(k < ntr)
    return np.column_stack(Y), np.array(is_tr), n_int


def reml_exact_profiles(K, Y, h):
    """Exact restricted log-likelihood, sigma^2 profiled, for every kappa (rows) and block (cols)."""
    m, p = h.shape
    lam, V = eigh(K)
    lam = np.maximum(lam, 0.0)
    VY, Vh = V.T @ Y, V.T @ h
    out = np.empty((len(KAP), Y.shape[1]))
    _, ldhh = np.linalg.slogdet(h.T @ h)
    for i, k in enumerate(KAP):
        dinv = 1.0 / (lam + k)
        ld = float(np.sum(np.log(lam + k)))
        A = (Vh * dinv[:, None]).T @ Vh
        bvec = (Vh * dinv[:, None]).T @ VY
        q = np.einsum('jb,j,jb->b', VY, dinv, VY) - np.einsum('pb,pb->b', bvec, np.linalg.solve(A, bvec))
        _, ldA = np.linalg.slogdet(A)
        df = m - p
        out[i] = -0.5 * (df * np.log(q / df) + ld + ldA - ldhh + df) - 0.5 * df * math.log(2 * math.pi)
    return out


def bracket_profiles(feats, Y, h, majorant):
    """Ld*(kappa) and U*(kappa) of the restricted likelihood, sigma^2 profiled (rows kappa, cols blocks)."""
    m, p = h.shape
    df = m - p
    Gam = feats.Gamma
    lr = C.LowRank(Gam)
    e = np.sqrt(feats.e2)
    trR = feats.trR
    _, ldhh = np.linalg.slogdet(h.T @ h)
    Ld = np.empty((len(KAP), Y.shape[1])); U = np.empty_like(Ld)
    for i, k in enumerate(KAP):
        sC = lambda X, c=k: lr.solve(X, c)
        extra, PY = C._reml_parts(sC, h, Y)
        ld = lr.logdet(k) + extra
        q = np.einsum('ib,ib->b', Y, PY)
        extra_t, PYt = C._reml_parts(lambda X, c=k + trR: lr.solve(X, c), h, Y)
        q0 = np.einsum('ib,ib->b', Y, PYt)
        q1 = q - (np.abs(PY).T @ e) ** 2
        bv, _ = majorant.beta(PY)
        qlo = np.maximum.reduce([q0, q1, q - bv])
        const = -0.5 * df * math.log(2 * math.pi)
        Ld[i] = -0.5 * (df * np.log(q / df) + ld + df) + const - 0.5 * float(np.sum(np.log1p(feats.e2 / k)))
        U[i] = -0.5 * (df * np.log(np.maximum(qlo, 1e-300) / df) + ld + df) + const
    return Ld, U


def refine(x, f):
    j = int(np.argmax(f))
    if 0 < j < len(f) - 1:
        y0, y1, y2 = f[j - 1], f[j], f[j + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            d = 0.5 * (y0 - y2) / den
            return float(x[j] + d * (x[1] - x[0])), float(y1 - 0.25 * (y0 - y2) * d), j
    return float(x[j]), float(f[j]), j


def analyse(Y, is_tr, n_int, Hs, mean='const', noise='free', n_cf=256, log=None, with_bracket=True, basins=None):
    a = np.arange(n_int) / n_int; b = (np.arange(n_int) + 1) / n_int
    des = C.Boxes(a, b)
    tt = np.arange(n_int) / n_int + 0.5 / n_int - 0.5
    h = np.ones((n_int, 1)) if mean == 'const' else np.c_[np.ones(n_int), tt]
    rows = []
    tr, te = np.where(is_tr)[0], np.where(~is_tr)[0]
    kap_idx = None if noise == 'free' else int(np.argmin(np.abs(LK + 6.0)))
    for H in Hs:
        t0 = time.perf_counter()
        K = des.K(H)
        ex = reml_exact_profiles(K, Y, h)                 # (G, B)
        tot_tr = ex[:, tr].sum(axis=1)
        if kap_idx is None:
            lk_hat, ll_tr, j = refine(LK, tot_tr)
        else:
            j = kap_idx; lk_hat, ll_tr = LK[j], tot_tr[j]
        ll_te = float(ex[j, te].sum())
        row = {'H': H, 'log10_kappa_hat': lk_hat, 'kappa_at_grid_edge': bool(j in (0, len(LK) - 1)),
               'exact_train': float(ll_tr), 'exact_train_grid': float(tot_tr[j]), 'exact_test': ll_te,
               'exact_train_gridmax': float(tot_tr.max())}
        if basins is not None:
            row['exact_basin_gridmax'] = {g: float(ex[:, idx].sum(axis=1).max()) for g, idx in basins.items()}
        if with_bracket:
            bas = C.CFBasis(H, n_cf)
            f = C.cf_features(des, bas)
            maj = C.CFMajorant(des, bas, f)
            Ld, U = bracket_profiles(f, Y, h, maj)
            Ltr, Utr = Ld[:, tr].sum(axis=1), U[:, tr].sum(axis=1)
            row.update({'D': bas.D, 'trR': f.trR, 'rho_at_kappa_hat': f.trR / (10 ** lk_hat),
                        'Ld_train_gridmax': float(Ltr.max()), 'U_train_gridmax': float(Utr.max()),
                        'Ld_train_at_j': float(Ltr[j]), 'U_train_at_j': float(Utr[j]),
                        'Ld_test_at_j': float(Ld[j, te].sum()), 'U_test_at_j': float(U[j, te].sum()),
                        'bracket_holds_train': bool(np.all(Ltr <= tot_tr + 1e-6 + 1e-9 * np.abs(tot_tr)) and
                                                    np.all(Utr >= tot_tr - 1e-6 - 1e-9 * np.abs(tot_tr))),
                        'max_viol': float(max(np.max(Ld - ex), np.max(ex - U)))})
            if basins is not None:
                row['Ld_basin_gridmax'] = {g: float(Ld[:, idx].sum(axis=1).max()) for g, idx in basins.items()}
                row['U_basin_gridmax'] = {g: float(U[:, idx].sum(axis=1).max()) for g, idx in basins.items()}
        row['seconds'] = time.perf_counter() - t0
        rows.append(row)
        if log:
            log('    H %.2f log10k %.2f%s train %.2f test %.2f%s' % (
                H, lk_hat, '*' if row['kappa_at_grid_edge'] else '', ll_tr, ll_te,
                (' | D %d rho %.2e Ld %.2f U %.2f' % (row['D'], row['rho_at_kappa_hat'], row['Ld_train_gridmax'],
                                                    row['U_train_gridmax'])) if with_bracket else ''))
    return rows


def summary(rows):
    best = max(rows, key=lambda r: r['exact_train'])
    r05 = [r for r in rows if abs(r['H'] - 0.5) < 1e-9][0]
    out = {'H_hat_train': best['H'], 'delta_train_vs_half': best['exact_train'] - r05['exact_train'],
           'delta_test_vs_half': best['exact_test'] - r05['exact_test']}
    if 'Ld_train_gridmax' in best:
        out['decided'] = bool(best['Ld_train_gridmax'] > r05['U_train_gridmax'])
        out['margin_L_hat_minus_U_half'] = best['Ld_train_gridmax'] - r05['U_train_gridmax']
        thr = max(r['Ld_train_gridmax'] for r in rows)
        inside = [r['H'] for r in rows if r['U_train_gridmax'] >= thr]
        out['enclosure_grid'] = [min(inside), max(inside)]
    return out


def main():
    log = Log('E7.log')
    bas, excl, files = load_basins()
    log('basins kept %d (%s); excluded (missing > 5%%): %s' % (len(bas), ', '.join(sorted(bas)), excl))
    res = {'experiment': 'E7', 'files': files, 'basins_kept': sorted(bas), 'excluded_missing_fraction': excl,
           'selection_rule': 'lowest gauge id per HUC-02 region, first 12 regions (04_DATA/README_data_sources.md); basins with '
                             'more than 5% missing daily flow in 1981-2014 dropped (rule fixed before any fit)', 'runs': {}}
    configs = [('main', dict(L=730, clim='all', centre='block', mean='const', noise='free')),
               ('clim_train', dict(L=730, clim='train', centre='block', mean='const', noise='free')),
               ('centre_train', dict(L=730, clim='train', centre='train', mean='const', noise='free')),
               ('trend', dict(L=730, clim='all', centre='block', mean='trend', noise='free')),
               ('noise_fixed', dict(L=730, clim='all', centre='block', mean='const', noise='fixed')),
               ('L365', dict(L=365, clim='all', centre='block', mean='const', noise='free')),
               ('L1460', dict(L=1460, clim='all', centre='block', mean='const', noise='free'))]
    for name, cfg in configs:
        t0 = time.perf_counter()
        Ys, trs, owners = [], [], []
        for g in sorted(bas):
            Yb, it, n_int = make_blocks(bas[g], cfg['L'], cfg['clim'], cfg['centre'])
            Ys.append(Yb); trs.append(it); owners += [g] * Yb.shape[1]
        Y = np.hstack(Ys); is_tr = np.concatenate(trs); owners = np.array(owners)
        basins = {g: np.where(owners == g)[0] for g in sorted(bas)}
        log('config %s: %d blocks (%d train), m = %d windows' % (name, Y.shape[1], int(is_tr.sum()), n_int))
        Hs = HGRID_COARSE
        rows = analyse(Y, is_tr, n_int, Hs, mean=cfg['mean'], noise=cfg['noise'], log=log,
                       with_bracket=(name in ('main', 'centre_train')), basins=basins if name == 'main' else None)
        sm = summary(rows)
        # fine H grid around the training maximum (exact and bracket)
        if name in ('main', 'centre_train'):
            Hb = sm['H_hat_train']
            fine = tuple(np.round(np.arange(max(0.5, Hb - 0.04), min(0.99, Hb + 0.045), 0.01), 2))
            rows_f = analyse(Y, is_tr, n_int, fine, mean=cfg['mean'], noise=cfg['noise'], log=log, with_bracket=True)
            allrows = sorted(rows + [r for r in rows_f if all(abs(r['H'] - q['H']) > 1e-9 for q in rows)], key=lambda r: r['H'])
            sm = summary(allrows)
            rows = allrows
        res['runs'][name] = {'config': cfg, 'n_blocks': int(Y.shape[1]), 'n_train': int(is_tr.sum()), 'm': n_int,
                             'rows': rows, 'summary': sm, 'seconds': time.perf_counter() - t0}
        if name == 'main':
            per = {}
            for g in sorted(bas):
                exb = {r['H']: r['exact_basin_gridmax'][g] for r in rows if 'exact_basin_gridmax' in r}
                Lb = {r['H']: r['Ld_basin_gridmax'][g] for r in rows if 'Ld_basin_gridmax' in r}
                Ub = {r['H']: r['U_basin_gridmax'][g] for r in rows if 'U_basin_gridmax' in r}
                Hhat = max(exb, key=exb.get)
                thr = max(Lb.values())
                inside = [H for H in Ub if Ub[H] >= thr]
                per[g] = {'H_hat_exact': Hhat, 'delta_vs_half_exact': exb[Hhat] - exb[0.5],
                          'decided_vs_half': bool(Lb[max(Lb, key=Lb.get)] > Ub[0.5]),
                          'enclosure': [min(inside), max(inside)]}
            res['runs'][name]['per_basin'] = per
            log('  per basin: ' + '; '.join('%s H %.2f d %.0f enc [%.2f,%.2f]%s' % (
                g, v['H_hat_exact'], v['delta_vs_half_exact'], v['enclosure'][0], v['enclosure'][1],
                '' if v['decided_vs_half'] else ' (undecided)') for g, v in per.items()))
        log('  summary', sm)
        save_json(res, 'E7_real_rivers.json')


if __name__ == '__main__':
    main()
