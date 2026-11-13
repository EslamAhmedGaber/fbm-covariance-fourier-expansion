"""A7.1: real borehole profiles (NOAA Huang-Pollack database) through the general expansion, validated against CRUTEM5.

Model per site s:  y_i = T0 + Gamma z_i + sigma_T I_H(f_{z_i, t_log}) + eps_i,  eps_i ~ N(0, sigma_eps_s^2),
flat prior on (T0, Gamma) (REML and the corresponding predictive formulas), dT = sigma_T B_H on 1500-2030,
kappa = 1e-6 m^2/s.  ONE basis per H serves all sites; every reading and target goes through the same procedure.
Target per site: W = mean dT over [t_log - 30, t_log] minus mean dT over [1901, 1930].
Hyperparameters: (H, sigma_T) pooled, maximising the summed REML profile likelihood over a random half of the sites
(training half); sigma_eps per site by REML.  CRUTEM5 is used only for validation.
Selection rules (fixed before any fit; metadata and coverage only): one attributed data block; t_log >= 1960;
>= 15 readings; shallowest reading <= 60 m; deepest >= 200 m; the 5x5 CRUTEM5 cell has >= 20 annual means
(>= 9 months each) in both 30-year windows.
Output: results/A7_borehole_realdata.json
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
from covfourier import heat                              # noqa: E402
import borehole_problem as BP                            # noqa: E402
import borehole_data as BD                               # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
CRU = os.path.join(BD.ROOT, '04_DATA', 'CRUTEM5', 'CRUTEM.5.2.0.0.anomalies.nc')
EARLY = (1901, 1930)
HS = (0.5, 0.55, 0.65, 0.75, 0.85, 0.95)
SIGMA_T_GRID = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8)
SIGMA_E_GRID = np.exp(np.linspace(np.log(0.003), np.log(0.5), 25))


# ----------------------------------------------------------------------------- instrumental data
class Crutem:
    def __init__(self, path=CRU):
        import h5py
        with h5py.File(path, 'r') as f:
            self.tas = f['tas'][:]
            self.lat = f['latitude'][:]; self.lon = f['longitude'][:]
            t = f['time'][:]
        self.tas[self.tas < -1e29] = np.nan
        days = t
        self.year = 1850 + np.floor((days - 0.5) / 365.2425).astype(int)
        # exact calendar: month index from start
        nm = len(t)
        self.yr_of_month = 1850 + np.arange(nm) // 12

    def cell(self, lat, lon):
        i = int(np.argmin(np.abs(self.lat - lat)))
        lon = ((lon + 180.0) % 360.0) - 180.0
        j = int(np.argmin(np.abs(self.lon - lon)))
        return i, j

    def annual(self, lat, lon, min_months=9):
        i, j = self.cell(lat, lon)
        x = self.tas[:, i, j]
        yrs = np.unique(self.yr_of_month)
        out = {}
        for y in yrs:
            v = x[self.yr_of_month == y]
            v = v[np.isfinite(v)]
            if len(v) >= min_months:
                out[int(y)] = float(v.mean())
        return out

    def window_change(self, lat, lon, recent, early=EARLY, min_years=20):
        a = self.annual(lat, lon)
        r = [a[y] for y in range(recent[0], recent[1] + 1) if y in a]
        e = [a[y] for y in range(early[0], early[1] + 1) if y in a]
        if len(r) < min_years or len(e) < min_years:
            return None
        # sampling sd of the difference of means from the interannual spread (independent years: conservative-ish)
        sd = math.sqrt(np.var(r, ddof=1) / len(r) + np.var(e, ddof=1) / len(e))
        return {'W': float(np.mean(r) - np.mean(e)), 'sd': sd, 'n_recent': len(r), 'n_early': len(e)}


# ----------------------------------------------------------------------------- per-site REML in data space
def site_terms(y, z, Z, E2, g, e2g, sT, sE):
    """REML log-likelihood and the predictive mean/variance of the target for one site."""
    m = len(y)
    Hn = np.stack([np.ones(m), z], axis=1)
    K = sT ** 2 * (Z @ Z.T + np.diag(E2)) + sE ** 2 * np.eye(m)
    cf = cho_factor(K, lower=True, check_finite=False)
    Ki_y = cho_solve(cf, y, check_finite=False); Ki_H = cho_solve(cf, Hn, check_finite=False)
    A = Hn.T @ Ki_H
    cA = cho_factor(A, lower=True, check_finite=False)
    beta = cho_solve(cA, Hn.T @ Ki_y, check_finite=False)
    r = y - Hn @ beta
    Ki_r = cho_solve(cf, r, check_finite=False)
    logdetK = 2 * np.sum(np.log(np.diag(cf[0]))); logdetA = 2 * np.sum(np.log(np.diag(cA[0])))
    ll = -0.5 * (r @ Ki_r + logdetK + logdetA + (m - 2) * math.log(2 * math.pi))
    k = sT ** 2 * (Z @ g)
    v = sT ** 2 * (g @ g + e2g)
    mu = float(k @ Ki_r)
    Ki_k = cho_solve(cf, k, check_finite=False)
    R = Hn.T @ Ki_k
    var = float(v - k @ Ki_k + R @ cho_solve(cA, R, check_finite=False))
    return ll, mu, var, beta, r


def best_sigma_e(y, z, Z, E2, g, e2g, sT):
    lls = [site_terms(y, z, Z, E2, g, e2g, sT, sE)[0] for sE in SIGMA_E_GRID]
    i = int(np.argmax(lls))
    return SIGMA_E_GRID[i], lls[i]


# ----------------------------------------------------------------------------- Huang et al. century trends
def huang_change(rates, t_log, recent, early=EARLY):
    """Piecewise-linear GST history implied by the database's century trends (K/100a, 16th..20th), zero in 1500,
    truncated at the logging year; returns mean(recent) - mean(early)."""
    if len(rates) != 5:
        return None
    knots = np.array([1500, 1600, 1700, 1800, 1900, 2000.0])
    def T(t):
        t = np.minimum(np.asarray(t, float), t_log)
        val = np.zeros_like(t)
        for c in range(5):
            a, b = knots[c], knots[c + 1]
            val += rates[c] / 100.0 * np.clip(np.minimum(t, b) - a, 0, None)
        return val
    tr = np.linspace(recent[0], recent[1], 301); te = np.linspace(early[0], early[1], 301)
    return float(np.mean(T(tr)) - np.mean(T(te)))


def select_sites(cru):
    rows = BD.load_all()
    sel, reasons = [], {}
    def rej(r):
        reasons[r] = reasons.get(r, 0) + 1
    for r in rows:
        if r['block_index'] is None:
            rej('no attributed block'); continue
        if r['log_year'] is None or r['log_year'] < 1960:
            rej('logged before 1960'); continue
        z = np.array(r['depth']); T = np.array(r['temp'])
        ok = np.isfinite(z) & np.isfinite(T) & (z >= 20.0) & (z <= 600.0)
        z, T = z[ok], T[ok]
        if len(z) < 15:
            rej('fewer than 15 readings'); continue
        if z.min() > 60.0:
            rej('shallowest reading below 60 m'); continue
        if z.max() < 200.0:
            rej('deepest reading above 200 m'); continue
        if r['lat'] is None or r['lon'] is None:
            rej('no coordinates'); continue
        tl = float(r['log_year'])
        recent = (int(math.floor(tl)) - 30, int(math.floor(tl)) - 1)
        w = cru.window_change(r['lat'], r['lon'], recent)
        if w is None:
            rej('CRUTEM5 cell coverage insufficient'); continue
        sel.append({'file': r['file'], 'site': r['site'], 'country': r['country'], 'lat': r['lat'], 'lon': r['lon'],
                    't_log': tl, 'z': z, 'T': T, 'recent': recent, 'inst': w,
                    'huang_W': huang_change(r['century_trends'], tl, recent)})
    return sel, reasons, len(rows)


def coordinates_for_sites(prep, sites):
    """All readings and all site targets through the same prepared basis (one call per batch)."""
    z_all = np.concatenate([s['z'] for s in sites]); t_all = np.concatenate([np.full(len(s['z']), s['t_log']) for s in sites])
    t0 = time.perf_counter()
    Z, E2 = prep.readings(z_all, t_all)
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    u = heat.midpoints(prep.N_targ)
    F = np.vstack([heat.epoch_mean_values(s['recent'][0], s['recent'][1] + 1, u, prep.win) -
                   heat.epoch_mean_values(EARLY[0], EARLY[1] + 1, u, prep.win) for s in sites])
    Zt, Et = [], []
    for i0 in range(0, len(F), 40):
        a, b_, _ = heat.project_samples(prep.bp_t, F[i0:i0 + 40]); Zt.append(a); Et.append(b_)
    Zt, Et = np.vstack(Zt), np.concatenate(Et)
    t_targ = time.perf_counter() - t0
    off = np.cumsum([0] + [len(s['z']) for s in sites])
    return Z, E2, Zt, Et, off, {'readings_s': t_read, 'targets_s': t_targ, 'n_readings': int(len(z_all))}


def main(n=256, seed=3, kappa_factors=(1.0,)):
    cru = Crutem()
    sites, reasons, n_files = select_sites(cru)
    rng = np.random.default_rng(seed)
    train = rng.permutation(len(sites))[: len(sites) // 2]
    is_train = np.zeros(len(sites), bool); is_train[train] = True
    print('files', n_files, 'selected', len(sites), 'rejections', reasons, flush=True)
    out = {'config': {'n': n, 'Hs': list(HS), 'sigma_T_grid': list(SIGMA_T_GRID), 'early_window': EARLY,
                      'recent_window': '[t_log-30, t_log-1]', 'kappa_m2_per_yr': heat.KAPPA_M2_PER_YR,
                      'window': [1500, 2030], 'seed': seed, 'selection_rejections': reasons, 'n_files': n_files,
                      'n_selected': len(sites)}, 'sites': [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in s.items() if k not in ('z', 'T')} | {'n_readings': int(len(s['z']))} for s in sites],
           'is_train': is_train.tolist(), 'by_H': {}}
    for H in HS:
        t0 = time.perf_counter()
        prep = BP.Prepared(H, n, N_cand=8192, N_targ=1 << 16)
        Z, E2, Zt, Et, off, tim = coordinates_for_sites(prep, sites)
        tim['prep_s'] = prep.prep_s
        # pooled sigma_T on the training half
        prof = {}
        for sT in SIGMA_T_GRID:
            tot = 0.0
            for i in np.where(is_train)[0]:
                sl = slice(off[i], off[i + 1])
                s = sites[i]
                _, ll = best_sigma_e(s['T'], s['z'], Z[sl], E2[sl], Zt[i], Et[i], sT)
                tot += ll
            prof[sT] = tot
        sT = max(prof, key=prof.get)
        per = []
        for i, s in enumerate(sites):
            sl = slice(off[i], off[i + 1])
            sE, ll = best_sigma_e(s['T'], s['z'], Z[sl], E2[sl], Zt[i], Et[i], sT)
            _, mu, var, beta, r = site_terms(s['T'], s['z'], Z[sl], E2[sl], Zt[i], Et[i], sT, sE)
            per.append({'sigma_e': float(sE), 'll': float(ll), 'mu': mu, 'sd': math.sqrt(max(var, 0.0)),
                        'T0': float(beta[0]), 'Gamma_K_per_km': float(1e3 * beta[1]),
                        'max_rel_E2': float(np.max(E2[sl] / np.maximum(np.sum(Z[sl] ** 2, 1), 1e-300)))})
        out['by_H'][str(H)] = {'sigma_T': sT, 'profile_ll_train': {str(k): v for k, v in prof.items()},
                               'll_train': float(prof[sT]), 'll_test': float(sum(p['ll'] for p, tr in zip(per, is_train) if not tr)),
                               'timing': tim, 'site': per, 'seconds': time.perf_counter() - t0}
        print('H', H, 'sigma_T', sT, 'll_train %.1f' % prof[sT], 'll_test %.1f' % out['by_H'][str(H)]['ll_test'],
              '%.1fs' % (time.perf_counter() - t0), flush=True)
        with open(os.path.join(RES, 'A7_borehole_realdata.json'), 'w') as fh:
            json.dump(out, fh, indent=0)


if __name__ == '__main__':
    main()
