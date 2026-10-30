"""A7.9: error of the instrumental validation target, estimated from the CRUTEM5 station archive alone.

The comparisons of A7.1-A7.6 treat the CRUTEM5 cell value as the air-temperature change at the borehole sites up to
interannual sampling noise of the cell series.  Station pairs show that a cell value is a much less exact target: two
stations 0-25 km apart differ in W by about 0.4 K (sd), and the difference grows with distance.  Model for the window
change of station k (reference windows 1961-1990 and 1901-1930, >= 20 annual means in each):
    W_k = F(x_k) + eta_k,   F Gaussian with Cov(F(x), F(y)) = sig2 exp(-|x-y|/L),   eta_k iid N(0, tau2),
F the regional change (it contains the weather of the windows, which nearby stations share), eta_k the changes
specific to a station (siting, relocation, urbanisation, homogenisation, instruments).  (tau2, sig2, L) are fitted by
least squares to the empirical semivariogram  tau2 + sig2 (1 - exp(-d/L))  of station pairs (equal weight per
distance bin).  The cell value of site s is approximated by the mean of the stations of its 5x5 cell with >= 10
annual means in each of the site's windows (nearest such station within 1000 km if the cell has none).  The error of
a country mean of cell values as an estimate of the mean of F over its sites is
    V = tau2 sum_k a_k^2 + [a; -b]^T C [a; -b],   a_k = mean_s 1{k in K_s}/|K_s|,   b_s = 1/n,
and a country mean is compared with  z = (mu - W_cell) / sqrt(sd^2 + V).  The interannual sd of the cell series is not
added: the weather of the windows is shared by the boreholes, the stations and F.  No borehole quantity enters V.
Output: results/A7_target_error.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import app7_borehole_realdata as V1                      # noqa: E402
import app7_station_validation as SV                     # noqa: E402

RES = V1.RES
REF_RECENT = (1961, 1990)
EDGES = (0, 25, 50, 100, 200, 400, 800, 1600, 3200)
MIN_YEARS_CELL = 10
R_FALLBACK = 1000.0


def variogram(st, recent=REF_RECENT):
    W = np.array([(lambda w: w['W'] if w else np.nan)(SV.window_change(a, recent)) for a in st['A']])
    m = np.isfinite(W)
    lat, lon, W = st['lat'][m], st['lon'][m], W[m]
    e = np.array(EDGES, float)
    s2 = np.zeros(len(e) - 1); cnt = np.zeros(len(e) - 1); dsum = np.zeros(len(e) - 1)
    for i in range(len(W) - 1):
        d = SV.km(lat[i], lon[i], lat[i + 1:], lon[i + 1:])
        b = np.digitize(d, e) - 1
        ok = (b >= 0) & (b < len(s2))
        np.add.at(s2, b[ok], 0.5 * (W[i] - W[i + 1:][ok]) ** 2); np.add.at(cnt, b[ok], 1); np.add.at(dsum, b[ok], d[ok])
    gam, dm = s2 / cnt, dsum / cnt
    best = None
    for L in np.exp(np.linspace(np.log(50.0), np.log(20000.0), 400)):
        X = np.stack([np.ones_like(dm), 1 - np.exp(-dm / L)], axis=1)
        coef, *_ = np.linalg.lstsq(X, gam, rcond=None)
        if np.any(coef < 0):
            continue
        r = float(np.sum((X @ coef - gam) ** 2))
        if best is None or r < best[0]:
            best = (r, float(coef[0]), float(coef[1]), float(L))
    _, tau2, sig2, L = best
    return {'n_stations': int(m.sum()), 'recent': recent, 'early': V1.EARLY, 'edges_km': list(EDGES),
            'pairs': cnt.astype(int).tolist(), 'mean_km': dm.tolist(), 'semivariance': gam.tolist(),
            'tau2': tau2, 'sig2': sig2, 'L_km': L, 'fit_rms': math.sqrt(best[0] / len(gam))}


def n_years(a, win):
    v = a[win[0] - SV.YEARS[0]:win[1] - SV.YEARS[0] + 1]
    return int(np.isfinite(v).sum())


def cell_stations(st, cru, site):
    i, j = cru.cell(site['lat'], site['lon'])
    clat, clon = float(cru.lat[i]), float(cru.lon[j])
    dlon = np.abs(((st['lon'] - clon + 180.0) % 360.0) - 180.0)
    box = np.where((np.abs(st['lat'] - clat) <= 2.5) & (dlon <= 2.5))[0]
    ok = [int(k) for k in box if n_years(st['A'][k], site['recent']) >= MIN_YEARS_CELL
          and n_years(st['A'][k], V1.EARLY) >= MIN_YEARS_CELL]
    if ok:
        return ok, False
    d = SV.km(site['lat'], site['lon'], st['lat'], st['lon'])
    for k in np.argsort(d):
        if d[k] > R_FALLBACK:
            break
        if n_years(st['A'][k], site['recent']) >= MIN_YEARS_CELL and n_years(st['A'][k], V1.EARLY) >= MIN_YEARS_CELL:
            return [int(k)], True
    return [], True


def target_variance(st, vg, sites_c, K):
    """Error variance of the mean of the sites' cell values as an estimate of the mean of F over the sites."""
    n = len(sites_c)
    used = sorted({k for Ks in K for k in Ks})
    pos = {k: i for i, k in enumerate(used)}
    a = np.zeros(len(used))
    for Ks in K:
        for k in Ks:
            a[pos[k]] += 1.0 / (len(Ks) * n)
    lat = np.concatenate([st['lat'][used], [s['lat'] for s in sites_c]])
    lon = np.concatenate([st['lon'][used], [s['lon'] for s in sites_c]])
    D = SV.km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    v = np.concatenate([a, -np.full(n, 1.0 / n)])
    C = vg['sig2'] * np.exp(-D / vg['L_km'])
    return float(vg['tau2'] * np.sum(a ** 2) + v @ C @ v)


def compare(C, V):
    out = {}
    for c, x in C.items():
        z = (x['mu'] - x['W_cell']) / math.sqrt(x['sd'] ** 2 + V[c])
        z_old = (x['mu'] - x['W_cell']) / math.sqrt(x['sd'] ** 2 + x['sd_inst'] ** 2)
        out[c] = x | {'sd_target': math.sqrt(V[c]), 'z': z, 'z_interannual_only': z_old}
    inside = sum(abs(v['z']) <= 1.96 for v in out.values())
    inside_old = sum(abs(v['z_interannual_only']) <= 1.96 for v in out.values())
    return out, {'n': len(out), 'inside95': inside, 'inside95_interannual_only': inside_old}


def main(H='0.5'):
    st = SV.station_index()
    vg = variogram(st)
    print('variogram: tau2 %.3f K^2 (tau %.2f K), sig2 %.3f K^2, L %.0f km, rms misfit %.3f' % (
        vg['tau2'], math.sqrt(vg['tau2']), vg['sig2'], vg['L_km'], vg['fit_rms']))
    cru = V1.Crutem()
    sites, _, _ = V1.select_sites(cru)
    countries = {}
    for i, s in enumerate(sites):
        countries.setdefault(s['country'], []).append(i)
    countries = {c: v for c, v in countries.items() if len(v) >= 10}
    K_all, fb_all = {}, {}
    for idx in countries.values():
        for i in idx:
            K_all[i], fb_all[i] = cell_stations(st, cru, sites[i])
    V, info = {}, {}
    for c, idx in countries.items():
        V[c] = target_variance(st, vg, [sites[i] for i in idx], [K_all[i] for i in idx])
        info[c] = {'n_sites': len(idx), 'n_stations': len({k for i in idx for k in K_all[i]}),
                   'median_stations_per_site': float(np.median([len(K_all[i]) for i in idx])),
                   'sites_single_station': int(sum(len(K_all[i]) == 1 for i in idx)),
                   'fallback_sites': int(sum(fb_all[i] for i in idx)), 'sd_target': math.sqrt(V[c])}
    models = {}
    base = json.load(open(os.path.join(RES, 'A7_borehole_regional.json')))['by_H'][H]['countries']
    models['regional'] = {c: {'mu': v['mu'], 'sd': v['sd'], 'W_cell': v['W_inst'], 'sd_inst': v['sd_inst']} for c, v in base.items()}
    p_adv = os.path.join(RES, 'A7_borehole_regional_adv.json')
    if os.path.exists(p_adv):
        adv = json.load(open(p_adv))['by_H'][H]['countries']
        models['regional+flow (pooled)'] = {c: {'mu': v['mu'], 'sd': v['sd'], 'W_cell': v['W_inst'], 'sd_inst': v['sd_inst']}
                                            for c, v in adv.items()}
    p_bc = os.path.join(RES, 'A7_borehole_regional_adv_by_country.json')
    if os.path.exists(p_bc):
        bc = json.load(open(p_bc))['countries']
        models['regional+flow (by country)'] = {c: {'mu': bc[c]['best']['mu'], 'sd': bc[c]['best']['sd'],
                                                    'W_cell': base[c]['W_inst'], 'sd_inst': base[c]['sd_inst']} for c in bc}
    out = {'config': {'H': H, 'reference_windows': [REF_RECENT, V1.EARLY], 'min_years_cell': MIN_YEARS_CELL,
                      'fallback_km': R_FALLBACK}, 'variogram': vg, 'countries_target': info, 'models': {}}
    for name, C in models.items():
        res, summ = compare(C, V)
        out['models'][name] = {'countries': res, 'summary': summ}
        print('%-28s inside 95%%: %d/%d (interannual-only target error: %d/%d)' % (
            name, summ['inside95'], summ['n'], summ['inside95_interannual_only'], summ['n']))
    # single sites: climate part of the regional posterior against the cell, target error of one site
    S = json.load(open(os.path.join(RES, 'A7_borehole_regional.json')))['by_H'][H]['sites']
    zs, zs_old, vs = [], [], []
    for idx in countries.values():
        for i in idx:
            r = S[sites[i]['file']]
            v = target_variance(st, vg, [sites[i]], [K_all[i]])
            vs.append(v)
            zs.append((r['mu_climate'] - r['W_inst']) / math.sqrt(r['sd_climate'] ** 2 + v))
            zs_old.append((r['mu_climate'] - r['W_inst']) / math.sqrt(r['sd_climate'] ** 2 + r['sd_inst'] ** 2))
    zs, zs_old = np.array(zs), np.array(zs_old)
    out['sites'] = {'n': len(zs), 'cover90': float(np.mean(np.abs(zs) <= 1.645)), 'cover50': float(np.mean(np.abs(zs) <= 0.6745)),
                    'z_sd': float(np.std(zs, ddof=1)), 'cover90_interannual_only': float(np.mean(np.abs(zs_old) <= 1.645)),
                    'cover50_interannual_only': float(np.mean(np.abs(zs_old) <= 0.6745)),
                    'z_sd_interannual_only': float(np.std(zs_old, ddof=1)), 'median_sd_target': float(np.median(np.sqrt(vs)))}
    print('sites (climate part): 90%% coverage %.2f (interannual only %.2f), 50%% coverage %.2f (%.2f), sd(z) %.2f (%.2f), median target sd %.2f K' % (
        out['sites']['cover90'], out['sites']['cover90_interannual_only'], out['sites']['cover50'], out['sites']['cover50_interannual_only'],
        out['sites']['z_sd'], out['sites']['z_sd_interannual_only'], out['sites']['median_sd_target']))
    print('%-26s %4s %4s %5s %6s | %s' % ('country', 'site', 'stn', 'tgt', 'cell', '  '.join('%-26s' % m for m in models)))
    for c in sorted(info, key=lambda c: base[c]['W_inst']):
        row = '  '.join('%5.2f+-%.2f z%+5.1f (%+5.1f)  ' % (out['models'][m]['countries'][c]['mu'], out['models'][m]['countries'][c]['sd'],
                                                       out['models'][m]['countries'][c]['z'], out['models'][m]['countries'][c]['z_interannual_only'])
                        for m in models)
        print('%-26s %4d %4d %5.2f %6.2f | %s' % (c, info[c]['n_sites'], info[c]['n_stations'], info[c]['sd_target'], base[c]['W_inst'], row))
    out['sensitivity'] = sensitivity(st, cru, sites, countries, base)
    out['china_split'] = china_split(st, cru, sites, vg, H)
    with open(os.path.join(RES, 'A7_target_error.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


def _fit_weighted(vg, w):
    dm, gam = np.array(vg['mean_km']), np.array(vg['semivariance'])
    best = None
    for L in np.exp(np.linspace(np.log(50.0), np.log(20000.0), 400)):
        X = np.stack([np.ones_like(dm), 1 - np.exp(-dm / L)], axis=1)
        coef, *_ = np.linalg.lstsq(X * w[:, None], gam * w, rcond=None)
        if np.any(coef < 0):
            continue
        r = float(np.sum(((X @ coef - gam) * w) ** 2))
        if best is None or r < best[0]:
            best = (r, float(coef[0]), float(coef[1]), float(L))
    return vg | {'tau2': best[1], 'sig2': best[2], 'L_km': best[3]}


def sensitivity(st, cru, sites, countries, base):
    """Country comparison of the regional model under other fitting choices of the target error."""
    global MIN_YEARS_CELL
    keep = MIN_YEARS_CELL
    runs = [('reference: 1961-1990, >= 10 years, equal bin weights', REF_RECENT, 10, None),
            ('cell stations with >= 20 years', REF_RECENT, 20, None),
            ('cell stations with >= 5 years', REF_RECENT, 5, None),
            ('bin weights sqrt(pairs)', REF_RECENT, 10, 'pairs'),
            ('reference windows 1951-1980', (1951, 1980), 10, None),
            ('reference windows 1971-2000', (1971, 2000), 10, None)]
    out = []
    for label, rec, my, wt in runs:
        MIN_YEARS_CELL = my
        vg = variogram(st, rec)
        if wt == 'pairs':
            vg = _fit_weighted(vg, np.sqrt(np.array(vg['pairs'], float)))
        zs = {}
        for c, idx in countries.items():
            K = [cell_stations(st, cru, sites[i])[0] for i in idx]
            V = target_variance(st, vg, [sites[i] for i in idx], K)
            zs[c] = (base[c]['mu'] - base[c]['W_inst']) / math.sqrt(base[c]['sd'] ** 2 + V)
        out.append({'label': label, 'tau': math.sqrt(vg['tau2']), 'sigma': math.sqrt(vg['sig2']), 'L_km': vg['L_km'],
                    'inside95': int(sum(abs(z) <= 1.96 for z in zs.values())), 'z': zs})
        print('  %-52s inside %d/13  outside %s' % (label, out[-1]['inside95'], {c: round(z, 1) for c, z in zs.items() if abs(z) > 1.96}))
    MIN_YEARS_CELL = keep
    return out


def china_split(st, cru, sites, vg, H, lat_split=31.0, n=256):
    """Diagnostic: China's holes span 1700 km; fit its southern and northern holes as separate regions."""
    import app7_borehole_realdata_v2 as V2
    import app7_borehole_regional as R1
    fit = json.load(open(os.path.join(RES, 'A7_borehole_realdata_v2.json')))
    R = json.load(open(os.path.join(RES, 'A7_borehole_regional.json')))['by_H'][H]
    Z, E2, Zt, Et, off, _ = V2.cached_coordinates(float(H), n, sites)
    sE = [p['sigma_e'] for p in fit['by_H'][H]['site']]
    idx = [i for i, s in enumerate(sites) if s['country'] == 'China']
    groups = {'all': idx, 'south': [i for i in idx if sites[i]['lat'] < lat_split],
              'north': [i for i in idx if sites[i]['lat'] >= lat_split]}
    out = {'lat_split': lat_split}
    for g, ii in groups.items():
        S = [R1.SiteR(sites[i], Z[off[i]:off[i + 1]], E2[off[i]:off[i + 1]], Zt[i], Et[i], sE[i]) for i in ii]
        _, _, reg = R1.country_fit(S, R['sigma_C'], R['sigma_L'], R['sigma_g'], predict=True)
        V = target_variance(st, vg, [sites[i] for i in ii], [cell_stations(st, cru, sites[i])[0] for i in ii])
        Wc = float(np.mean([sites[i]['inst']['W'] for i in ii]))
        lat = [sites[i]['lat'] for i in ii]; lon = [sites[i]['lon'] for i in ii]
        span = float(max(SV.km(lat[a], lon[a], lat[b], lon[b]) for a in range(len(ii)) for b in range(len(ii))))
        out[g] = {'n': len(ii), 'mu': reg['mu'], 'sd': reg['sd'], 'W_cell': Wc, 'sd_target': math.sqrt(V),
                  'z': (reg['mu'] - Wc) / math.sqrt(reg['sd'] ** 2 + V), 'span_km': span}
        print('  China %-5s n %2d  borehole %.2f +- %.2f  cell %.2f +- %.2f  z %+.1f  span %.0f km' % (
            g, len(ii), reg['mu'], reg['sd'], Wc, math.sqrt(V), out[g]['z'], span))
    return out


if __name__ == '__main__':
    main(*(sys.argv[1:2] or ['0.5']))
