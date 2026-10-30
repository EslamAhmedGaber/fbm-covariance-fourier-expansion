"""A7.8: borehole warming against the nearest CRUTEM5 station instead of the 5x5 degree cell.

Why: a borehole records the ground surface at one place.  The gridded cell is an average over every station in a
5x5 degree box, which dilutes local warming (for example urban warming around a city where the holes were drilled).
The regional model (A7.6) treats every country's holes as one climate plus local deviations; when all holes of a
country sit in a few cities, the shared part is the climate of those cities, and the like-for-like target is the
station record there.

Station series: CRUTEM.5.2.0.0 station files (monthly 2 m air temperature).  Annual anomaly = mean over >= 9 valid
months of (tas - monthly baseline); baseline = the file's 1961-1990 normals when all twelve are present, else the
station's own monthly means.  W_station = mean annual anomaly over the site's recent window [t_log-30, t_log-1] minus
the mean over 1901-1930, with >= 20 years in each window: the same rule and sampling sd as for the cell (A7.1).
For each validation site the nearest station with a valid W within R_MAX km (great-circle) is used.
Country means of the shared climate part are recomputed over the sites that have a station, so that the borehole
and station targets average the same sites.
Output: results/A7_station_validation[_adv].json; station annual anomalies cached in 04_DATA/CRUTEM5/derived/.
"""
import io
import json
import math
import os
import sys
import zipfile

import numpy as np
from scipy.linalg import cho_factor, cho_solve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import borehole_data as BD                               # noqa: E402
import app7_borehole_realdata as V1                      # noqa: E402
import app7_borehole_realdata_v2 as V2                   # noqa: E402
import app7_borehole_regional as R1                      # noqa: E402
import app7_borehole_regional_adv as RA                  # noqa: E402

RES = V1.RES
ZIP = os.path.join(BD.ROOT, '04_DATA', 'CRUTEM5', 'stations', 'CRUTEM.5.2.0.0.station_files.zip')
CACHE = os.path.join(BD.ROOT, '04_DATA', 'CRUTEM5', 'derived', 'crutem5_station_annual.npz')
YEARS = np.arange(1850, 2027)
R_MAX = 100.0
MIN_YEARS = 20


def station_index(path=ZIP, cache=CACHE):
    """Annual anomalies of every station (rows) and their coordinates; built once from the zip, then cached."""
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    import h5py
    zf = zipfile.ZipFile(path)
    names = sorted(n for n in zf.namelist() if n.endswith('.nc'))
    A = np.full((len(names), len(YEARS)), np.nan, dtype=np.float32)
    lat, lon = np.empty(len(names)), np.empty(len(names))
    ids, nms, ctry = [], [], []
    def txt(v):
        return v.decode(errors='replace') if isinstance(v, bytes) else str(v)
    for k, n in enumerate(names):
        with h5py.File(io.BytesIO(zf.read(n)), 'r') as f:
            lat[k] = float(f['latitude'][()]); lon[k] = float(f['longitude'][()])
            ids.append(txt(f.attrs.get('stationid', ''))); nms.append(txt(f.attrs.get('name', '')))
            ctry.append(txt(f.attrs.get('country', '')))
            x = f['tas'][:].astype(float); t = f['time'][:]
            nrm = f['tas_climatology_normal'][:].astype(float) if 'tas_climatology_normal' in f else np.full(12, np.nan)
        x[x < -1e29] = np.nan; nrm[nrm < -1e29] = np.nan
        d = np.datetime64('1850-01-01') + np.floor(t).astype('timedelta64[D]')
        yr = d.astype('datetime64[Y]').astype(int) + 1970
        mo = d.astype('datetime64[M]').astype(int) % 12
        if not np.all(np.isfinite(nrm)):
            nrm = np.array([np.nanmean(x[mo == m]) if np.any(np.isfinite(x[mo == m])) else np.nan for m in range(12)])
        an = x - nrm[mo]
        ok = np.isfinite(an) & (yr >= YEARS[0]) & (yr <= YEARS[-1])
        s = np.zeros(len(YEARS)); c = np.zeros(len(YEARS))
        np.add.at(s, yr[ok] - YEARS[0], an[ok]); np.add.at(c, yr[ok] - YEARS[0], 1)
        good = c >= 9
        A[k, good] = s[good] / c[good]
    out = {'A': A, 'years': YEARS, 'lat': lat, 'lon': lon, 'ids': np.array(ids), 'names': np.array(nms),
           'country': np.array(ctry)}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def window_change(a, recent, early=V1.EARLY, min_years=MIN_YEARS):
    """Same rule as V1.Crutem.window_change, for one row of annual anomalies."""
    r = a[recent[0] - YEARS[0]:recent[1] - YEARS[0] + 1]; e = a[early[0] - YEARS[0]:early[1] - YEARS[0] + 1]
    r, e = r[np.isfinite(r)], e[np.isfinite(e)]
    if len(r) < min_years or len(e) < min_years:
        return None
    sd = math.sqrt(np.var(r, ddof=1) / len(r) + np.var(e, ddof=1) / len(e))
    return {'W': float(np.mean(r) - np.mean(e)), 'sd': sd, 'n_recent': len(r), 'n_early': len(e)}


def km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def nearest_station(st, site, r_max=R_MAX):
    d = km(site['lat'], site['lon'], st['lat'], st['lon'])
    for k in np.argsort(d):
        if d[k] > r_max:
            return None
        w = window_change(st['A'][k], site['recent'])
        if w is not None:
            return w | {'km': float(d[k]), 'station': str(st['names'][k]), 'id': str(st['ids'][k])}
    return None


def climate_posterior(S, sC, sL, sG):
    """Posterior (mean, covariance) of the shared climate coordinates of one country (as in R1.country_fit)."""
    D = S[0].Z.shape[1]
    P = np.eye(D); b = np.zeros(D)
    for st in S:
        Pi, _ = st.local(sL, sG)
        P += sC ** 2 * (st.Z.T @ Pi(st.Z)); b += sC * (st.Z.T @ Pi(st.y))
    cP = cho_factor(P, lower=True, check_finite=False)
    return cho_solve(cP, b, check_finite=False), cho_solve(cP, np.eye(D), check_finite=False)


def summary(rows, key_mu, key_sd, key_w, key_wsd):
    z = np.array([(r[key_mu] - r[key_w]) / math.hypot(r[key_sd], r[key_wsd]) for r in rows])
    e = np.array([r[key_mu] - r[key_w] for r in rows])
    return {'n': len(rows), 'mean_z': float(z.mean()), 'sd_z': float(z.std(ddof=1)) if len(z) > 1 else None,
            'coverage95': float(np.mean(np.abs(z) < 1.96)), 'bias': float(e.mean()), 'rmse': float(np.sqrt(np.mean(e ** 2)))}


def main(model='base', H='0.5', n=256):
    st = station_index()
    fit = json.load(open(os.path.join(RES, 'A7_borehole_realdata_v2.json')))
    regf = 'A7_borehole_regional.json' if model == 'base' else 'A7_borehole_regional_adv.json'
    R = json.load(open(os.path.join(RES, regf)))['by_H'][H]
    sC, sL, sG = R['sigma_C'], R['sigma_L'], R['sigma_g']
    cru = V1.Crutem()
    sites, _, _ = V1.select_sites(cru)
    countries = {}
    for i, s in enumerate(sites):
        countries.setdefault(s['country'], []).append(i)
    countries = {c: v for c, v in countries.items() if len(v) >= R1.MIN_SITES}
    Z, E2, Zt, Et, off, _ = V2.cached_coordinates(float(H), n, sites)
    sE = [p['sigma_e'] for p in fit['by_H'][H]['site']]
    Site = R1.SiteR if model == 'base' else RA.SiteA
    out = {'config': {'model': model, 'regional_file': regf, 'H': H, 'n': n, 'r_max_km': R_MAX, 'min_years': MIN_YEARS,
                      'sigma_C': sC, 'sigma_L': sL, 'sigma_g': sG, 'sigma_q': R.get('sigma_q', 0.0),
                      'n_stations_in_archive': int(len(st['lat']))},
           'sites': {}, 'countries': {}}
    for c, idx in countries.items():
        S = [Site(sites[i], Z[off[i]:off[i + 1]], E2[off[i]:off[i + 1]], Zt[i], Et[i], sE[i]) for i in idx]
        if model != 'base':
            for x in S:
                x.sQ = R['sigma_q']
        mu, Sig = climate_posterior(S, sC, sL, sG)
        have, rows = [], []
        for j, i in enumerate(idx):
            s = sites[i]; ns = nearest_station(st, s)
            r = R['sites'][s['file']]
            row = {'country': c, 'lat': s['lat'], 'lon': s['lon'], 't_log': s['t_log'], 'mu': r['mu'], 'sd': r['sd'],
                   'mu_climate': r['mu_climate'], 'sd_climate': r['sd_climate'],
                   'W_cell': s['inst']['W'], 'sd_cell': s['inst']['sd']}
            if ns is not None:
                row |= {'W_station': ns['W'], 'sd_station': ns['sd'], 'station': ns['station'], 'station_id': ns['id'],
                        'km': ns['km']}
                have.append(j); rows.append(row)
            out['sites'][s['file']] = row
        rec = {'n': len(idx), 'n_with_station': len(have)}
        if have:
            cbar = np.mean([sC * S[j].g for j in have], axis=0)
            Ws = [rows[k]['W_station'] for k in range(len(rows))]; sds = [rows[k]['sd_station'] for k in range(len(rows))]
            Wc = [rows[k]['W_cell'] for k in range(len(rows))]; sdc = [rows[k]['sd_cell'] for k in range(len(rows))]
            rec |= {'mu': float(cbar @ mu), 'sd': math.sqrt(max(float(cbar @ Sig @ cbar), 0.0)),
                    'W_station': float(np.mean(Ws)), 'sd_station': float(math.sqrt(np.sum(np.square(sds))) / len(sds)),
                    'W_cell': float(np.mean(Wc)), 'sd_cell': float(math.sqrt(np.sum(np.square(sdc))) / len(sdc)),
                    'median_km': float(np.median([r['km'] for r in rows])),
                    'stations': sorted({r['station'] for r in rows})}
            rec['z_station'] = (rec['mu'] - rec['W_station']) / math.hypot(rec['sd'], rec['sd_station'])
            rec['z_cell'] = (rec['mu'] - rec['W_cell']) / math.hypot(rec['sd'], rec['sd_cell'])
        out['countries'][c] = rec
    rows = [r for r in out['sites'].values() if 'W_station' in r]
    out['site_summary'] = {'station': summary(rows, 'mu', 'sd', 'W_station', 'sd_station'),
                           'cell_same_sites': summary(rows, 'mu', 'sd', 'W_cell', 'sd_cell')}
    cs = [v for v in out['countries'].values() if 'z_station' in v]
    out['country_summary'] = {
        'station': {'n': len(cs), 'coverage95': float(np.mean([abs(v['z_station']) < 1.96 for v in cs])),
                    'rmse': float(np.sqrt(np.mean([(v['mu'] - v['W_station']) ** 2 for v in cs])))},
        'cell_same_sites': {'n': len(cs), 'coverage95': float(np.mean([abs(v['z_cell']) < 1.96 for v in cs])),
                            'rmse': float(np.sqrt(np.mean([(v['mu'] - v['W_cell']) ** 2 for v in cs])))}}
    name = 'A7_station_validation.json' if model == 'base' else 'A7_station_validation_adv.json'
    with open(os.path.join(RES, name), 'w') as fh:
        json.dump(out, fh, indent=0)
    print('model', model, 'stations in archive', len(st['lat']), 'sites with a station', len(rows), 'of', len(out['sites']))
    print('sites:', json.dumps(out['site_summary']))
    print('countries:', json.dumps(out['country_summary']))
    for c in sorted(cs and out['countries'], key=lambda c: out['countries'][c].get('W_cell', 0)):
        v = out['countries'][c]
        if 'mu' in v:
            print('  %-26s %3d/%3d  borehole %.2f +- %.2f   station %.2f (z %+.1f)   cell %.2f (z %+.1f)  %4.0f km' % (
                c, v['n_with_station'], v['n'], v['mu'], v['sd'], v['W_station'], v['z_station'], v['W_cell'],
                v['z_cell'], v['median_km']))


if __name__ == '__main__':
    main(*(sys.argv[1:2] or ['base']))
