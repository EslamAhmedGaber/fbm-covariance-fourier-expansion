"""A7.10: the urban air record behind the Japanese boreholes: JMA's published station series against CRUTEM5.

All 26 Japanese profiles lie in the Tokyo, Osaka and Nagoya metropolitan areas.  The annual mean temperatures
published by the Japan Meteorological Agency (past weather data, annual values; one decimal) are compared with the
CRUTEM.5.2.0.0 station series of the same stations, year by year, and through the validation target W (mean over the
30 years before logging minus 1901-1930).

Source of the JMA values (transcribed from the HTML tables on 30 October 2026):
  Tokyo  47662  https://www.data.jma.go.jp/obd/stats/etrn/view/annually_s.php?prec_no=44&block_no=47662
  Osaka  47772  https://www.data.jma.go.jp/obd/stats/etrn/view/annually_s.php?prec_no=62&block_no=47772
  Nagoya 47636  https://www.data.jma.go.jp/obd/stats/etrn/view/annually_s.php?prec_no=51&block_no=47636
Check of the transcription: where CRUTEM5 leaves a series unadjusted (Osaka throughout, Tokyo after 1970, Nagoya
after 1960) the two agree to within 0.05 K in every year, the rounding of the published values.
Licence: JMA website content, Public Data License (Version 1.0), compatible with CC BY 4.0
(https://www.jma.go.jp/jma/en/copyright.html).  Source: Japan Meteorological Agency website (URLs above).
Output: results/A7_jma_check.json
"""
import io
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import app7_station_validation as SV                     # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')


def _years(y0, vals):
    return dict(zip(range(y0, y0 + len(vals)), vals))


JMA = {
    '476620': {'name': 'Tokyo', 'windows': [(1968, 1997), (1971, 2000)],
               'annual': _years(1890, [15.0, 14.4, 14.0, 13.8, 14.8, 13.8, 14.0, 13.2, 14.0, 13.8, 13.6])
               | _years(1901, [13.9, 13.7, 13.7, 13.7, 13.5, 13.1, 13.5, 13.2, 13.6, 13.5, 14.4, 13.9, 13.4, 14.7, 14.2,
                               14.5, 13.6, 13.8, 14.1, 14.2, 13.6, 14.4, 14.2, 14.0, 13.8, 13.6, 14.1, 14.1, 14.3, 14.8])
               | _years(1931, [14.0, 14.6, 14.7, 13.9, 14.1, 14.1, 14.9, 14.5, 14.7, 14.8, 14.6, 15.0, 14.6, 14.2, 13.6,
                               14.9, 14.1, 15.2, 14.6, 15.1, 14.7, 14.6, 14.6, 14.8, 15.5, 14.7, 14.9, 15.2, 15.7, 15.4,
                               15.9, 15.4, 15.0, 15.3, 14.6, 15.5, 15.7, 15.6, 15.6, 15.2])
               | _years(1971, [15.0, 15.7, 15.7, 15.2, 15.6, 15.0, 15.8, 16.1, 16.9, 15.4, 15.0, 16.0, 15.7, 14.9, 15.7,
                               15.2, 16.3, 15.4, 16.4, 17.0, 16.4, 16.0, 15.5, 16.9, 16.3, 15.8, 16.7, 16.7, 17.0, 16.9])},
    '477720': {'name': 'Osaka', 'windows': [(1974, 2002)],
               'annual': _years(1901, [14.9, 14.9, 15.1, 14.9, 15.1, 14.5, 14.7, 14.6, 14.7, 14.7, 15.2, 15.1, 14.6, 15.9, 15.7,
                                       16.1, 14.7, 14.8, 15.3, 15.5, 14.9, 15.9, 15.6, 15.0, 14.9, 14.7, 15.1, 15.4, 15.3, 15.8])
               | _years(1974, [15.9, 16.3, 15.9, 16.7, 16.8, 17.1, 15.9, 15.8, 16.1, 16.5, 15.8, 16.6, 15.8, 16.8, 16.1,
                               16.9, 17.4, 17.1, 16.9, 16.3, 17.7, 16.6, 16.3, 16.8, 17.8, 17.3, 17.2, 17.1, 17.3, 16.9])},
    '476360': {'name': 'Nagoya', 'windows': [(1963, 1992)],
               'annual': _years(1901, [14.5, 14.5, 14.7, 14.6, 14.7, 14.1, 14.3, 14.0, 14.3, 14.3, 14.9, 14.6, 14.1, 15.3, 15.1,
                                       15.3, 14.0, 14.2, 14.8, 14.9, 14.0, 15.0, 14.5, 13.9, 13.7, 13.5, 13.9, 14.3, 14.1, 14.6])
               | _years(1963, [14.5, 15.4, 14.3, 14.8, 15.0, 14.6, 14.8, 14.5, 14.8, 15.1, 14.7, 14.3, 14.9, 14.4, 15.4,
                               15.4, 15.9, 14.7, 14.3, 15.2, 15.4, 14.8, 15.5, 14.9, 15.9, 15.0, 16.0, 16.6, 16.1, 15.9])},
}


def crutem_annual(zf, sid):
    import h5py
    n = [m for m in zf.namelist() if m.endswith('stations.%s.nc' % sid)][0]
    with h5py.File(io.BytesIO(zf.read(n)), 'r') as f:
        x = f['tas'][:].astype(float); t = f['time'][:]
    x[x < -1e29] = np.nan
    d = np.datetime64('1850-01-01') + np.floor(t).astype('timedelta64[D]')
    yr = d.astype('datetime64[Y]').astype(int) + 1970
    return {int(y): float(x[yr == y].mean()) for y in np.unique(yr) if np.isfinite(x[yr == y]).sum() == 12}


def window_W(a, recent, early=(1901, 1930)):
    r = [a[y] for y in range(recent[0], recent[1] + 1) if y in a]; e = [a[y] for y in range(early[0], early[1] + 1) if y in a]
    return float(np.mean(r) - np.mean(e)), len(r), len(e)


def main():
    zf = zipfile.ZipFile(SV.ZIP)
    out = {}
    for sid, J in JMA.items():
        C = crutem_annual(zf, sid)
        ys = sorted(set(J['annual']) & set(C))
        diff = {y: C[y] - J['annual'][y] for y in ys}
        dec = {}
        for d0 in range(1890, 2010, 10):
            v = [diff[y] for y in ys if d0 <= y < d0 + 10]
            if v:
                dec['%d-%d' % (d0, d0 + 9)] = {'mean': float(np.mean(v)), 'min': float(np.min(v)), 'max': float(np.max(v)), 'n': len(v)}
        rec = {'name': J['name'], 'years_compared': len(ys), 'crutem_minus_jma_by_decade': dec, 'windows': {}}
        for w in J['windows']:
            wj, nr, ne = window_W(J['annual'], w); wc, _, _ = window_W(C, w)
            rec['windows']['%d-%d' % w] = {'W_jma': wj, 'W_crutem5': wc, 'n_recent': nr, 'n_early': ne}
        out[J['name']] = rec
        print(J['name'], {k: '%+.2f' % v['mean'] for k, v in dec.items()})
        for k, v in rec['windows'].items():
            print('   W %s minus 1901-1930: JMA %.2f K, CRUTEM5 %.2f K' % (k, v['W_jma'], v['W_crutem5']))
    with open(os.path.join(RES, 'A7_jma_check.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
