"""Parser and structural profile of the NOAA/NCEI Huang-Pollack borehole files (huang2016 series).

Each file has a '#' header (site, coordinates, logging year, conductivity, geothermal gradient, the database's own
century trends) and a data section 'depth_m temp_meas'.  Some files contain several depth-temperature blocks separated
by blank lines.  We never guess: a block is attributed to the site only if its own least-squares gradient over its
deepest half matches the header gradient (|ratio - 1| <= 0.25) and it is the unique such block; otherwise the file is
reported as ambiguous and excluded.  The rule uses metadata and the block's own deep gradient only (no climate fit).
"""
import glob
import json
import math
import os
import re

import numpy as np

def _workspace_root():
    """First ancestor that contains 04_DATA (the workspace root); override with env BOREHOLE_ROOT."""
    if os.environ.get('BOREHOLE_ROOT'):
        return os.environ['BOREHOLE_ROOT']
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, '04_DATA')):
            return d
        d = os.path.dirname(d)
    raise FileNotFoundError('workspace root with 04_DATA not found; set BOREHOLE_ROOT')


ROOT = _workspace_root()
DATA = os.path.join(ROOT, '04_DATA', 'Borehole_NOAA_Huang', 'huang2016')


def _num(pattern, text):
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    head, _, body = txt.partition('\ndepth_m')
    meta = {
        'file': os.path.basename(path),
        'site': (re.search(r'Site_Name:\s*(\S+)', head) or [None, None])[1],
        'country': (re.search(r'Location:\s*(.+)', head) or [None, ''])[1].strip(),
        'lat': _num(r'Northernmost_Latitude:\s*(-?[\d.]+)', head),
        'lon': _num(r'Easternmost_Longitude:\s*(-?[\d.]+)', head),
        'log_year': _num(r'Date of logging \(Year\):\s*(-?[\d.]+)', head),
        'k_cond': _num(r'Thermal Conductivity \(W/m/K\):\s*(-?[\d.]+)', head),
        'gradient_K_per_km': _num(r'Geothermal Gradient \(K/km\):\s*(-?[\d.]+)', head),
        'pre1500_gst': _num(r'Pre-1500 Baseline GST \(oC\):\s*(-?[\d.]+)', head),
        'century_trends': [float(x) for x in re.findall(r'#\s+\d{2}th\s+(-?[\d.]+)', head)],
    }
    blocks, cur = [], []
    for line in body.splitlines()[1:]:
        s = line.strip()
        if not s:
            if cur:
                blocks.append(cur); cur = []
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            z, T = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if math.isfinite(z) and math.isfinite(T):
            cur.append((z, T))
    if cur:
        blocks.append(cur)
    return meta, [np.array(b) for b in blocks]


def deep_gradient(b):
    """Least-squares dT/dz (K/km) over the deeper half of a block (depth-sorted)."""
    b = b[np.argsort(b[:, 0])]
    h = b[len(b) // 2:] if len(b) >= 8 else b
    if len(h) < 3 or np.ptp(h[:, 0]) <= 0:
        return None
    return 1000.0 * float(np.polyfit(h[:, 0], h[:, 1], 1)[0])


def attribute(meta, blocks, tol=0.25):
    """Index of the unique block whose deep gradient matches the header gradient, else None (with reason)."""
    g0 = meta['gradient_K_per_km']
    if len(blocks) == 1:
        return 0, 'single block'
    if g0 is None or g0 <= 0:
        return None, 'several blocks and no header gradient'
    ok = [i for i, b in enumerate(blocks) if (deep_gradient(b) is not None and abs(deep_gradient(b) / g0 - 1) <= tol)]
    if len(ok) == 1:
        return ok[0], 'unique gradient match'
    return None, f'{len(ok)} gradient matches among {len(blocks)} blocks'


def load_all():
    rows = []
    for p in sorted(glob.glob(os.path.join(DATA, 'huang2016-*.txt'))):
        meta, blocks = parse(p)
        idx, why = attribute(meta, blocks)
        meta['n_blocks'] = len(blocks)
        meta['attribution'] = why
        meta['block_index'] = idx
        meta['block_gradients'] = [deep_gradient(b) for b in blocks]
        if idx is not None:
            b = blocks[idx]
            b = b[np.argsort(b[:, 0])]
            meta['depth'] = b[:, 0].tolist(); meta['temp'] = b[:, 1].tolist()
        rows.append(meta)
    return rows


if __name__ == '__main__':
    rows = load_all()
    nb = [r['n_blocks'] for r in rows]
    att = [r for r in rows if r['block_index'] is not None]
    single = [r for r in rows if r['n_blocks'] == 1]
    print('files', len(rows), 'single-block', len(single), 'multi-block', len(rows) - len(single))
    print('blocks histogram', {k: nb.count(k) for k in sorted(set(nb))})
    print('attributed', len(att), 'excluded', len(rows) - len(att))
    from collections import Counter
    print(Counter(r['attribution'] for r in rows))
    g = [(r['gradient_K_per_km'], r['block_gradients'][r['block_index']]) for r in single if r['gradient_K_per_km'] and r['block_gradients'][0]]
    ratio = np.array([b / a for a, b in g if a > 0])
    print('single-block deep-gradient / header-gradient: median %.3f, 5-95%% [%.3f, %.3f]' % (np.median(ratio), *np.percentile(ratio, [5, 95])))
    yrs = np.array([r['log_year'] for r in att if r['log_year']])
    print('log years', np.percentile(yrs, [0, 5, 25, 50, 75, 95, 100]))
    npts = np.array([len(r['depth']) for r in att]); zmin = np.array([min(r['depth']) for r in att]); zmax = np.array([max(r['depth']) for r in att])
    print('points', np.percentile(npts, [0, 25, 50, 75, 100]), 'zmin', np.percentile(zmin, [0, 50, 100]), 'zmax', np.percentile(zmax, [0, 50, 100]))
    rep = Counter(re.sub(r'-(19|20)\d\d$', '', r['site']) for r in rows if re.search(r'-(19|20)\d\d$', r['site'] or ''))
    print('repeat-logged site groups', len([k for k, v in rep.items() if v >= 2]), sorted(rep.items())[:40])
    out = os.path.join(ROOT, '04_DATA', 'Borehole_NOAA_Huang', 'PROFILE.json')
    with open(out, 'w') as fh:
        json.dump([{k: v for k, v in r.items() if k not in ('depth', 'temp')} | {'n_points': len(r.get('depth', []))} for r in rows], fh, indent=0)
    print('wrote', out)
