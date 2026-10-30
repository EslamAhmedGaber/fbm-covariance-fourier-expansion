"""Study W1: whole-line realization and the fixed expanding-window cell rule.

(a) normalization: 18 Gaussian packets exp(-a t^2 + i w t); weighted spectral
    quadrature against the closed physical-kernel value;
(b) the Toeplitz increment series used for step-function norms, against
    50-digit direct values at lags up to 65535;
(c) the cell rule Q_m (levels m = 2..8, window [-2^m, 2^m), width 2^-m) for five
    inputs and their exact covariance norms, with 8-point and 16-point
    Gauss-Legendre cell averages and with exact cell averages, plus successive
    differences ||Q_{m+1} f - Q_m f|| (exact averages).
Output: results/W1_whole_line.json
"""
import os, sys, json, time
import numpy as np
import mpmath as mp
from scipy.special import erfc, gamma, gammainc, gammaincc, beta as Beta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from covfourier.wholeline import (c_H, increments, step_norm2, cell_grid, cell_averages_gl,
                                  upsample, packet_norm2_closed, packet_norm2_spectral)

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
LEVELS = list(range(2, 9))


# ---------------------------------------------------------------- inputs
def f_gauss(t):
    return np.exp(-t * t)

def f_osc(t):
    return np.exp(-0.3 * t * t) * np.cos(8.0 * t + 0.2)

def f_cusp(t):
    return np.exp(-np.abs(t - 0.37))

def f_ind(t):
    return ((t >= -0.7) & (t < 1.2)).astype(float)

def f_sing(t):
    out = np.zeros_like(t)
    pos = t > 0
    out[pos] = t[pos] ** -0.55 * np.exp(-t[pos])
    return out


def exact_avg(name, m):
    a, b, d = cell_grid(m)
    if name == 'gauss':
        s = np.sqrt(np.pi) / 2
        out = np.where(a >= 0, s * (erfc(a) - erfc(b)), s * (erfc(-b) - erfc(-a)))
        return out / d
    if name == 'osc':        # smooth: 16-point rule is exact to rounding at these widths
        return cell_averages_gl(f_osc, m, q=16)
    if name == 'cusp':
        c = 0.37
        left = np.exp(-(c - b)) - np.exp(-(c - a))
        right = np.exp(-(a - c)) - np.exp(-(b - c))
        split = (1 - np.exp(-(c - a))) + (1 - np.exp(-(b - c)))
        out = np.where(b <= c, left, np.where(a >= c, right, split))
        return out / d
    if name == 'ind':
        lo, hi = -0.7, 1.2
        return np.clip(np.minimum(b, hi) - np.maximum(a, lo), 0, None) / d
    if name == 'sing':
        g = gamma(0.45)
        out = np.zeros(a.size)
        pos = a >= 0
        aa, bb = a[pos], b[pos]
        small = bb <= 1.0
        v = np.empty(aa.size)
        v[small] = g * (gammainc(0.45, bb[small]) - gammainc(0.45, aa[small]))
        v[~small] = g * (gammaincc(0.45, aa[~small]) - gammaincc(0.45, bb[~small]))
        out[pos] = v
        return out / d
    raise KeyError(name)


def exact_norm2(name, H):
    cH = c_H(H)
    if name == 'gauss':
        return cH * 2.0 ** (-H) * gamma(1 - H)
    if name == 'cusp':
        return cH * (2 / np.pi) * gamma(1 - H) * gamma(1 + H)
    if name == 'ind':
        return 1.9 ** (2 * H)
    if name == 'sing':
        return cH * gamma(0.45) ** 2 / (2 * np.pi) * Beta(1 - H, H - 0.55)
    if name == 'osc':
        mp.mp.dps = 30
        a = mp.mpf('0.3'); Hm = mp.mpf(H)
        cHm = mp.gamma(2 * Hm + 1) * mp.sin(mp.pi * Hm)
        g = lambda x: abs(x) ** (1 - 2 * Hm) * (mp.exp(-(x - 8) ** 2 / (2 * a)) + mp.exp(-(x + 8) ** 2 / (2 * a))
                                                + 2 * mp.cos(mp.mpf('0.4')) * mp.exp(-(x * x + 64) / (2 * a))) / (8 * a)
        return float(cHm * mp.quad(g, [-mp.inf, -14, -8, -2, 0, 2, 8, 14, mp.inf]))
    raise KeyError(name)


FUNCS = {'gauss': f_gauss, 'osc': f_osc, 'cusp': f_cusp, 'ind': f_ind, 'sing': f_sing}
LABEL = {'gauss': 'exp(-t^2)', 'osc': 'exp(-0.3 t^2) cos(8t+0.2)', 'cusp': 'exp(-|t-0.37|)',
         'ind': '1[-0.7,1.2)', 'sing': '1(t>0) t^-0.55 exp(-t)'}
HSETS = {'gauss': [0.5, 0.75, 0.95], 'osc': [0.5, 0.75, 0.95], 'cusp': [0.5, 0.75, 0.95],
         'ind': [0.5, 0.75, 0.95], 'sing': [0.75, 0.95]}


def main():
    t0 = time.time()
    out = {'description': __doc__, 'levels': LEVELS}
    # (a) packets
    pk = []
    for H in [0.5, 0.5001, 0.55, 0.75, 0.95, 0.99]:
        for (a, w) in [(0.3, 0.0), (1.0, 3.0), (2.7, 10.0)]:
            c = packet_norm2_closed(H, a, w); s = packet_norm2_spectral(H, a, w)
            pk.append({'H': H, 'a': a, 'omega': w, 'closed': float(c), 'spectral': float(s),
                       'rel_diff': float(abs(s - c) / abs(c))})
    out['packets'] = pk
    out['packets_max_rel_diff'] = max(p['rel_diff'] for p in pk)
    # (b) increments
    lags = [1, 2, 7, 8, 9, 100, 1000, 10000, 65535]
    inc = []
    mp.mp.dps = 50
    for H in [0.55, 0.65, 0.75, 0.85, 0.95]:
        rho, tail = increments(H, 65536)
        p = mp.mpf(2 * H)
        for k in lags:
            kk = mp.mpf(k)
            ref = ((kk + 1) ** p - 2 * kk ** p + (kk - 1) ** p) / 2
            inc.append({'H': H, 'k': k, 'value': float(rho[k]), 'rel_err': float(abs(rho[k] - ref) / abs(ref)),
                        'series_tail_bound_rel': tail})
    out['increments'] = inc
    out['increments_max_rel_err'] = max(r['rel_err'] for r in inc)
    # (c) cell rule
    cases = []
    for name, f in FUNCS.items():
        for H in HSETS[name]:
            ref = exact_norm2(name, H)
            rows = []
            prev_exact = None
            for m in LEVELS:
                a, b, d = cell_grid(m)
                rho, _ = increments(H, a.size)
                c8 = cell_averages_gl(f, m, q=8)
                c16 = cell_averages_gl(f, m, q=16)
                ce = exact_avg(name, m)
                n8 = step_norm2(c8, H, d, rho); n16 = step_norm2(c16, H, d, rho); ne = step_norm2(ce, H, d, rho)
                row = {'m': m, 'cells': int(a.size), 'width': d,
                       'rel_disc_gl8': (n8 - ref) / ref, 'rel_disc_gl16': (n16 - ref) / ref,
                       'rel_disc_exact_avg': (ne - ref) / ref,
                       'gl8_vs_gl16_rel': abs(n8 - n16) / ref}
                if prev_exact is not None:
                    dlt = ce - upsample(prev_exact)
                    row['succ_diff_rel'] = float(np.sqrt(max(step_norm2(dlt, H, d, rho), 0.0) / ref))
                prev_exact = ce
                rows.append(row)
            cases.append({'input': name, 'label': LABEL[name], 'H': H, 'norm2_exact': ref, 'levels': rows})
            last = rows[-1]
            print('%-6s H=%.2f  disc(m=8): gl8 %+.2e  gl16 %+.2e  exact %+.2e   succ %.2e' % (
                name, H, last['rel_disc_gl8'], last['rel_disc_gl16'], last['rel_disc_exact_avg'], last.get('succ_diff_rel', np.nan)), flush=True)
    out['cell_rule'] = cases
    out['runtime_s'] = time.time() - t0
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, 'W1_whole_line.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('packets max rel diff %.2e; increments max rel err %.2e; runtime %.1f s' % (
        out['packets_max_rel_diff'], out['increments_max_rel_err'], out['runtime_s']))


if __name__ == '__main__':
    main()
