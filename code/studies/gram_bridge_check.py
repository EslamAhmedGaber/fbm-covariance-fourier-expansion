"""Check: closed cosine Gram versus the whole-line spectral integral (zero extension).

<h_k, h_l>_H = c_H int |xi|^(1-2H) F(J h_k) conj F(J h_l) d xi, evaluated independently by panel
Gauss-Legendre on [0, 2e5] plus the analytic leading tail (covfourier.spectral_check).
Output: results/A1_gram_vs_spectral.json  (optional argument: other output path)
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
OUT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None
import math
from covfourier.gram import CosineGram
from covfourier import spectral_check as SC

PAIRS = [(0.55, 1.0, 0, 0), (0.55, 1.0, 0, 2), (0.55, 1.0, 0, 10), (0.55, 1.0, 1, 1), (0.55, 1.0, 1, 2), (0.55, 1.0, 1, 3), (0.55, 1.0, 2, 2), (0.55, 1.0, 3, 7), (0.55, 1.0, 4, 10), (0.55, 1.0, 5, 5), (0.55, 1.0, 11, 11), (0.55, 1.0, 20, 22), (0.55, 1.0, 29, 29), (0.55, 2.0, 0, 0), (0.55, 2.0, 0, 2), (0.55, 2.0, 0, 10), (0.55, 2.0, 1, 1), (0.55, 2.0, 1, 2), (0.55, 2.0, 1, 3), (0.55, 2.0, 2, 2), (0.55, 2.0, 3, 7), (0.55, 2.0, 4, 10), (0.55, 2.0, 5, 5), (0.55, 2.0, 11, 11), (0.55, 2.0, 20, 22), (0.55, 2.0, 29, 29), (0.75, 1.0, 0, 0), (0.75, 1.0, 0, 2), (0.75, 1.0, 0, 10), (0.75, 1.0, 1, 1), (0.75, 1.0, 1, 2), (0.75, 1.0, 1, 3), (0.75, 1.0, 2, 2), (0.75, 1.0, 3, 7), (0.75, 1.0, 4, 10), (0.75, 1.0, 5, 5), (0.75, 1.0, 11, 11), (0.75, 1.0, 20, 22), (0.75, 1.0, 29, 29), (0.75, 2.0, 0, 0), (0.75, 2.0, 0, 2), (0.75, 2.0, 0, 10), (0.75, 2.0, 1, 1), (0.75, 2.0, 1, 2), (0.75, 2.0, 1, 3), (0.75, 2.0, 2, 2), (0.75, 2.0, 3, 7), (0.75, 2.0, 4, 10), (0.75, 2.0, 5, 5), (0.75, 2.0, 11, 11), (0.75, 2.0, 20, 22), (0.75, 2.0, 29, 29), (0.95, 1.0, 0, 0), (0.95, 1.0, 0, 2), (0.95, 1.0, 0, 10), (0.95, 1.0, 1, 1), (0.95, 1.0, 1, 2), (0.95, 1.0, 1, 3), (0.95, 1.0, 2, 2), (0.95, 1.0, 3, 7), (0.95, 1.0, 4, 10), (0.95, 1.0, 5, 5), (0.95, 1.0, 11, 11), (0.95, 1.0, 20, 22), (0.95, 1.0, 29, 29), (0.95, 2.0, 0, 0), (0.95, 2.0, 0, 2), (0.95, 2.0, 0, 10), (0.95, 2.0, 1, 1), (0.95, 2.0, 1, 2), (0.95, 2.0, 1, 3), (0.95, 2.0, 2, 2), (0.95, 2.0, 3, 7), (0.95, 2.0, 4, 10), (0.95, 2.0, 5, 5), (0.95, 2.0, 11, 11), (0.95, 2.0, 20, 22), (0.95, 2.0, 29, 29)]

def hval(k, T):
    if k == 0:
        return T ** -0.5, T ** -0.5
    return math.sqrt(2 / T), (-1) ** k * math.sqrt(2 / T)

t0 = time.perf_counter()
rows = []
for H, T, k, l in PAIRS:
    Gd = CosineGram(H, T).dense(max(k, l) + 1)
    a0, aT = hval(k, T); b0, bT = hval(l, T)
    spec = SC.spectral_ip(lambda xi: SC.ft_cos(k, xi, T), lambda xi: SC.ft_cos(l, xi, T), H, T, Xi=2e5, ends=(a0, aT, b0, bT))
    rows.append({'H': H, 'T': T, 'k': k, 'l': l, 'closed': float(Gd[k, l]), 'spectral': float(spec), 'abs_diff': float(abs(Gd[k, l] - spec))})
out = {'check': 'closed cosine Gram vs independent spectral integral c_H int |xi|^(1-2H) F(Jh_k) conj F(Jh_l)',
       'method': 'panel Gauss-Legendre on [0, 2e5] + analytic leading tail; omitted remainder O(Xi^(-1-2H))',
       'rows': rows, 'max_abs_diff': max(r['abs_diff'] for r in rows), 'seconds': time.perf_counter() - t0}
json.dump(out, open(OUT or os.path.join(RES, 'A1_gram_vs_spectral.json'), 'w'), indent=1)
print('max abs diff %.3e over %d entries, %.0f s' % (out['max_abs_diff'], len(rows), out['seconds']))
