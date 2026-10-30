"""Study B2: double-precision projection against 90-digit reference values (D = 65, 129, 257).

Inputs sin(140 pi t) (H = 0.5, 0.75, 0.95) and t^-0.55 (H = 0.75, 0.95) with exact classical cosine
coefficients (moment interface, N_aux = 2^20) and exact norms. Reference: results/reference_highprecision_D257.json.
Output: results/B2_vs_90digit_reference.json  (optional argument: other output path)
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
OUT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None
from covfourier import gram as G, engine as EN, moments as MO

ref = json.load(open(os.path.join(RES, 'reference_highprecision_D257.json')))
refrows = {}
for r in ref['runs']:
    for inp in r['inputs']:
        for row in inp['rows']:
            refrows[(r['H'], inp['input'], row['D'])] = row
rows = []
Naux = 2 ** 20
for H in [0.5, 0.75, 0.95]:
    for D in [65, 129, 257]:
        t0 = time.perf_counter(); e = EN.CovarianceFourier(H, D - 2, 1.0, enrich=True, Naux=Naux); st = time.perf_counter() - t0
        f = MO.Sine(140 * np.pi); n2 = f.norm2(H, 1.0)
        r = e.project(f.cosine(Naux, 1.0))
        E2_sub = n2 - float(np.dot(r['coords'], r['coords']))
        rf = refrows[(H, 'high_frequency', D)]
        rows.append({'H': H, 'D': D, 'input': 'sin(140 pi t)', 'setup_s': st, 'E2_direct': r['error2'], 'E2_subtractive': E2_sub,
                     'ref_best_E2': rf['best_projection_error2'], 'ref_norm2': rf['norm2'], 'norm2': n2})
        print(H, D, 'sin  direct %.10e  ref %.10e' % (r['error2'], rf['best_projection_error2']), flush=True)
        if H > 0.55:
            p = -0.55; a = G.power_cosine_moments(p, Naux, 1.0); n2 = G.poly_poly(H, p, p, 1.0)
            r = e.project(a)
            E2_sub = n2 - float(np.dot(r['coords'], r['coords']))
            rf = refrows[(H, 'singular', D)]
            rows.append({'H': H, 'D': D, 'input': 't^-0.55', 'E2_direct_truncated': r['error2'], 'E2_subtractive': E2_sub,
                         'ref_best_E2': rf['best_projection_error2'], 'ref_norm2': rf['norm2'], 'norm2': n2})
            print(H, D, 't^-.55  subtractive %.10e  ref %.10e' % (E2_sub, rf['best_projection_error2']), flush=True)
json.dump({'check': 'double-precision engine vs 90-digit reference values (reference_highprecision_D257.json)', 'Naux': Naux, 'rows': rows},
          open(OUT or os.path.join(RES, 'B2_vs_90digit_reference.json'), 'w'), indent=1)
