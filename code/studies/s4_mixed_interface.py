"""S4: one interface for a mixed collection of inputs (brief item: one unchanged callable interface).

LargeProjector.from_callables is called on ONE unchanged list of eight callables -- smooth, oscillatory, kink, jump and
endpoint-singular -- once with the default regularity class (endpoint=None: N midpoint samples, discrete cosine transform,
a-posteriori endpoint correction; for inputs bounded near 0 and T) and once with the declared algebraic-endpoint class
(endpoint='graded'; graded end cells, no endpoint polynomial).  The class is declared per call and never inferred from the
input.  D = 1025 (n = 1023), N = 2^18, H in {0.55, 0.75, 0.95}, explicit residuals.
References (not used by the method): S1_scaling_study.json (exact classical coefficients and exact polynomial moments,
Naux = 2^18) for the bounded inputs, S3_endpoint_remedy.json (exact classical coefficients at N = 2^18) for the singular
inputs.  Writes results/S4_mixed_interface.json (about 3 minutes).
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from covfourier.large import LargeProjector             # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
N = 1 << 18
n = 1023

INPUTS = [  # (name, callable, local right-endpoint evaluator or None, reference source, reference key)
    ('exp(t)', lambda t: np.exp(np.asarray(t, float)), None, 'S1', 'exp(t)'),
    ('sin(140 pi t)', lambda t: np.sin(140 * np.pi * np.asarray(t, float)), None, 'S1', 'sin(140 pi t)'),
    ('sin(1000 pi t)', lambda t: np.sin(1000 * np.pi * np.asarray(t, float)), None, 'S1', 'sin(1000 pi t)'),
    ('|t-1/2|', lambda t: np.abs(np.asarray(t, float) - 0.5), None, 'S1', '|t-1/2|'),
    ('1[0,0.3)', lambda t: (np.asarray(t, float) < 0.3).astype(float), None, 'S1', '1[0,0.3)'),
    ('t^-0.25', lambda t: np.asarray(t, float) ** -0.25, None, 'S3', 't^-0.25'),
    ('(1-t)^-0.3', lambda t: (1 - np.asarray(t, float)) ** -0.3, lambda s: np.asarray(s, float) ** -0.3, 'S3', '(1-t)^-0.3'),
    ('t^-0.25+1[0,0.3)', lambda t: np.asarray(t, float) ** -0.25 + (np.asarray(t, float) < 0.3), None, 'S3', 't^-0.25+1[0,0.3)'),
]


def references(H):
    s1 = json.load(open(os.path.join(RES, 'S1_scaling_study.json')))
    s3 = json.load(open(os.path.join(RES, 'S3_endpoint_remedy.json')))
    ref = {}
    for r in s1['rows']:
        if r['H'] == H and r['n'] == n and r['Naux'] == N:
            ref[('S1', r['input'])] = (r['error2_enriched'], r['norm2_exact'])
    for r in s3['rows']:
        if r['H'] == H and r['n'] == n and r['N'] == N:
            ref[('S3', r['input'])] = (r['E2_exact_coef'], None, r['eD_true_bracket'])
    return ref


def run(Hs=(0.55, 0.75, 0.95), out='S4_mixed_interface.json'):
    data = {'note': __doc__, 'n': n, 'D': n + 2, 'N': N, 'inputs': [x[0] for x in INPUTS], 'rows': [], 'calls': []}
    fs = [x[1] for x in INPUTS]
    local = [(None, x[2]) if x[2] is not None else None for x in INPUTS]
    for H in Hs:
        ref = references(H)
        t0 = time.perf_counter(); lp = LargeProjector(H, n, N); t_setup = time.perf_counter() - t0
        res = {}
        for route, kw in (('default', {}), ('graded', {'endpoint': 'graded', 'local': local})):
            t0 = time.perf_counter(); r = lp.from_callables(fs, explicit=True, **kw); dt = time.perf_counter() - t0
            res[route] = r
            data['calls'].append({'H': H, 'route': route, 'seconds': dt, 'setup_s': t_setup, 'inputs': len(fs),
                                  'corrected': [bool(x) for x in lp.last_corrected] if route == 'default' else None})
        for i, (name, f, loc, src, key) in enumerate(INPUTS):
            rr = ref.get((src, key))
            row = {'H': H, 'input': name, 'reference_source': src, 'E2_reference': rr[0] if rr else None,
                   'bracket_reference': rr[2] if rr and len(rr) > 2 else None}
            for route in ('default', 'graded'):
                r = res[route]
                row[route] = {'norm2': float(r['norm2'][i]), 'E2_subtractive': float(r['error2'][i]), 'eta': float(r['eta'][i]),
                              'resolved': bool(r['resolved'][i]), 'E2_explicit': float(r['E2_aux_explicit'][i]),
                              'ratio_explicit_to_reference': (float(r['E2_aux_explicit'][i]) / rr[0]) if rr and rr[0] > 0 else None}
            row['rel_reference'] = (rr[0] / row['graded']['norm2']) if rr else None
            data['rows'].append(row)
            print('H %.2f %-18s ref %.3e | default ratio %s resolved %s | graded ratio %s resolved %s' % (
                H, name, rr[0] if rr else float('nan'),
                '%.4f' % row['default']['ratio_explicit_to_reference'] if row['default']['ratio_explicit_to_reference'] is not None else '-',
                row['default']['resolved'],
                '%.4f' % row['graded']['ratio_explicit_to_reference'] if row['graded']['ratio_explicit_to_reference'] is not None else '-',
                row['graded']['resolved']), flush=True)
        with open(os.path.join(RES, out), 'w') as fh:
            json.dump(data, fh, indent=1)
    return data


if __name__ == '__main__':
    run()
