"""A5 supplement: refinement in N of the computed brackets (error-contract item (i), the Fourier stage).

Repeats the w = 1/t cases at J = 1024, n = 2047 with N = 32768 auxiliary cosines (twice the N = 16384 of
A5_weighted_kl.json) and records the change of the bracket ends for L_w(1).  The change estimates the effect of the
Fourier stage (aliasing and omitted cosine tail) on the computed coordinates; it is an estimate, not a bound.
Writes results/A5_refinement_N32768.json (about 5 minutes).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import app5_weighted_kl as A5                            # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')


def run(Hs=(0.55, 0.75, 0.95), J=1024, N=32768, out='A5_refinement_N32768.json'):
    base = json.load(open(os.path.join(RES, 'A5_weighted_kl.json')))
    ref = {c['H']: c for c in base['cases'] if c['weight'] == 'inv_t' and c['J'] == J}
    data = {'note': __doc__, 'J': J, 'N': N, 'N_base': ref[Hs[0]]['N'], 'rows': []}
    for H in Hs:
        c = A5.run_case(H, 'inv_t', J, n=2047, N=N)
        lo, hi = c['laplace_bracket']['1.0']; lo0, hi0 = ref[H]['laplace_bracket']['1.0']
        row = {'H': H, 'bracket_N': [lo, hi], 'bracket_base': [lo0, hi0], 'change_lo': lo - lo0, 'change_hi': hi - hi0,
               'width_base': hi0 - lo0, 'max_change_over_width': max(abs(lo - lo0), abs(hi - hi0)) / (hi0 - lo0),
               'trace_gap_delta': c['trace_gap_delta'], 'trace_gap_delta_base': ref[H]['trace_gap_delta'],
               'seconds': c['setup_s'] + c['samples_s'] + c['project_s'] + c['eig_s']}
        data['rows'].append(row)
        print('H %.2f N %d bracket [%.10f, %.10f] base [%.10f, %.10f] change %.2e / %.2e (%.1f%% of width)' % (
            H, N, lo, hi, lo0, hi0, row['change_lo'], row['change_hi'], 100 * row['max_change_over_width']), flush=True)
        with open(os.path.join(RES, out), 'w') as fh:
            json.dump(data, fh, indent=1)
    return data


if __name__ == '__main__':
    run()
