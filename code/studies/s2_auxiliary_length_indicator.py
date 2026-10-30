"""S2 supplement: auxiliary length for a discontinuous input (indicator 1[0,0.3), H = 0.75, D = 1025).

For N = 32, 64 and 128 times (n + 1) the explicit residual energy is computed (a) from N midpoint samples through
LargeProjector.from_callables (default class) and (b) from exact classical coefficients truncated at N; both are compared
with a converged reference from exact coefficients at N = 2^21.  Writes results/S2_auxiliary_length_indicator.json.
"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from covfourier.large import LargeProjector             # noqa: E402
RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
H, n, TAU = 0.75, 1023, 0.3


def ind_coef(N):
    k = np.arange(N, dtype=float); a = np.empty(N); a[0] = TAU
    a[1:] = math.sqrt(2) * np.sin(k[1:] * np.pi * TAU) / (k[1:] * np.pi)
    return a


def exact_E2(lp, N):
    r = lp.project(ind_coef(N)[:, None], np.zeros((2, 1)), explicit=True)
    return float(r['E2_aux_explicit'][0])


def run(out='S2_auxiliary_length_indicator.json'):
    t0 = time.perf_counter(); lpR = LargeProjector(H, n, 1 << 21); ref = exact_E2(lpR, 1 << 21); tref = time.perf_counter() - t0
    data = {'note': __doc__, 'H': H, 'n': n, 'D': n + 2, 'reference_N': 1 << 21, 'E2_reference': ref, 'reference_s': tref, 'rows': []}
    for mult in (32, 64, 128):
        N = mult * (n + 1)
        lp = LargeProjector(H, n, N)
        rs = lp.from_callables([lambda t: (np.asarray(t, float) < TAU).astype(float)], explicit=True)
        Es = float(rs['E2_aux_explicit'][0]); Ex = exact_E2(lp, N)
        row = {'N_over_n_plus_1': mult, 'N': N, 'E2_sampled': Es, 'E2_exact_truncated': Ex,
               'sampled_rel_to_reference': Es / ref - 1, 'exact_truncated_rel_to_reference': Ex / ref - 1}
        data['rows'].append(row)
        print('N/(n+1) %3d  sampled %+.3f%%  exact truncated %+.3f%%' % (mult, 100 * row['sampled_rel_to_reference'], 100 * row['exact_truncated_rel_to_reference']), flush=True)
    with open(os.path.join(RES, out), 'w') as fh:
        json.dump(data, fh, indent=1)
    return data


if __name__ == '__main__':
    run()
