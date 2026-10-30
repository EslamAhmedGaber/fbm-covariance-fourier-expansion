"""A5 supplement: Monte Carlo reference for L_w(1) = E exp(-Q_w), Q_w = int_0^1 w B_H^2, w in {1, 1/t}.

Same estimator as app5_weighted_kl.monte_carlo: exact Davies-Harte paths on the grid t_j = j/N (N = 4096), right-endpoint
rule for Q_w, chunks of 1000 paths, seed 11.  For M = 2500 ... 40000 paths records the mean, the standard error (SE), the
full width 3.92 SE of the normal-approximation 95% confidence interval, and the wall time including the circulant setup.
The interval describes sampling error only; the time-discretization bias of the right-endpoint rule is not included.
Writes results/A5_mc_reference.json (about 6 minutes).
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from covfourier.comparators import DaviesHarte         # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
Z975 = 1.959963984540054


def mc(H, weight, M, N=4096, seed=11, a=1.0):
    t0 = time.perf_counter()
    dh = DaviesHarte(H, N)
    rng = np.random.default_rng(seed)
    t = np.arange(1, N + 1) / N
    wv = np.ones(N) if weight == 'one' else 1.0 / t
    vals = []
    for m0 in range(0, M, 1000):
        B = np.cumsum(dh.increments(min(1000, M - m0), rng), axis=1)
        vals.append(np.exp(-a * (np.sum(B ** 2 * wv[None, :], axis=1) / N)))
    v = np.concatenate(vals)
    se = float(v.std(ddof=1) / math.sqrt(v.size))
    return {'H': H, 'weight': weight, 'M': M, 'N': N, 'seconds': time.perf_counter() - t0, 'mean': float(v.mean()), 'se': se,
            'ci95_width': 2 * Z975 * se}


def run(Hs=(0.55, 0.75, 0.95), Ms=(2500, 5000, 10000, 20000, 40000), out='A5_mc_reference.json'):
    data = {'note': __doc__, 'rows': []}
    for weight in ('one', 'inv_t'):
        for H in Hs:
            for M in Ms:
                r = mc(H, weight, M)
                data['rows'].append(r)
                print('%-5s H %.2f M %6d mean %.5f se %.5f ci95 %.2e  %.1f s' % (weight, H, M, r['mean'], r['se'], r['ci95_width'], r['seconds']), flush=True)
                with open(os.path.join(RES, out), 'w') as fh:
                    json.dump(data, fh, indent=1)
    return data


if __name__ == '__main__':
    run()
