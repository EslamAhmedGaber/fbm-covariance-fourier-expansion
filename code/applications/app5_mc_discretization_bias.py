"""A4/A5 supplement: exact time-discretization bias of the Monte Carlo estimators.

The Monte Carlo comparators sample exact fBm values on a grid and replace Q_w = int_0^1 w B_H^2 by a quadrature sum
Q_N = sum_j c_j w(t_j) B_H(t_j)^2.  Q_N is a quadratic form of a Gaussian vector, so its Laplace transform is exact:
E exp(-a Q_N) = prod_j (1 + 2 a mu_j)^(-1/2), mu_j = eigenvalues of diag(sqrt(c w)) C diag(sqrt(c w)),
C_ij = (t_i^2H + t_j^2H - |t_i - t_j|^2H) / 2.  The difference to L_w(a) is the bias of the Monte Carlo estimator,
which its confidence interval does not contain.
  A5 rule (app5_weighted_kl, app5_mc_reference): t_j = j/N, j = 1..N, c_j = 1/N, N = 4096, w in {1, 1/t}.
  A4 rule (app4_analysis): t_j = j/N, trapezoid c_j = 1/N (j < N), 1/(2N) (j = N), N = 1024, w = 1.
Writes results/A5_mc_discretization_bias.json (under a minute).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')


def discrete_laplace(H, N, weight, rule, a=1.0):
    t = np.arange(1, N + 1) / N
    c = np.full(N, 1.0 / N)
    if rule == 'trapezoid':
        c[-1] = 0.5 / N
    w = np.ones(N) if weight == 'one' else 1.0 / t
    C = 0.5 * (t[:, None] ** (2 * H) + t[None, :] ** (2 * H) - np.abs(t[:, None] - t[None, :]) ** (2 * H))
    s = np.sqrt(c * w)
    mu = np.linalg.eigvalsh(s[:, None] * C * s[None, :])
    return float(np.exp(-0.5 * np.sum(np.log1p(2 * a * np.clip(mu, 0.0, None))))), float(mu.min())


def run(out='A5_mc_discretization_bias.json'):
    a5 = json.load(open(os.path.join(RES, 'A5_weighted_kl.json')))
    a4 = json.load(open(os.path.join(RES, 'A4_analysis.json')))
    data = {'note': __doc__, 'rows': []}
    br = {}
    for c in a5['cases']:
        if c['J'] == 1024:
            br[(c['weight'], c['H'])] = c['laplace_bracket']['1.0']
    for weight in ('one', 'inv_t'):
        for H in (0.55, 0.75, 0.95):
            t0 = time.perf_counter(); LN, mumin = discrete_laplace(H, 4096, weight, 'right'); dt = time.perf_counter() - t0
            lo, hi = br[(weight, H)]
            data['rows'].append({'study': 'A5', 'weight': weight, 'H': H, 'N': 4096, 'rule': 'right endpoint', 'L_N': LN,
                                 'bracket_J1024': [lo, hi], 'bias_interval': [LN - hi, LN - lo], 'min_mu': mumin, 'seconds': dt})
            print('A5 %-5s H %.2f L_N %.10f bracket [%.10f, %.10f] bias in [%.3e, %.3e]  %.1f s' % (weight, H, LN, lo, hi, LN - hi, LN - lo, dt), flush=True)
    # A4: H = 0.75, trapezoid, N = 1024 (Monte Carlo mean 0.73534 +- 0.00147, 95% CI half-width)
    t0 = time.perf_counter(); LN, mumin = discrete_laplace(0.75, 1024, 'one', 'trapezoid'); dt = time.perf_counter() - t0
    lo, hi = a4['H']['0.75']['laplace']['1.0']['improved']          # closed-form Galerkin bracket, J = 4096
    data['rows'].append({'study': 'A4', 'weight': 'one', 'H': 0.75, 'N': 1024, 'rule': 'trapezoid', 'L_N': LN,
                         'bracket_J4096': [lo, hi], 'bias_interval': [LN - hi, LN - lo], 'min_mu': mumin, 'seconds': dt})
    print('A4 one   H 0.75 L_N %.10f bias in [%.3e, %.3e]' % (LN, LN - hi, LN - lo), flush=True)
    with open(os.path.join(RES, out), 'w') as fh:
        json.dump(data, fh, indent=1)
    return data


if __name__ == '__main__':
    run()
