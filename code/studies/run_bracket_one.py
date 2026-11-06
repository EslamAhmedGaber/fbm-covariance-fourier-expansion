"""Law brackets for w = 1 (stored Galerkin spectra, J = 4096) and the exact H = 1/2 check, run ahead of the
weighted spectra.  Same functions as law_study.py; writes results/L2_brackets_one.json and L3_exact_bm.json."""
import json, math, os, sys, time
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import law_study as LS

z = np.load(os.path.join(LS.REPO, 'results', 'A4_eigs_J4096.npz'))
res = {}
for H in LS.H_ONE:
    t0 = time.perf_counter()
    mu = np.sort(z['mu_' + str(H).replace('.', 'p')])[::-1]
    b, eta = LS.lower_values('one', H, mu, 4096)
    tr = LS.trace_K('one', H); Delta = tr - b.sum()
    S = LS.ChiSquareLaw(b)
    qb = LS.law_bracket(S, Delta)
    chk = max(abs(S.cdf_imhof(v['lo']) - float(p)) for p, v in qb.items())
    chk_t = max(abs(S.cdf_talbot(v['lo'], M=24) - S.cdf_talbot(v['lo'], M=32)) for v in qb.values())
    var_lo = 2 * float(np.sum(b ** 2)); var_hi = var_lo + 4 * (b[0] + Delta) * Delta; var = 0.5 * (var_lo + var_hi)
    g = var / (2 * tr); nu = tr / g
    naive = {str(p): {'gauss': tr + math.sqrt(var) * stats.norm.ppf(p), 'chi2_two_moment': g * stats.chi2.ppf(p, nu)} for p in LS.PS}
    events = {}
    for alpha in (0.05, 0.01, 0.001):
        for name in ('gauss', 'chi2_two_moment'):
            x = naive[str(1 - alpha)][name]
            lo, hi = LS.cdf_bracket(S, Delta, [x])
            events[f'{name}_{alpha}'] = {'x': x, 'P_exceed': [1 - hi[0], 1 - lo[0]]}
    key = f'one_{H}'
    res[key] = {'weight': 'one', 'H': H, 'J': 4096, 'M': int(b.size), 'eta': eta, 'trace': tr, 'Delta': Delta,
                'var_bracket': [var_lo, var_hi], 'P_gauss_below_0': float(stats.norm.cdf(-tr / math.sqrt(var))),
                'quantiles': qb, 'naive': naive, 'events': events,
                'inversion_check_imhof_max_abs': chk, 'inversion_check_talbot_M24_M32': chk_t, 'seconds': time.perf_counter() - t0}
    print(key, 'Delta %.3e' % Delta, {p: '[%.10f, %.10f]' % (v['lo'], v['hi']) for p, v in qb.items()},
          'imhof %.1e talbot %.1e' % (chk, chk_t), '%.0fs' % (time.perf_counter() - t0), flush=True)
    json.dump(res, open(os.path.join(LS.RES, 'L2_brackets_one.json'), 'w'), indent=1)
