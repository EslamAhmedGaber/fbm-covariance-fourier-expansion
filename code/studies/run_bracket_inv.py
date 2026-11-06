"""Law brackets for w = 1/t (CovFourier coordinates of J = 1024 integrands; N = 16384 main, N = 32768 refinement),
CDF bands for Figure 6, and the exact H = 1/2 check.  Writes results/L2_brackets_inv.json, L2_cdf_band.json, L3_exact_bm.json."""
import json, math, os, sys, time
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import law_study as LS

sp = np.load(os.path.join(LS.RES, 'L1_spectra.npz'))
res = {}
for H in LS.H_INV:
    for N in (16384, 32768):
        t0 = time.perf_counter()
        key = f'inv_{H}_{N}'
        b, eta = LS.lower_values('inv_t', H, sp[key], 1024)
        tr = LS.trace_K('inv_t', H); Delta = tr - b.sum()
        S = LS.ChiSquareLaw(b)
        qb = LS.law_bracket(S, Delta)
        chk = max(abs(S.cdf_imhof(v['lo']) - float(p)) for p, v in qb.items())
        chk_t = max(abs(S.cdf_talbot(v['lo'], M=24) - S.cdf_talbot(v['lo'], M=32)) for v in qb.values())
        var_lo = 2 * float(np.sum(b ** 2)); var_hi = var_lo + 4 * (b[0] + Delta) * Delta; var = 0.5 * (var_lo + var_hi)
        g = var / (2 * tr); nu = tr / g
        naive = {str(p): {'gauss': tr + math.sqrt(var) * stats.norm.ppf(p), 'chi2_two_moment': g * stats.chi2.ppf(p, nu)} for p in LS.PS}
        events = {}
        if N == 16384:
            for alpha in (0.05, 0.01, 0.001):
                for name in ('gauss', 'chi2_two_moment'):
                    x = naive[str(1 - alpha)][name]
                    lo, hi = LS.cdf_bracket(S, Delta, [x])
                    events[f'{name}_{alpha}'] = {'x': x, 'P_exceed': [1 - hi[0], 1 - lo[0]]}
        res[key] = {'weight': 'inv_t', 'H': H, 'J': 1024, 'N': N, 'M': int(b.size), 'eta': eta, 'trace': tr, 'Delta': Delta,
                    'var_bracket': [var_lo, var_hi], 'P_gauss_below_0': float(stats.norm.cdf(-tr / math.sqrt(var))),
                    'quantiles': qb, 'naive': naive, 'events': events,
                    'inversion_check_imhof_max_abs': chk, 'inversion_check_talbot_M24_M32': chk_t, 'seconds': time.perf_counter() - t0}
        print(key, 'Delta %.3e' % Delta, {p: '[%.10f, %.10f]' % (v['lo'], v['hi']) for p, v in qb.items()},
              'imhof %.1e talbot %.1e' % (chk, chk_t), '%.0fs' % (time.perf_counter() - t0), flush=True)
        json.dump(res, open(os.path.join(LS.RES, 'L2_brackets_inv.json'), 'w'), indent=1)

# CDF bands (Figure 6)
band = {}
z = np.load(os.path.join(LS.REPO, 'results', 'A4_eigs_J4096.npz'))
for key, weight, H, mu, J, xmax in (('one_0.75', 'one', 0.75, z['mu_0p75'], 4096, 4.6), ('inv_0.75_16384', 'inv_t', 0.75, sp['inv_0.75_16384'], 1024, 7.0)):
    b, _ = LS.lower_values(weight, H, mu, J)
    Delta = LS.trace_K(weight, H) - b.sum()
    S = LS.ChiSquareLaw(b)
    xs = np.linspace(0.01, xmax, 80)
    lo, hi = LS.cdf_bracket(S, Delta, xs)
    band[key] = {'x': xs.tolist(), 'lo': lo, 'hi': hi, 'Delta': Delta}
    print('band', key, 'max width %.2e' % max(h - l for h, l in zip(hi, lo)), flush=True)
json.dump(band, open(os.path.join(LS.RES, 'L2_cdf_band.json'), 'w'))

# exact H = 1/2 check (needs L2_brackets_one.json)
one = json.load(open(os.path.join(LS.RES, 'L2_brackets_one.json')))
json.dump({**one, **res}, open(os.path.join(LS.RES, 'L2_brackets.json'), 'w'), indent=1)
LS.stage_exact()
