"""Third, independent inversion route for the finite chi-square laws: de Hoog's algorithm (mpmath.invertlaplace,
30 digits, Laplace transform evaluated in 30-digit arithmetic).  For each case it evaluates F_S at the computed lower
quantile ends q_S(p) (Talbot route) and reports |F_S(q_S(p)) - p|.  Writes results/L7_inversion_check.json."""
import json, os, sys, time
import numpy as np
import mpmath as mp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import law_study as LS

mp.mp.dps = 30
z = np.load(os.path.join(LS.REPO, 'results', 'A4_eigs_J4096.npz'))
sp = np.load(os.path.join(LS.RES, 'L1_spectra.npz'))
L2 = {}
for f in ('L2_brackets_one.json', 'L2_brackets_inv.json'):
    p = os.path.join(LS.RES, f)
    if os.path.exists(p):
        L2.update(json.load(open(p)))
cases = [(f'one_{H}', 'one', H, z['mu_' + str(H).replace('.', 'p')], 4096) for H in LS.H_ONE] + \
        [(f'inv_{H}_16384', 'inv_t', H, sp[f'inv_{H}_16384'], 1024) for H in LS.H_INV]
out = {}
for key, w, H, mu, J in cases:
    if key not in L2:
        continue
    b, _ = LS.lower_values(w, H, mu, J)
    bm = [mp.mpf(float(v)) for v in b if v > 0]
    def Fhat(s):
        return mp.exp(-mp.fsum(mp.log(1 + 2 * s * v) for v in bm) / 2) / s
    t0 = time.perf_counter()
    errs = {}
    for p, q in L2[key]['quantiles'].items():
        F = mp.invertlaplace(Fhat, q['lo'], method='dehoog')
        errs[p] = float(abs(F - mp.mpf(p)))
    out[key] = {'max_abs_err': max(errs.values()), 'by_p': errs, 'seconds': time.perf_counter() - t0}
    print(key, out[key], flush=True)
    json.dump(out, open(os.path.join(LS.RES, 'L7_inversion_check.json'), 'w'), indent=1)
