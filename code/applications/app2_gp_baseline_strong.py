"""A2 v2 supplement: a stronger exact-GP baseline, to test how sensitive the measured crossover is to the GP code.

Same windows, data model and targets as app2_noisy_reconstruction_v2.py (same seeds), H = 0.75.  Differences:
  * only the lower triangle of K is evaluated (all that the Cholesky factorization reads): half the kernel calls;
  * z <= 8: piecewise Chebyshev with gathered coefficients (Clenshaw, no per-panel masks);
  * z > 8 : asymptotic series in Horner form in w = z^-2 (one power per entry instead of eleven).
The covariance is validated entrywise against the v2 builder at m = 1000; posterior means and variances of the
25 targets are compared with the v2 GP at every m.  Output: results/A2_gp_baseline_strong.json
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.special import binom, ndtr
from scipy.linalg import cho_factor, cho_solve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import app2_noisy_reconstruction_v2 as V2                                          # noqa: E402

RES = V2.RES


class FasterAbsMoment(V2.FastAbsMoment):
    def __init__(self, p):
        super().__init__(p)
        self.coefT = np.ascontiguousarray(self.coef.T)                  # (DEG+1, NP)
        dfact, cs = 1.0, []
        for k in range(0, 11):
            if k > 0:
                dfact *= (2 * k - 1)
            cs.append(binom(self.p, 2 * k) * dfact)
        self.asym = np.array(cs)

    def __call__(self, mu, s):
        mu = np.asarray(mu, dtype=float); s = np.asarray(s, dtype=float)
        mu, s = np.broadcast_arrays(mu, s)
        amu = np.abs(mu)
        z = amu / s
        out = np.empty(z.shape)
        small = z <= self.ZMAX
        zs = z[small]
        idx = np.minimum((zs * (1 / self.h)).astype(np.int64), self.NP - 1)
        u = 2 * (zs * (1 / self.h) - idx) - 1
        b1 = np.zeros_like(u); b2 = np.zeros_like(u); u2 = 2 * u
        for k in range(self.DEG, 0, -1):
            b1, b2 = np.take(self.coefT[k], idx) + u2 * b1 - b2, b1
        val = np.take(self.coefT[0], idx) + u * b1 - b2
        out[small] = s[small] ** self.p * self.pref * val
        big = ~small
        w = (s[big] / amu[big]) ** 2
        acc = np.full(w.shape, self.asym[-1])
        for ck in self.asym[-2::-1]:
            acc = acc * w + ck
        out[big] = amu[big] ** self.p * acc
        return out


def lower_K(H, c, s, m, mabs, block=500):
    mi = mabs(c[:m], s[:m])
    K = np.empty((m, m))
    for i0 in range(0, m, block):
        i1 = min(m, i0 + block)
        K[i0:i1, :i1] = 0.5 * (mi[i0:i1, None] + mi[None, :i1] - mabs(c[i0:i1, None] - c[None, :i1],
                                                                       np.sqrt(s[i0:i1, None] ** 2 + s[None, :i1] ** 2)))
    return K


def run(H=0.75, ms=(1000, 2000, 4000, 8000), out='A2_gp_baseline_strong.json'):
    fa = FasterAbsMoment(2 * H); ref = V2.FastAbsMoment(2 * H)
    rng = np.random.default_rng(2)
    z = rng.uniform(0, 30, 2_000_000); sgm = rng.uniform(0.005, 0.04, z.size); mu = z * sgm
    t0 = time.perf_counter(); a = fa(mu, sgm); t_fast = time.perf_counter() - t0
    t0 = time.perf_counter(); b = ref(mu, sgm); t_ref = time.perf_counter() - t0
    data = {'note': __doc__, 'H': H, 'kernel_2e6_z_0_30': {'faster_s': t_fast, 'v2_table_s': t_ref,
                                                            'max_rel_diff': float(np.max(np.abs(a / b - 1)))}, 'cases': []}
    print(data['kernel_2e6_z_0_30'], flush=True)
    v2 = json.load(open(os.path.join(RES, 'A2_noisy_reconstruction_v2.json')))
    v2cases = {c['m']: c for c in v2['cases'][str(H)]}
    for m in ms:
        rng = np.random.default_rng(20261029 + m)
        c, s = V2.draw_windows(rng, m + V2.N_NEW)
        t0 = time.perf_counter()
        K = lower_K(H, c, s, m, fa)
        t_build = time.perf_counter() - t0
        # target rows (25 x m) with the same builder (cheap); timed as part of the build
        t0 = time.perf_counter()
        p = 2 * H; mi = fa(c, s); mp1 = V2.FastAbsMoment(p + 1)
        rows = [0.5 * (mi[:m] + 1.0 - fa(1.0 - c[:m], s[:m])),
                0.5 * (mi[:m] + 1 / (p + 1) - (mp1(c[:m], s[:m]) + mp1(1 - c[:m], s[:m])) / (p + 1))]
        var = [1.0, 1.0 / (p + 2)]
        for ts in V2.TSTARS:
            rows.append(0.5 * (mi[:m] + ts ** p - fa(c[:m] - ts, s[:m]))); var.append(ts ** p)
        cn, sn = c[m:], s[m:]
        Kn = 0.5 * (mi[m:, None] + mi[None, :m] - fa(cn[:, None] - c[None, :m], np.sqrt(sn[:, None] ** 2 + s[None, :m] ** 2)))
        vn = 0.5 * (2 * mi[m:] - fa(np.zeros_like(cn), np.sqrt(2) * sn))
        Kt = np.vstack(rows + [Kn]); vt = np.r_[np.array(var), vn]
        t_build += time.perf_counter() - t0
        rec = {'m': m}
        if m == ms[0]:
            Kref, _, _ = V2.gp_covariances(H, c, s, m, ref)
            tri = np.tril_indices(m)
            rec['K_lower_max_abs_diff_vs_v2'] = float(np.max(np.abs(K[tri] - Kref[tri])))
            del Kref
        K.flat[::m + 1] += V2.TAU ** 2
        t0 = time.perf_counter(); L = cho_factor(K, lower=True, overwrite_a=True, check_finite=False); t_fact = time.perf_counter() - t0
        y = np.tril(L[0]) @ rng.standard_normal(m)
        t0 = time.perf_counter()
        alpha = cho_solve(L, y, check_finite=False)
        mean = Kt @ alpha
        varp = vt - np.einsum('ij,ji->i', Kt, cho_solve(L, Kt.T, check_finite=False))
        t_solve = time.perf_counter() - t0
        del K, L
        rec.update({'build_s': t_build, 'factor_s': t_fact, 'solve_s': t_solve, 'total_s': t_build + t_fact + t_solve})
        v = v2cases.get(m)
        if v:
            rec['v2_gp_total_s'] = v['GP']['total_s']
            mt = v['matched']
            rec['cf_matched_n'] = mt.get('n'); rec['cf_matched_total_s'] = mt.get('cf_total_s')
            rec['strong_gp_over_cf'] = (rec['total_s'] / mt['cf_total_s']) if mt.get('n') else None
        data['cases'].append(rec)
        print(json.dumps(rec), flush=True)
        with open(os.path.join(RES, out), 'w') as fh:
            json.dump(data, fh, indent=1)
    return data


if __name__ == '__main__':
    run()
