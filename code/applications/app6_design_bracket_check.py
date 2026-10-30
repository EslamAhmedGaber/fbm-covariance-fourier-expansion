"""A6.0: exhaustive check of the design bracket and of the Frank-Wolfe lower bound where everything is computable.

Candidates: m Gaussian window sensors of the fBm path (the A2 sensors, exact closed-form covariance), noise tau.
Targets: B_H(1), int_0^1 B_H, B_H(1/2) and two new window averages.  Nuisance: beta = (offset, slope in the
window centre) with a Gaussian prior, added exactly to both models.  For every design S with |S| <= 3 (all subsets)
and several D, check   V0(S) <= V(S) <= U_post(S) <= U_unif(S)   per target, the monotonicity of V0 in D for nested
D, and LB_FW <= min_|S|=k V0(S) <= min_|S|=k V(S).  V is the exact posterior variance (dense GP, closed form).
Output: results/A6_design_bracket_check.json
"""
import itertools
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from covfourier.engine import CovarianceFourier                     # noqa: E402
from covfourier.batch import BatchProjector                         # noqa: E402
from covfourier.design import InfoDesign, exact_variances           # noqa: E402
import app2_noisy_reconstruction_v2 as V2                           # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')


def exact_model(H, c, s, cn, sn, mabs):
    """Exact covariances: K (m x m) of the sensor Wiener integrals, kt (m x q), var_t (q,) for the targets
    [B(1), int B, B(1/2), window(cn[0]), window(cn[1])] (rows of V2.gp_covariances, re-ordered)."""
    m = len(c)
    cc = np.r_[c, cn]; ss = np.r_[s, sn]
    K, Kt, vt = V2.gp_covariances(H, cc, ss, m, mabs)
    # rows of Kt: B(1), int B, B(.25), B(.5), B(.75), new windows...
    pick = [0, 1, 3, 5, 6]
    return K, Kt[pick].T.copy(), vt[pick].copy()


def coords(H, n, c, s, cn, sn, Naux):
    cf = CovarianceFourier(H, n, enrich=True, Naux=Naux)
    bp = BatchProjector(cf)
    Z, E2 = V2.cf_coords(bp, c, s)
    t = (np.arange(bp.N) + 0.5) / bp.N
    F = [np.ones_like(t), 1.0 - t, (t < 0.5).astype(float)]
    F = np.vstack([np.array(F), V2.window_integrands_values(cn, sn, t)])
    a_g, cc = bp.fourier_stage(F)
    r = bp.project(a_g, cc, norms=True)
    return Z, E2, r['coords'].T, np.clip(r['error2'], 0.0, None) + r['eta']


def main(m=30, kmax=3, seed=7, tau=0.01, Hs=(0.55, 0.75, 0.95), ns=(4, 8, 16, 32, 64, 128, 256)):
    rng = np.random.default_rng(seed)
    out = {'config': {'m': m, 'kmax': kmax, 'seed': seed, 'tau': tau, 'Hs': list(Hs), 'ns': list(ns),
                      'nuisance': 'beta=(offset, slope*(c-1/2)), Sigma_beta=diag(0.5^2, 1.0^2)'},
           'cases': []}
    for H in Hs:
        mabs = V2.FastAbsMoment(2 * H)
        c, s = V2.draw_windows(rng, m)
        cn, sn = V2.draw_windows(rng, 2)
        K, kt, vt = exact_model(H, c, s, cn, sn, mabs)
        Hn = np.stack([np.ones(m), c - 0.5], axis=1)
        Sb = np.diag([0.25, 1.0])
        q = len(vt)
        omega = 1.0 / vt
        sig2 = np.full(m, tau ** 2)
        subsets = [list(S) for k in range(1, kmax + 1) for S in itertools.combinations(range(m), k)]
        Vex = np.array([exact_variances(K[np.ix_(S, S)], kt[S], vt, sig2[S], Hn[S], Sb) for S in subsets])
        prev_V0 = None
        for n in ns:
            t0 = time.perf_counter()
            Z, E2, Zt, E2t = coords(H, n, c, s, cn, sn, Naux=max(1 << 14, 64 * n))
            D = Z.shape[1]
            A = np.hstack([Z, Hn])
            C = np.vstack([Zt.T, np.zeros((2, q))])
            Pi = np.zeros((D + 2, D + 2)); Pi[:D, :D] = np.eye(D); Pi[D:, D:] = np.linalg.inv(Sb)
            des = InfoDesign(A, sig2, C, omega=omega, Pi=Pi, e2=E2, e2_t=E2t)
            V0 = np.empty_like(Vex); Up = np.empty_like(Vex); Uu = np.empty_like(Vex)
            for r_, S in enumerate(subsets):
                b = des.bracket(S)
                V0[r_], Up[r_], Uu[r_] = b['V0'], b['U_post'], b['U_unif']
            rel = 1e-10                                                  # rounding of the dense exact GP itself
            lo_ok = bool(np.all(V0 <= Vex * (1 + rel) + 1e-15))
            up_ok = bool(np.all(Vex <= Up * (1 + rel) + 1e-15))
            ord_ok = bool(np.all(Up <= Uu * (1 + 1e-12) + 1e-15))
            mono = None if prev_V0 is None else bool(np.all(prev_V0 <= V0 * (1 + 1e-9) + 1e-15))
            prev_V0 = V0
            # Frank-Wolfe lower bound vs brute force (weighted objective) for k = kmax
            idx = [r_ for r_, S in enumerate(subsets) if len(S) == kmax]
            phi0 = V0[idx] @ omega; phiex = Vex[idx] @ omega
            fw = des.frank_wolfe(kmax, iters=400)
            g = des.greedy(kmax)
            case = {'H': H, 'n': n, 'D': D, 'rho_all_max': float(np.sum(E2 / sig2)),
                    'e2_max': float(E2.max()), 'e2t_max': float(E2t.max()),
                    'lower_holds': lo_ok, 'upper_holds': up_ok, 'U_post_le_U_unif': ord_ok, 'V0_monotone_in_D': mono,
                    'max_rel_width_post': float(np.max((Up - V0) / Vex)),
                    'median_rel_width_post': float(np.median((Up - V0) / Vex)),
                    'max_rel_gap_V0': float(np.max((Vex - V0) / Vex)),
                    'min_phi0_bruteforce': float(phi0.min()), 'min_phi_exact_bruteforce': float(phiex.min()),
                    'FW_LB': fw['LB'], 'FW_relaxed': fw['Phi0_relaxed'], 'FW_iterations': fw['iterations'],
                    'FW_LB_le_bruteforce': bool(fw['LB'] <= phi0.min() * (1 + 1e-12)),
                    'greedy_phi0': float(g['Phi0_history'][-1]),
                    'greedy_exact_phi': float(phiex[[r_ for r_, i in enumerate(idx) if sorted(subsets[i]) == sorted(g['S'])][0]]) if len(g['S']) == kmax else None,
                    'n_designs': len(subsets), 'seconds': time.perf_counter() - t0}
            out['cases'].append(case)
            print(json.dumps({k_: case[k_] for k_ in ('H', 'D', 'lower_holds', 'upper_holds', 'V0_monotone_in_D',
                                                      'max_rel_width_post', 'FW_LB_le_bruteforce')}), flush=True)
    out['all_hold'] = all(c_['lower_holds'] and c_['upper_holds'] and c_['U_post_le_U_unif'] and
                          c_['FW_LB_le_bruteforce'] and (c_['V0_monotone_in_D'] in (None, True)) for c_ in out['cases'])
    with open(os.path.join(RES, 'A6_design_bracket_check.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('all_hold', out['all_hold'])


if __name__ == '__main__':
    main()
