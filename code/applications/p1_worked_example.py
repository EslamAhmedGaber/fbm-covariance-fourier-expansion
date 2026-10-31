"""P1: worked example of the CovFourier procedure (Algorithm 1, Section 3 of the paper).

One small inference problem taken through the six steps, with observations of different regularity and one target,
all Wiener integrals of the same fBm path (H = 0.75, T = 1):

  step 1, express the problem through integrands
    y1 = B_H(0.3) + e                      f1 = 1_[0, 0.3)                          jump
    y2 = Gaussian-window average + e       f2 = Phi((0.6 - s)/0.02) - Phi(-0.4/0.02)  smooth, steep at 0.6
    y3 = fOU response X(1) + e             f3 = exp(-5 (1 - s))                     smooth, endpoint slopes
    y4 = heat-conduction response + e      f4 = erfc(0.3 / (2 sqrt(1 - s)))          physical kernel
    u  = mean of B_H over [0.8, 1]         g  = min(1, (1 - s)/0.2)                 kink at 0.8
  step 2, specify H, noise sd 0.05 and the accuracy request (relative width of the bracket for Var(u | y));
  step 3, prepare the basis once per (H, T, D, N);
  step 4, coordinates of all five integrands through the common coefficient rule (one call, from_callables);
  step 5, assess: residual energies, resolved flags, change under refinement of N; refine D if needed;
  step 6, use: posterior variance of u given y in D dimensions with the exact-model bracket of Proposition 6,
          against the exact value from the independent quadrature reference (covfourier.khref).
Output: results/P1_worked_example.json
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.special import erfc, ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from covfourier.engine import CovarianceFourier          # noqa: E402
from covfourier.batch import BatchProjector              # noqa: E402
from covfourier.khref import KHQuad                      # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
H, T, SIGMA = 0.75, 1.0, 0.05
NS = (16, 32, 64, 128, 256, 512)
N_AUX = 1 << 15
TOL = 1e-3                                                   # requested relative bracket width for Var(u | y)


def f1(s):
    return (np.asarray(s) < 0.3).astype(float)


def f2(s):
    return ndtr((0.6 - np.asarray(s)) / 0.02) - ndtr((0.6 - 1.0) / 0.02)


def f3(s):
    return np.exp(-5.0 * (1.0 - np.asarray(s)))


def f4(s):
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)
    m = s < 1.0
    out[m] = erfc(0.3 / (2.0 * np.sqrt(1.0 - s[m])))
    return out


def g(s):
    return np.minimum(1.0, (1.0 - np.asarray(s)) / 0.2)


INPUTS = [('y1', 'point value B_H(0.3)', 'jump', f1), ('y2', 'Gaussian-window average', 'steep, smooth', f2),
          ('y3', 'fractional OU response X(1)', 'smooth, endpoint slopes', f3),
          ('y4', 'heat-conduction response', 'physical kernel', f4), ('u', 'mean of B_H over [0.8, 1]', 'kink', g)]


def posterior(Gm, e, sigma=SIGMA):
    """Var(u | y) in D dimensions and the exact-model bracket of Proposition 6 (rows of Gm: y1..y4, u)."""
    Gy, gu = Gm[:4], Gm[4]
    A = Gy @ Gy.T + sigma ** 2 * np.eye(4)
    k = Gy @ gu
    a = np.linalg.solve(A, k)                                # posterior-mean weights a* of the projected model
    VD = float(gu @ gu - k @ a)
    ey, eu = e[:4], e[4]
    Vp = VD + (eu + float(np.abs(a) @ ey)) ** 2
    rho = float(np.sum(ey ** 2) / sigma ** 2)
    Vpp = ((1 + math.sqrt(rho)) * math.sqrt(VD) + eu) ** 2
    return {'V_D': VD, 'V_plus': Vp, 'V_plusplus': Vpp, 'rho': rho, 'weights': a.tolist(),
            'rel_width': (Vp - VD) / VD}


def exact_reference(h_max):
    vals = lambda v: np.vstack([f(v) for _, _, _, f in INPUTS])
    t0 = time.perf_counter()
    K = KHQuad(H, features=(0.3, 0.6, 0.8), h_max=h_max).gram(vals)
    return K, time.perf_counter() - t0


def main():
    out = {'config': {'H': H, 'T': T, 'noise_sd': SIGMA, 'N_aux': N_AUX, 'ns': NS, 'tol_rel_width': TOL,
                      'inputs': [{'name': a, 'what': b, 'type': c} for a, b, c, _ in INPUTS]}, 'by_D': []}
    # exact covariances (independent quadrature; two resolutions as a self-check)
    K1, t1 = exact_reference(1 / 256)
    K2, t2 = exact_reference(1 / 512)
    K = K2
    ref_change = float(np.max(np.abs(K1 - K2) / np.sqrt(np.outer(np.diag(K2), np.diag(K2)))))
    Ay = K[:4, :4] + SIGMA ** 2 * np.eye(4)
    Vex = float(K[4, 4] - K[4, :4] @ np.linalg.solve(Ay, K[:4, 4]))
    out['exact'] = {'K': K.tolist(), 'V_exact': Vex, 'norm2': np.diag(K).tolist(), 'self_check_rel': ref_change,
                    'seconds': t1 + t2}
    print('exact Var(u|y) = %.10f (quadrature self-check %.1e)' % (Vex, ref_change), flush=True)
    fs = [f for _, _, _, f in INPUTS]
    for n in NS:
        t0 = time.perf_counter()
        cf = CovarianceFourier(H, n, T=T, Naux=N_AUX)
        bp = BatchProjector(cf, N=N_AUX)
        t_prep = time.perf_counter() - t0
        t0 = time.perf_counter()
        r = bp.from_callables(fs)
        t_in = time.perf_counter() - t0
        # refinement of the Fourier stage: the same inputs at half the auxiliary length
        bph = BatchProjector(cf, N=N_AUX // 2)
        rh = bph.from_callables(fs)
        Gm = r['coords'].T                                   # (5, D)
        e2 = np.maximum(r['error2'], 0.0)
        post = posterior(Gm, np.sqrt(e2))
        cov_err = float(np.max(np.abs(Gm @ Gm.T - K) / np.sqrt(np.outer(np.diag(K), np.diag(K)))))
        entry = {'n': n, 'D': cf.D, 'prep_s': t_prep, 'per_input_s': t_in / len(fs),
                 'norm2': r['norm2'].tolist(), 'error2': r['error2'].tolist(), 'eta': r['eta'].tolist(),
                 'resolved': [bool(x) for x in r['resolved']], 'rel_error2': (r['error2'] / r['norm2']).tolist(),
                 'refinement_change_error2': np.abs(rh['error2'] - r['error2']).tolist(),
                 'refinement_change_coords': float(np.max(np.abs(rh['coords'] - r['coords']))),
                 'corrected': [bool(x) for x in bp.last_corrected], 'cov_max_rel_err': cov_err,
                 'posterior': post, 'V_exact_inside': bool(post['V_D'] <= Vex * (1 + 1e-12) and Vex <= post['V_plus'] * (1 + 1e-12)),
                 'meets_tol': bool(post['rel_width'] <= TOL)}
        out['by_D'].append(entry)
        print('D=%4d prep %.2fs per input %.3fs | rel e^2: %s | V_D %.8f V+ %.8f exact %.8f inside %s width %.1e%s' % (
            cf.D, t_prep, entry['per_input_s'], ' '.join('%.1e' % x for x in entry['rel_error2']),
            post['V_D'], post['V_plus'], Vex, entry['V_exact_inside'], post['rel_width'],
            '  <- meets tolerance' if entry['meets_tol'] else ''), flush=True)
    # observed decay orders of the residual energy (least squares in log-log over D >= 66)
    Ds = np.array([e['D'] for e in out['by_D']], float)
    sel = Ds >= 60
    out['observed_order_e2'] = {}
    for j, (name, _, _, _) in enumerate(INPUTS):
        y = np.array([e['rel_error2'][j] for e in out['by_D']])
        ok = sel & (y > 1e-26)
        out['observed_order_e2'][name] = float(np.polyfit(np.log(Ds[ok]), np.log(y[ok]), 1)[0]) if ok.sum() >= 2 else None
    print('observed orders of e_D^2:', {k: round(v, 2) if v is not None else None for k, v in out['observed_order_e2'].items()})
    with open(os.path.join(RES, 'P1_worked_example.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
