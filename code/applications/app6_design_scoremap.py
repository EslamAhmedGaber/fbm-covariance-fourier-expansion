"""A6.2b: value of a single reading over the (depth, time) candidate grid of A6.2 (for Figure 9b).

score_i = sum_t omega_t (a_i^T L c_t)^2 / (sigma_i^2 + a_i^T L a_i), L the posterior covariance after the 1985 log:
the reduction of Phi achieved by reading i alone (exact identity in the D-dimensional model).
Same configuration as app6_borehole_design.py; n = 256.  Output: results/A6_design_scoremap.json
"""
import json
import os
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import borehole_problem as BP                            # noqa: E402
import app6_borehole_design as E2                        # noqa: E402

RES = E2.RES


def main(n=256):
    zc, tc = BP.candidate_grid(dz=1.0, dt=0.25)
    prep = BP.Prepared(E2.H if hasattr(E2, 'H') else 0.75, n)
    P = BP.design_problem(prep, E2.Z_EXIST, E2.T_EXIST, zc, tc, sigma_T=E2.SIGMA_T, noise_sd=E2.NOISE, sb=E2.SB)
    cand = P['cand']
    L = cho_solve(cho_factor(cand.Pi, lower=True), np.eye(cand.d))
    AL = cand.A @ L
    u = np.einsum('ij,ij->i', AL, cand.A)
    W = AL @ cand.C
    score = (W ** 2 @ cand.omega) / (cand.s2 + u)
    zs = np.unique(zc); ts = np.unique(tc)
    out = {'z': zs.tolist(), 't': ts.tolist(), 'score_first': score.reshape(len(zs), len(ts)).ravel().tolist(),
           'per_target_first': {str(t): ((W[:, t] ** 2) / (cand.s2 + u)).reshape(len(zs), len(ts)).max(axis=1).tolist() for t in range(W.shape[1])},
           'best': {'z': float(zc[np.argmax(score)]), 't': float(tc[np.argmax(score)]), 'score': float(score.max())}}
    with open(os.path.join(RES, 'A6_design_scoremap.json'), 'w') as fh:
        json.dump(out, fh)
    print('best single reading', out['best'])


if __name__ == '__main__':
    main()
