"""A6.3: cost of certified design against the number of candidates m (one configuration per process).

Expansion: one basis (H = 0.75, n = 256), coordinates of the 1985 log, the m candidates and the three targets,
greedy k = 20, Frank-Wolfe (100 iterations), bracket.  Dense alternatives with the A1 competitor (two-triangle
Gauss-Jacobi product quadrature, n_q nodes, 1e-9 accuracy on these kernels at n_q = 1600; targets ~1e-6):
  dense greedy  - quadrature factors (q x n_q) for all candidates, then one exact column per greedy step;
  dense relaxation - one Frank-Wolfe step needs a Cholesky factorization of the m x m data covariance.
Peak working set from psutil (Windows peak_wset).  Usage:
  python app6_design_scaling.py expansion <dz> <dt>
  python app6_design_scaling.py dense <dz> <dt> <n_q>
Appends one JSON line to results/A6_design_scaling.jsonl.
"""
import json
import math
import os
import sys
import time

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from covfourier import heat                              # noqa: E402
from covfourier.design import DenseGreedy                # noqa: E402
from scipy.special import roots_jacobi                   # noqa: E402
import borehole_problem as BP                            # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
H, N_BASIS, K, SIGMA_T, NOISE, SB = 0.75, 256, 20, 1.0, 0.05, (10.0, 0.05)
Z_EXIST, T_EXIST = np.arange(20.0, 600.0 + 1e-9, 10.0), 1985.0


def peak_gib():
    mi = psutil.Process().memory_info()
    return getattr(mi, 'peak_wset', mi.rss) / 2 ** 30


def run_expansion(dz, dt):
    zc, tc = BP.candidate_grid(dz=dz, dt=dt)
    t0 = time.perf_counter()
    prep = BP.Prepared(H, N_BASIS)
    P = BP.design_problem(prep, Z_EXIST, T_EXIST, zc, tc, sigma_T=SIGMA_T, noise_sd=NOISE, sb=SB)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter(); g = P['cand'].greedy(K); t_g = time.perf_counter() - t0
    t0 = time.perf_counter()
    w0 = np.zeros(P['cand'].m); w0[g['S']] = 1.0
    fw = P['cand'].frank_wolfe(K, iters=100, w0=w0); t_fw = time.perf_counter() - t0
    b = P['full'].bracket(np.r_[np.arange(P['n_exist']), P['n_exist'] + np.array(g['S'])])
    return {'method': 'expansion', 'm': int(len(zc)), 'dz': dz, 'dt': dt, 'n': N_BASIS, 'build_s': t_build,
            'timing': P['timing'], 'greedy_s': t_g, 'fw_100_s': t_fw, 'Phi_D': b['Phi0'], 'Phi_U': b['Phi_U_post'],
            'LB': fw['LB'], 'peak_gib': peak_gib(), 'S': [int(i) for i in g['S']]}


class ProductQuadOracle:
    """Two-triangle Gauss-Jacobi factors of the exact covariance (comparators.triangle_gj_covariance, factored):
    <f,g> = H b int_0^1 x^b int_0^1 (1-y)^(b-1) [f(x) g(xy) + f(xy) g(x)] dy dx = Hb (A_w Bm^T + Bm A_w^T)."""

    def __init__(self, values, q, n, chunk=64):
        b = 2 * H - 1
        x, wx = roots_jacobi(n, 0.0, b); x = (x + 1) / 2; wx = wx / 2 ** (b + 1)
        if b >= 0.5:
            y, wv = roots_jacobi(n, b - 1.0, 0.0); y = (y + 1) / 2; wv = wv / 2 ** b
        else:
            v, wv = np.polynomial.legendre.leggauss(n); v = (v + 1) / 2; wv = wv / 2 / b; y = 1.0 - v ** (1.0 / b)
        self.A = np.empty((q, n)); self.Bm = np.empty((q, n))
        for i0 in range(0, q, chunk):
            idx = slice(i0, min(q, i0 + chunk))
            self.A[idx] = values(idx, x)
            XY = np.outer(x, y)
            self.Bm[idx] = np.stack([values(idx, XY[r]) @ wv for r in range(n)], axis=1)
        self.Aw = self.A * wx[None, :]
        self.c = H * b

    def col(self, j):
        return self.c * (self.Aw @ self.Bm[j] + self.Bm @ self.Aw[j])

    def diag(self):
        return 2 * self.c * np.einsum('qn,qn->q', self.Aw, self.Bm)


def run_dense(dz, dt, n_q):
    zc, tc = BP.candidate_grid(dz=dz, dt=dt)
    win = heat.Window()
    z_all = np.r_[Z_EXIST, zc]; t_all = np.r_[np.full(len(Z_EXIST), T_EXIST), tc]
    ne, m = len(Z_EXIST), len(zc)
    q = ne + m + 3
    def values(idx, u):
        rows = np.arange(q)[idx]
        out = np.empty((len(rows), len(np.atleast_1d(u))))
        rb = rows[rows < ne + m]
        if len(rb):
            out[:len(rb)] = heat.borehole_values(z_all[rb], t_all[rb], u, win)
        rt = rows[rows >= ne + m] - (ne + m)
        if len(rt):
            out[len(rb):] = BP.target_values(np.atleast_1d(u), win)[rt]
        return out
    t0 = time.perf_counter()
    orc = ProductQuadOracle(values, q, n_q)
    t_fact = time.perf_counter() - t0
    dg = orc.diag() * SIGMA_T ** 2
    tgt = np.arange(ne + m, q)
    kt = np.stack([orc.col(j) for j in tgt], axis=1) * SIGMA_T ** 2       # (q, 3)
    var_t = dg[tgt]
    Hn = np.stack([np.ones(ne + m), z_all], axis=1)
    sel = np.arange(ne + m)
    # weights: 1 / variance after the 1985 log (exact model), as in the expansion run
    dgr = DenseGreedy(lambda j: orc.col(j)[sel] * SIGMA_T ** 2, dg[sel], kt[sel], var_t, NOISE ** 2,
                      H=Hn, Sigma_beta=np.diag(np.array(SB) ** 2))
    r0 = dgr.run(0, forced=list(range(ne)))
    dgr.omega = 1.0 / r0['target_var']
    t0 = time.perf_counter()
    r = dgr.run(K, forced=list(range(ne)))
    t_g = time.perf_counter() - t0
    # one dense relaxation step: Cholesky of the (ne + m) x (ne + m) data covariance (measured only if it fits)
    t_chol = None
    if (ne + m) <= 6000:
        t0 = time.perf_counter()
        Kd = np.empty((ne + m, ne + m))
        for j in range(ne + m):
            Kd[:, j] = orc.col(j)[sel]
        t_assemble = time.perf_counter() - t0
        Kd = Kd * SIGMA_T ** 2 + Hn @ np.diag(np.array(SB) ** 2) @ Hn.T
        Kd[np.diag_indices_from(Kd)] += NOISE ** 2
        t0 = time.perf_counter(); np.linalg.cholesky(Kd); t_chol = time.perf_counter() - t0
        del Kd
    else:
        t_assemble = None
    S = [j - ne for j in r['S']]
    return {'method': 'dense_product_quadrature', 'm': int(m), 'dz': dz, 'dt': dt, 'n_q': n_q, 'factor_s': t_fact,
            'greedy_s': t_g, 'assemble_full_s': t_assemble, 'cholesky_s': t_chol, 'Phi_exact_greedy': r['Phi_history'][-1] * 1.0,
            'peak_gib': peak_gib(), 'S': [int(i) for i in S]}


if __name__ == '__main__':
    kind, dz, dt = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    res = run_expansion(dz, dt) if kind == 'expansion' else run_dense(dz, dt, int(sys.argv[4]))
    with open(os.path.join(RES, 'A6_design_scaling.jsonl'), 'a') as fh:
        fh.write(json.dumps(res) + '\n')
    print(json.dumps({k: v for k, v in res.items() if k != 'S'}))
