"""Tests for covfourier.design and covfourier.khref (run: python test_design.py, or pytest).

Each test checks a mathematical fact or an independent closed form, never a stored output.
"""
import itertools
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from covfourier.design import InfoDesign, DenseGreedy, exact_variances    # noqa: E402
from covfourier.khref import KHQuad                                       # noqa: E402
from covfourier.gram import poly_poly                                     # noqa: E402


def _toy(seed=0, m=14, D=5, M=7, q=2, p=1, resid=0.3):
    """Exact model in R^D (+) R^M: F_i = (gamma_i, r_i), f_t = (g_t, r_t); truncation keeps the first block."""
    rng = np.random.default_rng(seed)
    Gm = rng.standard_normal((m, D)); Rm = resid * rng.standard_normal((m, M))
    Gt = rng.standard_normal((q, D)); Rt = resid * rng.standard_normal((q, M))
    Hn = rng.standard_normal((m, p)); Sb = np.eye(p) * 0.7
    s2 = rng.uniform(0.05, 0.3, m)
    K = Gm @ Gm.T + Rm @ Rm.T
    kt = Gm @ Gt.T + Rm @ Rt.T
    vt = np.sum(Gt ** 2, 1) + np.sum(Rt ** 2, 1)
    A = np.hstack([Gm, Hn]); C = np.vstack([Gt.T, np.zeros((p, q))])
    Pi = np.eye(D + p); Pi[D:, D:] = np.linalg.inv(Sb)
    des = InfoDesign(A, s2, C, Pi=Pi, e2=np.sum(Rm ** 2, 1), e2_t=np.sum(Rt ** 2, 1))
    return des, K, kt, vt, s2, Hn, Sb


def test_bracket_all_subsets():
    """V0 <= V <= U_post <= U_unif for every design of size <= 3 (exact model with arbitrary residual Gram)."""
    for seed in range(4):
        des, K, kt, vt, s2, Hn, Sb = _toy(seed)
        for k in (1, 2, 3):
            for S in itertools.combinations(range(des.m), k):
                S = list(S)
                V = exact_variances(K[np.ix_(S, S)], kt[S], vt, s2[S], Hn[S], Sb)
                b = des.bracket(S)
                assert np.all(b['V0'] <= V * (1 + 1e-12)), (seed, S)
                assert np.all(V <= b['U_post'] * (1 + 1e-12)), (seed, S)
                assert np.all(b['U_post'] <= b['U_unif'] * (1 + 1e-12)), (seed, S)


def test_frank_wolfe_lower_bound_and_greedy():
    """LB_FW <= min over all k-subsets of Phi0; greedy value equals the recomputed objective of its design."""
    des, *_ = _toy(5, m=12)
    k = 3
    phi = [float(des.omega @ des.V0(list(S))) for S in itertools.combinations(range(des.m), k)]
    fw = des.frank_wolfe(k, iters=300)
    assert fw['LB'] <= min(phi) * (1 + 1e-12)
    g = des.greedy(k)
    assert abs(g['Phi0_history'][-1] - float(des.omega @ des.V0(g['S']))) < 1e-10 * g['Phi0_history'][0]
    assert g['Phi0_history'][-1] >= min(phi) * (1 - 1e-12)


def test_dense_greedy_matches_information_form():
    """With zero residuals the dense pivoted-Cholesky greedy and the information-form greedy coincide."""
    des, K, kt, vt, s2, Hn, Sb = _toy(3, resid=0.0)
    D = des.d - Hn.shape[1]
    G = des.A[:, :D]
    dg = DenseGreedy(lambda j: K[:, j], np.diag(K), kt, vt, s2, omega=des.omega, H=Hn, Sigma_beta=Sb)
    r1 = dg.run(5)
    r2 = des.greedy(5)
    assert r1['S'] == r2['S'], (r1['S'], r2['S'])
    assert np.allclose(r1['Phi_history'], r2['Phi0_history'], rtol=1e-9)
    # forced existing data: conditioning order does not change the final variances
    r3 = dg.run(0, forced=r2['S'])
    assert np.allclose(r3['target_var'], des.V0(r2['S']), rtol=1e-9)


def test_khref_closed_forms():
    """Independent quadrature: indicators (closed form) and polynomials (Beta-function closed form)."""
    for H in (0.55, 0.8):
        a, b = 0.3, 0.71
        Q = KHQuad(H, features=[a, b], h_max=1 / 32)
        M = Q.gram(lambda v: np.vstack([(v < a).astype(float), (v < b).astype(float)]))
        c = 0.5 * (a ** (2 * H) + b ** (2 * H) - abs(a - b) ** (2 * H))
        ex = np.array([[a ** (2 * H), c], [c, b ** (2 * H)]])
        assert np.max(np.abs(M - ex) / ex) < 1e-9, (H, M, ex)
        Q2 = KHQuad(H, h_max=1 / 32)
        M2 = Q2.gram(lambda v: np.vstack([np.ones_like(v), v ** 2]))
        ex2 = np.array([[poly_poly(H, i, j) for j in (0, 2)] for i in (0, 2)])
        assert np.max(np.abs(M2 - ex2) / ex2) < 1e-9, (H, M2, ex2)


def test_block_design_matches_full_model():
    """Independent blocks: BlockInfoDesign equals InfoDesign on the block-diagonal model (values, greedy, FW bound)."""
    from covfourier.design import BlockInfoDesign
    from scipy.linalg import block_diag
    rng = np.random.default_rng(9)
    nb, d, per = 3, 4, 5
    A = rng.standard_normal((nb * per, d)); blk = np.repeat(np.arange(nb), per)
    P = [np.eye(d) * (1 + s) for s in range(nb)]; C = [rng.standard_normal(d) for _ in range(nb)]
    s2 = rng.uniform(0.1, 0.5, nb * per)
    bd = BlockInfoDesign(A, blk, s2, P, C)
    Af = np.zeros((nb * per, nb * d))
    for i in range(nb * per):
        Af[i, blk[i] * d:(blk[i] + 1) * d] = A[i]
    full = InfoDesign(Af, s2, np.concatenate(C)[:, None], Pi=block_diag(*P))
    for S in ([0, 6, 11], [1, 2, 3], [14]):
        assert abs(bd.value(S)['Phi0'] - float(full.V0(S)[0])) < 1e-12
    assert bd.greedy(4)['S'] == full.greedy(4)['S']
    k = 3
    best = min(bd.value(list(S))['Phi0'] for S in itertools.combinations(range(nb * per), k))
    assert bd.frank_wolfe(k, iters=200)['LB'] <= best * (1 + 1e-12)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t(); print('PASS', t.__name__)
        except AssertionError as e:
            failed += 1; print('FAIL', t.__name__, e)
    print(f'{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
