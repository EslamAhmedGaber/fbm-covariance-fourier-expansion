"""Regression tests for covfourier (run: python test_covfourier.py, or pytest).

Each test checks a mathematical fact or an independent reference, never a stored output of the same code.
Total runtime about 1 minute on a laptop.
"""
import math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE); sys.path.insert(0, os.path.join(CODE, 'applications'))

from covfourier.gram import CosineGram, power_cosine_moments, poly_poly      # noqa: E402
from covfourier.engine import CovarianceFourier, midpoint_cosine_coefficients  # noqa: E402
from covfourier.batch import BatchProjector                                  # noqa: E402
from covfourier.comparators import ExpFamily, triangle_gj_covariance, DaviesHarte  # noqa: E402


def cH(H):
    return math.gamma(2 * H + 1) * math.sin(math.pi * H)


def test_weighted_gram_two_sided():
    """Spectrum of W^{-1/2} G_+ W^{-1/2} lies in [1/2, 1] (two-sided weighted cosine bounds, companion paper)."""
    for H in (0.55, 0.75, 0.95):
        G = CosineGram(H, 1.0).dense(257)[1:, 1:]
        w = cH(H) * (np.pi * np.arange(1, 257)) ** (1 - 2 * H)
        ev = np.linalg.eigvalsh(G / np.sqrt(np.outer(w, w)))
        assert ev.max() <= 1 + 1e-10 and ev.min() >= 0.5 - 1e-10, (H, ev.min(), ev.max())


def test_full_gram_lower_bound():
    """G - (1/2) diag(0, w) is positive semidefinite including the h_0 row."""
    for H in (0.6, 0.9):
        G = CosineGram(H, 1.0).dense(200)
        w = np.r_[0.0, cH(H) * (np.pi * np.arange(1, 200)) ** (1 - 2 * H)]
        assert np.linalg.eigvalsh(G - 0.5 * np.diag(w)).min() > -1e-12


def test_batch_matches_engine():
    """Batched endpoint-corrected projector and plain engine agree for a smooth input (large N)."""
    H = 0.75
    cf = CovarianceFourier(H, 64, enrich=True, Naux=1 << 14)
    f = lambda t: np.exp(np.asarray(t))
    r1 = cf.project(midpoint_cosine_coefficients(f, 1 << 14, 1.0))
    r2 = BatchProjector(cf).from_callables([f])
    assert np.max(np.abs(r1['coords'] - r2['coords'][:, 0])) < 1e-7
    assert r2['norm2'][0] > 0 and r2['error2'][0] < 1e-12 * r2['norm2'][0]          # e^t: D = 66 is resolved


def test_exact_polynomial_input():
    """f = t is represented exactly by the endpoint correction: norm^2 equals the closed form."""
    H = 0.65
    cf = CovarianceFourier(H, 32, enrich=True, Naux=1 << 12)
    r = BatchProjector(cf).from_callables([lambda t: np.asarray(t, float)])
    assert abs(r['norm2'][0] - poly_poly(H, 1, 1)) < 1e-10


def test_two_sided_error_bracket_t():
    """(1/2) tail <= E_J(t)^2 <= tail with tail = sum_{k>=J} w_k a_k^2 (classical coefficients of t)."""
    H, N = 0.75, 1 << 15
    a = power_cosine_moments(1.0, N, 1.0)
    w = np.r_[1.0, cH(H) * (np.pi * np.arange(1, N)) ** (1 - 2 * H)]
    cf = CovarianceFourier(H, 16, enrich=False, Naux=N)
    E2 = cf.project(a)['error2']
    tail = float(np.sum(w[16:] * a[16:] ** 2))
    assert 0.5 * tail * (1 - 1e-6) <= E2 <= tail * (1 + 1e-6), (E2, tail)


def test_kl_brownian_and_bounds():
    """KL Galerkin: exact Brownian eigenvalues; explicit eigenvalue bounds for H = 0.75."""
    import app4_kl_spectrum as A4
    mu = np.sort(np.linalg.eigvalsh(A4.galerkin_matrix(0.5, 512)))[::-1]
    n = np.arange(1, 21)
    assert np.max(np.abs(mu[:20] * ((n - 0.5) * np.pi) ** 2 - 1)) < 1e-6
    H = 0.75
    mu = np.sort(np.linalg.eigvalsh(A4.galerkin_matrix(H, 512)))[::-1]
    n = np.arange(1, 101)
    assert np.all(mu[:100] >= 0.5 * cH(H) * (n * np.pi) ** (-1 - 2 * H))
    assert np.all(mu[1:100] <= cH(H) * ((n[1:] - 1) * np.pi) ** (-1 - 2 * H))


def test_m_abs():
    import app2_noisy_reconstruction as A2
    assert A2.check_m_abs(0.75) < 1e-12


def test_triangle_vs_exact():
    fam = ExpFamily(np.array([2 - 5j, 0.5 - 1j, 3.0 + 0j, 1.0 + 0j]), np.array([1j, 1.0, 1.0, 0.5]), s0=1.0)
    for H in (0.55, 0.85):
        K, _ = triangle_gj_covariance(fam.values, H, 128)
        Kex = fam.exact_covariance(H, dps=30)
        assert np.linalg.norm(K - Kex) / np.linalg.norm(Kex) < 1e-10, H


def test_davies_harte_variance():
    """Var B_H(1) = 1: sample estimate within 4 standard errors (fixed seed)."""
    dh = DaviesHarte(0.7, 256)
    X = dh.increments(20000, np.random.default_rng(3))
    v = np.var(X.sum(axis=1))
    assert abs(v - 1.0) < 4 * math.sqrt(2 / 20000)


def test_pcg_safe_zero_and_status():
    from covfourier.solvers import pcg_safe
    x, info = pcg_safe(lambda v: v, np.zeros(7), np.ones(7))
    assert info['status'] == 'zero_rhs' and info['iterations'] == 0 and not np.any(x)
    x, info = pcg_safe(lambda v: 2 * v, np.ones(7), np.ones(7))
    assert info['status'] == 'converged' and info['true_rel_residual'] < 1e-13
    from covfourier.engine import pcg
    x, it = pcg(lambda v: v, np.zeros(4), np.ones(4))
    assert it == 0 and not np.any(x)


def test_schur_spectrum_and_white():
    from covfourier.solvers import SchurCosine
    for H in (0.6, 0.9):
        G = CosineGram(H, 1.0); n = 300
        sc = SchurCosine(G, n); Gd = G.dense(n)
        Gp = Gd[1:, 1:] - np.outer(Gd[1:, 0], Gd[0, 1:]) / Gd[0, 0]
        ev = np.linalg.eigvalsh(Gp / np.sqrt(np.outer(sc.w, sc.w)))
        assert ev.min() >= 0.5 - 1e-12 and ev.max() <= 1 + 1e-12, (ev.min(), ev.max())
        B = Gd @ np.random.default_rng(0).standard_normal((n, 3))
        Z = sc.white(B)
        ref = B.T @ np.linalg.solve(Gd, B)
        assert np.max(np.abs(Z.T @ Z - ref)) < 1e-12 * np.max(np.abs(ref))
        x, info = sc.solve(B[:, 0])
        assert info['status'] == 'converged' and info['iterations'] <= 20
        assert np.max(np.abs(x - np.linalg.solve(Gd, B[:, 0]))) < 1e-10 * np.max(np.abs(x))


def test_large_matches_batch():
    from covfourier.large import LargeProjector
    H, n, N = 0.75, 128, 4096
    fs = [lambda t: np.exp(np.asarray(t)), lambda t: np.sin(40 * np.asarray(t)) + np.asarray(t) ** 2]
    lp = LargeProjector(H, n, N).from_callables(fs)
    cf = CovarianceFourier(H, n, enrich=True, Naux=N)
    bp = BatchProjector(cf).from_callables(fs)
    K1 = lp['coords'].T @ lp['coords']; K2 = bp['coords'].T @ bp['coords']
    assert np.max(np.abs(K1 - K2)) < 1e-11 * np.max(np.abs(K2))
    assert np.max(np.abs(lp['norm2'] - bp['norm2'])) < 1e-12 * np.max(lp['norm2'])


def test_certified_bracket_exp():
    """Declared C2 for e^t on [0,1]: |a_k| <= sqrt2 (|f'(0)|+|f'(1)|+V(f'))/omega_k^2 = sqrt2*2e/omega_k^2."""
    from covfourier.large import certified_bracket
    from covfourier.engine import midpoint_cosine_coefficients
    H, J, N = 0.75, 16, 4096
    a_hat = midpoint_cosine_coefficients(lambda t: np.exp(np.asarray(t)), N, 1.0)
    a_ex = midpoint_cosine_coefficients(lambda t: np.exp(np.asarray(t)), 1 << 16, 1.0)
    # true ||f - S_J f||^2 from a long expansion (reference only)
    from covfourier.gram import CosineGram
    G = CosineGram(H, 1.0); e = a_ex.copy(); e[:J] = 0.0
    true = float(e @ G.matvec(e))
    lo, up, status = certified_bracket(a_hat, a_hat[:J], H, 1.0, math.sqrt(2) * 2 * math.e)
    assert lo <= true <= up and 'certified' in status, (lo, true, up)


def _exact_power_coefficients(p, kmax):
    import mpmath as mp
    mp.mp.dps = 30
    out = [1.0 / (p + 1)]
    for k in range(1, kmax + 1):
        out.append(float(mp.sqrt(2) / (p + 1) * mp.hyp1f2((p + 1) / 2, 0.5, (p + 3) / 2, -(k * mp.pi) ** 2 / 4)))
    return np.array(out)


def test_graded_endpoint_coefficients():
    """Graded end-cell rule recovers the classical coefficients of t^p (p = -1/4, -0.4) from point values."""
    from scipy.fft import dct
    from covfourier.endpoint import graded_cosine_coefficients
    N = 2048
    tm = (np.arange(N) + 0.5) / N
    for p in (-0.25, -0.4):
        ex = _exact_power_coefficients(p, 40)
        ag = graded_cosine_coefficients(lambda t: np.asarray(t) ** p, N, 1.0, M=32, q=8, over=4)[:41]
        y = dct(tm ** p, type=2); au = math.sqrt(2) * y / (2 * N); au[0] = y[0] / (2 * N)
        eg, eu = np.max(np.abs(ag - ex)), np.max(np.abs(au[:41] - ex))
        assert eg < 1e-5 and eg < 1e-2 * eu, (p, eg, eu)


def test_power_coefficient_bound():
    """|a_k(t^p) - sqrt2 Gamma(1+p) cos(pi(1+p)/2) (k pi)^(-1-p)| <= 2 sqrt2 |p| (k pi)^(-2), k = 1..40."""
    for p in (-0.25, -0.4, -0.7):
        ex = _exact_power_coefficients(p, 40)[1:]
        k = np.arange(1, 41)
        asym = math.sqrt(2) * math.gamma(1 + p) * math.cos(math.pi * (1 + p) / 2) * (k * np.pi) ** (-1 - p)
        assert np.all(np.abs(ex - asym) <= 2 * math.sqrt(2) * abs(p) * (k * np.pi) ** -2.0), p


def test_explicit_residual_and_reorthogonalization():
    """Explicit residual: equals the subtractive one when that is resolved (indicator); converges in N for a
    stiff input whose subtractive residual is below its rounding allowance; enrichment directions orthogonal."""
    from covfourier.large import LargeProjector
    vals = []
    for N in (4096, 8192):
        lp = LargeProjector(0.75, 128, N)
        t = (np.arange(N) + 0.5) / N
        F = np.stack([((t >= 0) & (t < 0.3)).astype(float), np.sin(40 * t) + t ** 2])
        r = lp.project(*lp.fourier_stage(F), explicit=True)
        assert lp.orthogonality_defect < 1e-15, lp.orthogonality_defect
        assert abs(r['E2_aux_explicit'][0] / r['error2'][0] - 1) < 1e-9
        assert r['error2'][1] < r['eta'][1]                    # subtractive value unresolved for the stiff input
        vals.append(r['E2_aux_explicit'][1])
    assert vals[1] > 0 and abs(vals[0] / vals[1] - 1) < 1e-3, vals


def test_wholeline_normalization_and_steps():
    """Whole-line realization: spectral and physical norms of Gaussian packets agree; step norms are exact."""
    from covfourier.wholeline import packet_norm2_closed, packet_norm2_spectral, increments, step_norm2
    for H, a, w in ((0.5, 1.0, 3.0), (0.75, 0.3, 0.0), (0.95, 2.7, 10.0), (0.99, 1.0, 3.0)):
        c = packet_norm2_closed(H, a, w); s = packet_norm2_spectral(H, a, w)
        assert abs(s - c) <= 1e-25 * abs(c), (H, a, w)
    import mpmath as mp
    mp.mp.dps = 40
    for H in (0.55, 0.95):
        rho, _ = increments(H, 70000)
        for k in (3, 8, 65535):
            ref = (((mp.mpf(k) + 1) ** (2 * H) - 2 * mp.mpf(k) ** (2 * H) + (mp.mpf(k) - 1) ** (2 * H)) / 2)
            assert abs(rho[k] - float(ref)) <= 1e-14 * abs(float(ref)), (H, k)
    for H in (0.5, 0.7, 0.9):          # 1[0.25,0.75) + 2 * 1[1,1.5) on cells of width 1/4
        cvec = np.array([0, 1, 1, 0, 2, 2], float)
        exact = 0.5 ** (2 * H) + 4 * 0.5 ** (2 * H) + 2 * (abs(1 - 0.75) ** (2 * H) + abs(0.25 - 1.5) ** (2 * H)
                                                           - abs(0.75 - 1.5) ** (2 * H) - abs(0.25 - 1) ** (2 * H))
        assert abs(step_norm2(cvec, H, 0.25) - exact) < 1e-13 * exact, H


def test_unified_interface_graded_route_and_resolved_flag():
    from covfourier.large import LargeProjector
    from covfourier import endpoint as EP
    lp = LargeProjector(0.75, 63, 4096)
    f = lambda t: np.asarray(t, float) ** -0.25
    g = lambda t: (1 - np.asarray(t, float)) ** -0.3
    gl = lambda s: np.asarray(s, float) ** -0.3
    r = lp.from_callables([f, g], endpoint='graded', explicit=True, local=[None, (None, gl)])
    A = np.stack([EP.graded_cosine_coefficients(f, 4096, M=64, q=8, over=8),
                  EP.graded_cosine_coefficients(g, 4096, M=64, q=8, over=8, f_right_local=gl)], axis=1)
    r2 = lp.project(A, np.zeros((2, 2)), explicit=True)
    assert np.array_equal(r['coords'], r2['coords']) and np.array_equal(r['E2_aux_explicit'], r2['E2_aux_explicit'])
    # default route: an input inside V_D has a rounding-level subtractive energy, flagged as unresolved
    r3 = lp.from_callables([lambda t: 1.0 + 0.0 * np.asarray(t, float), lambda t: np.sin(40 * np.pi * np.asarray(t, float))])
    assert (not r3['resolved'][0]) and r3['resolved'][1], (r3['error2'], r3['eta'])


def test_endpoint_rule_options():
    """rule='ratio' reproduces the fixed switch of the first A1 run; both rules agree on smooth inputs."""
    from covfourier.batch import BatchProjector
    cf = CovarianceFourier(0.75, 32, enrich=True, Naux=1024)
    t = (np.arange(1024) + 0.5) / 1024
    F = np.vstack([np.exp(t), np.cos(3 * t) + t ** 2])
    out = {}
    for key, kw in (('post', {}), ('ratio', {'rule': 'ratio', 'max_ratio': 100.0}),
                    ('always', {'rule': 'ratio', 'max_ratio': np.inf}), ('never', {'rule': 'ratio', 'max_ratio': 0.0})):
        bp = BatchProjector(cf, **kw)
        a, c = bp.fourier_stage(F)
        out[key] = (a, c, bp.last_corrected.copy())
    assert out['always'][2].all() and not out['never'][2].any()
    assert out['post'][2].all() and np.array_equal(out['post'][0], out['ratio'][0]) and np.array_equal(out['post'][1], out['ratio'][1])
    assert np.array_equal(out['never'][1], np.zeros((2, 2)))
    try:
        BatchProjector(cf, rule='other')
    except ValueError:
        pass
    else:
        raise AssertionError('invalid rule accepted')


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
