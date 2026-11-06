"""Law of quadratic functionals of fBm from the CovFourier spectral bracket (release v1.3.0).

Q_w = int_0^1 w(t) B_H(t)^2 dt = sum_n lambda_n xi_n^2  (KL of sqrt(w) K sqrt(w); xi_n iid N(0,1)).
Given computed lower eigenvalues 0 <= b_n <= lambda_n (n <= M) and the exact trace tr K_w:
    S = sum_{n<=M} b_n xi_n^2 <= Q,   R = Q - S >= 0,   E R = Delta = tr K_w - sum b_n,
    every coefficient of R is <= Delta, so  P(R > eps) <= exp(-(sqrt(eps/Delta) - 1)^2 / 2)  (eps > Delta).
Hence  F_S(x - eps) - exp(-(sqrt(eps/Delta)-1)^2/2) <= F_Q(x) <= F_S(x)  for every x, eps > Delta,
and   q_S(p) <= q_Q(p) <= min_eps { q_S(p + exp(...)) + eps }.
F_S is a finite weighted chi-square law; it is inverted by the fixed Talbot rule (route 1) and checked by
Imhof's formula with adaptive quadrature (route 2).

Stages (results cached in ../results):
  spectra   w = 1: stored Galerkin eigenvalues (J = 4096, closed-form matrix, A4);
            w = 1/t: CovFourier coordinates of J = 1024 integrands (A5 procedure), N = 16384 and 32768
  bracket   CDF and quantile brackets, naive Gaussian / two-moment chi-square, event probabilities
  exact     H = 1/2, w = 1: exact law  E e^{-sQ} = (cosh sqrt(2s))^{-1/2}
  disc      exact law of the discretized functional (grid rule of the Monte Carlo) for several grids
  mc        Monte Carlo, exact Davies-Harte paths, 10^6 paths
Usage: python law_study.py [stage ...]     (default: all stages in order)
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy import integrate, optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(HERE, '..', 'covfourier')):          # inside the repository: code/studies/
    REPO = os.path.dirname(os.path.dirname(HERE)); ROOT = REPO
    RES = os.path.join(REPO, 'results'); TABLES = os.path.join(RES, 'law_tables')
else:                                                                # outside the repository: set COVFOURIER_REPO to its root
    ROOT = os.path.dirname(HERE); REPO = os.environ.get('COVFOURIER_REPO', os.path.dirname(ROOT))
    RES = os.path.join(ROOT, 'results'); TABLES = os.path.join(ROOT, 'manuscript')
sys.path.insert(0, os.path.join(REPO, 'code'))
sys.path.insert(0, os.path.join(REPO, 'code', 'applications'))
os.makedirs(RES, exist_ok=True)

PS = (0.5, 0.9, 0.95, 0.99, 0.999)
H_ONE = (0.5, 0.75, 0.95)
H_INV = (0.55, 0.75, 0.95)


def c_H(H):
    return math.gamma(2 * H + 1) * math.sin(math.pi * H)


def trace_K(weight, H):
    return 1 / (2 * H + 1) if weight == 'one' else 1 / (2 * H)


# ----------------------------------------------------------------------------- inversion of finite chi-square laws
class ChiSquareLaw:
    """Law of S = sum b_j xi_j^2 (+ optional exact log-Laplace transform)."""

    def __init__(self, b=None, loglap=None, mean=None):
        self.b = None if b is None else np.asarray(b, float)
        self._loglap = loglap
        self.mean = float(np.sum(self.b)) if mean is None else mean

    def loglap(self, s):                      # log E exp(-s S), s complex array
        if self._loglap is not None:
            return self._loglap(s)
        out = np.zeros(s.shape, complex)
        for c in np.array_split(self.b, max(1, self.b.size // 512)):
            out -= 0.5 * np.log1p(2 * s[..., None] * c).sum(-1)
        return out

    def cdf_talbot(self, x, M=32):
        """Fixed Talbot rule (Abate-Valko 2004) for F(x) = L^{-1}[E e^{-sS}/s](x)."""
        x = float(x)
        if x <= 0:
            return 0.0
        r = 2 * M / (5 * x)
        th = np.arange(1, M) * np.pi / M
        cot = 1 / np.tan(th)
        s = r * th * (cot + 1j)
        sig = th + (th * cot - 1) * cot
        F = np.exp(self.loglap(s) + x * s) / s
        F0 = math.exp(float(self.loglap(np.array([r + 0j]))[0].real) + r * x) / r
        return float((r / M) * (0.5 * F0 + np.sum((F * (1 + 1j * sig)).real)))

    def cdf_imhof(self, x):
        """Imhof (1961): P(S > x) = 1/2 + (1/pi) int_0^inf sin(theta(u)) / (u rho(u)) du (route 2)."""
        b = self.b

        def f(u):
            th = 0.5 * np.sum(np.arctan(b * u)) - 0.5 * x * u
            lr = 0.25 * np.sum(np.log1p((b * u) ** 2))
            return math.sin(th) * math.exp(-lr) / u if lr < 700 else 0.0
        edges = np.concatenate([[0.0], np.geomspace(1e-3, 1e7, 61)])
        tot = 0.0
        for a, c in zip(edges[:-1], edges[1:]):
            tot += integrate.quad(f, a, c, limit=500, epsabs=1e-15, epsrel=1e-13)[0]
        return 1.0 - (0.5 + tot / math.pi)

    def cdf(self, x):
        return self.cdf_talbot(x)

    def quantile(self, p, lo=1e-6, hi=60.0):
        return optimize.brentq(lambda x: self.cdf(x) - p, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=300)


def tail_bound(eps, Delta):
    """P(R > eps) for R = sum c_n xi_n^2, sum c_n = Delta, by the Chernoff bound (closed-form optimum)."""
    if eps <= Delta:
        return 1.0
    return math.exp(-0.5 * (math.sqrt(eps / Delta) - 1) ** 2)


def law_bracket(S, Delta, ps=PS):
    """Quantile brackets for Q from S and Delta."""
    out = {}
    for p in ps:
        qlo = S.quantile(p)
        h = 1e-6 * max(qlo, 1e-3)
        dens = (S.cdf(qlo + h) - S.cdf(qlo - h)) / (2 * h)
        # minimise eps + tail_bound(eps)/dens over eps, then evaluate exactly
        g = lambda le: math.exp(le) + tail_bound(math.exp(le), Delta) / dens
        r = optimize.minimize_scalar(g, bounds=(math.log(1.0001 * Delta), math.log(1e4 * Delta + 1e-30)), method='bounded')
        best = None
        for eps in np.exp(r.x) * np.array([0.7, 0.85, 1.0, 1.2, 1.5]):
            d = tail_bound(eps, Delta)
            if p + d >= 1:
                continue
            qhi = S.quantile(p + d) + eps
            if best is None or qhi < best[0]:
                best = (qhi, eps, d)
        out[str(p)] = {'lo': qlo, 'hi': best[0], 'eps': best[1], 'tail': best[2], 'density': dens}
    return out


def cdf_bracket(S, Delta, xs):
    lo, hi = [], []
    for x in xs:
        up = S.cdf(x)
        best = 0.0
        for eps in Delta * np.geomspace(2, 3e3, 60):
            if x - eps <= 0:
                break
            best = max(best, S.cdf(x - eps) - tail_bound(eps, Delta))
        lo.append(max(best, 0.0)); hi.append(up)
    return lo, hi


# ----------------------------------------------------------------------------- stages
def stage_spectra():
    out = {}
    z = np.load(os.path.join(REPO, 'results', 'A4_eigs_J4096.npz'))
    for H in H_ONE:
        mu = np.sort(z['mu_' + str(H).replace('.', 'p')])[::-1]
        out[f'one_{H}'] = mu
    import app5_weighted_kl as A5
    meta = {}
    for H in H_INV:
        for N in (16384, 32768):
            t0 = time.perf_counter()
            r = A5.run_case(H, 'inv_t', 1024, n=2047, N=N)
            # recompute the full spectrum (run_case stores only the first 50)
            from covfourier.large import LargeProjector
            lp = LargeProjector(H, 2047, N)
            F = A5.F_samples('inv_t', 1024, N)
            Zs = []
            for i0 in range(0, 1024, 256):
                a_g, c = lp.fourier_stage(F[i0:i0 + 256])
                Zs.append(lp.project(a_g, c)['coords'].T)
            Zc = np.vstack(Zs)
            mu = np.linalg.svd(Zc, compute_uv=False) ** 2
            out[f'inv_{H}_{N}'] = mu
            meta[f'inv_{H}_{N}'] = {'frob': float(np.sum(Zc ** 2)), 'trace': trace_K('inv_t', H),
                                    'delta_runcase': r['trace_gap_delta'], 'seconds': time.perf_counter() - t0,
                                    'laplace_1': r['laplace_bracket']['1.0']}
            print('spectrum inv_t', H, N, meta[f'inv_{H}_{N}'], flush=True)
    np.savez(os.path.join(RES, 'L1_spectra.npz'), **out)
    json.dump(meta, open(os.path.join(RES, 'L1_spectra_meta.json'), 'w'), indent=1)


def lower_values(weight, H, mu, J):
    """b_n = max(mu_n - eta, l_n) (w = 1, Theorem 3 lower bounds) or mu_n - eta (w = 1/t); eta covers eigensolver rounding."""
    mu = np.sort(np.asarray(mu))[::-1]
    eta = 10 * J * np.finfo(float).eps * mu[0]
    b = np.clip(mu - eta, 0, None)
    if weight == 'one':
        n = np.arange(1, b.size + 1)
        b = np.maximum(b, 0.5 * c_H(H) * (n * np.pi) ** (-1 - 2 * H))
    return b, eta


def stage_bracket():
    sp = np.load(os.path.join(RES, 'L1_spectra.npz'))
    cases = [('one', H, f'one_{H}', 4096) for H in H_ONE] + [('inv_t', H, f'inv_{H}_16384', 1024) for H in H_INV] \
        + [('inv_t', H, f'inv_{H}_32768', 1024) for H in H_INV]
    res = {}
    for weight, H, key, J in cases:
        t0 = time.perf_counter()
        b, eta = lower_values(weight, H, sp[key], J)
        tr = trace_K(weight, H)
        Delta = tr - b.sum()
        S = ChiSquareLaw(b)
        qb = law_bracket(S, Delta)
        # route-2 check of the inversion at the bracket ends
        chk = max(abs(S.cdf_imhof(v['lo']) - float(p)) for p, v in qb.items())
        chk_t = max(abs(S.cdf_talbot(v['lo'], M=24) - S.cdf_talbot(v['lo'], M=32)) for v in qb.values())
        var_lo = 2 * float(np.sum(b ** 2)); var_hi = var_lo + 4 * (b[0] + Delta) * Delta
        var = 0.5 * (var_lo + var_hi)
        naive = {}
        g = var / (2 * tr); nu = tr / g
        for p in PS:
            qG = tr + math.sqrt(var) * stats.norm.ppf(p)
            qW = g * stats.chi2.ppf(p, nu)
            naive[str(p)] = {'gauss': qG, 'chi2_two_moment': qW}
        events = {}
        for alpha in (0.05, 0.01, 0.001):
            xG = naive[str(1 - alpha)]['gauss']; xW = naive[str(1 - alpha)]['chi2_two_moment']
            for name, x in (('gauss', xG), ('chi2_two_moment', xW)):
                lo, hi = cdf_bracket(S, Delta, [x])
                events[f'{name}_{alpha}'] = {'x': x, 'P_exceed': [1 - hi[0], 1 - lo[0]]}
        res[key] = {'weight': weight, 'H': H, 'J': J, 'M': int(b.size), 'eta': eta, 'trace': tr, 'Delta': Delta,
                    'var_bracket': [var_lo, var_hi], 'P_gauss_below_0': float(stats.norm.cdf(-tr / math.sqrt(var))),
                    'quantiles': qb, 'naive': naive, 'events': events,
                    'inversion_check_imhof_max_abs': chk, 'inversion_check_talbot_M24_M32': chk_t,
                    'seconds': time.perf_counter() - t0}
        print(key, 'Delta %.3e' % Delta, {p: '[%.10f, %.10f]' % (v['lo'], v['hi']) for p, v in qb.items()},
              'imhof chk %.1e talbot chk %.1e' % (chk, chk_t), flush=True)
        json.dump(res, open(os.path.join(RES, 'L2_brackets.json'), 'w'), indent=1)
    # CDF band for the figure (w = 1, H = 0.75; w = 1/t, H = 0.75)
    band = {}
    for key, weight, H, J in (('one_0.75', 'one', 0.75, 4096), ('inv_0.75_16384', 'inv_t', 0.75, 1024)):
        b, _ = lower_values(weight, H, sp[key], J)
        Delta = trace_K(weight, H) - b.sum()
        S = ChiSquareLaw(b)
        xs = np.linspace(0.005, 4.5 if weight == 'one' else 9.0, 120)
        lo, hi = cdf_bracket(S, Delta, xs)
        band[key] = {'x': xs.tolist(), 'lo': lo, 'hi': hi, 'Delta': Delta}
    json.dump(band, open(os.path.join(RES, 'L2_cdf_band.json'), 'w'))


def stage_exact():
    """H = 1/2, w = 1: exact law; checks the inversion and that the brackets contain the exact quantiles."""
    def loglap(s):
        # product over the exact spectrum lambda_n = ((n-1/2) pi)^-2, n <= 2e5, plus the first two tail cumulants
        n = np.arange(1, 200001)
        lam = 1 / ((n - 0.5) * np.pi) ** 2
        out = np.zeros(s.shape, complex)
        for c in np.array_split(lam, 400):
            out -= 0.5 * np.log1p(2 * s[..., None] * c).sum(-1)
        t1 = 0.5 - lam.sum(); t2 = 1 / 6 - (lam ** 2).sum()
        return out - s * t1 + s * s * t2
    E = ChiSquareLaw(loglap=loglap, mean=0.5)
    br = json.load(open(os.path.join(RES, 'L2_brackets.json')))['one_0.5']['quantiles']
    out = {}
    for p in PS:
        q = E.quantile(p)
        # closed form on the right half-plane check: Talbot nodes have Re s < 0 near theta -> pi, so compare a CDF value
        out[str(p)] = {'exact': q, 'lo': br[str(p)]['lo'], 'hi': br[str(p)]['hi'],
                       'inside': bool(br[str(p)]['lo'] <= q <= br[str(p)]['hi'])}
        print('exact H=1/2', p, out[str(p)], flush=True)
    # independent check of the exact CDF at one point: Laplace transform identity int e^{-ax} dF = (cosh sqrt(2a))^{-1/2}
    a, X = 1.0, 14.0
    xg, wg = np.polynomial.legendre.leggauss(40)
    lap_num = math.exp(-a * X)
    for lo_, hi_ in ((0.0, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, X)):
        xx = 0.5 * (hi_ - lo_) * xg + 0.5 * (hi_ + lo_)
        lap_num += a * 0.5 * (hi_ - lo_) * float(np.sum(wg * np.exp(-a * xx) * np.array([E.cdf(v) for v in xx])))
    out['laplace_check'] = {'numerical': float(lap_num), 'exact': float(np.cosh(math.sqrt(2)) ** -0.5)}
    print('Laplace check', out['laplace_check'], flush=True)
    json.dump(out, open(os.path.join(RES, 'L3_exact_bm.json'), 'w'), indent=1)


def fbm_cov(H, t):
    T, S = np.meshgrid(t, t, indexing='ij')
    return 0.5 * (T ** (2 * H) + S ** (2 * H) - np.abs(T - S) ** (2 * H))


def disc_eigs(weight, H, N):
    t = np.arange(1, N + 1) / N
    if weight == 'one':                    # trapezoidal rule, B(0) = 0
        w = np.full(N, 1.0 / N); w[-1] = 0.5 / N
    else:                                  # right-endpoint rule for w = 1/t (as in the Monte Carlo of Section 7.3)
        w = 1.0 / (N * t)
    C = fbm_cov(H, t)
    sw = np.sqrt(w)
    lam = np.linalg.eigvalsh(sw[:, None] * C * sw[None, :])
    return np.sort(np.clip(lam, 0, None))[::-1]


def stage_disc():
    res = {}
    plan = [('one', H, N) for H in (0.75, 0.95) for N in (1024, 2048, 4096, 8192)] + \
           [('inv_t', H, N) for H in H_INV for N in (1024, 2048, 4096, 8192)]
    for weight, H, N in plan:
        t0 = time.perf_counter()
        lam = disc_eigs(weight, H, N)
        S = ChiSquareLaw(lam)
        qs = {str(p): S.quantile(p) for p in PS}
        res[f'{weight}_{H}_{N}'] = {'quantiles': qs, 'mean': float(lam.sum()), 'seconds': time.perf_counter() - t0}
        print('disc', weight, H, N, qs, '%.0fs' % (time.perf_counter() - t0), flush=True)
        json.dump(res, open(os.path.join(RES, 'L4_discretized.json'), 'w'), indent=1)


def stage_mc(M=10 ** 6, seed=20261106):
    from covfourier.comparators import DaviesHarte
    out_name = os.environ.get('MC_OUT', 'L5_monte_carlo.json')
    res = json.load(open(os.path.join(RES, out_name))) if os.path.exists(os.path.join(RES, out_name)) else {}
    plan = [('one', H, 1024) for H in H_ONE] + [('inv_t', H, 4096) for H in H_INV]
    if os.environ.get('MC_CASES'):                       # e.g. MC_CASES=inv_t:0.55  (parallel runs)
        sel = [tuple(c.split(':')) for c in os.environ['MC_CASES'].split(',')]
        plan = [c for c in plan if (c[0], str(c[1])) in sel]
    for weight, H, N in plan:
        t0 = time.perf_counter()
        dh = DaviesHarte(H, N)
        rng = np.random.default_rng(seed + int(1000 * H) + (0 if weight == 'one' else 7))
        t = np.arange(1, N + 1) / N
        if weight == 'one':
            w = np.full(N, 1.0 / N); w[-1] = 0.5 / N
        else:
            w = 1.0 / (N * t)
        Q = np.empty(M)
        bs = 4000 if N <= 1024 else 1000
        for m0 in range(0, M, bs):
            B = np.cumsum(dh.increments(min(bs, M - m0), rng), axis=1)
            Q[m0:m0 + B.shape[0]] = (B ** 2) @ w
        Q.sort()
        qs = {}
        for p in PS:
            k = M * p; h = 1.959964 * math.sqrt(M * p * (1 - p))
            qs[str(p)] = {'q': float(Q[int(round(k)) - 1]), 'ci': [float(Q[max(0, int(math.floor(k - h)) - 1)]),
                                                                    float(Q[min(M - 1, int(math.ceil(k + h)) - 1)])]}
        res[f'{weight}_{H}_{N}'] = {'M': M, 'quantiles': qs, 'mean': float(Q.mean()), 'seconds': time.perf_counter() - t0,
                                    'ecdf_x': Q[::max(1, M // 2000)].tolist()}
        print('mc', weight, H, N, {p: '%.5f [%.5f, %.5f]' % (v['q'], *v['ci']) for p, v in qs.items()},
              '%.0fs' % (time.perf_counter() - t0), flush=True)
        json.dump(res, open(os.path.join(RES, out_name), 'w'))


STAGES = {'spectra': stage_spectra, 'bracket': stage_bracket, 'exact': stage_exact, 'disc': stage_disc, 'mc': stage_mc}

if __name__ == '__main__':
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '6')
    for name in (sys.argv[1:] or list(STAGES)):
        t0 = time.perf_counter()
        STAGES[name]()
        print(f'== stage {name} done in {time.perf_counter() - t0:.0f}s', flush=True)
