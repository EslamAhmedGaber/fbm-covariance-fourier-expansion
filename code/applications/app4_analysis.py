"""A4 analysis: KL eigenvalues of fBm on [0,1] far beyond the published numerical range.

Outputs results/A4_analysis.json (and A4_eigs_J4096.npz with the eigenvalue arrays).

Contents
  * Galerkin (cosine basis) eigenvalues mu_n(J), J = 1024, 2048, 4096: J-convergence by n.
  * Chigansky-Kleptsyna (SPA 128 (2018), Thm 2.1) two-term law nu_n_hat; empirical residual
    rho_n = nu_n - nu_n_hat and the O(1/n) coefficient (fit kappa1/n + kappa2/n^2).
  * Analytic eigenvalue bounds (Theorem 'KL eigenvalues' of the paper, from the two-sided weighted cosine bounds):
        l_n = (1/2) c_H (n pi)^(-1-2H)  <=  lambda_n  <=  u_n = c_H ((n-1) pi)^(-1-2H)   (n >= 2).
  * Nystrom (midpoint) comparator, N = 1024, 2048, 4096.
  * Certified Laplace-transform bracket for L(a) = E exp(-a ||B_H||^2), basic and improved
    (improved: head lower bounds max(mu_n, l_n), tail lower bounds l_n with exact Hurwitz tail sum).
  * Monte Carlo comparator (Davies-Harte paths + trapezoid) for L(1).
"""
import json, math, os, sys, time
import numpy as np
import mpmath as mp
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import app4_kl_spectrum as A4                      # noqa: E402
from covfourier.comparators import DaviesHarte     # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')


def cH(H):
    return math.gamma(2 * H + 1) * math.sin(math.pi * H)


def galerkin_eigs(H, J):
    A = A4.galerkin_matrix(H, J)
    tr = float(np.trace(A))
    mu = eigh(A, eigvals_only=True, driver='evr', overwrite_a=True, check_finite=False)
    return np.sort(mu)[::-1], 1.0 / (2 * H + 1) - tr


def nystrom_eigs(H, N, nmax):
    t = (np.arange(N) + 0.5) / N
    R = 0.5 * (t[:, None] ** (2 * H) + t[None, :] ** (2 * H) - np.abs(t[:, None] - t[None, :]) ** (2 * H)) / N
    lam = eigh(R, eigvals_only=True, driver='evr', overwrite_a=True, check_finite=False,
               subset_by_index=[N - nmax, N - 1])
    return np.sort(lam)[::-1]


def hurwitz_tail(H, J):
    """sum_{n>J} l_n = (1/2) c_H pi^(-1-2H) zeta(1+2H, J+1)."""
    return 0.5 * cH(H) * math.pi ** (-1 - 2 * H) * float(mp.zeta(1 + 2 * H, J + 1))


def laplace_improved(H, mu, delta, a, nmax_tail=2_000_000):
    J = len(mu)
    n = np.arange(1, J + 1, dtype=float)
    ln = 0.5 * cH(H) * (n * np.pi) ** (-1 - 2 * H)
    head = np.maximum(mu, ln)
    # upper bound on L: lower bounds on every lambda_n (finite tail sum is still a valid bound)
    nt = np.arange(J + 1, nmax_tail + 1, dtype=float)
    lt = 0.5 * cH(H) * (nt * np.pi) ** (-1 - 2 * H)
    s_up = np.sum(np.log1p(2 * a * head)) + np.sum(np.log1p(2 * a * lt))
    upper = math.exp(-0.5 * s_up)
    # lower bound: exact missing mass after using the improved lower bounds
    delta2 = delta - float(np.sum(head - mu)) - hurwitz_tail(H, J)
    s_lo = np.sum(np.log1p(2 * a * head)) + np.sum(np.log1p(2 * a * lt))
    lower = math.exp(-0.5 * s_lo - a * max(delta2, 0.0))
    # (the tail beyond nmax_tail is omitted in s_lo: compensate by adding its mass to delta2)
    rest = hurwitz_tail(H, nmax_tail)
    lower = lower * math.exp(-a * rest)
    return lower, upper, delta2


def mc_laplace(H, a, M, N=1024, batch=5000, seed=7):
    dh = DaviesHarte(H, N)
    rng = np.random.default_rng(seed)
    vals = []
    t0 = time.perf_counter()
    for _ in range(M // batch):
        X = dh.increments(batch, rng)                  # (batch, N) increments on grid 1/N
        B = np.concatenate([np.zeros((batch, 1)), np.cumsum(X, axis=1)], axis=1)
        I = (np.sum(B[:, 1:-1] ** 2, axis=1) + 0.5 * B[:, -1] ** 2) / N
        vals.append(np.exp(-a * I))
    v = np.concatenate(vals)
    return float(v.mean()), float(1.96 * v.std(ddof=1) / math.sqrt(len(v))), time.perf_counter() - t0


def run(Hs=(0.5, 0.55, 0.65, 0.75, 0.85, 0.95), Js=(1024, 2048, 4096), out='A4_analysis.json'):
    data = {'note': __doc__, 'H': {}}
    arrays = {}
    T0 = time.perf_counter()
    for H in Hs:
        rec = {}
        mus, deltas, times = {}, {}, {}
        for J in Js:
            t0 = time.perf_counter()
            mus[J], deltas[J] = galerkin_eigs(H, J)
            times[J] = time.perf_counter() - t0
        Jm = Js[-1]; mu = mus[Jm]
        arrays[f'mu_{H}'] = mu
        rec['delta_J'] = {str(J): deltas[J] for J in Js}
        rec['seconds_build_plus_eig'] = {str(J): times[J] for J in Js}
        # J-convergence (observed increments; mu_n(J) is nondecreasing in J)
        conv = {}
        for nmax in (50, 100, 300, 1000):
            conv[str(nmax)] = {f'{Js[i]}->{Js[i+1]}': float(np.max((mus[Js[i + 1]][:nmax] - mus[Js[i]][:nmax]) / mus[Js[i + 1]][:nmax]))
                               for i in range(len(Js) - 1)}
        rec['rel_increment_max'] = conv
        rec['monotone_violation_min'] = float(min(np.min(mus[Js[i + 1]][:Js[i]] - mus[Js[i]]) for i in range(len(Js) - 1)))
        # C-K comparison
        n = np.arange(1, 1001, dtype=float)
        nu = (cH(H) / mu[:1000]) ** (1 / (2 * H + 1))
        nuh = A4.ck_nu(H, n)
        rho = nu - nuh
        rec['ck_rho_times_n'] = {str(int(k)): float(rho[int(k) - 1] * k) for k in (1, 2, 5, 10, 20, 40, 100, 200, 500, 1000)}
        sel = (n >= 50) & (n <= 500)
        X = np.c_[1 / n[sel], 1 / n[sel] ** 2]
        coef, *_ = np.linalg.lstsq(X, rho[sel], rcond=None)
        rec['ck_fit_kappa1_kappa2'] = coef.tolist()
        lam_ck = cH(H) * nuh ** (-2 * H - 1)
        rec['ck_two_term_rel_err'] = {str(int(k)): float(abs(lam_ck[int(k) - 1] - mu[int(k) - 1]) / mu[int(k) - 1])
                                      for k in (1, 5, 10, 40, 100, 1000)}
        arrays[f'rho_{H}'] = rho
        if H == 0.5:
            ex = 1.0 / ((n - 0.5) * np.pi) ** 2
            rec['bm_rel_err'] = {str(int(k)): float(abs(mu[int(k) - 1] - ex[int(k) - 1]) / ex[int(k) - 1]) for k in (1, 10, 100, 300, 1000)}
        # analytic certified bounds
        ln = 0.5 * cH(H) * (n * np.pi) ** (-1 - 2 * H)
        un = cH(H) * ((n[1:] - 1) * np.pi) ** (-1 - 2 * H)
        rec['analytic_bounds'] = {'min_mu_over_l (n<=1000)': float(np.min(mu[:1000] / ln)),
                                  'max_mu_over_u (2<=n<=1000)': float(np.max(mu[1:1000] / un)),
                                  'mu_over_l_at_1000': float(mu[999] / ln[999]),
                                  'mu_over_u_at_1000': float(mu[999] / un[-1])}
        # Nystrom comparator
        nys = {}
        for N in (1024, 2048, 4096):
            t0 = time.perf_counter()
            lamN = nystrom_eigs(H, N, 600)
            tN = time.perf_counter() - t0
            nuN = (cH(H) / lamN) ** (1 / (2 * H + 1))
            nys[str(N)] = {'seconds': tN,
                           'rel_err': {str(k): float(abs(lamN[k - 1] - mu[k - 1]) / mu[k - 1]) for k in (1, 10, 50, 100, 200, 500)},
                           'rho_times_n': {str(k): float((nuN[k - 1] - A4.ck_nu(H, k)) * k) for k in (10, 50, 100, 200, 500)}}
            arrays[f'nys_{H}_{N}'] = lamN
        rec['nystrom'] = nys
        # Laplace brackets at the largest J
        lap = {}
        for a in (0.1, 1.0, 10.0, 100.0):
            lo0, hi0 = A4.laplace_bounds(mu, max(deltas[Jm], 0.0), a)
            lo1, hi1, d2 = laplace_improved(H, mu, deltas[Jm], a)
            lap[str(a)] = {'basic': [lo0, hi0], 'improved': [max(lo0, lo1), min(hi0, hi1)], 'residual_mass': d2}
            if H == 0.5:
                lap[str(a)]['exact'] = float(np.cosh(math.sqrt(2 * a)) ** -0.5)
        rec['laplace'] = lap
        data['H'][str(H)] = rec
        print('H', H, 'done', 'delta', {k: '%.2e' % v for k, v in rec['delta_J'].items()},
              'kappa', ['%.4f' % c for c in coef], 'elapsed %.0fs' % (time.perf_counter() - T0), flush=True)
    # Monte Carlo comparator at H = 0.75, a = 1
    mean, ci, tmc = mc_laplace(0.75, 1.0, 100_000)
    data['mc_H075_a1'] = {'M': 100_000, 'N': 1024, 'mean': mean, 'ci95_halfwidth': ci, 'seconds': tmc}
    data['seconds_total'] = time.perf_counter() - T0
    with open(os.path.join(RES, out), 'w') as fh:
        json.dump(data, fh, indent=1)
    np.savez_compressed(os.path.join(RES, 'A4_eigs_J4096.npz'), **{k.replace('.', 'p'): v for k, v in arrays.items()})
    return data


if __name__ == '__main__':
    run()
