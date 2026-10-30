"""A2 v2: many heterogeneous noisy window sensors, exact GP against covariance-Fourier (CF) shared coordinates.

Changes against the first version (app2_noisy_reconstruction.py):
  * sensor/reference alignment.  Sensors are y_i = int_0^1 phi_i(t) B_H(t) dt + eps_i with phi_i the N(c_i, s_i^2)
    density.  As Wiener integrals the exact integrand is F_i(u) = Phi((c_i - u)/s_i) - Phi((c_i - 1)/s_i).
    Centres are drawn in [10 s_i, 1 - 10 s_i], so the window mass outside [0, 1] is at most 2 Phi(-10) = 1.5e-23,
    and the closed-form (full-line) covariance used by the GP equals the covariance of the truncated sensors
    up to that relative amount.  The first version allowed 5 s_i margins (mass up to 2.9e-7) and used Phi((c - u)/s).
  * the CF sensor residuals I_H(F_i - P F_i) are Gaussian, independent of the retained coordinates and correlated
    across sensors; replacing their covariance R by diag(E_i^2) is an APPROXIMATION.  Its size is reported via
    trace(R)/tau^2 = sum_i E_i^2 / tau^2, which bounds the relative perturbation of the noise covariance.
  * GP baseline strengthened: the kernel E|N(mu, s^2)|^p uses a piecewise-Chebyshev table of 1F1(-p/2; 1/2; -z^2/2)
    (verified against scipy and mpmath), the m x m covariance is built in row blocks into one array and factored
    in place (LAPACK potrf).  This makes the measured crossover conservative for CF.
  * matched accuracy: for each (H, m) the CF level is the smallest n whose smooth-target errors (B(1), int B,
    20 new windows) are <= TOL in posterior-mean sd units and in relative variance; indicator targets B(t*)
    (slow D^-H class) are reported separately and are not used for matching.
  * larger experiments: GP head-to-head to the largest m the free memory allows (checked at run time);
    CF-only beyond, with GP cost there EXTRAPOLATED from a fit to the measured GP times (labelled as such).
"""
import ctypes
import json
import math
import os
import sys
import time

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.special import hyp1f1, gamma, binom, ndtr
from scipy.linalg import cho_factor, cho_solve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from covfourier.engine import CovarianceFourier                                   # noqa: E402
from covfourier.batch import BatchProjector                                       # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
TOL = 1e-6
TAU = 0.01
TSTARS = (0.25, 0.5, 0.75)
N_NEW = 20


def avail_gib():
    class MS(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong), ('a', ctypes.c_ulonglong), ('b', ctypes.c_ulonglong),
                    ('c', ctypes.c_ulonglong), ('d', ctypes.c_ulonglong), ('e', ctypes.c_ulonglong)]
    s = MS(); s.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / 2 ** 30


class FastAbsMoment:
    """E|X|^p, X ~ N(mu, s^2): s^p 2^{p/2} Gamma((p+1)/2)/sqrt(pi) 1F1(-p/2; 1/2; -z^2/2), z = |mu|/s.

    z <= 8: piecewise Chebyshev table (32 panels of width 1/4, degree 18) of the 1F1 factor;
    z > 8 : the asymptotic series of the first version (11 terms).  Accuracy is measured in check()."""
    ZMAX, NP, DEG = 8.0, 32, 18

    def __init__(self, p):
        self.p = float(p)
        self.h = self.ZMAX / self.NP
        self.pref = 2 ** (p / 2) * gamma((p + 1) / 2) / math.sqrt(math.pi)
        x = np.cos(np.pi * (np.arange(self.DEG + 1) + 0.5) / (self.DEG + 1))
        self.coef = np.empty((self.NP, self.DEG + 1))
        for j in range(self.NP):
            z = (j + (x + 1) / 2) * self.h
            self.coef[j] = C.chebfit(x, hyp1f1(-p / 2, 0.5, -z ** 2 / 2), self.DEG)

    def factor(self, z):
        z = np.asarray(z, dtype=float)
        out = np.empty(z.shape)
        small = z <= self.ZMAX
        if np.any(small):
            zs = z[small]
            idx = np.minimum((zs / self.h).astype(np.int64), self.NP - 1)
            u = 2 * (zs / self.h - idx) - 1
            val = np.empty(zs.shape)
            for j in np.unique(idx):
                mk = idx == j
                val[mk] = C.chebval(u[mk], self.coef[j])
            out[small] = val
        if np.any(~small):
            zz = z[~small]; acc = np.zeros_like(zz); dfact = 1.0
            for k in range(0, 11):
                if k > 0:
                    dfact *= (2 * k - 1)
                acc += binom(self.p, 2 * k) * dfact * zz ** (-2 * k)
            out[~small] = zz ** self.p * acc / self.pref          # so that s^p pref factor = |mu|^p acc
        return out

    def __call__(self, mu, s):
        mu = np.asarray(mu, dtype=float); s = np.asarray(s, dtype=float)
        mu, s = np.broadcast_arrays(mu, s)
        return s ** self.p * self.pref * self.factor(np.abs(mu) / s)

    def check(self):
        import mpmath as mp
        rng = np.random.default_rng(1)
        z = np.r_[rng.uniform(0, 8, 200000), np.linspace(0, 8, 2001)]
        ref = hyp1f1(-self.p / 2, 0.5, -z ** 2 / 2)
        e_scipy = float(np.max(np.abs(self.factor(z) / ref - 1)))
        worst = 0.0
        for mu, s in ((0.3, 0.01), (0.05, 0.02), (0.7, 0.03), (0.001, 0.005), (0.2, 0.024), (0.12, 0.015), (0.04, 0.01), (0.15, 0.019)):
            f = lambda x: abs(x) ** self.p * mp.npdf(x, mu, s)
            pts = sorted({0.0, mu - 10 * s, mu, mu + 10 * s})
            ex = mp.quad(f, [-mp.inf] + pts + [mp.inf])
            worst = max(worst, abs(float(self(mu, s)) / float(ex) - 1))
        return {'max_rel_vs_scipy_hyp1f1_z_le_8': e_scipy, 'max_rel_vs_mpmath_8_cases': worst}


def draw_windows(rng, k):
    s = np.exp(rng.uniform(math.log(0.005), math.log(0.03), k))
    lo = np.maximum(0.15, 10 * s); hi = np.minimum(0.85, 1 - 10 * s)
    c = lo + (hi - lo) * rng.uniform(0, 1, k)
    return c, s


def outside_mass(c, s):
    return ndtr(-c / s) + ndtr(-(1 - c) / s)


def gp_covariances(H, c, s, m, mabs, block=500):
    """Exact sensor covariance K (m x m, built in row blocks in place), target rows (targets x m), target variances."""
    p = 2 * H
    mi = mabs(c, s)
    K = np.empty((m, m))
    for i0 in range(0, m, block):
        i1 = min(m, i0 + block)
        K[i0:i1] = 0.5 * (mi[i0:i1, None] + mi[None, :m] - mabs(c[i0:i1, None] - c[None, :m],
                                                                 np.sqrt(s[i0:i1, None] ** 2 + s[None, :m] ** 2)))
    mp1 = FastAbsMoment(p + 1)
    rows = [0.5 * (mi[:m] + 1.0 - mabs(1.0 - c[:m], s[:m])),
            0.5 * (mi[:m] + 1 / (p + 1) - (mp1(c[:m], s[:m]) + mp1(1 - c[:m], s[:m])) / (p + 1))]
    var = [1.0, 1.0 / (p + 2)]
    for ts in TSTARS:
        rows.append(0.5 * (mi[:m] + ts ** p - mabs(c[:m] - ts, s[:m]))); var.append(ts ** p)
    cn, sn = c[m:], s[m:]
    Kn = 0.5 * (mi[m:, None] + mi[None, :m] - mabs(cn[:, None] - c[None, :m], np.sqrt(sn[:, None] ** 2 + s[None, :m] ** 2)))
    vn = mi[m:]                                   # Var(int phi B) = (m + m - m(0, sqrt2 s))/2 ; computed below exactly
    vn = 0.5 * (2 * mi[m:] - mabs(np.zeros_like(cn), np.sqrt(2) * sn))
    return K, np.vstack(rows + [Kn]), np.r_[np.array(var), vn]


def window_integrands_values(c, s, t):
    return ndtr((c[:, None] - t[None, :]) / s[:, None]) - ndtr((c[:, None] - 1.0) / s[:, None])


def cf_coords(bp, c, s, chunk=1000):
    t = (np.arange(bp.N) + 0.5) / bp.N
    Zs, Es = [], []
    for i0 in range(0, len(c), chunk):
        F = window_integrands_values(c[i0:i0 + chunk], s[i0:i0 + chunk], t)
        a_g, cc = bp.fourier_stage(F)
        r = bp.project(a_g, cc, norms=True)
        Zs.append(r['coords'].T); Es.append(np.clip(r['error2'], 0.0, None) + r['eta'])
    return np.vstack(Zs), np.concatenate(Es)


def cf_targets(bp, cn, sn):
    t = (np.arange(bp.N) + 0.5) / bp.N
    F = [np.ones_like(t), 1.0 - t] + [(t < ts).astype(float) for ts in TSTARS]
    F = np.vstack([np.array(F), window_integrands_values(cn, sn, t)])
    a_g, cc = bp.fourier_stage(F)
    r = bp.project(a_g, cc, norms=True)
    return r['coords'].T, np.clip(r['error2'], 0.0, None) + r['eta']


def cf_posterior(Z, E2s, y, Wt, E2t):
    noise = TAU ** 2 + E2s                               # diagonal residual model (approximation; see docstring)
    Zs = Z / np.sqrt(noise)[:, None]
    P = np.eye(Z.shape[1]) + Zs.T @ Zs
    Lp = cho_factor(P, lower=True, check_finite=False)
    xi = cho_solve(Lp, Zs.T @ (y / np.sqrt(noise)))
    return Wt @ xi, np.einsum('ij,ji->i', Wt, cho_solve(Lp, Wt.T)) + E2t


def run_case(H, m, seed, ns=(64, 128, 256, 512), mabs=None):
    rng = np.random.default_rng(seed)
    c, s = draw_windows(rng, m + N_NEW)
    out = {'m': m, 'max_outside_mass': float(outside_mass(c, s).max())}
    t0 = time.perf_counter()
    K, Kt, vt = gp_covariances(H, c, s, m, mabs)
    t_build = time.perf_counter() - t0
    K.flat[::m + 1] += TAU ** 2
    t0 = time.perf_counter()
    L = cho_factor(K, lower=True, overwrite_a=True, check_finite=False)
    t_fact = time.perf_counter() - t0
    y = np.tril(L[0]) @ rng.standard_normal(m)            # data from the exact model (not timed for any method)
    t0 = time.perf_counter()
    alpha = cho_solve(L, y, check_finite=False)
    mean_gp = Kt @ alpha
    var_gp = vt - np.einsum('ij,ji->i', Kt, cho_solve(L, Kt.T, check_finite=False))
    t_solve = time.perf_counter() - t0
    del K, L
    out['GP'] = {'build_s': t_build, 'factor_s': t_fact, 'solve_s': t_solve, 'total_s': t_build + t_fact + t_solve}
    out['CF'] = []
    nt = 2 + len(TSTARS)
    smooth = np.r_[0, 1, np.arange(nt, nt + N_NEW)]
    ind = np.arange(2, nt)
    sd = np.sqrt(var_gp)
    for n in ns:
        t0 = time.perf_counter()
        cf = CovarianceFourier(H, n, enrich=True, Naux=8 * n)
        bp = BatchProjector(cf)
        t_prep = time.perf_counter() - t0
        t0 = time.perf_counter(); Z, E2s = cf_coords(bp, c[:m], s[:m]); t_coords = time.perf_counter() - t0
        t0 = time.perf_counter(); Wt, E2t = cf_targets(bp, c[m:], s[m:]); t_tg = time.perf_counter() - t0
        t0 = time.perf_counter(); mean_cf, var_cf = cf_posterior(Z, E2s, y, Wt, E2t); t_up = time.perf_counter() - t0
        em = np.abs(mean_cf - mean_gp) / sd
        ev = np.abs(var_cf / var_gp - 1)
        rec = {'n': n, 'D': cf.D, 'N': bp.N, 'prep_s': t_prep, 'coords_s': t_coords, 'targets_s': t_tg, 'update_s': t_up,
               'total_s': t_prep + t_coords + t_tg + t_up,
               'smooth_mean_err_sd_max': float(em[smooth].max()), 'smooth_var_rel_err_max': float(ev[smooth].max()),
               'indicator_mean_err_sd_max': float(em[ind].max()), 'indicator_var_rel_err_max': float(ev[ind].max()),
               'B1_mean_err_sd': float(em[0]), 'intB_mean_err_sd': float(em[1]),
               'windows_mean_err_sd_max': float(em[nt:].max()), 'windows_var_rel_err_max': float(ev[nt:].max()),
               'sum_E2_over_tau2': float(E2s.sum() / TAU ** 2), 'max_E2_over_tau2': float(E2s.max() / TAU ** 2)}
        out['CF'].append(rec)
        print('  H %.2f m %6d n %4d total %6.2fs (coords %5.2f) | GP %6.2fs | smooth mean %.1e var %.1e | ind mean %.1e | sumE2/tau2 %.1e'
              % (H, m, n, rec['total_s'], t_coords, out['GP']['total_s'], rec['smooth_mean_err_sd_max'],
                 rec['smooth_var_rel_err_max'], rec['indicator_mean_err_sd_max'], rec['sum_E2_over_tau2']), flush=True)
    ok = [r for r in out['CF'] if r['smooth_mean_err_sd_max'] <= TOL and r['smooth_var_rel_err_max'] <= TOL]
    out['matched'] = ({'tol': TOL, 'n': ok[0]['n'], 'cf_total_s': ok[0]['total_s'], 'gp_total_s': out['GP']['total_s'],
                       'gp_over_cf': out['GP']['total_s'] / ok[0]['total_s']} if ok else {'tol': TOL, 'n': None})
    return out


def run(Hs=(0.55, 0.75, 0.95), ms=(500, 1000, 2000, 4000, 8000), m_try=(10000,), cf_only=(16000, 32000, 64000, 128000),
        out='A2_noisy_reconstruction_v2.json'):
    data = {'note': __doc__, 'tol': TOL, 'tau': TAU, 'kernel_checks': {}, 'cases': {}, 'cf_only': [], 'skipped': []}
    tabs = {H: FastAbsMoment(2 * H) for H in Hs}
    for H in Hs:
        data['kernel_checks'][str(H)] = tabs[H].check()
    # speed of the tabulated kernel against scipy hyp1f1 (same inputs)
    z = np.random.default_rng(2).uniform(0, 8, 2_000_000)
    t0 = time.perf_counter(); tabs[0.75].factor(z); t_tab = time.perf_counter() - t0
    t0 = time.perf_counter(); hyp1f1(-0.75, 0.5, -z ** 2 / 2); t_sc = time.perf_counter() - t0
    data['kernel_speed_2e6'] = {'table_s': t_tab, 'scipy_hyp1f1_s': t_sc}
    print('kernel checks', data['kernel_checks'], data['kernel_speed_2e6'], flush=True)

    def dump():
        with open(os.path.join(RES, out), 'w') as fh:
            json.dump(data, fh, indent=1)
    for H in Hs:
        data['cases'][str(H)] = []
        for m in tuple(ms) + tuple(m_try):
            need = 8 * m * m / 2 ** 30 * 1.0 + 8 * 500 * m * 6 / 2 ** 30 + 0.4
            if m in m_try and (H != 0.75 or avail_gib() < need):
                data['skipped'].append({'H': H, 'm': m, 'needed_GiB': need, 'available_GiB': avail_gib()}); continue
            data['cases'][str(H)].append(run_case(H, m, seed=20261029 + m, mabs=tabs[H]))
            dump()
    # CF-only beyond the dense-GP memory limit (H = 0.75, matched level from the largest head-to-head case)
    last = data['cases']['0.75'][-1]
    n_star = last['matched']['n'] or 256
    gp = [(c['m'], c['GP']['total_s']) for c in data['cases']['0.75']]
    M = np.array([[mm ** 2, mm ** 3] for mm, _ in gp]); tt = np.array([t for _, t in gp])
    # non-negative least squares (identical to the unconstrained fit whenever that fit is non-negative, as in the
    # reported run); an unconstrained fit to noisy timings can give b < 0 and negative extrapolated times.
    from scipy.optimize import nnls
    coef = nnls(M, tt)[0]
    data['gp_fit_H075'] = {'model': 't = a m^2 + b m^3 (least squares on measured m)', 'a': float(coef[0]), 'b': float(coef[1])}
    for m in cf_only:
        rng = np.random.default_rng(7 + m)
        c, s = draw_windows(rng, m + N_NEW)
        t0 = time.perf_counter(); cf = CovarianceFourier(0.75, n_star, enrich=True, Naux=8 * n_star); bp = BatchProjector(cf); tp = time.perf_counter() - t0
        t0 = time.perf_counter(); Z, E2s = cf_coords(bp, c[:m], s[:m]); tc = time.perf_counter() - t0
        t0 = time.perf_counter(); Wt, E2t = cf_targets(bp, c[m:], s[m:]); tt_ = time.perf_counter() - t0
        y = np.random.default_rng(3).standard_normal(m) * 0.5         # timing only (posterior cost does not depend on y)
        t0 = time.perf_counter(); cf_posterior(Z, E2s, y, Wt, E2t); tu = time.perf_counter() - t0
        rec = {'m': m, 'n': n_star, 'prep_s': tp, 'coords_s': tc, 'targets_s': tt_, 'update_s': tu, 'total_s': tp + tc + tt_ + tu,
               'gp_total_s_EXTRAPOLATED': float(coef[0] * m ** 2 + coef[1] * m ** 3), 'dense_K_GiB': 8 * m * m / 2 ** 30,
               'max_outside_mass': float(outside_mass(c, s).max()), 'sum_E2_over_tau2': float(E2s.sum() / TAU ** 2)}
        data['cf_only'].append(rec); dump()
        print('  CF-only m %6d n %d total %.1fs (coords %.1f) | GP extrapolated %.0fs, dense K %.1f GiB'
              % (m, n_star, rec['total_s'], tc, rec['gp_total_s_EXTRAPOLATED'], rec['dense_K_GiB']), flush=True)
    dump()
    return data


if __name__ == '__main__':
    run()
