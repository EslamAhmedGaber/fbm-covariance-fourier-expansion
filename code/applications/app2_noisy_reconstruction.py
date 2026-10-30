"""A2: reconstruction from many heterogeneous noisy measurements (Gaussian-process regression).

Sensors  y_i = int phi_i(t) B_H(t) dt + eps_i,  phi_i = N(c_i, sig_i^2) density (window inside (0,1)),
         eps_i ~ N(0, tau^2).  As Wiener integrals: y_i = int F_i dB_H + eps_i, F_i(s) = Phi((c_i - s)/sig_i).
Exact covariances (truncation of the Gaussian windows below 1e-14 by construction):
         Cov(int phi_i B, int phi_j B) = (m(c_i, s_i) + m(c_j, s_j) - m(c_i - c_j, sqrt(s_i^2 + s_j^2))) / 2,
         m(mu, s) = E|N(mu, s^2)|^p, p = 2H  (Kummer function; asymptotic series for |mu|/s > 8).
Targets  B(1) (integrand 1), int_0^1 B (integrand 1 - s), B(t*) for t* in {1/4, 1/2, 3/4} (indicator
         integrands: adverse, slow D^-H convergence), 20 new window averages.
Methods  GP   exact dense posterior: build m x m covariance, Cholesky, O(m^3);
         CF   covariance-Fourier shared coordinates: y ~ Z xi + eps, xi ~ N(0, I_D), posterior in
              D dimensions, O(m D^2); target variance adds the target residual energy E_D^2
              (residual is H-orthogonal to the span); sensor residual energies inflate the noise.
"""
import json, math, os, sys, time
import numpy as np
from scipy.special import hyp1f1, gamma, binom, ndtr
from scipy.linalg import cho_factor, cho_solve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from covfourier.engine import CovarianceFourier                                   # noqa: E402
from covfourier.batch import BatchProjector                                       # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')


def m_abs(mu, s, p):
    """E|X|^p, X ~ N(mu, s^2), vectorized; s > 0."""
    mu = np.asarray(mu, dtype=float); s = np.asarray(s, dtype=float)
    mu, s = np.broadcast_arrays(mu, s)
    z = np.abs(mu) / s
    out = np.empty(mu.shape)
    small = z <= 8.0
    if np.any(small):
        zz = z[small]
        out[small] = s[small] ** p * 2 ** (p / 2) * gamma((p + 1) / 2) / math.sqrt(math.pi) * hyp1f1(-p / 2, 0.5, -zz ** 2 / 2)
    if np.any(~small):
        zz = z[~small]; acc = np.zeros_like(zz); dfact = 1.0
        for k in range(0, 11):
            if k > 0:
                dfact *= (2 * k - 1)
            acc += binom(p, 2 * k) * dfact * zz ** (-2 * k)
        out[~small] = np.abs(mu[~small]) ** p * acc
    return out


class Windows:
    def __init__(self, c, s):
        self.c = np.asarray(c, float); self.s = np.asarray(s, float)

    def __len__(self):
        return len(self.c)

    def callables(self):
        return [lambda t, c=c, s=s: ndtr((c - np.asarray(t, float)) / s) for c, s in zip(self.c, self.s)]


def exact_blocks(H, W, tstars):
    """Exact joint covariance pieces: sensors (m x m), targets x sensors, target variances."""
    p = 2 * H
    c, s = W.c, W.s
    mi = m_abs(c, s, p)
    K = 0.5 * (mi[:, None] + mi[None, :] - m_abs(c[:, None] - c[None, :], np.sqrt(s[:, None] ** 2 + s[None, :] ** 2), p))
    rows, var = [], []
    # B(1)
    rows.append(0.5 * (mi + 1.0 - m_abs(1.0 - c, s, p))); var.append(1.0)
    # int_0^1 B dt :  int_0^1 R(t,u) du = (t^p + 1/(p+1) - (t^(p+1) + (1-t)^(p+1))/(p+1)) / 2
    rows.append(0.5 * (mi + 1 / (p + 1) - (m_abs(c, s, p + 1) + m_abs(1 - c, s, p + 1)) / (p + 1)))
    var.append(1.0 / (p + 2))
    for ts in tstars:
        rows.append(0.5 * (mi + ts ** p - m_abs(c - ts, s, p))); var.append(ts ** p)
    return K, np.array(rows), np.array(var)


def sensor_coords(bp, c, s, chunk=1000):
    """Shared coordinates of window sensors F_i(t) = Phi((c_i - t)/s_i), batched in chunks."""
    N = bp.N
    t = (np.arange(N) + 0.5) / N
    Zs, Es = [], []
    for i0 in range(0, len(c), chunk):
        F = ndtr((c[i0:i0 + chunk, None] - t[None, :]) / s[i0:i0 + chunk, None])
        a_g, cc = bp.fourier_stage(F)
        r = bp.project(a_g, cc, norms=True)
        Zs.append(r['coords'].T); Es.append(np.clip(r['error2'], 0.0, None) + r['eta'])
    return np.vstack(Zs), np.concatenate(Es)


def target_integrands(tstars):
    f = [lambda t: np.ones_like(np.asarray(t, float)), lambda t: 1.0 - np.asarray(t, float)]
    f += [lambda t, ts=ts: (np.asarray(t, float) < ts).astype(float) for ts in tstars]
    return f


def run_case(H, m, rng, ns=(128, 256, 512), tau=0.01, tstars=(0.25, 0.5, 0.75), n_new=20):
    c = rng.uniform(0.15, 0.85, m + n_new); s = np.exp(rng.uniform(math.log(0.005), math.log(0.03), m + n_new))
    Wall = Windows(c, s)
    # ---------------- exact dense GP (timed: covariance build + Cholesky + solves)
    t0 = time.perf_counter()
    Kall, Rt, vt = exact_blocks(H, Wall, tstars)
    K = Kall[:m, :m]
    Kt = np.r_[Rt[:, :m], Kall[m:, :m]]                 # targets x sensors
    vtar = np.r_[vt, np.diag(Kall)[m:]]
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    Ky = K + tau ** 2 * np.eye(m)
    cf_ = cho_factor(Ky, lower=True, overwrite_a=True)
    t_fact = time.perf_counter() - t0
    # data: y ~ N(0, K + tau^2 I) drawn with the same exact factor (not part of any method's cost)
    y = np.tril(cf_[0]) @ rng.standard_normal(m)
    t0 = time.perf_counter()
    alpha = cho_solve(cf_, y)
    mean_gp = Kt @ alpha
    var_gp = vtar - np.einsum('ij,ji->i', Kt, cho_solve(cf_, Kt.T))
    t_solve = time.perf_counter() - t0 + t_fact
    out = {'m': m, 'GP': {'build_s': t_build, 'factor_solve_s': t_solve, 'total_s': t_build + t_solve}, 'CF': []}
    # ---------------- covariance-Fourier low-rank posterior
    for n in ns:
        Naux = 8 * n
        t0 = time.perf_counter()
        cf = CovarianceFourier(H, n, enrich=True, Naux=Naux)
        bp = BatchProjector(cf)
        t_prep = time.perf_counter() - t0
        t0 = time.perf_counter()
        Z, E2s = sensor_coords(bp, c[:m], s[:m])
        t_coords = time.perf_counter() - t0
        t0 = time.perf_counter()
        r = bp.from_callables(target_integrands(tstars) + Windows(c[m:], s[m:]).callables())
        Wt = r['coords'].T; E2t = np.clip(r['error2'], 0.0, None) + r['eta']
        t_targets = time.perf_counter() - t0
        t0 = time.perf_counter()
        noise = tau ** 2 + E2s
        Zs = Z / np.sqrt(noise)[:, None]
        P = np.eye(cf.D) + Zs.T @ Zs
        Lp = cho_factor(P, lower=True)
        xi = cho_solve(Lp, Zs.T @ (y / np.sqrt(noise)))
        mean_cf = Wt @ xi
        var_cf = np.einsum('ij,ji->i', Wt, cho_solve(Lp, Wt.T)) + E2t
        t_update = time.perf_counter() - t0
        sd = np.sqrt(var_gp)
        errm = np.abs(mean_cf - mean_gp) / sd
        errv = np.abs(var_cf / var_gp - 1)
        rec = {'n': n, 'D': cf.D, 'Naux': Naux, 'prep_s': t_prep, 'coords_s': t_coords, 'targets_s': t_targets,
               'update_s': t_update, 'total_s': t_prep + t_coords + t_targets + t_update,
               'max_sensor_E2_rel': float(np.max(E2s / np.diag(K))),
               'mean_err_sd_units': {'B1': float(errm[0]), 'intB': float(errm[1]),
                                     'B_tstar_max': float(errm[2:2 + len(tstars)].max()),
                                     'windows_max': float(errm[2 + len(tstars):].max())},
               'var_rel_err': {'B1': float(errv[0]), 'intB': float(errv[1]),
                               'B_tstar_max': float(errv[2:2 + len(tstars)].max()),
                               'windows_max': float(errv[2 + len(tstars):].max())},
               'target_E2': E2t[:2 + len(tstars)].tolist()}
        out['CF'].append(rec)
        print('  m %5d n %4d coords %.2fs update %.3fs | GP %.2fs | mean err (sd) B1 %.1e intB %.1e Bt %.1e win %.1e | var err win %.1e Bt %.1e'
              % (m, n, t_coords, t_update, t_build + t_solve, errm[0], errm[1], errm[2:5].max(), errm[5:].max(),
                 errv[5:].max(), errv[2:5].max()), flush=True)
    return out


def check_m_abs(H=0.75):
    import mpmath as mp
    p = 2 * H; worst = 0.0
    for mu, s in ((0.3, 0.01), (0.05, 0.02), (0.7, 0.03), (0.001, 0.005), (0.2, 0.024), (0.12, 0.015)):
        f = lambda x: abs(x) ** p * mp.npdf(x, mu, s)
        pts = sorted({0.0, mu - 10 * s, mu, mu + 10 * s})          # cusp at 0 must be a breakpoint
        ex = mp.quad(f, [-mp.inf] + pts + [mp.inf])
        worst = max(worst, abs(float(m_abs(mu, s, p)) / float(ex) - 1))
    return worst


def run(Hs=(0.55, 0.75, 0.95), ms=(250, 500, 1000, 2000, 4000), out='A2_noisy_reconstruction.json'):
    data = {'note': __doc__, 'm_abs_check_max_rel_err': {str(H): check_m_abs(H) for H in Hs}, 'cases': {}}
    print('m_abs check', data['m_abs_check_max_rel_err'], flush=True)
    for H in Hs:
        rng = np.random.default_rng(20261029)
        data['cases'][str(H)] = []
        for m in ms:
            data['cases'][str(H)].append(run_case(H, m, rng))
            with open(os.path.join(RES, out), 'w') as fh:
                json.dump(data, fh, indent=1)
    # CF-only scaling where the dense GP is memory-infeasible on this machine (16 GiB, ~1.4 GiB free)
    sc = []
    for m in (8000, 16000, 32000):
        rng = np.random.default_rng(7)
        c = rng.uniform(0.15, 0.85, m); s = np.exp(rng.uniform(math.log(0.005), math.log(0.03), m))
        cf = CovarianceFourier(0.75, 512, enrich=True, Naux=4096)
        bp = BatchProjector(cf)
        t0 = time.perf_counter()
        Z, _ = sensor_coords(bp, c, s)
        tc = time.perf_counter() - t0
        t0 = time.perf_counter(); P = np.eye(cf.D) + Z.T @ Z / 1e-4; cho_factor(P); tu = time.perf_counter() - t0
        sc.append({'m': m, 'coords_s': tc, 'update_s': tu, 'dense_gp_bytes': 8 * m * m})
        print('  scaling m %d coords %.1fs update %.2fs (dense K would need %.1f GB)' % (m, tc, tu, 8 * m * m / 1e9), flush=True)
    data['cf_scaling_H075_n512'] = sc
    with open(os.path.join(RES, out), 'w') as fh:
        json.dump(data, fh, indent=1)
    return data


if __name__ == '__main__':
    run()
