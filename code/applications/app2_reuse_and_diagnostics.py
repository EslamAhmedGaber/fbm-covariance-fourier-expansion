"""A2 supplement: residual-energy diagnostic (Proposition 5), latent-path reconstruction and incremental reuse.

python app2_reuse_and_diagnostics.py [diagnostic|reconstruction|incremental|all]

  diagnostic      H = 0.75, m = 2000 window sensors, n in {16, 32, 64, 128, 256}, tau in {0.003, 0.01, 0.03}; 200 data
                  sets drawn from the exact model.  Compares exact and working-model posterior means of 25 linear targets
                  with Proposition 5: KL <= rho^2 (1 + rho) / 4 and the root-mean-square mean bounds.
                  Writes results/A2_residual_diagnostic.json.
  reconstruction  H = 0.75, m = 500 sensors, tau = 0.01, n = 256; 200 latent paths (Davies-Harte on a 2^14 grid, sensors by
                  trapezoidal quadrature of the window integrals); posterior of B_H(t) at t = k/256 with pointwise 95% bands;
                  coverage and agreement with the exact GP.  Writes results/A2_reconstruction.json.
  incremental     H = 0.75, n = 256: add 2000 sensors and 25 targets to an 8000-sensor posterior.  The expansion reuses its
                  basis (new coordinates, rank update of the D x D precision); the exact GP extends its Cholesky factor by
                  blocks.  Minimum of 3 repetitions.  Writes results/A2_incremental_reuse.json.
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import app2_noisy_reconstruction_v2 as A                                   # noqa: E402
from covfourier.engine import CovarianceFourier                            # noqa: E402
from covfourier.batch import BatchProjector                                # noqa: E402
from covfourier.comparators import DaviesHarte                             # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
H = 0.75


def kernel_block(mabs, mi_i, ci, si, mi_j, cj, sj):
    """Exact covariance of window sensors (closed form through E|N(mu, s^2)|^{2H})."""
    return 0.5 * (mi_i[:, None] + mi_j[None, :] - mabs(ci[:, None] - cj[None, :], np.sqrt(si[:, None] ** 2 + sj[None, :] ** 2)))


def model_maps(Gam, E2s, tau, Wt):
    """Working model: noise tau^2 + E_i^2 (diagonal residual model).  Returns the map y -> posterior means of Wt Z and the
    posterior variances of Wt Z (without the target residual energies)."""
    noise = tau ** 2 + E2s
    Zs = Gam / np.sqrt(noise)[:, None]
    P = np.eye(Gam.shape[1]) + Zs.T @ Zs
    Lp = cho_factor(P, lower=True, check_finite=False)
    Aw = cho_solve(Lp, Wt.T).T
    return (Aw @ Zs.T) / np.sqrt(noise)[None, :], np.einsum('ij,ij->i', Aw, Wt)


def diagnostic(ns=(16, 32, 64, 128, 256), taus=(0.003, 0.01, 0.03), m=2000, R=200, seed=20261030):
    rng = np.random.default_rng(seed)
    c, s = A.draw_windows(rng, m + A.N_NEW)
    mabs = A.FastAbsMoment(2 * H)
    K, Kt, vt = A.gp_covariances(H, c, s, m, mabs)
    nt = Kt.shape[0]; ind = [2, 3, 4]; smooth = [i for i in range(nt) if i not in ind]
    cf_data = {}
    for n in ns:
        cf = CovarianceFourier(H, n, enrich=True, Naux=8 * n); bp = BatchProjector(cf)
        Gam, E2s = A.cf_coords(bp, c[:m], s[:m]); Wt, E2t = A.cf_targets(bp, c[m:], s[m:])
        cf_data[n] = (Gam, E2s, Wt, E2t)
    out = {'note': __doc__.split('\n\n')[0], 'H': H, 'm': m, 'R': R, 'targets_smooth': len(smooth), 'targets_indicator': len(ind), 'rows': []}
    for tau in taus:
        L = cholesky(K + tau ** 2 * np.eye(m), lower=True, check_finite=False)
        Y = L @ rng.standard_normal((m, R))
        alpha = cho_solve((L, True), Y, check_finite=False)
        m_exact = Kt @ alpha
        for n in ns:
            Gam, E2s, Wt, E2t = cf_data[n]
            rho = float(E2s.sum() / tau ** 2)
            Rres = K - Gam @ Gam.T
            sq = 1.0 / np.sqrt(tau ** 2 + E2s)
            mu = np.linalg.eigvalsh(sq[:, None] * (Rres - np.diag(E2s)) * sq[None, :])
            kl = float(0.5 * np.sum(mu - np.log1p(mu)))
            meanmap, sQ2 = model_maps(Gam, E2s, tau, Wt)
            m_model = meanmap @ Y
            m_proj = Wt @ (Gam.T @ alpha)
            sQ = np.sqrt(sQ2)
            rms_p = np.sqrt(np.mean((m_proj - m_model) ** 2, axis=1)) / sQ
            rms_f = np.sqrt(np.mean((m_exact - m_model) ** 2, axis=1)) / sQ
            b_p = rho * math.sqrt((1 + rho) / 2)
            b_f = b_p + np.sqrt(E2t) / sQ
            row = {'n': n, 'D': int(Gam.shape[1]), 'tau': tau, 'rho': rho, 'KL': kl, 'KL_bound': rho ** 2 * (1 + rho) / 4,
                   'min_eig_R_over_tau2': float(np.linalg.eigvalsh(Rres)[0] / tau ** 2),
                   'max_rel_diag_R_vs_E2': float(np.max(np.abs(np.diag(Rres) - E2s) / E2s)),
                   'bound_projected': b_p,
                   'rms_projected_smooth_max': float(rms_p[smooth].max()), 'rms_projected_indicator_max': float(rms_p[ind].max()),
                   'rms_full_smooth_max': float(rms_f[smooth].max()), 'rms_full_indicator_max': float(rms_f[ind].max()),
                   'ratio_projected_max': float(rms_p.max() / b_p) if b_p > 0 else None,
                   'ratio_full_max': float(np.max(rms_f / b_f)),
                   'eD_over_sQ_smooth_max': float((np.sqrt(E2t) / sQ)[smooth].max()), 'eD_over_sQ_indicator_max': float((np.sqrt(E2t) / sQ)[ind].max())}
            out['rows'].append(row)
            print('tau %.3f n %3d rho %.2e KL %.2e (bound %.2e) | rms proj smooth %.1e ind %.1e (bound %.1e) | full smooth %.1e ind %.1e | ratios %.2f %.2f'
                  % (tau, n, rho, kl, row['KL_bound'], row['rms_projected_smooth_max'], row['rms_projected_indicator_max'], b_p,
                     row['rms_full_smooth_max'], row['rms_full_indicator_max'], row['ratio_projected_max'] or 0, row['ratio_full_max']), flush=True)
    with open(os.path.join(RES, 'A2_residual_diagnostic.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    return out


def reconstruction(m=500, tau=0.01, n=256, R=200, NG=2 ** 14, KT=256, seed=20261101):
    rng = np.random.default_rng(seed)
    c, s = A.draw_windows(rng, m)
    p = 2 * H; mabs = A.FastAbsMoment(p); mi = mabs(c, s)
    tg = np.arange(1, KT + 1) / KT
    K = kernel_block(mabs, mi, c, s, mi, c, s)
    Kt = 0.5 * (tg[:, None] ** p + mi[None, :] - mabs(c[None, :] - tg[:, None], np.broadcast_to(s[None, :], (KT, m))))
    vt = tg ** p
    Lg = cho_factor(K + tau ** 2 * np.eye(m), lower=True, check_finite=False)
    Agp = cho_solve(Lg, Kt.T).T
    var_gp = vt - np.einsum('ij,ij->i', Agp, Kt)
    cf = CovarianceFourier(H, n, enrich=True, Naux=8 * n); bp = BatchProjector(cf)
    Gam, E2s = A.cf_coords(bp, c, s)
    t = (np.arange(bp.N) + 0.5) / bp.N
    r = bp.project(*bp.fourier_stage((t[None, :] < tg[:, None]).astype(float)), norms=True)
    Wt = r['coords'].T; E2t = np.clip(r['error2'], 0.0, None) + r['eta']
    meanmap, sQ2 = model_maps(Gam, E2s, tau, Wt)
    var_cf = sQ2 + E2t
    dh = DaviesHarte(H, NG)
    B = np.cumsum(dh.increments(R, rng), axis=1)                         # B(j / NG), j = 1..NG
    tgrid = np.arange(1, NG + 1) / NG
    w = np.full(NG, 1.0 / NG); w[-1] *= 0.5                               # trapezoid on [0, 1], B(0) = 0
    Phi = norm.pdf(tgrid[None, :], c[:, None], s[:, None]) * w[None, :]
    Y = Phi @ B.T + tau * rng.standard_normal((m, R))
    M_cf = meanmap @ Y; M_gp = Agp @ Y
    Bt = B[:, (np.arange(1, KT + 1) * (NG // KT)) - 1].T                  # latent values at t_k
    cov_cf = np.abs(Bt - M_cf) <= 1.959964 * np.sqrt(var_cf)[:, None]
    cov_gp = np.abs(Bt - M_gp) <= 1.959964 * np.sqrt(var_gp)[:, None]
    inner = (tg >= 0.15) & (tg <= 0.85)
    out = {'note': 'latent-path reconstruction, H = 0.75', 'm': m, 'tau': tau, 'n': n, 'D': int(Gam.shape[1]), 'R': R, 'NG': NG,
           'coverage_cf': float(cov_cf.mean()), 'coverage_gp': float(cov_gp.mean()),
           'coverage_cf_per_t_range': [float(cov_cf.mean(axis=1).min()), float(cov_cf.mean(axis=1).max())],
           'coverage_cf_inner': float(cov_cf[inner].mean()), 'coverage_cf_outer': float(cov_cf[~inner].mean()),
           'max_mean_diff_cf_gp_over_sd_gp': float(np.max(np.abs(M_cf - M_gp) / np.sqrt(var_gp)[:, None])),
           'max_rel_var_diff_cf_gp': float(np.max(np.abs(var_cf / var_gp - 1))),
           'rms_error_over_sd_cf': float(np.sqrt(np.mean(((Bt - M_cf) / np.sqrt(var_cf)[:, None]) ** 2))),
           'sd_range_inner': [float(np.sqrt(var_cf[inner]).min()), float(np.sqrt(var_cf[inner]).max())],
           'sd_at_t1': float(np.sqrt(var_cf[-1])), 'sum_E2_over_tau2': float(E2s.sum() / tau ** 2),
           'example': {'t': tg.tolist(), 'latent': Bt[:, 0].tolist(), 'mean_cf': M_cf[:, 0].tolist(), 'sd_cf': np.sqrt(var_cf).tolist(),
                       'mean_gp': M_gp[:, 0].tolist(), 'latent_fine_t': tgrid[::16].tolist(), 'latent_fine': B[0, ::16].tolist(),
                       'centres': c.tolist(), 'widths': s.tolist()}}
    print({k: v for k, v in out.items() if k != 'example'}, flush=True)
    with open(os.path.join(RES, 'A2_reconstruction.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    return out


def incremental(m0=8000, dm=2000, tau=0.01, n=256, reps=3, seed=20261102):
    rng = np.random.default_rng(seed)
    c, s = A.draw_windows(rng, m0 + dm + A.N_NEW)
    mabs = A.FastAbsMoment(2 * H); mi = mabs(c, s); m1 = m0 + dm
    o, a_, tq = slice(0, m0), slice(m0, m1), slice(m1, m1 + A.N_NEW)
    T = {k: [] for k in ('cf_prep', 'cf_coords0', 'cf_update0', 'cf_coords_add', 'cf_update_add', 'cf_targets', 'cf_means',
                         'gp_build0', 'gp_factor0', 'gp_build_add', 'gp_solve_add', 'gp_factor_add', 'gp_targets', 'gp_means')}
    z = rng.standard_normal(m1)
    for rep in range(reps):
        t0 = time.perf_counter(); cf = CovarianceFourier(H, n, enrich=True, Naux=8 * n); bp = BatchProjector(cf); T['cf_prep'].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); Z0, E0 = A.cf_coords(bp, c[o], s[o]); T['cf_coords0'].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); Zs0 = Z0 / np.sqrt(tau ** 2 + E0)[:, None]; P = np.eye(Z0.shape[1]) + Zs0.T @ Zs0
        Lp = cho_factor(P, lower=True, check_finite=False); T['cf_update0'].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); Z1, E1 = A.cf_coords(bp, c[a_], s[a_]); T['cf_coords_add'].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); Zs1 = Z1 / np.sqrt(tau ** 2 + E1)[:, None]; P = P + Zs1.T @ Zs1
        Lp = cho_factor(P, lower=True, check_finite=False); T['cf_update_add'].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); Wt, E2t = A.cf_targets(bp, c[tq], s[tq]); T['cf_targets'].append(time.perf_counter() - t0)
        # GP: initial factor, then block extension
        t0 = time.perf_counter(); K00 = np.empty((m0, m0))
        for i0 in range(0, m0, 500):
            i1 = min(m0, i0 + 500); K00[i0:i1] = kernel_block(mabs, mi[i0:i1], c[i0:i1], s[i0:i1], mi[o], c[o], s[o])
        T['gp_build0'].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); K00.flat[::m0 + 1] += tau ** 2; L0 = cholesky(K00, lower=True, overwrite_a=True, check_finite=False)
        T['gp_factor0'].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); K01 = kernel_block(mabs, mi[o], c[o], s[o], mi[a_], c[a_], s[a_]); K11 = kernel_block(mabs, mi[a_], c[a_], s[a_], mi[a_], c[a_], s[a_])
        T['gp_build_add'].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); L21T = solve_triangular(L0, K01, lower=True, check_finite=False); T['gp_solve_add'].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); S = K11 + tau ** 2 * np.eye(dm) - L21T.T @ L21T; L11 = cholesky(S, lower=True, check_finite=False)
        T['gp_factor_add'].append(time.perf_counter() - t0)
        # data from the exact model with the extended factor, targets and posterior means
        y = np.r_[L0 @ z[:m0], L21T.T @ z[:m0] + L11 @ z[m0:]]
        t0 = time.perf_counter(); xi = cho_solve(Lp, Zs0.T @ (y[o] / np.sqrt(tau ** 2 + E0)) + Zs1.T @ (y[a_] / np.sqrt(tau ** 2 + E1)))
        mean_cf = Wt @ xi; var_cf = np.einsum('ij,ji->i', Wt, cho_solve(Lp, Wt.T)) + E2t; T['cf_means'].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        mp1 = A.FastAbsMoment(2 * H + 1)
        rows = [0.5 * (mi[:m1] + 1.0 - mabs(1.0 - c[:m1], s[:m1])),
                0.5 * (mi[:m1] + 1 / (2 * H + 1) - (mp1(c[:m1], s[:m1]) + mp1(1 - c[:m1], s[:m1])) / (2 * H + 1))]
        var = [1.0, 1.0 / (2 * H + 2)]
        for ts in A.TSTARS:
            rows.append(0.5 * (mi[:m1] + ts ** (2 * H) - mabs(c[:m1] - ts, s[:m1]))); var.append(ts ** (2 * H))
        Kn = kernel_block(mabs, mi[tq], c[tq], s[tq], mi[:m1], c[:m1], s[:m1])
        Kt = np.vstack(rows + [Kn]); vt = np.r_[np.array(var), 0.5 * (2 * mi[tq] - mabs(np.zeros(A.N_NEW), np.sqrt(2) * s[tq]))]
        T['gp_targets'].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        u0 = solve_triangular(L0, y[o], lower=True, check_finite=False); u1 = solve_triangular(L11, y[a_] - L21T.T @ u0, lower=True, check_finite=False)
        V0 = solve_triangular(L0, Kt[:, o].T, lower=True, check_finite=False); V1 = solve_triangular(L11, Kt[:, a_].T - L21T.T @ V0, lower=True, check_finite=False)
        mean_gp = V0.T @ u0 + V1.T @ u1; var_gp = vt - np.sum(V0 ** 2, axis=0) - np.sum(V1 ** 2, axis=0)
        T['gp_means'].append(time.perf_counter() - t0)
        del K00, L0, K01, L21T
    tmin = {k: float(min(v)) for k, v in T.items()}
    sd = np.sqrt(var_gp)
    out = {'note': 'incremental reuse, H = 0.75', 'm0': m0, 'added_sensors': dm, 'targets': int(Wt.shape[0]), 'n': n, 'D': int(Z0.shape[1]),
           'tau': tau, 'reps': reps, 'times_min_s': tmin,
           'cf_add_sensors_s': tmin['cf_coords_add'] + tmin['cf_update_add'],
           'gp_add_sensors_s': tmin['gp_build_add'] + tmin['gp_solve_add'] + tmin['gp_factor_add'],
           'cf_targets_s': tmin['cf_targets'] + tmin['cf_means'], 'gp_targets_s': tmin['gp_targets'] + tmin['gp_means'],
           'cf_initial_s': tmin['cf_prep'] + tmin['cf_coords0'] + tmin['cf_update0'], 'gp_initial_s': tmin['gp_build0'] + tmin['gp_factor0'],
           'max_mean_diff_over_sd': float(np.max(np.abs(mean_cf - mean_gp) / sd)),
           'max_mean_diff_over_sd_smooth': float(np.max(np.delete(np.abs(mean_cf - mean_gp) / sd, [2, 3, 4]))),
           'max_rel_var_diff_smooth': float(np.max(np.delete(np.abs(var_cf / var_gp - 1), [2, 3, 4])))}
    print(json.dumps({k: v for k, v in out.items() if k != 'times_min_s'}, indent=1), json.dumps(tmin), flush=True)
    with open(os.path.join(RES, 'A2_incremental_reuse.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if part in ('diagnostic', 'all'):
        diagnostic()
    if part in ('reconstruction', 'all'):
        reconstruction()
    if part in ('incremental', 'all'):
        incremental()
