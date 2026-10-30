"""Shared builder for borehole design and inference problems on top of the general expansion.

Prior: surface-temperature history dT(s) = sigma_T B_H((s - t0)/L) on the window [t0, t1], nuisance beta = (T0, Gamma)
(surface intercept in K, geothermal gradient in K/m) with an independent Gaussian prior, white measurement noise.
Every reading and every target is a Wiener integral; all of them go through ONE prepared CovarianceFourier (n, Naux)
and the paper's per-input rule (endpoint-corrected Fourier stage + projection).  Candidate readings use N_cand
midpoint samples; targets with interior square-root cusps (heat content) use the larger N_targ, following the
refinement rule of Algorithm 1 (increase N while the result changes).  Both projectors share the same basis.
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from covfourier.engine import CovarianceFourier          # noqa: E402
from covfourier.batch import BatchProjector              # noqa: E402
from covfourier.design import InfoDesign                 # noqa: E402
from covfourier import heat                              # noqa: E402

EPOCH_T1 = ((1850.0, 1900.0), (1961.0, 1990.0))          # 20th-century warming
EPOCH_T2 = ((1961.0, 1990.0), (1991.0, 2020.0))          # recent warming
HEAT_T3 = (1960.0, 2020.0)                               # ground heat gain, 1e8 J/m^2


def target_values(u, win):
    """Rows: T1 = mean(1961-90) - mean(1850-1900); T2 = mean(1991-2020) - mean(1961-90); T3 = Q(2020) - Q(1960)."""
    t1 = heat.epoch_mean_values(*EPOCH_T1[1], u, win) - heat.epoch_mean_values(*EPOCH_T1[0], u, win)
    t2 = heat.epoch_mean_values(*EPOCH_T2[1], u, win) - heat.epoch_mean_values(*EPOCH_T2[0], u, win)
    t3 = (heat.heat_content_values(HEAT_T3[1], u, win) - heat.heat_content_values(HEAT_T3[0], u, win)) / 1e8
    return np.vstack([t1, t2, t3])


TARGET_NAMES = ['T1: mean 1961-90 minus 1850-1900 (K)', 'T2: mean 1991-2020 minus 1961-90 (K)',
                'T3: ground heat gain 1960-2020 (1e8 J/m^2)']


class Prepared:
    """One basis (H, n, Naux) and two projectors (candidates, targets) on the same basis."""

    def __init__(self, H, n, N_cand=8192, N_targ=1 << 18, win=None):
        t0 = time.perf_counter()
        self.H, self.n, self.win = H, n, win or heat.Window()
        self.cf = CovarianceFourier(H, n, enrich=True, Naux=max(N_targ, N_cand))
        self.bp_c = BatchProjector(self.cf, N=N_cand)
        self.bp_t = BatchProjector(self.cf, N=N_targ, dense_rows_limit=2e6)   # FFT products: no n x N_targ block
        self.N_cand, self.N_targ = N_cand, N_targ
        self.prep_s = time.perf_counter() - t0

    @property
    def D(self):
        return self.n + 2

    def readings(self, z, t_log, chunk=300):
        """Coordinates (m, D) and conservative residual energies (m,) of readings at depths z (m), times t_log (yr)."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        t_log = np.broadcast_to(np.asarray(t_log, dtype=float), z.shape)
        u = heat.midpoints(self.N_cand)
        Zs, Es = [], []
        for i0 in range(0, len(z), chunk):
            F = heat.borehole_values(z[i0:i0 + chunk], t_log[i0:i0 + chunk], u, self.win)
            Z, E2, _ = heat.project_samples(self.bp_c, F)
            Zs.append(Z); Es.append(E2)
        return np.vstack(Zs), np.concatenate(Es)

    def targets(self):
        F = target_values(heat.midpoints(self.N_targ), self.win)
        Z, E2, _ = heat.project_samples(self.bp_t, F)
        return Z, E2


def design_problem(prep, z_exist, t_exist, z_cand, t_cand, sigma_T=1.0, noise_sd=0.02, sb=(10.0, 0.05),
                   omega='posterior'):
    """InfoDesign over candidates with the existing log folded into the prior precision, plus a second
    InfoDesign over [existing; candidates] for brackets of E u S.  Returns dict."""
    t0 = time.perf_counter()
    Ze, Ee = prep.readings(z_exist, t_exist)
    t_ex = time.perf_counter() - t0
    t0 = time.perf_counter()
    Zc, Ec = prep.readings(z_cand, t_cand)
    t_cand_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    Zt, Et = prep.targets()
    t_targ = time.perf_counter() - t0
    D = prep.D
    def rows(Z, z):
        return np.hstack([sigma_T * Z, np.ones((len(z), 1)), np.asarray(z, float)[:, None]])
    Ae, Ac = rows(Ze, z_exist), rows(Zc, z_cand)
    C = np.vstack([sigma_T * Zt.T, np.zeros((2, Zt.shape[0]))])
    Pi = np.eye(D + 2); Pi[D, D] = 1.0 / sb[0] ** 2; Pi[D + 1, D + 1] = 1.0 / sb[1] ** 2
    s2e = np.full(len(z_exist), noise_sd ** 2); s2c = np.full(len(z_cand), noise_sd ** 2)
    P_exist = Pi + (Ae.T / s2e) @ Ae
    full = InfoDesign(np.vstack([Ae, Ac]), np.r_[s2e, s2c], C, Pi=Pi,
                      e2=sigma_T ** 2 * np.r_[Ee, Ec], e2_t=sigma_T ** 2 * Et)
    v_exist = full.V0(np.arange(len(z_exist)))
    w = 1.0 / v_exist if omega == 'posterior' else np.ones(len(v_exist))
    full.omega = w
    cand = InfoDesign(Ac, s2c, C, omega=w, Pi=P_exist, e2=sigma_T ** 2 * Ec, e2_t=sigma_T ** 2 * Et)
    return {'cand': cand, 'full': full, 'n_exist': len(z_exist), 'omega': w, 'V_exist_D': v_exist,
            'E2_exist': Ee, 'E2_cand': Ec, 'E2_targ': Et,
            'timing': {'existing_s': t_ex, 'candidates_s': t_cand_s, 'targets_s': t_targ, 'prep_s': prep.prep_s}}


def candidate_grid(dz=1.0, dt=0.25, z0=20.0, z1=600.0, ta=2021.0, tb=2030.0):
    zs = np.arange(z0, z1 + 1e-9, dz)
    ts = np.arange(ta, tb + 1e-9, dt)
    Zg, Tg = np.meshgrid(zs, ts, indexing='ij')
    return Zg.ravel(), Tg.ravel()
