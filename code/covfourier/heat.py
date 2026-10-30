"""Heat-conduction response kernels: subsurface temperatures as Wiener integrals of a surface-temperature history.

Physics (homogeneous half-space z >= 0, diffusivity kappa; Carslaw & Jaeger 1959, Sec. 2.4).  If the ground surface
temperature is T0 + dT(s) with dT = 0 before the window start t0, the subsurface temperature at depth z and time t is
    T(z, t) = T0 + Gamma z + int_{t0}^{t} erfc( z / (2 sqrt(kappa (t - s))) ) d dT(s),
with Gamma the steady geothermal gradient.  With the prior dT(s) = sigma_T B_H(u), u = (s - t0)/L on [t0, t0 + L],
    T(z, t) - T0 - Gamma z = sigma_T I_H(f_{z,t}),   f_{z,t}(u) = erfc( z / (2 sqrt(kappa L (u_t - u))) ) 1{u < u_t}.
For z > 0, f_{z,t} vanishes to all orders as u -> u_t, so it is C^infinity on [0, 1] (no endpoint class needed).
Targets used here, as integrands on [0, 1]:
    epoch mean of dT over [A, B]:   (1/(b - a)) (b - max(u, a))_+          (piecewise linear; kinks at a, b)
    ground heat content at time t:  rho_c * 2 sqrt(kappa L (u_t - u) / pi) 1{u < u_t}   (J m^-2 per unit sigma_T)
(the depth integral of the erfc kernel is 2 sqrt(kappa tau / pi)).  Units: years, metres, kelvin.
"""
import math

import numpy as np
from scipy.special import erfc

KAPPA_M2_PER_YR = 1.0e-6 * 365.25 * 86400.0       # 1e-6 m^2/s, the conventional value in borehole climatology
RHO_C = 2.5e6                                      # J m^-3 K^-1 (k = 2.5 W/m/K with kappa = 1e-6 m^2/s)


class Window:
    """Reconstruction window [t0, t1] in years; rescaled time u = (t - t0)/L."""

    def __init__(self, t0=1500.0, t1=2030.0):
        self.t0, self.t1 = float(t0), float(t1)
        self.L = self.t1 - self.t0

    def u(self, t):
        return (np.asarray(t, dtype=float) - self.t0) / self.L


def borehole_values(z, t_log, u, win, kappa=KAPPA_M2_PER_YR):
    """Integrand values f_{z_i, t_i}(u_j), shape (len(z), len(u)).  z in m, t_log in years."""
    z = np.atleast_1d(np.asarray(z, dtype=float))[:, None]
    uL = np.atleast_1d(win.u(t_log)).astype(float)
    uL = np.broadcast_to(uL, z.shape[:1])[:, None]
    tau = (uL - np.asarray(u, dtype=float)[None, :]) * win.L        # years before logging
    out = np.zeros(np.broadcast_shapes(z.shape, tau.shape))
    pos = tau > 0
    arg = np.where(pos, z / (2.0 * np.sqrt(kappa * np.where(pos, tau, 1.0))), np.inf)
    out[pos] = erfc(arg[pos])
    return out


def epoch_mean_values(A, B, u, win):
    """Integrand of the mean of dT over [A, B] (years), per unit sigma_T."""
    a, b = float(win.u(A)), float(win.u(B))
    u = np.asarray(u, dtype=float)
    return np.clip(b - np.maximum(u, a), 0.0, None) / (b - a)


def heat_content_values(t, u, win, kappa=KAPPA_M2_PER_YR, rho_c=RHO_C):
    """Integrand of the transient ground heat content at time t (J m^-2 per unit sigma_T)."""
    tau = (float(win.u(t)) - np.asarray(u, dtype=float)) * win.L
    return rho_c * 2.0 * np.sqrt(np.clip(tau, 0.0, None) * kappa / math.pi)


def features(win, t_logs=(), epochs=(), heat_times=()):
    """Feature points (rescaled) for the independent quadrature reference."""
    f = [float(win.u(t)) for t in t_logs] + [float(win.u(t)) for e in epochs for t in e] + \
        [float(win.u(t)) for t in heat_times]
    return sorted(set(x for x in f if 0.0 < x < 1.0))


def project_samples(bp, F, explicit=False):
    """The paper's per-input rule applied to sampled integrands (identical to from_callables):
    endpoint-corrected Fourier stage, then projection.  bp: BatchProjector or LargeProjector.
    Returns coordinates (q, D), conservative residual energies (q,) = max(E2, 0) + eta, and the raw output."""
    a_g, c = bp.fourier_stage(F)
    if hasattr(bp, 'explicit_residual'):
        r = bp.project(a_g, c, explicit=explicit)
    else:
        r = bp.project(a_g, c, norms=True)
    E2 = np.clip(r['error2'], 0.0, None) + r['eta']                  # subtractive energy plus rounding allowance
    if explicit and 'E2_aux_explicit' in r:
        # explicit residual inside the auxiliary space plus the declared polynomial-tail bound beyond N
        E2 = (np.sqrt(np.clip(r['E2_aux_explicit'], 0.0, None)) + np.sqrt(r['tau2_poly_tail_bound'])) ** 2
    return r['coords'].T, E2, r


def midpoints(N):
    return (np.arange(N) + 0.5) / N
