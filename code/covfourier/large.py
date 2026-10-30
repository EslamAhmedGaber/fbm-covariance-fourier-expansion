"""Matrix-free projector for any n: endpoint-corrected Fourier stage + white coordinates.

Same mathematics as covfourier.batch.BatchProjector, but no dense factorization. The cosine
white coordinates come from covfourier.solvers.SchurCosine (proven spectrum [1/2,1]; degree-20
Chebyshev for x^{-1/2}). The two enrichment directions w~_j are the cosine-truncated residuals of
t/T and (t/T)^2 against V_n. Since w~_j is H-orthogonal to V_n,
rho_j = <f, w~_j> = (G dw_j)^T a_g + dw_j^T (Q c), so no solve with G_n is needed per input.
Cost per batch of q inputs: one FFT mat-vec of length N, plus about 20 of length n.
The procedure is identical for every input; nothing depends on what the input is.

Error sources, reported separately:
  (i)   sampling / coefficient approximation (DCT of N midpoints; endpoint correction);
  (ii)  finite auxiliary space (N cosines; truncated enrichment directions);
  (iii) linear algebra (Chebyshev and PCG tolerances, reported);
  (iv)  floating point.
"""
import math
import numpy as np
from scipy.fft import dct
from scipy.linalg import cholesky, solve_triangular

from .gram import CosineGram, power_cosine_moments, poly_cos_moments, poly_poly
from .batch import gram_matmat, _ParityKernel, _endpoint_weights, endpoint_corrected_dct
from .solvers import SchurCosine


class LargeProjector:
    def __init__(self, H, n, N, T=1.0, deg=8, reorth=True):
        # reorth=False skips the re-orthogonalization solve (ablation only; see studies/s2_reorthogonalization_ablation.py)
        H, T, n, N = float(H), float(T), int(n), int(N)
        if not (0.5 <= H < 1.0) or not (T > 0):
            raise ValueError('need 1/2 <= H < 1, T > 0')
        if n < 2 or N < 4 * n:
            raise ValueError('need n >= 2 and N >= 4n')
        self.H, self.T, self.n, self.N = H, T, n, N
        self.G = CosineGram(H, T); self.G.m(N)
        self.kN = _ParityKernel(N)
        self.sc = SchurCosine(self.G, n)
        aq = np.stack([power_cosine_moments(float(j), N, T) / T ** j for j in (1, 2)], axis=1)
        Gaq = gram_matmat(self.G, aq, self.kN)
        X = np.empty((n, 2)); self.enrich_solve_info = []
        for j in range(2):
            X[:, j], info = self.sc.solve(Gaq[:n, j]); self.enrich_solve_info.append(info)
        d = aq.copy(); d[:n] -= X
        Gd = gram_matmat(self.G, d, self.kN)
        # one re-orthogonalization step: the PCG residual (G d)[:n] is amplified by the normalization
        # R^{-1} ~ n^{1+H}; a second solve removes it to second order
        if reorth:
            for j in range(2):
                dx, info = self.sc.solve(Gd[:n, j]); self.enrich_solve_info.append(info)
                d[:n, j] -= dx
            Gd = gram_matmat(self.G, d, self.kN)
        S = 0.5 * (d.T @ Gd + (d.T @ Gd).T)
        Rt = cholesky(S, lower=False)
        Ri = solve_triangular(Rt, np.eye(2), lower=False)
        self.dw, self.Gdw = d @ Ri, Gd @ Ri
        self.orthogonality_defect = float(np.max(np.abs(self.Gdw[:n])))   # should be ~ solver tolerance
        self.QN = np.stack([poly_cos_moments(H, j, N, T) / T ** j for j in (1, 2)], axis=1)
        self.P = np.array([[poly_poly(H, i, j, T) / T ** (i + j) for j in (1, 2)] for i in (1, 2)])
        self.w = _endpoint_weights(deg); self.deg = deg

    @property
    def D(self):
        return self.n + 2

    def fourier_stage(self, F, max_ratio=1e5):
        # cap 1e5: the explicit residual (explicit_residual) has no subtractive cancellation, so the
        # 1e3 cap that protects BatchProjector's subtractive residual is not needed here; measured on
        # sin(1000 pi t) (D = 4097, 16385): cap 1e3 -> E2 1.1-11x too large, cap 1e5 -> ratio 1.00 to the moment route.
        F = np.atleast_2d(np.asarray(F, dtype=float))
        if F.shape[1] != self.N:
            raise ValueError('sample count must equal N')
        if not np.all(np.isfinite(F)):
            raise ValueError('non-finite samples')
        a, c, use = endpoint_corrected_dct(F, self.T, self.w, self.deg, max_ratio)
        self.last_corrected = use
        return a, c

    def project(self, a_g, c, explicit=False):
        n = self.n
        Ga = gram_matmat(self.G, a_g, self.kN)
        Qc = self.QN @ c
        b = Ga[:n] + Qc[:n]
        Z = self.sc.white(b)
        rho = self.Gdw.T @ a_g + self.dw.T @ Qc
        coords = np.vstack([Z, rho])
        t1 = np.einsum('kq,kq->q', a_g, Ga); t2 = 2 * np.einsum('kq,kq->q', a_g, Qc)
        t3 = np.einsum('iq,ij,jq->q', c, self.P, c)
        energy = np.sum(coords ** 2, axis=0)
        nrm = t1 + t2 + t3
        out = {'coords': coords, 'energy': energy, 'norm2': nrm, 'error2': nrm - energy,
               'eta': 64 * np.finfo(float).eps * (np.abs(t1) + np.abs(t2) + np.abs(t3) + energy)}
        out['resolved'] = out['error2'] > out['eta']             # subtractive energies at or below eta are unresolved
        if explicit:
            out.update(self.explicit_residual(a_g, c))
        return out

    def explicit_residual(self, a_g, c):
        """Residual energy without the subtractive floor, inside the auxiliary space (N cosines).

        a_fN = a_g + a_p,N (polynomial part truncated at N), projected onto V_n + span(w~):
        e = a_fN - [x; 0] - dw rho_N,  E2_aux = e^T G e  (no cancellation).
        Also returns a rigorous upper bound tau2 for the H-norm of the polynomial tail beyond N
        (upper weighted cosine bound applied to sum_{k>=N} a_k(p) h_k, whose h_0 coefficient is zero).
        E2_aux omits the input's own tail beyond N; |E2_true - E2_aux| <= 2 sqrt(E2_aux tau2) + tau2
        when the input's tail beyond N equals the polynomial tail (g-tail neglected; declared)."""
        from .gram import power_cosine_moments
        n, N = self.n, self.N
        aq = np.stack([power_cosine_moments(float(j), N, self.T) / self.T ** j for j in (1, 2)], axis=1)
        afN = a_g + aq @ c
        GafN = gram_matmat(self.G, afN, self.kN)
        q = afN.shape[1]
        X = np.empty((n, q)); infos = []
        for j in range(q):
            X[:, j], info = self.sc.solve(GafN[:n, j]); infos.append(info['iterations'])
        rhoN = self.Gdw.T @ afN
        e = afN - self.dw @ rhoN; e[:n] -= X
        Ge = gram_matmat(self.G, e, self.kN)
        E2aux = np.einsum('kq,kq->q', e, Ge)
        # tail bound for p beyond N: |a_k(t/T)| <= 2 sqrt(2T)/(k pi)^2, |a_k((t/T)^2)| <= 2 sqrt(2T)/(k pi)^2 (T = 1 scaling)
        cH = math.gamma(2 * self.H + 1) * math.sin(math.pi * self.H)
        s_ = 3 + 2 * self.H
        amp = 2 * math.sqrt(2 * self.T) * (np.abs(c[0]) + np.abs(c[1]))      # |a_k(p)| <= amp / (k pi)^2 (k >= 1)
        zeta_tail = N ** (-s_) + N ** (1 - s_) / (s_ - 1)
        tau2 = cH * amp ** 2 * (math.pi / self.T) ** (1 - 2 * self.H) * math.pi ** (-4) * zeta_tail
        return {'E2_aux_explicit': E2aux, 'tau2_poly_tail_bound': tau2, 'explicit_pcg_iterations': infos}

    def from_callables(self, fs, endpoint=None, explicit=False, local=None, M=64, q=8, over=8):
        """One documented entry point for point-evaluable inputs on [0, T].

        endpoint=None (default): N midpoint samples, discrete cosine transform and the a posteriori endpoint
            correction of fourier_stage; for inputs that are bounded near 0 and T (jumps and kinks allowed).
        endpoint='graded': for inputs declared to satisfy the algebraic endpoint assumption of the paper,
            |f^(m)(t)| <= C dist(t, {0, T})^(-alpha - m) for m <= 2 with alpha < H: classical coefficients from
            covfourier.endpoint.graded_cosine_coefficients (M graded end cells, q-point rules, over-times
            oversampling) and no endpoint polynomial.  local[i] = (f_left_local, f_right_local) optionally evaluates
            input i in endpoint-local coordinates (distance to the endpoint), which avoids the rounding of T - s.
        The route is a declared regularity class; it is never inferred from the input or keyed to a function name.
        Returns the dict of project() (coords, energy, norm2, error2, eta, resolved); explicit=True adds the
        explicit auxiliary residual E2_aux_explicit and the polynomial tail bound.
        """
        if endpoint is None:
            t = self.T * (np.arange(self.N) + 0.5) / self.N
            F = np.stack([np.broadcast_to(np.asarray(f(t), dtype=float), t.shape) for f in fs])
            return self.project(*self.fourier_stage(F), explicit=explicit)
        if endpoint != 'graded':
            raise ValueError("endpoint must be None or 'graded'")
        from .endpoint import graded_cosine_coefficients
        local = list(local) if local is not None else [None] * len(fs)
        cols = []
        for f, loc in zip(fs, local):
            fl, fr = loc if loc is not None else (None, None)
            cols.append(graded_cosine_coefficients(f, self.N, self.T, M=M, q=q, over=over, f_left_local=fl, f_right_local=fr))
        A = np.stack(cols, axis=1)
        return self.project(A, np.zeros((2, A.shape[1])), explicit=explicit)


def cosine_tail_bound(C2, J, H, T=1.0):
    """Rigorous bound on sum_{k>=J} w_k a_k^2 when |a_k| <= C2 / omega_k^2 for all k >= J.

    For f with f' of bounded variation (e.g. f in C^2): integrating by parts twice gives
    |a_k| <= sqrt(2/T) (|f'(0)| + |f'(T)| + V(f')) / omega_k^2, i.e. C2 = sqrt(2/T)(...).
    sum_{k>=J} c_H omega_k^{1-2H} C2^2 omega_k^{-4} <= c_H C2^2 (pi/T)^{-3-2H} [J^{-3-2H} + J^{-2-2H}/(2+2H)]."""
    cH = math.gamma(2 * H + 1) * math.sin(math.pi * H)
    s = 3 + 2 * H
    return cH * C2 ** 2 * (math.pi / T) ** (-s) * (J ** (-s) + J ** (1 - s) / (s - 1))


def certified_bracket(a_hat, x, H, T, C2):
    """Bracket for ||f - g||_H^2, g = sum_{k<len(x)} x_k h_k, from N midpoint-DCT coefficients a_hat.

    Status 'certified (exact arithmetic)' when the declared bound C2 (|a_k| <= C2/omega_k^2 for all
    k >= 1) is valid: the aliasing error delta of every a_hat_k and the tail beyond N are then bounded.
    Without such a bound (C2=None) the same formulas are returned as an ESTIMATE (status 'unresolved'),
    because finitely many samples cannot see, e.g., cos(N pi t / T)."""
    a_hat = np.asarray(a_hat, float); N = len(a_hat)
    xx = np.zeros(N); xx[:len(x)] = x
    beta = 2 * H - 1
    cH = math.gamma(2 * H + 1) * math.sin(math.pi * H)
    k = np.arange(1, N); w = cH * (k * np.pi / T) ** (1 - 2 * H)
    dif = np.abs(a_hat[1:] - xx[1:])
    if C2 is None:
        lo = 0.5 * np.sum(w * dif ** 2)
        up = (abs(a_hat[0] - xx[0]) * T ** (beta / 2) + math.sqrt(np.sum(w * dif ** 2))) ** 2
        return lo, up, 'unresolved (finite-sample estimate; no tail information)'
    dlt = midpoint_alias_bound(C2, N, N, T)
    lo = 0.5 * np.sum(w * np.clip(dif - dlt, 0, None) ** 2)
    tail = cosine_tail_bound(C2, N, H, T)
    up = ((abs(a_hat[0] - xx[0]) + math.sqrt(2) * dlt) * T ** (beta / 2) + math.sqrt(np.sum(w * (dif + dlt) ** 2) + tail)) ** 2
    return lo, up, 'certified in exact arithmetic under the declared bound C2'


def midpoint_alias_bound(C2, N, kmax, T=1.0):
    """Rigorous bound on |a_hat_k - a_k| (midpoint DCT-II aliasing), k < kmax <= N, if |a_m| <= C2/omega_m^2 for m >= N.

    The DCT-II of N midpoint samples gives a_hat_k = a_k + sum_{m>=1} (-1)^m (a_{2mN-k} + a_{2mN+k}).
    Each alias index is at least 2mN - k > (2m-1)N, so the sum is bounded by
    2 C2 (T/pi)^2 sum_m ((2m-1)N)^{-2} = 2 C2 (T/pi)^2 (pi^2/8) / N^2."""
    return 2 * C2 * (T / math.pi) ** 2 * (math.pi ** 2 / 8) / N ** 2
