"""Batched, endpoint-corrected Fourier stage and projection (a-posteriori endpoint correction).

Why: with the plain midpoint DCT the classical cosine coefficients of a smooth non-periodic input
carry an aliasing error O(N^-2), and truncating the k^-2 coefficient tail at Naux perturbs every
H-inner product by O(Naux^-(1+2H)).  Both errors sit OUTSIDE the projection, so the residual
energies E_i^2 do not certify the covariance (observed in A1: floor 2e-7 at Naux = 4096).

Remedy (general; no input-specific choices):
  1. Endpoint correction.  From the same midpoint samples, estimate f'(0), f'(T) by one-sided
     polynomial extrapolation (degree 8).  Write f = g + p, p = c1 (t/T) + c2 (t/T)^2 with
     p'(0) = f'(0), p'(T) = f'(T).  Then g'(0) = g'(T) ~ 0, its cosine coefficients decay like k^-4
     and the midpoint DCT of g has aliasing O(N^-4).
  2. Exact polynomial part.  All H-inner products involving p use the closed forms
     <h_k, t^j>_H (all k < N) and <t^i, t^j>_H; nothing about p is truncated.
  3. Batched projection.  b = G[:n, :N] a_g + Q[:n] c through a dense block of Gram rows (BLAS);
     enrichment coefficients rho = Gdw^T a_g + dw^T (Q c) - Gdw[:n]^T x (no per-output long matvec);
     ||f||^2 = a_g^T G a_g + 2 a_g.(Q c) + c^T P c (one batched FFT matvec).
The projection is the exact H-orthogonal projection onto V_n + span{w~_1, w~_2} (w~ as in the
engine), so  Sigma - Sigma_hat = R R^T  up to the O(N^-4) aliasing of g.
"""
import math
import numpy as np
from scipy.fft import dct, rfft, irfft, next_fast_len
from scipy.linalg import solve_triangular

from .gram import power_cosine_moments, poly_cos_moments, poly_poly

SQ2 = math.sqrt(2.0)


def gram_rows(G, n, N, block=256):
    """Rows 0..n-1, columns 0..N-1 of the closed cosine Gram (same formulas as CosineGram.dense)."""
    if G.b == 0.0:
        out = np.zeros((n, N)); out[np.arange(n), np.arange(n)] = 1.0
        return out
    m = G.m(N); p = m.real; r = m.imag; C = G.C
    k = np.arange(N, dtype=float)
    out = np.zeros((n, N))
    for s in range(1, n, block):
        e = min(s + block, n)
        K = k[s:e, None]; L = k[None, 1:]
        same = ((K + L) % 2 == 0) & (K != L)
        with np.errstate(divide='ignore', invalid='ignore'):
            val = (2 * C / np.pi) * (L * r[None, 1:] - K * r[s:e, None]) / (K ** 2 - L ** 2)
        out[s:e, 1:] = np.where(same, val, 0.0)
    kk = np.arange(1, n)
    out[kk, kk] = C * (p[kk] - (1 - G.b) * r[kk] / (kk * np.pi))
    out[0, 0] = G.T ** G.b
    ev = np.arange(2, N, 2)
    out[0, ev] = -SQ2 * C * r[ev] / (np.pi * ev)
    evr = ev[ev < n]
    out[evr, 0] = -SQ2 * C * r[evr] / (np.pi * evr)
    return out


class _ParityKernel:
    """Cached FFTs of the parity-restricted 1/d kernel used by the Gram mat-vec (length-N vectors)."""
    def __init__(self, N):
        n = N - 1
        self.n = n
        d = np.arange(-2 * n, 2 * n + 1)
        q = np.zeros(len(d))
        msk = (d != 0) & (d % 2 == 0)
        q[msk] = 1.0 / d[msk]
        self.L = (2 * n + 1) + len(q) - 1
        self.nf = next_fast_len(self.L, real=True)
        self.Q = rfft(q, self.nf)

    def conv(self, X, sign):
        """Column-wise version of CosineGram._parity_conv; X has shape (n, q)."""
        n = self.n
        ext = np.zeros((2 * n + 1, X.shape[1]))
        ext[n + 1:] = X
        ext[:n] = sign * X[::-1]
        full = irfft(rfft(ext, self.nf, axis=0) * self.Q[:, None], self.nf, axis=0)[:self.L]
        return full[3 * n + 1: 4 * n + 1]


def gram_matmat(G, A, kernel=None):
    """G[:N,:N] @ A for A of shape (N, q) (batched FFT mat-vec; same algebra as CosineGram.matvec)."""
    A = np.asarray(A, dtype=float)
    N = A.shape[0]
    if G.b == 0.0:
        return A.copy()
    kernel = kernel or _ParityKernel(N)
    m = G.m(N); p = m.real; r = m.imag; C = G.C
    out = np.empty_like(A)
    out[0] = G.T ** G.b * A[0]
    k = np.arange(1, N, dtype=float)[:, None]
    V = A[1:]; rr = r[1:, None]
    odd = kernel.conv(rr * V, -1.0)
    even = kernel.conv(V, +1.0)
    out[1:] = C * (p[1:, None] + G.b * rr / (np.pi * k)) * V + (C / np.pi) * (odd - rr * even)
    ev = (np.arange(1, N) % 2 == 0)
    g0 = -SQ2 * C * r[1:][ev] / (np.pi * k[ev, 0])
    out[0] += g0 @ V[ev]
    out[1:][ev] += g0[:, None] * A[0][None, :]
    return out


def _endpoint_weights(deg=8):
    u = np.arange(deg + 1) + 0.5
    V = np.vander(u, deg + 1, increasing=True).T          # V[m, j] = u_j^m
    rhs = np.zeros(deg + 1); rhs[1] = 1.0
    return np.linalg.solve(V, rhs)                          # sum_j w_j u_j^m = delta_{m1}



def endpoint_corrected_dct(F, T, w, deg, max_ratio=1e3, rule='a_posteriori'):
    """Shared Fourier stage.  F: (q, N) midpoint samples.  Returns a_g (N, q), c (2, q), mask.

    The quadratic endpoint correction p = c1 t/T + c2 (t/T)^2 (from one-sided derivative estimates)
    is applied to an input only if it is useful, judged a posteriori from that input's own data:
      (1) it lowers the high-frequency tail energy sum_{k >= N/2} a_k^2 of the DCT, and
      (2) max|p| <= max_ratio * max|f|, which bounds the cancellation loss in the norms.
    Inputs with endpoint singularities (huge derivative estimates) or stiff layers beyond the ratio
    are therefore left uncorrected.  The rule depends on the samples only, never on the input's identity.

    rule='ratio' reproduces the fixed switch of the first A1 run (used there with max_ratio = 100): the
    correction is applied unless max|p| > max_ratio * max|f|, without the tail-energy test."""
    F = np.atleast_2d(np.asarray(F, dtype=float))
    q, N = F.shape
    h = T / N
    d0 = (F[:, :deg + 1] @ w) / h
    d1 = -(F[:, ::-1][:, :deg + 1] @ w) / h
    c1 = d0 * T
    c2 = (d1 - d0) * T / 2
    x = (np.arange(N) + 0.5) / N
    Pv = c1[:, None] * x[None, :] + c2[:, None] * x[None, :] ** 2
    a_f = math.sqrt(T / N) * dct(F, type=2, norm='ortho', axis=1)
    a_g = math.sqrt(T / N) * dct(F - Pv, type=2, norm='ortho', axis=1)
    tail_f = np.sum(a_f[:, N // 2:] ** 2, axis=1); tail_g = np.sum(a_g[:, N // 2:] ** 2, axis=1)
    ratio = np.max(np.abs(Pv), axis=1) / np.maximum(np.max(np.abs(F), axis=1), 1e-300)
    if rule == 'a_posteriori':
        use = (tail_g < tail_f) & (ratio <= max_ratio) & np.isfinite(ratio)
    elif rule == 'ratio':
        use = ~(ratio > max_ratio)
    else:
        raise ValueError("rule must be 'a_posteriori' or 'ratio'")
    a = np.where(use[:, None], a_g, a_f)
    c1 = np.where(use, c1, 0.0); c2 = np.where(use, c2, 0.0)
    return a.T, np.vstack([c1, c2]), use

class BatchProjector:
    """Endpoint-corrected Fourier stage + batched projection for a prepared CovarianceFourier
    (solver='cholesky', enrich=True)."""

    def __init__(self, cf, N=None, deg=8, dense_rows_limit=4e7, max_ratio=1e3, rule='a_posteriori'):
        # max_ratio: cap of the a-posteriori endpoint-correction rule (see endpoint_corrected_dct): the
        # correction is applied to an input only if it lowers that input's DCT tail energy AND
        # |p|_inf <= max_ratio |f|_inf.  A fixed switch max_ratio = 100 gave a stiff A1
        # error of 9.6e-8 at H = 0.75, n = 1024, N = 65536, and no cap 3.6e-9; the rule with
        # cap 1e3 is used for A1_memory_responses_v2.json.  The cancellation is accounted for by 'eta'.
        # rule='ratio' (with max_ratio = 100) is the fixed switch of the first A1 run; see endpoint_corrected_dct.
        self.max_ratio = float(max_ratio)
        if rule not in ('a_posteriori', 'ratio'):
            raise ValueError("rule must be 'a_posteriori' or 'ratio'")
        self.rule = rule
        if cf.L is None or not cf.enrich:
            raise ValueError('needs a cholesky-prepared, enriched CovarianceFourier')
        self.cf = cf
        self.N = cf.Naux if N is None else int(N)
        if self.N > cf.Naux:
            raise ValueError('N must not exceed Naux of the prepared engine')
        H, T, n, N = cf.H, cf.T, cf.n, self.N
        self.T = T
        self.P = np.array([[poly_poly(H, i, j, T) / T ** (i + j) for j in (1, 2)] for i in (1, 2)])
        self.QN = np.stack([poly_cos_moments(H, j, N, T) / T ** j for j in (1, 2)], axis=1)    # (N, 2) exact
        self.Grows = gram_rows(cf.G, n, N) if n * N <= dense_rows_limit else None
        self.kernel = _ParityKernel(N)
        self.dw = cf.dw[:N]; self.Gdw = cf.Gdw[:N]
        self.dwQ = self.dw.T @ self.QN                                                           # (2, 2)
        self.w = _endpoint_weights(deg)
        self.deg = deg

    def fourier_stage(self, F):
        """F: (q, N) midpoint samples.  Returns a_g (N, q) and polynomial coefficients c (2, q)."""
        F = np.atleast_2d(np.asarray(F, dtype=float))
        if F.shape[1] != self.N:
            raise ValueError('sample count must equal N')
        if not np.all(np.isfinite(F)):
            raise ValueError('non-finite samples')
        a, c, use = endpoint_corrected_dct(F, self.T, self.w, self.deg, self.max_ratio, self.rule)
        self.last_corrected = use
        return a, c

    def project(self, a_g, c, norms=True):
        cf = self.cf; n = cf.n
        Qc = self.QN @ c                                          # (N, q): <h_k, p>_H, exact
        if self.Grows is not None:
            b = self.Grows @ a_g + Qc[:n]
        else:
            b = gram_matmat(cf.G, a_g, self.kernel)[:n] + Qc[:n]
        C = solve_triangular(cf.L, b, lower=True)                 # cosine-orthonormal coordinates
        X = solve_triangular(cf.L.T, C, lower=False)
        rho = self.Gdw.T @ a_g + self.dw.T @ Qc - self.Gdw[:n].T @ X
        coords = np.vstack([C, rho])                              # (D, q)
        out = {'coords': coords, 'energy': np.sum(coords ** 2, axis=0)}
        if norms:
            Ga = gram_matmat(cf.G, a_g, self.kernel)
            t1 = np.einsum('kq,kq->q', a_g, Ga); t2 = 2 * np.einsum('kq,kq->q', a_g, Qc)
            t3 = np.einsum('iq,ij,jq->q', c, self.P, c)
            nrm = t1 + t2 + t3
            out['norm2'] = nrm
            out['error2'] = nrm - out['energy']
            # rounding allowance for the subtractive (Pythagoras) residual energy: the endpoint
            # polynomial can be much larger than f (stiff inputs), so E^2 = ||f||^2 - ||coords||^2
            # loses about log10(scale/||f||^2) digits.  eta = 64 eps * scale (heuristic, conservative).
            out['eta'] = 64 * np.finfo(float).eps * (np.abs(t1) + np.abs(t2) + np.abs(t3) + out['energy'])
            out['resolved'] = out['error2'] > out['eta']          # energies at or below eta are unresolved, not small
        return out

    def from_callables(self, fs, norms=True):
        N = self.N
        t = self.T * (np.arange(N) + 0.5) / N
        F = np.stack([np.broadcast_to(np.asarray(f(t), dtype=float), t.shape) for f in fs])
        a_g, c = self.fourier_stage(F)
        return self.project(a_g, c, norms=norms)
