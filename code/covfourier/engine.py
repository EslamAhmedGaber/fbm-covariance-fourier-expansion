"""Covariance-Fourier engine: one fixed procedure for every admissible input on [0,T].

Mathematical object (Theorem 'general expansion' of the paper): projection in the
fBm integrand space H onto  V_D = span{h_0, t/T, (t/T)^2, h_1, ..., h_{D-3}}
(or onto the cosine span alone when enrich=False).

Numerical realization:
  1. Fourier stage   a = classical cosine coefficients of the input, by one DCT-II
                     of N midpoint samples (exactly as for ordinary Fourier series);
                     or supplied directly (moment interface).
  2. Metric stage    b = (G a)_{k<n} with the closed cosine Gram applied by FFT,
                     c = L^{-1} b (cosine-orthonormal coordinates; G_n = L L^T is
                     well conditioned, cond <= 1e5 for H <= 0.99).
  3. Enrichment      the two polynomial directions enter through their explicit
                     cosine-residual vectors d_j (exact classical coefficients), so the
                     ill-conditioned augmented Gram (cond ~ D^(2H+2)) is never factored.
  4. Error stage     the residual f - f_D is represented by its own cosine coefficient
                     vector e and its squared norm is the FFT quadratic form e^T G e
                     (no subtraction of nearly equal energies).
In exact arithmetic the cosine part is the projection onto V_n.  The enrichment directions are
cosine-truncated (Naux terms) polynomial residuals, so the enriched projection is onto
V_n + span{t_Naux, t^2_Naux}, which approaches the space V_D as Naux grows.
The coordinates are white coordinates of a cosine-orthonormal basis, NOT the polynomial-first
prefix coordinates of Definition 'general expansion' (they differ by a rotation within V_D).
Safeguards: see covfourier.solvers (Schur-complement PCG with proven kappa <= 2 and
matrix-free white coordinates).  The old solver='pcg' route below preconditions the FULL block
(including h_0) with diag(G); the theorem's bound does not cover that matrix directly.
"""
import math
import time
import numpy as np
from scipy.fft import dct
from scipy.linalg import cholesky, solve_triangular, cho_solve
from .gram import CosineGram, power_cosine_moments, poly_cos_moments, poly_poly


def midpoint_cosine_coefficients(f, N, T=1.0, chunk=1 << 16):
    """Classical cosine coefficients of the midpoint interpolant (DCT-II).

    a_k = sqrt(T/N) * DCT-II_ortho(f(t_j))_k,  t_j = T (j + 1/2)/N."""
    vals = np.empty(N)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        t = T * (np.arange(s, e) + 0.5) / N
        v = np.asarray(f(t), dtype=float)
        vals[s:e] = np.broadcast_to(v, t.shape)
    if not np.all(np.isfinite(vals)):
        raise ArithmeticError('non-finite sample at a midpoint')
    return math.sqrt(T / N) * dct(vals, type=2, norm='ortho')


def pcg(matvec, b, diag, tol=1e-14, maxit=500):
    """Preconditioned CG for the SPD cosine Gram block (diagonal preconditioner).

    By the two-sided bound (1/2) W <= G_+ <= W (k >= 1 block), the diagonally scaled
    matrix has condition number bounded independently of its size."""
    nb = np.linalg.norm(b)
    if nb == 0.0:                       # zero right-hand side returns zero, no 0/0
        return np.zeros_like(b), 0
    x = np.zeros_like(b); r = b.copy(); z = r / diag; p = z.copy(); rz = np.dot(r, z)
    for it in range(1, maxit + 1):
        Ap = matvec(p); pAp = np.dot(p, Ap)
        if not np.isfinite(pAp) or pAp <= 0:
            raise ArithmeticError('PCG breakdown (non-positive curvature)')
        alpha = rz / pAp
        x += alpha * p; r -= alpha * Ap
        if np.linalg.norm(r) <= tol * nb:
            return x, it
        z = r / diag; rz_new = np.dot(r, z); p = z + (rz_new / rz) * p; rz = rz_new
    raise ArithmeticError('PCG did not converge in maxit iterations (use covfourier.solvers.pcg_safe for a status return)')


class _PCGFactor:
    """Drop-in for the dense Cholesky route when n is large (projection only)."""
    def __init__(self, G, n):
        self.G, self.n = G, n
        m = G.m(n); p = m.real; r = m.imag; k = np.arange(1, n)
        d = np.empty(n); d[0] = G.T ** G.b
        d[1:] = G.C * (p[1:] - (1 - G.b) * r[1:] / (k * np.pi)) if G.b else 1.0
        self.diag = d
        self.iterations = []

    def solve(self, b):
        x, it = pcg(self.G.matvec, b, self.diag)
        self.iterations.append(it)
        return x


class CovarianceFourier:
    """Prepared once for (H, T, n, enrich, Naux); reused for every input.

    solver='cholesky' (dense, gives cosine-orthonormal coordinates) or 'pcg'
    (FFT mat-vec + diagonal preconditioner; projection, energies and residuals only)."""

    def __init__(self, H, n, T=1.0, enrich=True, Naux=1 << 18, solver='cholesky'):
        t0 = time.perf_counter()
        self.H, self.T, self.n, self.enrich = float(H), float(T), int(n), bool(enrich)
        self.Naux = int(Naux)
        self.solver = solver
        self.record_cosine_only = False
        if self.Naux < 4 * self.n:
            raise ValueError('Naux should be well above n (>= 4n)')
        self.G = CosineGram(self.H, self.T)
        self.G.m(self.Naux)                       # Kummer grid once
        if solver == 'cholesky':
            Gn = self.G.dense(self.n)
            self.L = cholesky(Gn, lower=True, overwrite_a=False)
            self.cond_cosine = float(np.linalg.cond(Gn)) if self.n <= 2048 else None
            del Gn
            self._solve = lambda b: cho_solve((self.L, True), b)
        else:
            self.L = None
            self.cond_cosine = None
            self._pcg = _PCGFactor(self.G, self.n)
            self._solve = lambda b: (np.stack([self._pcg.solve(b[:, j]) for j in range(b.shape[1])], axis=1)
                                     if b.ndim == 2 else self._pcg.solve(b))
        self.D = self.n + (2 if self.enrich else 0)
        if self.enrich:
            T_ = self.T
            # Classical coefficients of q_j = (t/T)^j are exact; all inner products are taken
            # in the same Naux-term cosine representation used for inputs (Galerkin consistency).
            aq = np.stack([power_cosine_moments(float(j), self.Naux, T_) / T_ ** j for j in (1, 2)], axis=1)
            Gaq = np.stack([self.G.matvec(aq[:, j]) for j in (0, 1)], axis=1)
            Q = Gaq[:self.n]
            X = self._solve(Q)                                # G_n^{-1} Q
            d = aq.copy()
            d[:self.n] -= X                                   # cosine residuals of q_1, q_2
            Gd = np.stack([self.G.matvec(d[:, j]) for j in (0, 1)], axis=1)
            S = d.T @ Gd
            S = 0.5 * (S + S.T)
            self.S = S
            self.R = cholesky(S, lower=False)                 # S = R^T R
            Rinv = solve_triangular(self.R, np.eye(2), lower=False)
            self.dw = d @ Rinv                                # residual directions, orthonormal in H
            self.Gdw = Gd @ Rinv
            self.Q, self.X, self.Rinv = Q, X, Rinv
            # diagnostics: exact closed-form moments and the cancellation-prone Schur form
            Qex = np.stack([poly_cos_moments(self.H, j, self.n, T_) / T_ ** j for j in (1, 2)], axis=1)
            P = np.array([[poly_poly(self.H, i, j, T_) / T_ ** (i + j) for j in (1, 2)] for i in (1, 2)])
            self.moment_truncation_error = float(np.max(np.abs(Qex - Q)))
            self.schur_subtractive = P - Qex.T @ self._solve(Qex)
        self.setup_seconds = time.perf_counter() - t0

    # ---------------------------------------------------------------- core
    def project(self, a):
        """a: classical cosine coefficient vector (length <= Naux).  Returns dict."""
        a = np.asarray(a, dtype=float)
        if len(a) > self.Naux:
            raise ValueError('coefficient vector longer than Naux')
        full = np.zeros(self.Naux)
        full[:len(a)] = a
        Ga = self.G.matvec(full)
        b = Ga[:self.n]
        if self.L is not None:
            c = solve_triangular(self.L, b, lower=True)       # cosine-orthonormal coordinates
            x = solve_triangular(self.L.T, c, lower=False)    # G_n^{-1} b
            energy_cos = float(np.dot(c, c))
        else:
            x = self._solve(b); c = None
            energy_cos = float(np.dot(b, x))
        e = full.copy()
        e[:self.n] -= x                                        # residual of cosine projection
        out = {'b': b, 'c': c, 'x_cos': x}
        if self.enrich and self.record_cosine_only:
            out['error2_cosine_only'] = float(np.dot(e, self.G.matvec(e)))
        if self.enrich:
            rho = self.Gdw.T @ e                              # <(I-P_n) f, w_j>
            e = e - self.dw @ rho
            y = self.Rinv @ rho                               # coefficients of q_j
            xr = self._solve(b - self.Q @ y)                  # cosine part of f_D
            out.update({'rho': rho, 'poly_coef': y, 'x_raw_cos': xr})
            out['coords'] = None if c is None else np.r_[c, rho]
            out['energy'] = energy_cos + float(np.dot(rho, rho))
        else:
            out['coords'] = c
            out['energy'] = energy_cos
        Ge = self.G.matvec(e)
        out['error2'] = float(np.dot(e, Ge))
        out['norm2_aux'] = float(np.dot(full, Ga))
        out['residual_coefficients'] = e
        return out

    def from_callable(self, f, N=None, check=True):
        """Same steps for every callable: DCT at N midpoints (default Naux), project.

        check=True repeats at N/2 (nested) and N-1 (non-nested) and reports the
        changes of coordinates and of the squared error as numerical diagnostics."""
        N = self.Naux if N is None else int(N)
        t0 = time.perf_counter()
        res = self.project(midpoint_cosine_coefficients(f, N, self.T)[:min(N, self.Naux)])
        res['N'] = N
        if check:
            diag = {}
            for M in (N // 2, N - 1):
                r2 = self.project(midpoint_cosine_coefficients(f, M, self.T)[:min(M, self.Naux)])
                if res['coords'] is not None:
                    cc = float(np.linalg.norm(r2['coords'] - res['coords']))
                else:   # energy-norm change of the projection, sqrt(dx^T G dx), via b
                    dx = r2['x_cos'] - res['x_cos']
                    cc = float(math.sqrt(max(np.dot(dx, self.G.matvec(dx)), 0.0)))
                diag[M] = {'coord_change': cc,
                           'error2_change': float(abs(r2['error2'] - res['error2']))}
            res['refinement'] = diag
        res['seconds'] = time.perf_counter() - t0
        return res

    def evaluate(self, res, t):
        """Pointwise values of the approximant f_D (ordinary-function inputs)."""
        t = np.asarray(t, dtype=float)
        k = np.arange(self.n)
        H = np.where(k == 0, 1 / math.sqrt(self.T), math.sqrt(2 / self.T))[None, :] * \
            np.cos(np.pi * np.outer(t, k) / self.T)
        if self.enrich:
            v = H @ res['x_raw_cos']
            y = res['poly_coef']
            return v + y[0] * (t / self.T) + y[1] * (t / self.T) ** 2
        return H @ res['x_cos']
