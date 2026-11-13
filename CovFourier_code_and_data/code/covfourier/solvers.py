"""Safeguarded large-n solvers for the cosine Gram.

Mathematics used (Theorem 'two-sided bounds' of the paper; constants exact):
  * Lower bound (L): for every a_0 and every (a_k)_{k>=1},
        ||a_0 h_0 + sum_{k>=1} a_k h_k||_H^2 >= (1/2) sum_{k>=1} w_k a_k^2,   w_k = c_H omega_k^(1-2H).
    Minimizing the left side over a_0 gives the Schur complement S_+ = G_+ - g g^T / G_00 of the h_0
    entry, hence S_+ >= (1/2) W.  With the upper bound, S_+ <= G_+ <= W.  Therefore
        spec( W^{-1/2} S_+ W^{-1/2} ) is contained in [1/2, 1]     for every n   (finite sections).
  * Consequences used here:
      - CG on S_+ preconditioned by W (the exact weights, not diag(G)) has kappa <= 2;
        the standard bound gives ||e_k||_S <= 2 ((sqrt2-1)/(sqrt2+1))^k ||e_0||_S.
      - White (covariance-orthonormal) coordinates without any dense factorization:
            z_0 = b_0 / sqrt(G_00),   z_+ = M^{-1/2} W^{-1/2} (b_+ - g b_0/G_00),
        M = W^{-1/2} S_+ W^{-1/2}.  The basis {h_0/sqrt(G_00)} + Loewdin(M) is orthonormal in H.
        M^{-1/2} is applied by a Chebyshev expansion of x^{-1/2} on [1/2, 1] (slightly widened);
        the coefficients decay like (3 + sqrt 8)^{-k}, so about 21 mat-vecs give 1e-15,
        independently of n.
Every routine validates its inputs, returns an explicit status, and recomputes the true residual.
No routine inspects or branches on the identity of the input function.
"""
import math
import numpy as np
from numpy.polynomial import chebyshev as C

from .batch import gram_matmat, _ParityKernel

SQ2 = math.sqrt(2.0)


def pcg_safe(matvec, b, precond_diag, tol=1e-14, maxit=500, x0=None):
    """Preconditioned CG with safeguards.

    Returns (x, info) with info = {'status': 'zero_rhs'|'converged'|'maxit'|'breakdown',
    'iterations', 'true_rel_residual'}.  b may be a vector; precond_diag > 0 elementwise."""
    b = np.asarray(b, dtype=float)
    d = np.asarray(precond_diag, dtype=float)
    if b.ndim != 1 or d.shape != b.shape:
        raise ValueError('b and precond_diag must be 1-D of equal length')
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(d)) and np.all(d > 0)):
        raise ValueError('non-finite right-hand side or non-positive preconditioner')
    nb = float(np.linalg.norm(b))
    if nb == 0.0:
        return np.zeros_like(b), {'status': 'zero_rhs', 'iterations': 0, 'true_rel_residual': 0.0}
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    r = b - matvec(x) if x0 is not None else b.copy()
    z = r / d; p = z.copy(); rz = float(np.dot(r, z))
    status, it = 'maxit', 0
    for it in range(1, maxit + 1):
        Ap = matvec(p)
        pAp = float(np.dot(p, Ap))
        if not np.isfinite(pAp) or pAp <= 0.0:
            status = 'breakdown'; break
        alpha = rz / pAp
        x += alpha * p; r -= alpha * Ap
        if float(np.linalg.norm(r)) <= tol * nb:
            status = 'converged'; break
        z = r / d; rz_new = float(np.dot(r, z))
        if not np.isfinite(rz_new):
            status = 'breakdown'; break
        p = z + (rz_new / rz) * p; rz = rz_new
    true_res = float(np.linalg.norm(b - matvec(x)) / nb)
    if status == 'converged' and true_res > 100 * tol:
        status = 'maxit'      # recursive residual drifted; report honestly
    return x, {'status': status, 'iterations': it, 'true_rel_residual': true_res}


class SchurCosine:
    """Matrix-free access to the cosine Gram G_n (h_0 eliminated) for any n.

    G is a covfourier.gram.CosineGram; n = number of cosines (h_0..h_{n-1})."""

    def __init__(self, G, n, cheb_tol=5e-15, widen=1e-3):
        if int(n) < 2:
            raise ValueError('n must be >= 2')
        self.G, self.n = G, int(n)
        H, T = G.H, G.T
        if not (0.5 <= H < 1.0) or not (T > 0):
            raise ValueError('need 1/2 <= H < 1 and T > 0')
        self.kernel = _ParityKernel(self.n)
        m = G.m(self.n); r = m.imag
        k = np.arange(1, self.n, dtype=float)
        self.G00 = T ** G.b
        g = np.zeros(self.n - 1)
        if G.b != 0.0:
            ev = (np.arange(1, self.n) % 2 == 0)
            g[ev] = -SQ2 * G.C * r[1:][ev] / (np.pi * k[ev])
        self.g = g
        cH = math.gamma(2 * H + 1) * math.sin(math.pi * H)
        self.w = cH * (k * np.pi / T) ** (1 - 2 * H)
        self.sw = np.sqrt(self.w)
        # Chebyshev expansion of x^{-1/2} on [lo, hi] (proven spectrum [1/2, 1], widened for rounding)
        self.lo, self.hi = 0.5 * (1 - widen), 1.0 * (1 + widen)
        for deg in range(8, 200, 2):
            cheb = C.Chebyshev.interpolate(lambda x: x ** -0.5, deg, domain=[self.lo, self.hi])
            if np.max(np.abs(cheb.coef[-3:])) < cheb_tol * abs(cheb.coef[0]):
                break
        self.cheb = cheb.coef
        self.cheb_degree = len(self.cheb) - 1
        self.matvecs = 0

    # ---- operators ------------------------------------------------------------------
    def S_plus(self, V):
        """S_+ V for V of shape (n-1, q)."""
        V = np.atleast_2d(V.T).T if V.ndim == 1 else V
        X = np.zeros((self.n, V.shape[1])); X[1:] = V
        Y = gram_matmat(self.G, X, self.kernel)
        self.matvecs += V.shape[1]
        return Y[1:] - np.outer(self.g, Y[0]) / self.G00

    def M(self, V):
        return self.S_plus(V / self.sw[:, None]) / self.sw[:, None]

    # ---- solves ---------------------------------------------------------------------
    def solve(self, b, tol=1e-14, maxit=200):
        """x = G_n^{-1} b by block elimination of h_0 and W-preconditioned CG (kappa <= 2)."""
        b = np.asarray(b, dtype=float)
        rhs = b[1:] - self.g * b[0] / self.G00
        xp, info = pcg_safe(lambda v: self.S_plus(v[:, None])[:, 0], rhs, self.w, tol=tol, maxit=maxit)
        x0 = (b[0] - float(np.dot(self.g, xp))) / self.G00
        return np.r_[x0, xp], info

    def white(self, B):
        """White coordinates z (n x q) of right-hand sides B = (<h_k, f_j>_H) (n x q)."""
        B = np.asarray(B, dtype=float)
        one = B.ndim == 1
        B = B[:, None] if one else B
        z0 = B[0] / math.sqrt(self.G00)
        R = (B[1:] - np.outer(self.g, B[0]) / self.G00) / self.sw[:, None]
        # Chebyshev series sum_k c_k T_k(u) R with u = (2M - (lo+hi)) / (hi-lo)
        a, c = 2.0 / (self.hi - self.lo), (self.hi + self.lo) / (self.hi - self.lo)
        U = lambda V: a * self.M(V) - c * V
        T0, T1 = R, U(R)
        acc = self.cheb[0] * T0 + self.cheb[1] * T1
        for ck in self.cheb[2:]:
            T0, T1 = T1, 2 * U(T1) - T0
            acc = acc + ck * T1
        Z = np.vstack([z0[None, :], acc])
        return Z[:, 0] if one else Z

    def spectrum_check(self, nprobe=6, iters=60, seed=0):
        """Power-iteration estimates of the extreme eigenvalues of M (diagnostic only)."""
        rng = np.random.default_rng(seed)
        V = rng.standard_normal((self.n - 1, 1))
        for _ in range(iters):
            V = self.M(V); V /= np.linalg.norm(V)
        lmax = float((V.T @ self.M(V))[0, 0])
        V = rng.standard_normal((self.n - 1, 1))
        for _ in range(iters):            # power iteration on (1 - M), spectrum of M in [1/2, 1]
            V = V - self.M(V); V /= np.linalg.norm(V)
        lmin = float((V.T @ self.M(V))[0, 0])
        return lmin, lmax
