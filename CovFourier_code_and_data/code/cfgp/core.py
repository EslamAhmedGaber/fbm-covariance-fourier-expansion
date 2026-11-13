"""cfgp.core -- Gaussian-process learning from many noisy linear functionals of one fBm path.

Model (Section 2 of the paper):  y_i = sigma * I_H(F_i) + h_i^T beta + tau * eps_i,  i = 1..m,  on [0, T].
  K_il = <F_i, F_l>_H (exact covariance of the Wiener integrals I_H(F_i)),  1/2 <= H < 1.

CovFourier features (engine of the CovFourier package v1.3.0, unchanged): an H-orthonormal family of D = n + 2 functions
  phi_j = (L^{-1} h)_j, j < n  (Cholesky whitening of the closed-form cosine Gram, cited from CosChaos)
  phi_{n}, phi_{n+1} = H-orthonormal residual directions of t, t^2 (finite cosine sums, Naux = 4n terms).
Coordinates gamma_i = (<F_i, phi_j>_H)_j are computed from EXACT covariance moments b_ik = <F_i, h_k>_H
(closed forms for point values and box-window averages; the N-term representation for smooth kernels), so
  K = Gamma Gamma^T + R,  R = Gram of the residuals F_i - P_D F_i  (PSD),  R_ii = e_i^2 = ||F_i||_H^2 - |gamma_i|^2.

Evidence bracket (Section 4): with C = Q + tau^2 I, Q = Gamma Gamma^T (+ nuisance), v = C^{-1} y,
  L  = log N(y; 0, C) - tr R / (2 tau^2)                                      (collapsed lower bound)
  U  = -1/2 logdet C - 1/2 q_lo - m/2 log 2 pi,   q_lo <= y^T (K + tau^2 I)^{-1} y,
  q_lo = max( y^T (C + tr R I)^{-1} y ,  y^T C^{-1} y - (|v|^T e)^2 ,  y^T C^{-1} y - beta(v) ),
  beta(v) >= v^T R v  computed from the cosine representation of the residuals (exact Gram on N cosines,
  L2 tail through Lambda_N of the weighted cosine inequality, cited from CosChaos).
All quantities cost O(m D^2) time and O(m D) memory (beta: O(m N) with N = 4n).
"""
import math
import time

import numpy as np
from scipy.linalg import cholesky, solve_triangular, cho_solve, eigh

from covfourier.kummer import M as KM
from covfourier.engine import CovarianceFourier
from covfourier.gram import CosineGram, poly_cos_moments, poly_poly
from covfourier.batch import gram_matmat, endpoint_corrected_dct, _endpoint_weights

LOG2PI = math.log(2.0 * math.pi)
SQ2 = math.sqrt(2.0)


# ----------------------------------------------------------------------------------------------- constants
def c_star(H):
    """c*_H = Gamma(2H+1) sin(pi H) (CosChaos, Lemma 3)."""
    return math.gamma(2.0 * H + 1.0) * math.sin(math.pi * H)


def Lambda_J(H, J, T=1.0):
    """Tail constant of the weighted cosine inequality: ||g||_H^2 <= Lambda_J ||g||_{L2}^2 for g _|_ h_0..h_{J-1}."""
    return c_star(H) * T ** (2 * H - 1) * (J * math.pi) ** (1 - 2 * H)


def weights_wk(H, N, T=1.0):
    """w_k = c*_H (k pi / T)^{1-2H}, k = 1..N-1 (index 0 -> 0)."""
    k = np.arange(N, dtype=float)
    w = np.zeros(N)
    w[1:] = c_star(H) * (k[1:] * math.pi / T) ** (1 - 2 * H)
    return w


# ----------------------------------------------------------------------------------------------- designs
class Points:
    """Point values Y_i = B_H(t_i), integrands F_i = 1_[0, t_i)."""
    kind = 'point'

    def __init__(self, t, T=1.0):
        self.t = np.asarray(t, dtype=float).ravel()
        self.T = float(T)
        self.m = self.t.size
        self._order = np.argsort(self.t, kind='stable')

    def loc(self):
        return self.t

    def subset(self, idx):
        return Points(self.t[idx], self.T)

    def norms(self, H):
        return self.t ** (2 * H)

    def K(self, H, other=None):
        s = self.t if other is None else other.t
        t = self.t
        return 0.5 * (t[:, None] ** (2 * H) + s[None, :] ** (2 * H) - np.abs(t[:, None] - s[None, :]) ** (2 * H))

    def cross_points(self, H, z):
        """Cov(Y_i, B_H(z_j)), shape (m, M)."""
        z = np.asarray(z, dtype=float)
        t = self.t
        return 0.5 * (t[:, None] ** (2 * H) + z[None, :] ** (2 * H) - np.abs(t[:, None] - z[None, :]) ** (2 * H))

    def moments(self, H, N, chunk=256):
        """Exact b_ik = <1_[0,t_i), h_k>_H, k < N (shape m x N)."""
        T = self.T
        b = 2 * H - 1
        k = np.arange(N, dtype=float)
        w = k * math.pi / T
        Pu = math.sqrt(2.0 / T) * (T ** (b + 1) * KM(b + 1.0, w * T).real / (b + 1))
        out = np.empty((self.m, N))
        for s in range(0, self.m, chunk):
            tau = self.t[s:s + chunk][:, None]
            e = np.exp(1j * w[None, :] * tau)
            p1 = e * tau ** (b + 1) * KM(b + 1.0, -w[None, :] * tau) / (b + 1)
            p2 = e * (T - tau) ** (b + 1) * KM(b + 1.0, w[None, :] * (T - tau)) / (b + 1)
            blk = H * (Pu[None, :] + math.sqrt(2.0 / T) * (p1 - p2).real)
            tt = tau[:, 0]
            blk[:, 0] = 0.5 * (tt ** (2 * H) + T ** (2 * H) - (T - tt) ** (2 * H)) / math.sqrt(T)
            out[s:s + chunk] = blk
        return out

    def cosine(self, N):
        """Classical cosine coefficients a_ik = int F_i h_k (shape m x N)."""
        T = self.T
        k = np.arange(N, dtype=float)
        w = k * math.pi / T
        out = np.empty((self.m, N))
        out[:, 0] = self.t / math.sqrt(T)
        out[:, 1:] = math.sqrt(2.0 / T) * np.sin(np.outer(self.t, w[1:])) / w[None, 1:]
        return out

    def l2_quad(self, V):
        """||sum_i v_i F_i||_{L2}^2 for each column of V (m x B), exactly, O(m log m)."""
        V = np.atleast_2d(np.asarray(V, dtype=float).T).T if np.ndim(V) == 1 else np.asarray(V, dtype=float)
        o = self._order
        ts = self.t[o]
        Vs = V[o]
        S = np.cumsum(Vs[::-1], axis=0)[::-1]              # suffix sums S_j = sum_{i>=j} v_(i)
        dt = np.diff(np.r_[0.0, ts])
        return (dt[:, None] * S ** 2).sum(axis=0)


class Boxes:
    """Box-window averages Y_i = (1/w_i) int_{a_i}^{b_i} B_H(s) ds; integrands are ramps."""
    kind = 'box'

    def __init__(self, a, b, T=1.0):
        self.a = np.asarray(a, dtype=float).ravel()
        self.b = np.asarray(b, dtype=float).ravel()
        if np.any(self.b <= self.a) or np.any(self.a < 0) or np.any(self.b > T):
            raise ValueError('need 0 <= a < b <= T')
        self.T = float(T)
        self.m = self.a.size
        self.w = self.b - self.a

    def loc(self):
        return 0.5 * (self.a + self.b)

    def subset(self, idx):
        return Boxes(self.a[idx], self.b[idx], self.T)

    def _A(self, H):
        return (self.b ** (2 * H + 1) - self.a ** (2 * H + 1)) / ((2 * H + 1) * self.w)

    def norms(self, H):
        return self._A(H) - self.w ** (2 * H) / ((2 * H + 1) * (2 * H + 2))

    def K(self, H, other=None):
        o = self if other is None else other
        G = lambda x: np.abs(x) ** (2 * H + 2) / ((2 * H + 1) * (2 * H + 2))
        a1, b1, w1 = self.a[:, None], self.b[:, None], self.w[:, None]
        a2, b2, w2 = o.a[None, :], o.b[None, :], o.w[None, :]
        Phi = G(b1 - a2) + G(a1 - b2) - G(b1 - b2) - G(a1 - a2)
        return 0.5 * (self._A(H)[:, None] + o._A(H)[None, :]) - Phi / (2.0 * w1 * w2)

    def cross_points(self, H, z):
        z = np.asarray(z, dtype=float)[None, :]
        g = lambda x: np.sign(x) * np.abs(x) ** (2 * H + 1) / (2 * H + 1)
        a, b, w = self.a[:, None], self.b[:, None], self.w[:, None]
        return 0.5 * (z ** (2 * H) + self._A(H)[:, None] - (g(b - z) - g(a - z)) / w)

    def moments(self, H, N, chunk=128):
        """Exact b_ik = <F_i, h_k>_H for the ramp integrands (closed form through M_{beta+2})."""
        T = self.T
        be = 2 * H - 1
        k = np.arange(N, dtype=float)
        om = k * math.pi / T
        Pu = math.sqrt(2.0 / T) * (T ** (be + 1) * KM(be + 1.0, om * T).real / (be + 1))
        out = np.empty((self.m, N))

        def J(c):
            return np.exp(1j * om[None, :] * c) * c ** (be + 2) * KM(be + 2.0, -om[None, :] * c) / (be + 2)

        def J2(c):
            return np.exp(1j * om[None, :] * c) * (T - c) ** (be + 2) * KM(be + 2.0, om[None, :] * (T - c)) / (be + 2)

        for s in range(0, self.m, chunk):
            a = self.a[s:s + chunk][:, None]
            b = self.b[s:s + chunk][:, None]
            w = b - a
            S = (J(b) - J(a) + J2(b) - J2(a)).real / ((be + 1) * w)
            blk = H * (Pu[None, :] + math.sqrt(2.0 / T) * S)
            aa, bb, ww = a[:, 0], b[:, 0], w[:, 0]
            p = 2 * H + 1
            blk[:, 0] = 0.5 * ((bb ** p - aa ** p) / (p * ww) + T ** (2 * H)
                               + ((T - bb) ** p - (T - aa) ** p) / (p * ww)) / math.sqrt(T)
            out[s:s + chunk] = blk
        return out

    def cosine(self, N):
        T = self.T
        k = np.arange(N, dtype=float)
        om = k * math.pi / T
        out = np.empty((self.m, N))
        out[:, 0] = (self.a + 0.5 * self.w) / math.sqrt(T)
        oa = np.outer(self.a, om[1:]); ob = np.outer(self.b, om[1:])
        out[:, 1:] = math.sqrt(2.0 / T) * (np.cos(oa) - np.cos(ob)) / (self.w[:, None] * om[None, 1:] ** 2)
        return out

    def l2_quad(self, V):
        """||sum_i v_i F_i||_{L2}^2 exactly: g is continuous and piecewise linear with breaks at a_i, b_i."""
        V = np.asarray(V, dtype=float)
        if V.ndim == 1:
            V = V[:, None]
        T = self.T
        p = np.unique(np.r_[0.0, T, self.a, self.b])
        oa = np.argsort(self.a, kind='stable'); ob = np.argsort(self.b, kind='stable')
        a_s, b_s = self.a[oa], self.b[ob]
        # cumulative sums over windows sorted by a and by b
        def cums(X, order):
            return np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X[order], axis=0)])
        Xv = V
        Xb = V * (self.b / self.w)[:, None]
        Xi = V / self.w[:, None]
        Cv_a, Cb_a, Ci_a = cums(Xv, oa), cums(Xb, oa), cums(Xi, oa)
        Cv_b, Cb_b, Ci_b = cums(Xv, ob), cums(Xb, ob), cums(Xi, ob)
        tot_v = Cv_a[-1]
        na_lt = np.searchsorted(a_s, p, side='left')       # #{a_i < p}
        nb_le = np.searchsorted(b_s, p, side='right')      # #{b_i <= p}
        S1 = tot_v[None, :] - Cv_a[na_lt]                  # sum_{a_i >= p} v_i
        S2 = Cb_a[na_lt] - Cb_b[nb_le]                     # sum_{a_i < p < b_i} v_i b_i / w_i
        S3 = Ci_a[na_lt] - Ci_b[nb_le]
        g = S1 + S2 - p[:, None] * S3
        d = np.diff(p)[:, None]
        return (d * (g[:-1] ** 2 + g[:-1] * g[1:] + g[1:] ** 2) / 3.0).sum(axis=0)


def _poly_indicator(H, z, j, T=1.0):
    """<1_[0,z), (t/T)^j>_H for integer j >= 1 (closed form), z array."""
    z = np.asarray(z, dtype=float)
    be = 2 * H - 1
    tot = T ** (j + be + 1) / (j + be + 1)
    left = z ** (j + be + 1) * math.gamma(j + 1) * math.gamma(be + 1) / math.gamma(j + be + 2)
    right = np.zeros_like(z)
    for r in range(j + 1):
        right += math.comb(j, r) * z ** (j - r) * (T - z) ** (r + be + 1) / (r + be + 1)
    return H * (tot + left - right) / T ** j


class Smooth:
    """Smooth integrands given by samples at N midpoints; represented exactly (the 'N-model') by
    f = sum_{k<N} a_k h_k + c_1 (t/T) + c_2 (t/T)^2 from the endpoint-corrected DCT of covfourier.batch.
    The representation is H-independent; all covariances below are exact for it."""
    kind = 'smooth'

    def __init__(self, F_samples, T=1.0, deg=8, max_ratio=1e3, rule='a_posteriori', loc=None, meta=None):
        F = np.atleast_2d(np.asarray(F_samples, dtype=float))
        self.T = float(T)
        self.N = F.shape[1]
        self.m = F.shape[0]
        w = _endpoint_weights(deg)
        a, c, use = endpoint_corrected_dct(F, self.T, w, deg, max_ratio, rule)
        self.A = np.ascontiguousarray(a)            # (N, m)
        self.C = np.ascontiguousarray(c)            # (2, m)
        self.corrected = use
        self._loc = loc
        self.meta = meta or {}
        self._cache = {}

    def loc(self):
        return self._loc

    def subset(self, idx):
        s = Smooth.__new__(Smooth)
        s.T, s.N = self.T, self.N
        s.A = np.ascontiguousarray(self.A[:, idx]); s.C = np.ascontiguousarray(self.C[:, idx])
        s.m = s.A.shape[1]
        s.corrected = None if self.corrected is None else np.asarray(self.corrected)[idx]
        s._loc = None if self._loc is None else np.asarray(self._loc)[idx]
        s.meta = self.meta; s._cache = {}
        return s

    def _prep(self, H):
        if H not in self._cache:
            G = CosineGram(H, self.T)
            QN = np.stack([poly_cos_moments(H, j, self.N, self.T) / self.T ** j for j in (1, 2)], axis=1)
            P = np.array([[poly_poly(H, i, j, self.T) / self.T ** (i + j) for j in (1, 2)] for i in (1, 2)])
            GA = gram_matmat(G, self.A)
            self._cache = {H: (G, QN, P, GA)}
        return self._cache[H]

    def moments(self, H, N=None):
        G, QN, P, GA = self._prep(H)
        b = GA + QN @ self.C                        # (N, m)
        N = self.N if N is None else N
        if N > self.N:
            raise ValueError('N-model has only %d cosines' % self.N)
        return np.ascontiguousarray(b[:N].T)

    def norms(self, H):
        G, QN, P, GA = self._prep(H)
        return (np.einsum('km,km->m', self.A, GA) + 2 * np.einsum('km,kj,jm->m', self.A, QN, self.C)
                + np.einsum('im,ij,jm->m', self.C, P, self.C))

    def K(self, H, other=None):
        if other is not None:
            raise NotImplementedError
        G, QN, P, GA = self._prep(H)
        X = self.A.T @ QN @ self.C
        Kx = self.A.T @ GA + X + X.T + self.C.T @ P @ self.C
        return 0.5 * (Kx + Kx.T)

    def cross_points(self, H, z):
        z = np.asarray(z, dtype=float)
        bz = Points(z, self.T).moments(H, self.N)                    # (M, N)
        pz = np.stack([_poly_indicator(H, z, j, self.T) for j in (1, 2)], axis=0)   # (2, M)
        return self.A.T @ bz.T + self.C.T @ pz

    def cosine(self, N=None):
        raise NotImplementedError('smooth kind uses the exact N-model majorant')


# ----------------------------------------------------------------------------------------------- CF basis
class CFBasis:
    """Prepared CovFourier family for (H, n): coordinates from exact moments and the cosine
    representation Phi (Naux x D) of the orthonormal functions."""

    def __init__(self, H, n, Naux=None, enrich=True, T=1.0):
        t0 = time.perf_counter()
        self.H, self.n, self.T = float(H), int(n), float(T)
        self.Naux = int(Naux or 4 * n)
        self.enrich = bool(enrich)
        self.cf = CovarianceFourier(self.H, self.n, T=self.T, enrich=self.enrich, Naux=self.Naux)
        self.L = self.cf.L
        self.D = self.cf.D
        self.G = self.cf.G
        self.dw = None
        if self.enrich:
            # one re-orthogonalization pass of the residual directions against h_0..h_{n-1} (the engine's
            # single pass leaves a cross defect ~1e-11 at n = 256, amplified by 1/||(I-P_n) t^j||_H)
            d = self.cf.dw.copy()
            for _ in range(2):
                Gd = gram_matmat(self.G, d)
                X = cho_solve((self.L, True), Gd[:self.n])
                d[:self.n] -= X
                Gd = gram_matmat(self.G, d)
                S = d.T @ Gd
                Rc = cholesky(0.5 * (S + S.T), lower=False)
                d = d @ solve_triangular(Rc, np.eye(2), lower=False)
            self.dw = d
        self.setup_seconds = time.perf_counter() - t0

    def coords(self, B):
        """B: (m, >= Naux) exact moments  ->  Gamma (m, D)."""
        n = self.n
        c = solve_triangular(self.L, B[:, :n].T, lower=True).T
        if not self.enrich:
            return c
        rho = B[:, :self.Naux] @ self.dw
        return np.hstack([c, rho])

    def phi_coeffs(self, Gm):
        """Cosine coefficients (rows, length Naux) of sum_j Gm[r, j] phi_j for each row r of Gm (r x D)."""
        n = self.n
        out = np.zeros((Gm.shape[0], self.Naux))
        out[:, :n] = solve_triangular(self.L.T, Gm[:, :n].T, lower=False).T
        if self.enrich:
            out += Gm[:, n:] @ self.dw.T
        return out

    def orthonormality_defect(self):
        """max |Phi^T G Phi - I| (diagnostic)."""
        Phi = self.phi_coeffs(np.eye(self.D)).T          # (Naux, D)
        GPhi = gram_matmat(self.G, Phi)
        M = Phi.T @ GPhi
        return float(np.max(np.abs(M - np.eye(self.D))))


class Features:
    """Generic features: Gamma (m x r) and residual energies e2 (m,) of a PSD-ordered approximation
    K = Gamma Gamma^T + R, R >= 0, diag(R) = e2.  name, prep/proj seconds, diagnostics."""

    def __init__(self, Gamma, e2, name='', valid=True, info=None):
        self.Gamma = np.asarray(Gamma, dtype=float)
        e2 = np.asarray(e2, dtype=float)
        self.n_negative = int(np.sum(e2 < 0))
        self.min_e2 = float(e2.min()) if e2.size else 0.0
        self.e2 = np.maximum(e2, 0.0)
        self.name = name
        self.valid = valid
        self.info = info or {}

    @property
    def trR(self):
        return float(self.e2.sum())


def cf_features(design, basis, H=None, B=None, norms=None):
    """CovFourier features of a design for a prepared basis (exact moments)."""
    H = basis.H if H is None else H
    t0 = time.perf_counter()
    if B is None:
        B = design.moments(H, basis.Naux)
    Gam = basis.coords(B)
    nrm = design.norms(H) if norms is None else norms
    e2 = nrm - np.einsum('ij,ij->i', Gam, Gam)
    f = Features(Gam, e2, name='CF' if basis.enrich else 'COS', info={'proj_seconds': time.perf_counter() - t0,
                                                                       'prep_seconds': basis.setup_seconds})
    return f


# ----------------------------------------------------------------------------------------------- majorant
class CFMajorant:
    """beta(v) >= v^T R v for the CovFourier residuals (points, boxes: exact N-cosine Gram + L2 tail through
    Lambda_N; smooth: exact Gram of the N-model, no tail)."""

    def __init__(self, design, basis, feats, Nmaj=None):
        self.design, self.basis, self.feats = design, basis, feats
        self.H = basis.H
        self.kind = design.kind
        if self.kind == 'smooth':
            self.N = design.N
            G, QN, P, GA = design._prep(self.H)
            self.G, self.QN, self.P = G, QN, P
        else:
            self.N = int(Nmaj or basis.Naux)
            self.Acos = np.ascontiguousarray(design.cosine(self.N).T)    # (N, m)
            self.G = basis.G
            self.LamN = Lambda_J(self.H, self.N, design.T)

    def beta(self, V):
        V = np.asarray(V, dtype=float)
        if V.ndim == 1:
            V = V[:, None]
        Gam = self.feats.Gamma
        GtV = Gam.T @ V                                     # (D, B)
        Pc = self.basis.phi_coeffs(GtV.T).T                # (Naux, B)
        if self.kind == 'smooth':
            E = self.design.A @ V                          # (N, B)
            E[:Pc.shape[0]] -= Pc
            cbar = self.design.C @ V                       # (2, B)
            GE = gram_matmat(self.G, E)
            val = (np.einsum('kb,kb->b', E, GE) + 2 * np.einsum('kb,kj,jb->b', E, self.QN, cbar)
                   + np.einsum('ib,ij,jb->b', cbar, self.P, cbar))
            return np.maximum(val, 0.0), {'tail': np.zeros(V.shape[1])}
        AV = self.Acos @ V                                 # (N, B)
        E = AV.copy()
        E[:Pc.shape[0]] -= Pc
        GE = gram_matmat(self.G, E)
        head = np.maximum(np.einsum('kb,kb->b', E, GE), 0.0)
        l2 = self.design.l2_quad(V)
        tail_l2 = np.maximum(l2 - np.einsum('kb,kb->b', AV, AV), 0.0)
        tail = self.LamN * tail_l2
        return (np.sqrt(head) + np.sqrt(tail)) ** 2, {'tail': tail, 'head': head}


class ExactMajorant:
    """beta(v) = v^T K v - |Gamma^T v|^2 exactly (needs K; reference only, O(m^2))."""

    def __init__(self, K, feats):
        self.K, self.feats = K, feats

    def beta(self, V):
        V = np.asarray(V, dtype=float)
        if V.ndim == 1:
            V = V[:, None]
        GtV = self.feats.Gamma.T @ V
        return np.maximum(np.einsum('ib,ib->b', V, self.K @ V) - np.einsum('jb,jb->b', GtV, GtV), 0.0), {}


# ----------------------------------------------------------------------------------------------- algebra
class LowRank:
    """C(c) = F F^T + c I via a thin SVD of F (m x r)."""

    def __init__(self, F):
        F = np.asarray(F, dtype=float)
        self.m, self.r = F.shape
        if self.r == 0:
            self.U = np.zeros((self.m, 0)); self.lam = np.zeros(0)
        else:
            U, s, _ = np.linalg.svd(F, full_matrices=False)
            self.U, self.lam = U, s ** 2
            self.r = s.size                    # min(m, number of features)

    def logdet(self, c):
        return (self.m - self.r) * math.log(c) + float(np.sum(np.log(self.lam + c)))

    def proj(self, Y):
        UY = self.U.T @ Y
        perp = Y - self.U @ UY
        return UY, perp

    def solve(self, Y, c, UYp=None):
        UY, perp = self.proj(Y) if UYp is None else UYp
        return perp / c + self.U @ (UY / (self.lam + c)[:, None])

    def quad(self, Y, c, UYp=None):
        UY, perp = self.proj(Y) if UYp is None else UYp
        return np.einsum('ib,ib->b', perp, perp) / c + np.einsum('jb,jb->b', UY, UY / (self.lam + c)[:, None])


class DiagLowRank:
    """C = F F^T + diag(d) (Woodbury)."""

    def __init__(self, F, d):
        self.F = np.asarray(F, dtype=float); self.d = np.asarray(d, dtype=float)
        Fd = self.F / self.d[:, None]
        S = np.eye(self.F.shape[1]) + self.F.T @ Fd
        self.Ls = cholesky(S, lower=True)
        self.Fd = Fd

    def logdet(self):
        return float(np.sum(np.log(self.d))) + 2.0 * float(np.sum(np.log(np.diag(self.Ls))))

    def solve(self, Y):
        Y = np.asarray(Y, dtype=float)
        Yd = Y / self.d[:, None] if Y.ndim == 2 else Y / self.d
        X = cho_solve((self.Ls, True), self.F.T @ Yd)
        return Yd - self.Fd @ X

    def quad(self, Y):
        return np.einsum('ib,ib->b', Y, self.solve(Y))


def _reml_parts(solveC, h, Y):
    """Given a solver for C, return (logdet(h^T C^-1 h) - logdet(h^T h), P y) for the restricted likelihood."""
    Ch = solveC(h)
    M = h.T @ Ch
    Lm = cholesky(0.5 * (M + M.T), lower=True)
    CY = solveC(Y)
    PY = CY - Ch @ cho_solve((Lm, True), h.T @ CY)
    Lh = cholesky(h.T @ h, lower=True)
    return 2 * float(np.sum(np.log(np.diag(Lm)))) - 2 * float(np.sum(np.log(np.diag(Lh)))), PY


def exact_evidence(K, Y, tau2, h=None, Sigma_beta=None, reml=False):
    """Exact log marginal likelihood (dense Cholesky) for each column of Y."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    m = K.shape[0]
    A = K + tau2 * np.eye(m)
    if h is not None and Sigma_beta is not None and not reml:
        A = A + h @ Sigma_beta @ h.T
    Lc = cholesky(A, lower=True)
    ld = 2.0 * float(np.sum(np.log(np.diag(Lc))))
    solve = lambda X: cho_solve((Lc, True), X)
    if reml:
        p = h.shape[1]
        extra, PY = _reml_parts(solve, h, Y)
        q = np.einsum('ib,ib->b', Y, PY)
        return -0.5 * (ld + extra) - 0.5 * q - 0.5 * (m - p) * LOG2PI
    q = np.einsum('ib,ib->b', Y, solve(Y))
    return -0.5 * ld - 0.5 * q - 0.5 * m * LOG2PI


def bracket(feats, Y, tau2, h=None, Sigma_beta=None, reml=False, majorant=None, working=True):
    """Evidence bracket and working-model evidence for each column of Y (m x B).

    Returns dict with arrays (B,): L, U, U0 (trace), U1 (per-vector diagonal), U2 (majorant, if given), W,
    and scalars rho = tr R / tau2, logdet parts, timing."""
    t0 = time.perf_counter()
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    m, B = Y.shape
    Gam = feats.Gamma
    F = Gam
    if h is not None and Sigma_beta is not None and not reml:
        F = np.hstack([Gam, h @ np.linalg.cholesky(Sigma_beta)])
    lr = LowRank(F)
    trR = feats.trR
    out = {'rho': trR / tau2, 'trR': trR}
    UYp = lr.proj(Y)
    if reml:
        p = h.shape[1]
        sC = lambda X: lr.solve(X, tau2)
        extra, PY = _reml_parts(sC, h, Y)
        ld = lr.logdet(tau2) + extra
        q = np.einsum('ib,ib->b', Y, PY)
        v = PY
        sCt = lambda X: lr.solve(X, tau2 + trR)
        extra_t, PYt = _reml_parts(sCt, h, Y)
        q0 = np.einsum('ib,ib->b', Y, PYt)
        nconst = 0.5 * (m - p) * LOG2PI
    else:
        ld = lr.logdet(tau2)
        v = lr.solve(Y, tau2, UYp)
        q = np.einsum('ib,ib->b', Y, v)
        q0 = lr.quad(Y, tau2 + trR, UYp)
        nconst = 0.5 * m * LOG2PI
    e = np.sqrt(feats.e2)
    q1 = q - (np.abs(v).T @ e) ** 2
    qlo = np.maximum(q0, q1)
    out['L'] = -0.5 * ld - 0.5 * q - nconst - trR / (2 * tau2)
    # diagonal refinement (Hadamard: logdet(I + R/tau2) <= sum log(1 + e_i^2/tau2); the collapsed bound of
    # Titsias 2025 has this form); always >= L.  Valid for REML by Cauchy interlacing.
    out['Ld'] = -0.5 * ld - 0.5 * q - nconst - 0.5 * float(np.sum(np.log1p(feats.e2 / tau2)))
    out['U0'] = -0.5 * ld - 0.5 * q0 - nconst
    out['U1'] = -0.5 * ld - 0.5 * q1 - nconst
    if majorant is not None:
        bv, binfo = majorant.beta(v)
        q2 = q - bv
        out['U2'] = -0.5 * ld - 0.5 * q2 - nconst
        out['beta'] = bv
        out['beta_tail'] = binfo.get('tail')
        qlo = np.maximum(qlo, q2)
    out['U'] = -0.5 * ld - 0.5 * qlo - nconst
    out['logC'] = -0.5 * ld - 0.5 * q - nconst
    out['vRv_proxy'] = (np.abs(v).T @ e) ** 2
    if working:
        dW = feats.e2 + tau2
        if h is not None and Sigma_beta is not None and not reml:
            dl = DiagLowRank(F, dW)
            out['W'] = -0.5 * dl.logdet() - 0.5 * dl.quad(Y) - nconst
        else:
            dl = DiagLowRank(Gam, dW)
            if reml:
                extra_w, PYw = _reml_parts(dl.solve, h, Y)
                out['W'] = -0.5 * (dl.logdet() + extra_w) - 0.5 * np.einsum('ib,ib->b', Y, PYw) - nconst
            else:
                out['W'] = -0.5 * dl.logdet() - 0.5 * dl.quad(Y) - nconst
    out['seconds'] = time.perf_counter() - t0
    return out


# ----------------------------------------------------------------------------------------------- baselines
def ip_features(design, H, z, K_diag=None, jitters=(0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4)):
    """Inducing points u_j = B_H(z_j): Gamma = K_fu L_uu^{-T} (Nystrom, PSD ordering kept with jitter)."""
    t0 = time.perf_counter()
    z = np.asarray(z, dtype=float)
    Pz = Points(z, design.T)
    Kuu = Pz.K(H)
    Kfu = design.cross_points(H, z)
    md = float(np.mean(np.diag(Kuu)))
    lu, used = None, None
    for j in jitters:
        try:
            lu = cholesky(Kuu + j * md * np.eye(len(z)), lower=True)
            if np.all(np.isfinite(lu)):
                used = j
                break
        except Exception:
            continue
    ev = np.linalg.eigvalsh(Kuu)
    cond = float(ev[-1] / max(ev[0], 1e-300)) if ev[0] > 0 else float('inf')
    Gam = solve_triangular(lu, Kfu.T, lower=True).T
    kd = design.norms(H) if K_diag is None else K_diag
    e2 = kd - np.einsum('ij,ij->i', Gam, Gam)
    return Features(Gam, e2, name='IP-VFE', info={'jitter_rel': used, 'cond_Kuu': cond,
                                                  'seconds': time.perf_counter() - t0, 'M': len(z)})


def oracle_features(evals, evecs, D, K_diag):
    """Top-D eigenpairs of K (needs K): the floor tr R >= sum_{n>D} lambda_n."""
    lam = evals[::-1][:D]; V = evecs[:, ::-1][:, :D]
    Gam = V * np.sqrt(np.maximum(lam, 0.0))[None, :]
    e2 = K_diag - np.einsum('ij,ij->i', Gam, Gam)
    return Features(Gam, e2, name='ORACLE', info={'floor': float(np.sum(np.maximum(evals[::-1][D:], 0.0)))})


def cos_classical_features(design, H, D):
    """Classical cosine truncation (not an H-projection): Q = A_D G_D A_D^T, residual not PSD in general."""
    A = design.cosine(D)
    G = CosineGram(H, design.T).dense(D)
    Lg = cholesky(G, lower=True)
    Gam = A @ Lg
    e2 = design.norms(H) - np.einsum('ij,ij->i', Gam, Gam)
    return Features(Gam, e2, name='COS-classical', valid=False)


# ----------------------------------------------------------------------------------------------- profiles
def profile_bounds(lr_or_feats, Y, kappas, trR=None, e=None, majorant=None, h=None, reml=False, beta_cache=None):
    """Profile over sigma^2 (closed form) of L, U and log N(y; 0, sigma^2 (Q + kappa I)) on a grid of kappa.

    Q in sigma = 1 units.  Returns arrays (len(kappas), B) of L*, U*, C* (plain low-rank evidence) and
    sigma2 estimates."""
    raise NotImplementedError
