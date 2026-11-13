"""Goal-oriented linear-Gaussian measurement design on covariance coordinates, with brackets for the exact objective.

Model (exact).  Unknowns: the fBm path (through its Wiener integrals) and a nuisance vector beta ~ N(0, Sigma_beta).
Candidate measurement i:  y_i = I_H(F_i) + h_i^T beta + eps_i,  eps_i ~ N(0, sigma_i^2) independent.
Target t:                 u_t = I_H(f_t) + b_t^T beta.
Truncated model.  With P_D the covariance-orthogonal projection onto V_D, coordinates gamma(F) of P_D F and residual
energies e(F)^2 = ||F - P_D F||_H^2, put x = (Z, beta) in R^d, d = D + p, a_i = (gamma(F_i), h_i), c_t = (gamma(f_t), b_t).
The truncated model drops the residuals:  y_i = a_i^T x + eps_i,  u_t = c_t^T x,  prior precision Pi = blockdiag(I_D, Sigma_beta^-1).

Design bracket (paper, Proposition 'design bracket').  For every design S,
    V0_t(S) <= V_t(S) <= V0_t(S) + (e_t + sum_{i in S} |a*_i| e_i)^2 <= V0_t(S) + (e_t + sqrt(rho_S V0_t(S)))^2,
V_t = exact posterior variance, V0_t = c_t^T P(S)^{-1} c_t with P(S) = Pi + sum_{i in S} a_i a_i^T / sigma_i^2,
a* = Sigma^-1 A_S P(S)^-1 c_t the truncated-model weights, rho_S = sum_{i in S} e_i^2 / sigma_i^2.
Gap certificate.  Phi0(w) = sum_t omega_t c_t^T P(w)^-1 c_t is convex on w in [0,1]^m; any Frank-Wolfe iterate gives a
lower bound on min_{|S|<=k} Phi0(S) <= min_{|S|<=k} Phi(S).  Hence Phi(S_hat) - min Phi <= U(S_hat) - LB.

Everything here is dense linear algebra in d dimensions plus O(m d) passes over the candidate rows; nothing depends
on what the candidates are.  Existing (already collected) data enter through P_existing.
"""
import math
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular, eigh


def _chol(P):
    return cho_factor(0.5 * (P + P.T), lower=True, check_finite=False)


class InfoDesign:
    """A: (m, d) candidate rows; sigma2: (m,) noise variances; C: (d, q) target rows; omega: (q,) weights;
    Pi: (d, d) prior precision (or P_existing including already collected data); e2: (m,) candidate residual
    energies; e2_t: (q,) target residual energies (both may be zero, e.g. for a dense reference model)."""

    def __init__(self, A, sigma2, C, omega=None, Pi=None, e2=None, e2_t=None):
        self.A = np.ascontiguousarray(A, dtype=float)
        self.m, self.d = self.A.shape
        self.s2 = np.broadcast_to(np.asarray(sigma2, dtype=float), (self.m,)).copy()
        self.C = np.asarray(C, dtype=float).reshape(self.d, -1)
        self.q = self.C.shape[1]
        self.omega = np.ones(self.q) if omega is None else np.asarray(omega, dtype=float)
        self.Pi = np.eye(self.d) if Pi is None else np.asarray(Pi, dtype=float)
        self.e2 = np.zeros(self.m) if e2 is None else np.asarray(e2, dtype=float)
        self.e2_t = np.zeros(self.q) if e2_t is None else np.asarray(e2_t, dtype=float)
        if np.any(self.s2 <= 0):
            raise ValueError('noise variances must be positive')

    # ------------------------------------------------------------------ objective for a set
    def precision(self, S):
        S = np.asarray(S, dtype=int)
        As = self.A[S] / np.sqrt(self.s2[S])[:, None]
        return self.Pi + As.T @ As

    def V0(self, S):
        """Truncated-model posterior variances of the q targets for design S (prior/existing data included)."""
        cf = _chol(self.precision(S))
        X = cho_solve(cf, self.C, check_finite=False)
        return np.einsum('dq,dq->q', self.C, X)

    def bracket(self, S):
        """Per-target V0, a-posteriori upper bound and uniform (Cauchy-Schwarz) upper bound for design S."""
        S = np.asarray(S, dtype=int)
        cf = _chol(self.precision(S))
        X = cho_solve(cf, self.C, check_finite=False)                 # P^-1 c_t
        v0 = np.einsum('dq,dq->q', self.C, X)
        a_star = (self.A[S] @ X) / self.s2[S][:, None]                # (|S|, q) truncated-model weights
        e = np.sqrt(np.maximum(self.e2[S], 0.0))
        et = np.sqrt(np.maximum(self.e2_t, 0.0))
        up_post = v0 + (et + np.abs(a_star).T @ e) ** 2
        rho = float(np.sum(self.e2[S] / self.s2[S]))
        up_unif = v0 + (et + np.sqrt(rho * v0)) ** 2
        return {'V0': v0, 'U_post': up_post, 'U_unif': up_unif, 'rho': rho,
                'Phi0': float(self.omega @ v0), 'Phi_U_post': float(self.omega @ up_post),
                'Phi_U_unif': float(self.omega @ up_unif)}

    # ------------------------------------------------------------------ greedy (information form)
    def greedy(self, k, exclude=()):
        """Greedy minimisation of Phi0 = sum_t omega_t V0_t over designs of size k (each candidate at most once).

        Maintains L = P^-1, u_i = a_i^T L a_i and W = A L C; one step costs O(m d)."""
        t0 = time.perf_counter()
        L = cho_solve(_chol(self.Pi), np.eye(self.d), check_finite=False)
        L = 0.5 * (L + L.T)
        u = np.einsum('ij,ij->i', self.A @ L, self.A)
        W = self.A @ (L @ self.C)                                      # (m, q)
        cw = self.C.copy()
        avail = np.ones(self.m, dtype=bool)
        avail[list(exclude)] = False
        S, hist = [], []
        phi = float(self.omega @ np.einsum('dq,dq->q', self.C, L @ self.C))
        hist.append(phi)
        for _ in range(int(k)):
            score = (W ** 2 @ self.omega) / (self.s2 + u)
            score[~avail] = -np.inf
            j = int(np.argmax(score))
            if not np.isfinite(score[j]) or score[j] <= 0:
                break
            v = L @ self.A[j]
            den = self.s2[j] + float(self.A[j] @ v)
            Av = self.A @ v
            vc = v @ cw                                                # (q,)
            L -= np.outer(v, v) / den
            u -= Av ** 2 / den
            W -= np.outer(Av, vc) / den
            phi -= float(score[j])
            S.append(j); avail[j] = False; hist.append(phi)
        return {'S': S, 'Phi0_history': hist, 'seconds': time.perf_counter() - t0}

    # ------------------------------------------------------------------ convex relaxation (Frank-Wolfe)
    def _phi_grad(self, P):
        cf = _chol(P)
        X = cho_solve(cf, self.C * np.sqrt(self.omega)[None, :], check_finite=False)    # P^-1 C_w
        phi = float(np.sum((self.C * np.sqrt(self.omega)[None, :]) * X))
        G = self.A @ X                                                 # (m, q)
        score = np.einsum('mq,mq->m', G, G) / self.s2                  # = -dPhi/dw_i
        return phi, score

    def frank_wolfe(self, k, iters=300, w0=None, tol=0.0, verbose=False):
        """Frank-Wolfe with exact line search on Phi0(w), w in [0,1]^m, sum(w) <= k.

        Returns the best certified lower bound  LB = max_t [Phi0(w_t) - g_t]  (g_t the Frank-Wolfe duality gap) on
        min_{|S|<=k} Phi0(S), the final relaxed value, and the history."""
        t0 = time.perf_counter()
        k = int(k)
        w = np.zeros(self.m) if w0 is None else np.asarray(w0, dtype=float).copy()
        Cw = self.C * np.sqrt(self.omega)[None, :]
        As = self.A / np.sqrt(self.s2)[:, None]
        P = self.Pi + (As * w[:, None]).T @ As
        best_lb, hist = -np.inf, []
        for it in range(int(iters)):
            phi, score = self._phi_grad(P)
            top = np.argpartition(-score, k - 1)[:k] if k < self.m else np.arange(self.m)
            s = np.zeros(self.m); s[top] = 1.0
            gap = float(score @ w - score[top].sum())                  # grad^T (s - w) = -gap_FW ... sign below
            fw_gap = -gap                                              # = grad^T (w - s) >= 0
            lb = phi - fw_gap
            best_lb = max(best_lb, lb)
            hist.append((phi, lb, fw_gap))
            if verbose and it % 25 == 0:
                print(it, phi, lb, fw_gap, flush=True)
            if fw_gap <= tol * max(abs(phi), 1e-300):
                break
            Ps = self.Pi + As[top].T @ As[top]
            Dlt = Ps - P
            R = np.linalg.cholesky(0.5 * (P + P.T))
            Ri = solve_triangular(R, np.eye(self.d), lower=True)
            lam, Q = eigh(Ri @ Dlt @ Ri.T)
            ch = Q.T @ (Ri @ Cw)
            c2 = np.sum(ch ** 2, axis=1)
            dphi = lambda g: -float(np.sum(c2 * lam / (1.0 + g * lam) ** 2))
            if dphi(1.0) <= 0:
                gam = 1.0
            else:
                lo, hi = 0.0, 1.0
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if dphi(mid) > 0:
                        hi = mid
                    else:
                        lo = mid
                gam = 0.5 * (lo + hi)
            w = (1 - gam) * w + gam * s
            P = (1 - gam) * P + gam * Ps
        phi, _ = self._phi_grad(P)
        return {'LB': float(best_lb), 'Phi0_relaxed': phi, 'w': w, 'history': hist, 'iterations': len(hist),
                'seconds': time.perf_counter() - t0}


# ---------------------------------------------------------------------- exact (dense) reference objective
def exact_variances(K_SS, k_St, var_t, sigma2_S, H_S=None, Sigma_beta=None):
    """Exact posterior variances of targets given y_S for the full (untruncated) Gaussian model.

    K_SS: exact covariance of the I_H(F_i), i in S; k_St: (|S|, q) exact cross-covariances with the targets'
    Wiener integrals; var_t: (q,) prior variances of the targets; nuisance enters as H_S Sigma_beta H_S^T
    (targets carry no nuisance)."""
    Ky = np.array(K_SS, dtype=float)
    if H_S is not None:
        Ky = Ky + H_S @ Sigma_beta @ H_S.T
    Ky[np.diag_indices_from(Ky)] += sigma2_S
    cf = _chol(Ky)
    X = cho_solve(cf, k_St, check_finite=False)
    return var_t - np.einsum('sq,sq->q', k_St, X)


class DenseGreedy:
    """Exact greedy on the untruncated model with on-demand covariance columns (pivoted-Cholesky updates).

    col(j) -> exact Cov(I_H(F_i), I_H(F_j)) for all candidates i (array of length m);
    diag:  (m,) exact Var(I_H(F_i));  kt: (m, q) exact Cov(I_H(F_i), I_H(f_t));  var_t: (q,) target variances.
    A Gaussian nuisance beta ~ N(0, Sigma_beta) with rows H (m, p) is added to the prior covariance exactly.
    forced: candidates conditioned on first (already collected data), in the given order.
    Memory O(m k); one step costs one exact column plus O(m k)."""

    def __init__(self, col, diag, kt, var_t, sigma2, omega=None, H=None, Sigma_beta=None):
        self.col = col
        self.diag = np.asarray(diag, dtype=float)
        self.m = len(self.diag)
        self.kt = np.asarray(kt, dtype=float).reshape(self.m, -1)
        self.var_t = np.asarray(var_t, dtype=float)
        self.s2 = np.broadcast_to(np.asarray(sigma2, dtype=float), (self.m,)).copy()
        self.omega = np.ones(len(self.var_t)) if omega is None else np.asarray(omega, dtype=float)
        self.H = None if H is None else np.asarray(H, dtype=float)
        self.Sb = None if Sigma_beta is None else np.asarray(Sigma_beta, dtype=float)

    def run(self, k, forced=()):
        t0 = time.perf_counter()
        cvar = self.diag + self.s2
        if self.H is not None:
            HS = self.H @ self.Sb
            cvar = cvar + np.einsum('mp,mp->m', HS, self.H)
        ccov = self.kt.copy()
        tvar = self.var_t.copy()
        G = []
        chosen = []
        hist = [float(self.omega @ tvar)]
        order = list(forced)
        n_forced = len(order)
        steps = n_forced + int(k)
        for step in range(steps):
            if step < n_forced:
                j = int(order[step])
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    score = (ccov ** 2 @ self.omega) / cvar
                score[chosen] = -np.inf
                j = int(np.argmax(score))
            c = np.array(self.col(j), dtype=float)
            if self.H is not None:
                c = c + HS @ self.H[j]
            c[j] += self.s2[j]
            for g in G:
                c -= g * g[j]
            piv = math.sqrt(cvar[j])
            g = c / piv
            gt = ccov[j] / piv
            cvar = cvar - g ** 2
            ccov = ccov - np.outer(g, gt)
            tvar = tvar - gt ** 2
            G.append(g)
            chosen.append(j)
            hist.append(float(self.omega @ tvar))
        return {'S': chosen[n_forced:], 'forced': chosen[:n_forced], 'Phi_history': hist, 'target_var': tvar,
                'seconds': time.perf_counter() - t0}


class BlockInfoDesign:
    """Information-form design for independent blocks (e.g. boreholes with independent histories) and one target
    that is a weighted sum of per-block targets:  u = sum_s c_s^T x_s.  Block s has prior/existing precision P_s
    (d x d) and target row c_s; candidate i belongs to block blk[i] with row A[i] and noise variance sigma2[i].
    Phi(S) = Var(u | y_S) in the truncated model = sum_s c_s^T P_s(S)^-1 c_s.  Greedy and Frank-Wolfe as in
    InfoDesign; memory O(n_blocks d^2 + m d)."""

    def __init__(self, A, blk, sigma2, P_blocks, C_blocks, e2=None, e2_t=None):
        self.A = np.ascontiguousarray(A, dtype=float)
        self.blk = np.asarray(blk, dtype=int)
        self.m, self.d = self.A.shape
        self.s2 = np.broadcast_to(np.asarray(sigma2, dtype=float), (self.m,)).copy()
        self.P = [np.asarray(P, dtype=float) for P in P_blocks]
        self.C = [np.asarray(c, dtype=float) for c in C_blocks]
        self.nb = len(self.P)
        self.e2 = np.zeros(self.m) if e2 is None else np.asarray(e2, dtype=float)
        self.e2_t = np.zeros(self.nb) if e2_t is None else np.asarray(e2_t, dtype=float)
        self.members = [np.where(self.blk == s)[0] for s in range(self.nb)]

    def _phi_block(self, s, P):
        cf = _chol(P)
        x = cho_solve(cf, self.C[s], check_finite=False)
        return float(self.C[s] @ x), x

    def value(self, S):
        """Phi0(S) and the bracket for the summed target: blocks are independent, so the exact variance is the sum of
        the block variances and the a-posteriori bracket of Proposition 'design bracket' is applied block by block."""
        S = np.asarray(S, dtype=int)
        phi, up = 0.0, 0.0
        for s in range(self.nb):
            Ss = S[self.blk[S] == s]
            P = self.P[s].copy()
            if len(Ss):
                As = self.A[Ss] / np.sqrt(self.s2[Ss])[:, None]
                P = P + As.T @ As
            v, x = self._phi_block(s, P)
            et = math.sqrt(max(float(self.e2_t[s]), 0.0))
            ae = 0.0
            if len(Ss):
                a_star = (self.A[Ss] @ x) / self.s2[Ss]
                ae = float(np.abs(a_star) @ np.sqrt(np.maximum(self.e2[Ss], 0.0)))
            phi += v; up += v + (et + ae) ** 2
        return {'Phi0': phi, 'Phi_U_post': up}

    def greedy(self, k):
        t0 = time.perf_counter()
        L = [cho_solve(_chol(P), np.eye(self.d), check_finite=False) for P in self.P]
        u = np.empty(self.m); W = np.empty(self.m)
        for s in range(self.nb):
            idx = self.members[s]
            AL = self.A[idx] @ L[s]
            u[idx] = np.einsum('ij,ij->i', AL, self.A[idx]); W[idx] = AL @ self.C[s]
        avail = np.ones(self.m, dtype=bool)
        phi = sum(float(self.C[s] @ L[s] @ self.C[s]) for s in range(self.nb))
        S, hist = [], [phi]
        for _ in range(int(k)):
            score = W ** 2 / (self.s2 + u)
            score[~avail] = -np.inf
            j = int(np.argmax(score))
            if not np.isfinite(score[j]) or score[j] <= 0:
                break
            s = self.blk[j]; idx = self.members[s]
            v = L[s] @ self.A[j]; den = self.s2[j] + float(self.A[j] @ v)
            Av = self.A[idx] @ v; vc = float(v @ self.C[s])
            L[s] -= np.outer(v, v) / den
            u[idx] -= Av ** 2 / den; W[idx] -= Av * vc / den
            phi -= float(score[j]); S.append(j); avail[j] = False; hist.append(phi)
        return {'S': S, 'Phi0_history': hist, 'seconds': time.perf_counter() - t0}

    def frank_wolfe(self, k, iters=100, w0=None):
        """Frank-Wolfe on the relaxation; returns the best certified lower bound on min_{|S|<=k} Phi0(S)."""
        t0 = time.perf_counter()
        k = int(k)
        w = np.zeros(self.m) if w0 is None else np.asarray(w0, dtype=float).copy()
        As = self.A / np.sqrt(self.s2)[:, None]
        def P_of(wv, s):
            idx = self.members[s]
            return self.P[s] + (As[idx] * wv[idx][:, None]).T @ As[idx]
        Ps = [P_of(w, s) for s in range(self.nb)]
        best, hist = -np.inf, []
        for it in range(int(iters)):
            phi, score = 0.0, np.empty(self.m)
            for s in range(self.nb):
                v, x = self._phi_block(s, Ps[s]); phi += v
                idx = self.members[s]
                score[idx] = (self.A[idx] @ x) ** 2 / self.s2[idx]
            top = np.argpartition(-score, k - 1)[:k]
            fw_gap = float(score[top].sum() - score @ w)
            best = max(best, phi - fw_gap); hist.append((phi, phi - fw_gap, fw_gap))
            if fw_gap <= 0:
                break
            sv = np.zeros(self.m); sv[top] = 1.0
            changed = sorted(set(self.blk[top]) | set(self.blk[w > 0]))
            Pt = {s: P_of(sv, s) for s in changed}
            eig = {}
            for s in changed:
                R = np.linalg.cholesky(0.5 * (Ps[s] + Ps[s].T)); Ri = solve_triangular(R, np.eye(self.d), lower=True)
                lam, Q = eigh(Ri @ (Pt[s] - Ps[s]) @ Ri.T); ch = Q.T @ (Ri @ self.C[s])
                eig[s] = (lam, ch ** 2)
            dphi = lambda g: -sum(float(np.sum(c2 * lam / (1.0 + g * lam) ** 2)) for lam, c2 in eig.values())
            if dphi(1.0) <= 0:
                gam = 1.0
            else:
                lo, hi = 0.0, 1.0
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if dphi(mid) > 0:
                        hi = mid
                    else:
                        lo = mid
                gam = 0.5 * (lo + hi)
            w = (1 - gam) * w + gam * sv
            for s in changed:
                Ps[s] = (1 - gam) * Ps[s] + gam * Pt[s]
        return {'LB': float(best), 'history': hist, 'iterations': len(hist), 'w': w, 'seconds': time.perf_counter() - t0}
