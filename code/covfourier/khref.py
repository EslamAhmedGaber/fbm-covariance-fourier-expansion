"""Independent reference for fBm Wiener-integral covariances <f, g>_H on [0, 1] (shares no formula with the expansion).

<f, g>_H = int_0^1 f(u) (K_H g)(u) du,   (K_H g)(u) = H beta int_0^1 g(v) |u - v|^(beta - 1) dv,   beta = 2H - 1 > 0.

Composite Gauss-Legendre quadrature on panels split at the declared feature points of the integrands (logging times,
window ends) and at the singular point v = u, subdivided to width <= h_max and graded geometrically towards every
split point.  On the two inner subpanels that touch v = u the weight w^(beta-1) is removed by the substitution
w = delta x^(1/beta), which avoids Gauss-Jacobi nodes with exponents near -1.  The outer integral uses the same
composite rule with grading towards 0, 1 and the features ((K_H g)(u) ~ u^beta near the endpoints).
Cost: O(n_outer * n_inner) evaluations per batch of functions; used as the exact reference and as the dense-GP
covariance builder for physical kernels.  At H = 1/2, <f, g>_H = int f g.
"""
import math

import numpy as np


def _gl(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def _graded_panels(a, b, toward_a, toward_b, h_max, levels, sigma):
    """Panel edges on [a, b]: uniform subdivision to width <= h_max, then geometric grading of the end panels
    towards a and/or b (levels refinements with ratio sigma)."""
    L = b - a
    if L <= 0:
        return np.array([a, b])
    n = max(1, int(math.ceil(L / h_max)))
    h = L / n
    inner = list(a + L * np.arange(1, n) / n)
    if toward_a:
        inner += [a + h * sigma ** k for k in range(1, levels + 1)]
    if toward_b:
        inner += [b - h * sigma ** k for k in range(1, levels + 1)]
    # the end points are exactly a and b (the singular-point test compares them with ==)
    inner = sorted(x for x in set(inner) if a < x < b)
    return np.array([a] + inner + [b])


class KHQuad:
    def __init__(self, H, features=(), h_max=1.0 / 128, levels=14, sigma=0.3, n_gl=10, n_sing=12,
                 outer_h_max=None, outer_levels=None):
        self.H = float(H)
        if not (0.5 <= self.H < 1.0):
            raise ValueError('need 1/2 <= H < 1')
        self.b = 2.0 * self.H - 1.0
        f = sorted(set([0.0, 1.0] + [float(x) for x in features if 0.0 < x < 1.0]))
        self.features = np.array(f)
        self.h_max, self.levels, self.sigma = h_max, levels, sigma
        self.xg, self.wg = _gl(n_gl)
        self.xs, self.ws = _gl(n_sing)
        oh = outer_h_max or h_max
        ol = outer_levels or levels
        edges = []
        for a, c in zip(f[:-1], f[1:]):
            edges.append(_graded_panels(a, c, True, True, oh, ol, sigma))
        e = np.unique(np.concatenate(edges))
        lo, hi = e[:-1], e[1:]
        self.u = (lo[:, None] + (hi - lo)[:, None] * self.xg[None, :]).ravel()
        self.wu = ((hi - lo)[:, None] * self.wg[None, :]).ravel()

    def _inner_nodes(self, u):
        """Nodes and weights (including the kernel |u - v|^(beta-1)) for int_0^1 g(v)|u-v|^(beta-1) dv."""
        b = self.b
        pts = sorted(set(list(self.features) + [u]))
        V, W = [], []
        for a, c in zip(pts[:-1], pts[1:]):
            ta = (a == u); tb = (c == u)
            e = _graded_panels(a, c, True, True, self.h_max, self.levels, self.sigma)
            for lo, hi in zip(e[:-1], e[1:]):
                if (ta and lo == a) or (tb and hi == c):          # subpanel touching the singular point
                    dlt = hi - lo
                    x = self.xs
                    wv = dlt * x ** (1.0 / b)                       # distance to u
                    v = u + wv if ta else u - wv
                    V.append(v); W.append((dlt ** b / b) * self.ws)
                else:
                    v = lo + (hi - lo) * self.xg
                    V.append(v); W.append((hi - lo) * self.wg * np.abs(u - v) ** (b - 1.0))
        return np.concatenate(V), np.concatenate(W)

    def apply(self, values, chunk=1):
        """(K_H g_j)(u_k) at the outer nodes for all j.  values(v) -> array (q, len(v))."""
        if self.b == 0.0:
            return values(self.u)
        out = []
        for u in self.u:
            v, w = self._inner_nodes(float(u))
            out.append(values(v) @ w)
        return self.H * self.b * np.stack(out, axis=1)               # (q, n_outer)

    def gram(self, values_f, values_g=None):
        """Matrix <f_i, g_j>_H (q_f x q_g); values_g defaults to values_f (symmetrised)."""
        KG = self.apply(values_g or values_f)
        Fu = values_f(self.u)
        M = (Fu * self.wu[None, :]) @ KG.T
        if values_g is None:
            M = 0.5 * (M + M.T)
        return M

    @property
    def n_outer(self):
        return len(self.u)
