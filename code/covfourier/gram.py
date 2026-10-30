"""Closed-form covariance inner products for fBm Wiener integrands on [0,T].

<f,g>_H = H(2H-1) int int f(s) g(t) |s-t|^(2H-2) ds dt  (H > 1/2),  L^2 at H = 1/2.

From the companion manuscript (closed cosine Gram):
    G_00 = T^b,  G_kk = C [p_k - (1-b) r_k/(k pi)],
    G_kl = (2C/pi)(l r_l - k r_k)/(k^2-l^2)  (k != l, k+l even, k,l >= 1),
    G_0l = -sqrt2 C r_l/(pi l)  (l even >= 2),  zero if k+l odd,
with b = 2H-1, C = 2H T^b and p_k + i r_k = M_b(k pi) = 1F1(b; b+1; i k pi).

Derived here:
  * an O(N log N) mat-vec with G (odd/even-extension convolutions), giving the
    whole vector G a, not only the quadratic form;
  * polynomial-cosine moments <h_l, t^j>_H for all l through the same Kummer
    family M_a, a in {b, b+1, ..., b+j}, via the explicit kernel image K_H t^j.
"""
import math
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from scipy.special import beta as _B
from .kummer import M

SQ2 = math.sqrt(2.0)


def _check(H, T):
    H = float(H); T = float(T)
    if not (0.5 <= H < 1.0) or not T > 0:
        raise ValueError('require 1/2 <= H < 1 and T > 0')
    return H, T


def kummer_grid(H, N):
    """M_b(k pi) for k = 0..N-1 (index 0 -> 1)."""
    b = 2.0 * H - 1.0
    k = np.arange(N, dtype=float)
    if b == 0.0:
        return np.ones(N, dtype=complex)
    return M(b, np.pi * k)


class CosineGram:
    """Closed cosine Gram (companion manuscript) with dense and structured (FFT) access."""

    def __init__(self, H, T=1.0):
        self.H, self.T = _check(H, T)
        self.b = 2.0 * self.H - 1.0
        self.C = 2.0 * self.H * self.T ** self.b
        self._m = np.ones(1, dtype=complex)

    def m(self, N):
        if len(self._m) < N:
            self._m = kummer_grid(self.H, max(N, 2 * len(self._m)))
        return self._m[:N]

    def dense(self, N, block=512):
        if self.b == 0.0:
            return np.eye(N)
        m = self.m(N); p = m.real; r = m.imag; C = self.C
        k = np.arange(N, dtype=float)
        G = np.zeros((N, N))
        for s in range(1, N, block):          # row blocks: O(block*N) temporaries
            e = min(s + block, N)
            K = k[s:e, None]; L = k[None, 1:]
            same = ((K + L) % 2 == 0) & (K != L)
            with np.errstate(divide='ignore', invalid='ignore'):
                val = (2 * C / np.pi) * (L * r[None, 1:] - K * r[s:e, None]) / (K ** 2 - L ** 2)
            G[s:e, 1:] = np.where(same, val, 0.0)
        kk = np.arange(1, N)
        G[kk, kk] = C * (p[kk] - (1 - self.b) * r[kk] / (kk * np.pi))
        G[0, 0] = self.T ** self.b
        ev = np.arange(2, N, 2)
        G[0, ev] = G[ev, 0] = -SQ2 * C * r[ev] / (np.pi * ev)
        return G

    @staticmethod
    def _parity_conv(x, sign):
        """y_k = sum_{l>=1, l!=k, l=k mod 2} x_l/(k-l) + sign*sum_{l>=1, l=k mod 2} x_l/(k+l),
        for k = 1..n (x has length n, x[i] <-> index i+1).  FFT linear convolution."""
        n = len(x)
        ext = np.zeros(2 * n + 1)
        ext[n + 1:] = x
        ext[:n] = sign * x[::-1]
        d = np.arange(-2 * n, 2 * n + 1)
        q = np.zeros(len(d))
        msk = (d != 0) & (d % 2 == 0)
        q[msk] = 1.0 / d[msk]
        L = len(ext) + len(q) - 1
        nf = next_fast_len(L, real=True)
        full = irfft(rfft(ext, nf) * rfft(q, nf), nf)[:L]
        # full[i] = sum_j ext[j] q[i-j]; ext index j <-> coefficient index j-n,
        # q index <-> lag d + 2n.  Output coefficient index k = i - 3n.
        return full[3 * n + 1: 4 * n + 1]

    def matvec(self, a):
        a = np.asarray(a, dtype=float)
        N = len(a)
        if self.b == 0.0:
            return a.copy()
        m = self.m(N); p = m.real; r = m.imag; C = self.C
        out = np.empty(N)
        out[0] = self.T ** self.b * a[0]
        if N == 1:
            return out
        k = np.arange(1, N, dtype=float)
        v = a[1:]; rr = r[1:]
        odd = self._parity_conv(rr * v, -1.0)     # (A - B)(r v)
        even = self._parity_conv(v, +1.0)         # (A + B) v
        out[1:] = C * (p[1:] + self.b * rr / (np.pi * k)) * v + (C / np.pi) * (odd - rr * even)
        ev = (np.arange(1, N) % 2 == 0)
        g0 = -SQ2 * C * rr[ev] / (np.pi * k[ev])
        out[0] += np.dot(g0, v[ev])
        out[1:][ev] += g0 * a[0]
        return out

    def quad(self, a):
        a = np.asarray(a, dtype=float)
        return float(np.dot(a, self.matvec(a)))


def power_cosine_moments(q, N, T=1.0):
    """P_l(q) = int_0^T s^q h_l(s) ds, l = 0..N-1, q > -1 (classical cosine coefficients of s^q)."""
    l = np.arange(N, dtype=float)
    out = np.empty(N)
    out[0] = T ** (q + 0.5) / (q + 1.0)
    if N > 1:
        out[1:] = math.sqrt(2.0 / T) * (T ** (q + 1.0) * M(q + 1.0, np.pi * l[1:]).real / (q + 1.0))
    return out


def poly_cos_moments(H, j, N, T=1.0):
    """<h_l, t^j>_H for l = 0..N-1 and integer j >= 0."""
    H, T = _check(H, T)
    b = 2.0 * H - 1.0
    if b == 0.0:
        return power_cosine_moments(float(j), N, T)
    sign = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    acc = _B(j + 1.0, b) * power_cosine_moments(j + b, N, T)
    for m in range(j + 1):
        cm = math.comb(j, m) / (b + m)
        for i in range(j - m + 1):
            coef = cm * math.comb(j - m, i) * T ** (j - m - i) * (-1.0) ** i
            acc = acc + coef * sign * power_cosine_moments(b + m + i, N, T)
    return H * b * acc


def poly_poly(H, i, j, T=1.0):
    """<t^i, t^j>_H for real exponents i, j > -H (mixed power inner products)."""
    H, T = _check(H, T)
    b = 2.0 * H - 1.0
    if b == 0.0:
        return T ** (i + j + 1.0) / (i + j + 1.0)
    return H * b * T ** (i + j + 2 * H) / (i + j + 2 * H) * (_B(i + 1.0, b) + _B(j + 1.0, b))


def raw_cosine_coefficients(D, T=1.0, N=None):
    """Classical cosine coefficients (rows: raw functions h0, t/T, (t/T)^2, h1..h_{D-3};
    columns l = 0..N-1).  Polynomial rows are exact P_l(j)/T^j; cosine rows are unit vectors."""
    N = D - 2 if N is None else N
    R = np.zeros((D, N))
    R[0, 0] = 1.0
    R[1] = power_cosine_moments(1.0, N, T) / T
    R[2] = power_cosine_moments(2.0, N, T) / T ** 2
    for k in range(1, D - 2):
        if k < N:
            R[k + 2, k] = 1.0
    return R


def augmented_gram(H, D, T=1.0, cos_gram=None):
    """Gram of the nested raw sequence [h0, t/T, (t/T)^2, h1, ..., h_{D-3}]."""
    H, T = _check(H, T)
    if D < 3:
        raise ValueError('D >= 3 required')
    n = D - 2
    cg = CosineGram(H, T) if cos_gram is None else cos_gram
    Gc = cg.dense(n)
    idx = [0] + list(range(3, D))
    A = np.zeros((D, D))
    A[np.ix_(idx, idx)] = Gc
    for j in (1, 2):
        q = poly_cos_moments(H, j, n, T) / T ** j
        A[j, idx] = q
        A[idx, j] = q
        for i in (1, 2):
            A[j, i] = poly_poly(H, j, i, T) / T ** (i + j)
    return A
