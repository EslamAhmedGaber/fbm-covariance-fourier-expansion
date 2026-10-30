"""Exact covariance moments for validation inputs (never used by the production rule).

For each input family: classical cosine coefficients a_k = int f h_k, covariance
moments b_k = <f, h_k>_H and the norm ||f||_H^2, in double precision through the
Kummer family M_a.  These are independent references for the general procedure.
"""
import math
import numpy as np
from scipy.special import hyp1f1 as _hyp1f1_real, beta as _B
from .kummer import M
from .gram import power_cosine_moments, poly_cos_moments, poly_poly, _check


def _Fbeta_real(b, z, T):
    """F_b(z) = int_0^T u^(b-1) e^{z u} du for real z."""
    return T ** b * _hyp1f1_real(b, b + 1.0, z * T) / b


def _Fbeta_imag(b, w, T):
    """F_b(i w) for real w (vector)."""
    return T ** b * M(b, np.asarray(w, dtype=float) * T) / b


class Exponential:
    """f(t) = exp(a t), real a != 0."""
    def __init__(self, a):
        self.a = float(a)

    def __call__(self, t):
        return np.exp(self.a * np.asarray(t, dtype=float))

    def cosine(self, N, T=1.0):
        a = self.a; k = np.arange(N); w = k * np.pi / T
        out = math.sqrt(2.0 / T) * a * (np.exp(a * T) * (-1.0) ** k - 1.0) / (a * a + w * w)
        out[0] = (math.exp(a * T) - 1.0) / a / math.sqrt(T)
        return out

    def norm2(self, H, T=1.0):
        H, T = _check(H, T); a = self.a; b = 2 * H - 1
        if b == 0:
            return (math.exp(2 * a * T) - 1) / (2 * a)
        return H * b / a * (math.exp(2 * a * T) * _Fbeta_real(b, -a, T) - _Fbeta_real(b, a, T))

    def moments(self, H, N, T=1.0):
        H, T = _check(H, T); a = self.a; b = 2 * H - 1
        if b == 0:
            return self.cosine(N, T)
        k = np.arange(N); w = k * np.pi / T
        Fa, Fma = _Fbeta_real(b, a, T), _Fbeta_real(b, -a, T)
        Fp, Fm = _Fbeta_imag(b, w, T), _Fbeta_imag(b, -w, T)
        z = a + 1j * w
        J = H * b / z * (np.exp(a * T) * (-1.0) ** k * (Fma + Fm) - Fa - Fp)
        out = math.sqrt(2.0 / T) * J.real
        F0 = T ** b / b
        out[0] = H * b / a * (math.exp(a * T) * (Fma + F0) - Fa - F0) / math.sqrt(T)
        return out


class Indicator:
    """f = 1_[0,tau)."""
    def __init__(self, tau):
        self.tau = float(tau)

    def __call__(self, t):
        return (np.asarray(t, dtype=float) < self.tau).astype(float)

    def cosine(self, N, T=1.0):
        k = np.arange(N); w = k * np.pi / T
        out = np.empty(N)
        out[0] = self.tau / math.sqrt(T)
        out[1:] = math.sqrt(2.0 / T) * np.sin(w[1:] * self.tau) / w[1:]
        return out

    def norm2(self, H, T=1.0):
        return self.tau ** (2 * float(H))

    def moments(self, H, N, T=1.0):
        H, T = _check(H, T); b = 2 * H - 1; tau = self.tau
        if b == 0:
            return self.cosine(N, T)
        k = np.arange(N); w = k * np.pi / T
        e = np.exp(1j * w * tau)
        p1 = e * tau ** (b + 1) * M(b + 1.0, -w * tau) / (b + 1)
        p2 = e * (T - tau) ** (b + 1) * M(b + 1.0, w * (T - tau)) / (b + 1)
        # Q_k = int_0^T h_k(u) H [u^b + sgn(tau-u)|tau-u|^b] du  (companion-manuscript notation)
        Pu = np.sqrt(2.0 / T) * (T ** (b + 1) * M(b + 1.0, w * T).real / (b + 1))
        out = H * (Pu + math.sqrt(2.0 / T) * (p1 - p2).real)
        out[0] = 0.5 * (tau ** (2 * H) + T ** (2 * H) - (T - tau) ** (2 * H)) / math.sqrt(T)
        return out


def _J_imag(H, nu1, nu2, T):
    """Bilinear <e^{i nu1 .}, e^{i nu2 .}>_H for real nu1, nu2 (exponential moments, Supplementary Material)."""
    b = 2 * H - 1
    F = lambda w: T ** b * complex(M(b, np.array([w * T]))[0]) / b
    s = nu1 + nu2
    if abs(s) > 1e-12:
        z = 1j * s
        return H * b / z * (np.exp(z * T) * (F(-nu1) + F(-nu2)) - F(nu1) - F(nu2))
    # b = -a branch: H b T^(b+1) int_0^1 x^(b-1)(1-x)[e^{a T x} + e^{-a T x}] dx
    x = nu1 * T
    Mb = complex(M(b, np.array([x]))[0]); Mb1 = complex(M(b + 1, np.array([x]))[0])
    I = Mb / b - Mb1 / (b + 1)                  # int_0^1 x^(b-1)(1-x) e^{i x s} dx
    return H * b * T ** (b + 1) * (I + np.conj(I))


class Sine:
    """f(t) = sin(nu t)."""
    def __init__(self, nu):
        self.nu = float(nu)

    def __call__(self, t):
        return np.sin(self.nu * np.asarray(t, dtype=float))

    def cosine(self, N, T=1.0):
        nu = self.nu; k = np.arange(N); w = k * np.pi / T
        def term(x):
            x = np.asarray(x, dtype=float)
            out = np.zeros_like(x)
            nz = np.abs(x) > 1e-14
            out[nz] = (1 - np.cos(x[nz] * T)) / x[nz]
            return out
        a = math.sqrt(2.0 / T) * 0.5 * (term(nu + w) + term(nu - w))
        a[0] = (1 - math.cos(nu * T)) / nu / math.sqrt(T)
        return a

    def norm2(self, H, T=1.0):
        H, T = _check(H, T)
        if H == 0.5:
            nu = self.nu
            return T / 2 - math.sin(2 * nu * T) / (4 * nu)
        nu = self.nu
        v = -_J_imag(H, nu, nu, T) / 4 - _J_imag(H, -nu, -nu, T) / 4 + _J_imag(H, nu, -nu, T) / 2
        return float(np.real(v))


class Power:
    """f = t^p, p > -H (moments only for integer p; cosine coefficients for all p > -1)."""
    def __init__(self, p):
        self.p = float(p)

    def __call__(self, t):
        return np.asarray(t, dtype=float) ** self.p

    def cosine(self, N, T=1.0):
        return power_cosine_moments(self.p, N, T)

    def norm2(self, H, T=1.0):
        return poly_poly(H, self.p, self.p, T)

    def moments(self, H, N, T=1.0):
        if abs(self.p - round(self.p)) > 1e-14 or self.p < 0:
            raise NotImplementedError('double-precision moments only for nonnegative integer powers')
        return poly_cos_moments(H, int(round(self.p)), N, T)
