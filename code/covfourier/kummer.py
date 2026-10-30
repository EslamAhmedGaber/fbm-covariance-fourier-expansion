"""Double-precision evaluation of M_a(x) = 1F1(a; a+1; i x), a > 0, x real.

Independent implementation (NumPy, SciPy and mpmath only).

Identity used throughout:  M_a(x) = a * int_0^1 s^(a-1) exp(i x s) ds.
Hence  int_0^T s^q exp(i w s) ds = T^(q+1) M_(q+1)(w T)/(q+1)  for q > -1.

Evaluation strategy (all vectorized):
  |x| <= X_SWITCH : piecewise Chebyshev interpolation on [0, X_SWITCH]
                    (15 panels of width 4, 32 nodes each), tabulated once per a
                    from 30-digit mpmath values.  M_a is entire with
                    |d^n M_a/dx^n| <= 1, so the interpolation error is far
                    below double rounding.  (A Gauss-Jacobi rule was rejected:
                    scipy's nodes for weight s^(a-1), a ~ 0.02, lose accuracy as
                    the node count grows.)
  |x| >  X_SWITCH : for a0 = a - floor(a) in (0,1): the convergent-remainder
                    asymptotic series (integration by parts on the tail)
                        M_a0(x) = Gamma(a0+1) e^{i pi a0/2} x^-a0
                                  + a0 e^{ix} sum_{j<J} (1-a0)_j /(i x)^(j+1) + R_J,
                        |R_J| <= a0 prod_{j=1}^{J-1}(j-a0) x^-J ;
                    then the upward recurrence  M_{a+1} = (a+1)(e^{ix} - M_a)/(i x),
                    which is stable for |x| >= X_SWITCH.  a0 = 0 uses M_1 exactly.
  x < 0           : M_a(-x) = conj(M_a(x)).
"""
import math
import numpy as np
import mpmath as mp
from scipy.special import gamma as _gamma

X_SWITCH = 60.0
_PANEL = 4.0
_CHEB = 32
_J = 24
_cache = {}


def _table(a):
    """Chebyshev coefficients of M_a on panels [4p, 4p+4], p = 0..14."""
    key = round(float(a), 15)
    if key not in _cache:
        npan = int(round(X_SWITCH / _PANEL))
        theta = np.pi * (np.arange(_CHEB) + 0.5) / _CHEB
        u = np.cos(theta)
        coeffs = np.empty((npan, _CHEB), dtype=complex)
        with mp.workdps(30):
            am = mp.mpf(repr(float(a)))
            for p in range(npan):
                xs = _PANEL * p + 0.5 * _PANEL * (u + 1.0)
                vals = np.array([complex(mp.hyp1f1(am, am + 1, 1j * mp.mpf(repr(float(x))))) for x in xs])
                c = (2.0 / _CHEB) * (np.cos(np.outer(np.arange(_CHEB), theta)) @ vals)
                c[0] *= 0.5
                coeffs[p] = c
        _cache[key] = coeffs
    return _cache[key]


def _small(a, x):
    coeffs = _table(a)
    p = np.minimum((x / _PANEL).astype(int), coeffs.shape[0] - 1)
    u = 2.0 * (x - _PANEL * p) / _PANEL - 1.0
    # Clenshaw recurrence, vectorized over points with per-point coefficient rows
    b1 = np.zeros(x.shape, dtype=complex)
    b2 = np.zeros(x.shape, dtype=complex)
    for k in range(_CHEB - 1, 0, -1):
        b1, b2 = 2.0 * u * b1 - b2 + coeffs[p, k], b1
    return u * b1 - b2 + coeffs[p, 0]


def _large_fractional(a0, x):
    """a0 in (0,1); x >= X_SWITCH > 0 (array)."""
    ix = 1j * x
    term = 1.0 / ix
    series = term.copy()
    for j in range(1, _J):
        term = term * (j - a0) / ix
        series = series + term
    return _gamma(a0 + 1.0) * np.exp(0.5j * np.pi * a0) * x ** (-a0) + a0 * np.exp(1j * x) * series


def remainder_bound(a0, x, J=_J):
    """Analytic bound of the truncated asymptotic series, 0 < a0 < 1."""
    p = a0
    for j in range(1, J):
        p *= (j - a0)
    return p * np.asarray(x, dtype=float) ** (-J)


def M(a, x):
    """M_a(x) = 1F1(a; a+1; i x) for scalar a > 0 and real array x."""
    a = float(a)
    if not a > 0:
        raise ValueError('a must be positive')
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    out = np.empty(x.shape, dtype=complex)
    small = ax <= X_SWITCH
    if np.any(small):
        out[small] = _small(a, ax[small])
    if np.any(~small):
        xl = ax[~small]
        n = int(math.floor(a))
        a0 = a - n
        if a0 < 1e-14:            # integer a: start from M_1 exactly
            cur = (np.exp(1j * xl) - 1.0) / (1j * xl)
            start = 1.0
            n -= 1
        else:
            cur = _large_fractional(a0, xl)
            start = a0
        e = np.exp(1j * xl)
        for m in range(n):
            nxt = start + m + 1.0
            cur = nxt * (e - cur) / (1j * xl)
        out[~small] = cur
    return np.where(x < 0, np.conj(out), out)


def power_exp_moment(q, w, T=1.0):
    """int_0^T s^q exp(i w s) ds for q > -1 (vectorized in w)."""
    w = np.asarray(w, dtype=float)
    return T ** (q + 1.0) * M(q + 1.0, w * T) / (q + 1.0)
