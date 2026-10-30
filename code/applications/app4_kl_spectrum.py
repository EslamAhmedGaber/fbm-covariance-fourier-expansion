"""A4: certified Karhunen-Loeve spectrum of fBm and E exp(-a int_0^1 B_H(t)^2 dt).

Covariance operator (K phi)(t) = int_0^1 R_H(s,t) phi(s) ds on L^2(0,1), R_H the fBm covariance.
Galerkin (Rayleigh-Ritz) on the nested cosine spaces V_J = span{h_0..h_{J-1}}:
    A_ij = <h_i, K h_j>_{L2} = E[(int h_i B)(int h_j B)] = <H_i, H_j>_H,
    H_i(s) = int_s^1 h_i,  H_0 = 1-s,  H_i = -sqrt2 sin(i pi s)/(i pi)  (i >= 1).
Certified facts (exact arithmetic):
  * mu_n(J) <= lambda_n, nondecreasing in J (min-max, nested spaces);
  * sum_n (lambda_n - mu_n(J)) = delta_J := 1/(2H+1) - tr A_J  (trace of K is int t^{2H}),
    hence 0 <= lambda_n - mu_n(J) <= delta_J for every n;
  * L(a) := E exp(-a ||B_H||_2^2) = prod (1+2 a lambda_n)^(-1/2) lies in
    [ L_J(a) exp(-a delta_J), L_J(a) ],  L_J(a) = prod_{n<=J} (1+2 a mu_n)^(-1/2).
The integer-sine block of A is evaluated in closed form through M_b (no truncation);
the h_0 row uses the explicit kernel image K_H t^j.
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from covfourier.kummer import M                    # noqa: E402
from covfourier.gram import poly_poly              # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')


def sine_gram(H, J, block=512):
    """S_ij = <sin(i pi .), sin(j pi .)>_H on [0,1], i,j = 1..J  (closed form, derived here).

    With p_i + i r_i = M_b(i pi):
      S_ij = (H/pi) [ (r_i + r_j)/(i+j) - (r_i - r_j)/(i-j) ]   (i != j, i = j mod 2),
      S_ij = 0                                                  (i + j odd),
      S_ii = H [ p_i + r_i/(i pi) ] - H b/(b+1) Re M_{b+1}(i pi).
    Derivation: bilinear exponential moments (Supplementary Material) with a, b = +-i u."""
    b = 2 * H - 1
    if b == 0:
        return 0.5 * np.eye(J)
    i = np.arange(1, J + 1, dtype=float)
    m = M(b, np.pi * i); p = m.real; r = m.imag
    S = np.zeros((J, J))
    for s0 in range(0, J, block):
        s1 = min(s0 + block, J)
        I = i[s0:s1, None]; Jj = i[None, :]
        same = ((I + Jj) % 2 == 0) & (I != Jj)
        with np.errstate(divide='ignore', invalid='ignore'):
            val = (H / np.pi) * ((r[s0:s1, None] + r[None, :]) / (I + Jj)
                                 - (r[s0:s1, None] - r[None, :]) / (I - Jj))
        S[s0:s1] = np.where(same, val, 0.0)
    d = H * (p + r / (np.pi * i)) - H * b / (b + 1) * M(b + 1, np.pi * i).real
    S[np.arange(J), np.arange(J)] = d
    return 0.5 * (S + S.T)


def h0_row(H, J):
    """<H_0, H_j>_H for j = 0..J, H_0 = 1 - s (closed form)."""
    b = 2 * H - 1
    out = np.empty(J + 1)
    out[0] = poly_poly(H, 0, 0) - 2 * poly_poly(H, 0, 1) + poly_poly(H, 1, 1)
    j = np.arange(1, J + 1, dtype=float); x = np.pi * j
    sgn = (-1.0) ** np.arange(1, J + 1)
    if b == 0:
        s1 = (1 - sgn) / x                                  # int sin
        st = -sgn / x                                       # int t sin = -cos(x)/x
        val = s1 - st
    else:
        Im = lambda q: (M(q + 1, x) / (q + 1)).imag         # int_0^1 t^q sin(x t) dt
        Ir = lambda q: -sgn * Im(q)                          # int_0^1 (1-t)^q sin(x t) dt
        one = H * (Im(b) + Ir(b))                           # <1, sin>  via K_H 1 = H(t^b + r^b)
        # K_H t = H b [ B(2,b) t^{1+b} + t r^b / b + r^{1+b}/(1+b) ],  t r^b = r^b - r^{1+b}
        from scipy.special import beta as B
        tt = H * b * (B(2, b) * Im(1 + b) + (Ir(b) - Ir(1 + b)) / b + Ir(1 + b) / (1 + b))
        val = one - tt
    out[1:] = -(math.sqrt(2) / x) * val                     # H_j = -sqrt2 sin(j pi s)/(j pi)
    return out


def galerkin_matrix(H, J):
    """A_ij = <H_i, H_j>_H, i, j = 0..J-1."""
    A = np.empty((J, J))
    S = sine_gram(H, J - 1)
    i = np.arange(1, J, dtype=float)
    A[1:, 1:] = 2.0 * S / (np.pi ** 2 * np.outer(i, i))
    r = h0_row(H, J - 1)
    A[0, :] = r; A[:, 0] = r
    return A


def ck_nu(H, n, order=2):
    """Chigansky-Kleptsyna (SPA 2018, eq. 2.3) nu_n without the O(1/n) term."""
    ell = math.sin(math.pi / 2 * (H - 0.5) / (H + 0.5)) / math.sin(math.pi / 2 / (H + 0.5))
    nu = (n - 0.5) * np.pi
    if order >= 2:
        nu = nu + (1 - 2 * H) * np.pi / 4 + math.asin(ell / math.sqrt(1 + ell ** 2))
    return nu


def ck_lambda(H, n, order=2):
    return math.sin(math.pi * H) * math.gamma(2 * H + 1) * ck_nu(H, n, order) ** (-2 * H - 1)


def laplace_bounds(mu, delta, a):
    logL = -0.5 * np.sum(np.log1p(2 * a * np.clip(mu, 0, None)))
    return math.exp(logL - a * delta), math.exp(logL)


def run(Hs=(0.5, 0.55, 0.65, 0.75, 0.85, 0.95), Js=(64, 128, 256, 512, 1024, 2048),
        avals=(0.1, 1.0, 10.0, 100.0), out='A4_kl_spectrum.json'):
    data = {'application': 'A4 certified KL spectrum of fBm and Laplace transform of ||B_H||_2^2',
            'rows': [], 'eigen': {}}
    t0 = time.perf_counter()
    for H in Hs:
        for J in Js:
            t1 = time.perf_counter()
            A = galerkin_matrix(H, J)
            tb = time.perf_counter() - t1
            mu = np.sort(np.linalg.eigvalsh(A))[::-1]
            te = time.perf_counter() - t1 - tb
            delta = 1.0 / (2 * H + 1) - float(np.trace(A))
            row = {'H': H, 'J': J, 'delta_J': delta, 'build_seconds': tb, 'eig_seconds': te,
                   'min_eig': float(mu[-1]), 'laplace': {}}
            for a in avals:
                lo, hi = laplace_bounds(mu, max(delta, 0.0), a)
                row['laplace'][str(a)] = [lo, hi]
            if H == 0.5:
                n = np.arange(1, 51)
                exact = 1.0 / ((n - 0.5) * np.pi) ** 2
                row['bm_max_rel_err_first50'] = float(np.max(np.abs(mu[:50] - exact) / exact))
                row['bm_laplace_exact'] = {str(a): float(np.cosh(math.sqrt(2 * a)) ** -0.5) for a in avals}
            data['rows'].append(row)
            data['eigen'][f'{H}_{J}'] = mu[:min(J, 1024)].tolist()
            print('H', H, 'J', J, 'delta %.3e' % delta, 'L(1) in [%.10f, %.10f]' % tuple(row['laplace']['1.0']),
                  'build %.1fs eig %.1fs' % (tb, te), flush=True)
    data['seconds'] = time.perf_counter() - t0
    with open(os.path.join(RES, out), 'w') as fh:
        json.dump(data, fh)
    return data


if __name__ == '__main__':
    run()
