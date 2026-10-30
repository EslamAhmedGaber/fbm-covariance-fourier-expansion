"""Numerical checks of the two-sided weighted cosine bound (Theorem 'two-sided bounds' of the paper).

  (a) A3: extreme eigenvalues of W^{-1/2} G_+ W^{-1/2} (N = 2048; H = 0.75 also N = 4096) and the
      tail-block ratios lambda_max(G_{J:,J:}) / w_J;
  (b) A5: for seven inputs, H in {0.55, 0.75, 0.95}, T in {1, 2}, J = 1..256 (378 cases), the ratio of
      E_J(f)^2 to the weighted classical tail sum_{k>=J} w_k a_k^2 (proven band [1/2, 1]);
  (c) A4 checks: Schur complement of the constant mode (weighted spectrum), 40-digit eigenvalues at N = 40,
      the closed formula for K_lambda against direct quadrature, the per-lambda inequality
      a^T K_lambda a >= (1/2) sum_{k>=1} 2 lambda a_k^2/(lambda^2+omega_k^2) as a minimum eigenvalue, and
      inf_x (1 - F(x)) on a grid.
Outputs: results/A3_weighted_inequality.json, results/A5_two_sided_rates_inputs.json,
         results/A4_two_sided_bound_checks.json   (optional argument: output directory)
"""
import os, sys, json, math, time
import numpy as np
import mpmath as mp
from scipy.linalg import cholesky, solve_triangular, eigh
from numpy.polynomial.legendre import leggauss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from covfourier import gram as G
from covfourier import moments as MO

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else RES


def cH(H):
    return math.gamma(2 * H + 1) * math.sin(math.pi * H)


def weights(H, N, T=1.0):
    k = np.arange(1, N)
    return cH(H) * (k * np.pi / T) ** (1 - 2 * H)


def part_a():
    rows = []
    for H, N in [(0.501, 2048), (0.55, 2048), (0.65, 2048), (0.75, 2048), (0.85, 2048), (0.95, 2048), (0.99, 2048), (0.75, 4096)]:
        Gd = G.CosineGram(H, 1.0).dense(N)
        w = weights(H, N)
        lam = np.linalg.eigvalsh(Gd[1:, 1:] / np.sqrt(np.outer(w, w)))
        tails = {}
        for J in [1, 8, 64, 512]:
            lt = np.linalg.eigvalsh(Gd[J:, J:])[-1]
            tails[J] = float(lt / (cH(H) * (J * np.pi) ** (1 - 2 * H)))
        rows.append({'H': H, 'N': N, 'max_eig_weighted': float(lam[-1]), 'min_eig_weighted': float(lam[0]),
                     'tail_block_ratio': tails, 'cond_cosine_gram': float(np.linalg.cond(Gd))})
        print('A3', H, N, 'eig range [%.6f, %.12f]' % (lam[0], lam[-1]), flush=True)
    json.dump({'check': 'weighted cosine Gram W^{-1/2} G_+ W^{-1/2}: spectrum in [1/2, 1]; tail-block bound', 'rows': rows},
              open(os.path.join(OUTDIR, 'A3_weighted_inequality.json'), 'w'), indent=1)


def part_b():
    def weighted_tail(a, H, T, J_list):
        Nbig = a.size
        k = np.arange(Nbig); w = np.zeros(Nbig); w[1:] = cH(H) * (k[1:] * np.pi / T) ** (1 - 2 * H)
        s = np.cumsum((w * a * a)[::-1])[::-1]
        return {J: float(s[J]) for J in J_list}
    checks = []
    N = 2048; Js = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for H in [0.55, 0.75, 0.95]:
        for T in [1.0, 2.0]:
            Gd = G.CosineGram(H, T).dense(N); L = cholesky(Gd, lower=True)
            funcs = {'t': MO.Power(1), 't2': MO.Power(2), 'exp(t)': MO.Exponential(1.0), 'exp(-5t)': MO.Exponential(-5.0),
                     '1[0,0.3T)': MO.Indicator(0.3 * T), '1[0,0.5T)': MO.Indicator(0.5 * T), '1[0,0.71T)': MO.Indicator(0.71 * T)}
            for name, f in funcs.items():
                c = solve_triangular(L, f.moments(H, N, T), lower=True)
                n2 = f.norm2(H, T)
                tailc = np.cumsum((c * c)[::-1])[::-1]
                wt = weighted_tail(f.cosine(2 ** 20, T), H, T, Js + [N])
                for J in Js:
                    lo_E2 = tailc[J]                 # E_J^2 >= sum_{k=J}^{N-1} c_k^2
                    hi_E2 = tailc[J] + wt[N]         # E_N^2 <= weighted tail beyond N (upper bound)
                    checks.append({'H': H, 'T': T, 'f': name, 'J': J, 'E2_lo': float(lo_E2), 'E2_hi': float(hi_E2),
                                   'E2_subtractive': float(n2 - np.dot(c[:J], c[:J])), 'weighted_tail': wt[J],
                                   'ratio_lo': float(lo_E2 / wt[J]), 'ratio_hi': float(hi_E2 / wt[J])})
    mn = min(c['ratio_lo'] for c in checks); mx = max(c['ratio_hi'] for c in checks)
    print('A5 ratio range over %d cases: [%.4f, %.4f]' % (len(checks), mn, mx), flush=True)
    json.dump({'check': 'two-sided bound E_J^2 in [1/2, 1] x sum_{k>=J} w_k a_k^2 (classical cosine coefficients)',
               'E2_bracket': 'lower = sum_{k=J}^{N-1} c_k^2 (orthonormal tail), upper = lower + weighted tail beyond N',
               'N': N, 'cases': checks, 'min_ratio_lo': mn, 'max_ratio_hi': mx},
              open(os.path.join(OUTDIR, 'A5_two_sided_rates_inputs.json'), 'w'), indent=1)


def K_closed(lam, K, T=1.0):
    """(K_lambda)_{kl} = int int exp(-lam|u-v|) h_k(u) h_l(v) du dv, closed form (k, l < K)."""
    k = np.arange(K); om = k * np.pi / T
    n = np.full(K, math.sqrt(2 / T)); n[0] = T ** -0.5
    x = lam * T
    d = lam ** 2 + om ** 2
    M = np.diag(2 * lam / d)
    same = (k[:, None] % 2) == (k[None, :] % 2)
    fac = 2 * lam ** 2 * (1 - (-1.0) ** k[:, None] * math.exp(-x)) * n[:, None] * n[None, :] / (d[:, None] * d[None, :])
    return M - np.where(same, fac, 0.0)


def K_quad(lam, K, T=1.0, q=160):
    """Independent 2-D Gauss-Legendre quadrature on the two triangles (kink on the diagonal)."""
    x, w = leggauss(q)
    u = 0.5 * T * (x + 1); wu = 0.5 * T * w
    k = np.arange(K); om = k * np.pi / T
    n = np.full(K, math.sqrt(2 / T)); n[0] = T ** -0.5
    out = np.zeros((K, K))
    for ui, wi in zip(u, wu):               # inner variable v in [0, u] and [u, T], each mapped to GL nodes
        for a, b in ((0.0, ui), (ui, T)):
            v = a + (b - a) * 0.5 * (x + 1); wv = (b - a) * 0.5 * w
            ker = np.exp(-lam * np.abs(ui - v)) * wv
            hv = n[:, None] * np.cos(np.outer(om, v))
            hu = n * np.cos(om * ui)
            out += wi * np.outer(hu, hv @ ker)
    return out


def part_c():
    rows_s = []
    for H in [0.501, 0.55, 0.75, 0.95, 0.99]:
        N = 2048
        Gd = G.CosineGram(H, 1.0).dense(N)
        w = weights(H, N)
        S0 = Gd[1:, 1:] - np.outer(Gd[1:, 0], Gd[0, 1:]) / Gd[0, 0]
        lam = np.linalg.eigvalsh(S0 / np.sqrt(np.outer(w, w)))
        rows_s.append({'H': H, 'N': N, 'schur_weighted_min': float(lam[0]), 'schur_weighted_max': float(lam[-1])})
        print('Schur', H, '[%.6f, %.12f]' % (lam[0], lam[-1]), flush=True)
    hp = []
    for Hs in ['0.75', '0.95']:
        N = 40
        with mp.workdps(40):
            Hm = mp.mpf(Hs); b = 2 * Hm - 1; C = 2 * Hm
            m_ = [mp.hyp1f1(b, b + 1, 1j * mp.pi * kk) for kk in range(N)]
            p = [mp.re(z) for z in m_]; r = [mp.im(z) for z in m_]
            cHm = mp.gamma(2 * Hm + 1) * mp.sin(mp.pi * Hm)
            S = mp.matrix(N - 1, N - 1)
            for i in range(1, N):
                for j in range(1, N):
                    if i == j:
                        g = C * (p[i] - (1 - b) * r[i] / (i * mp.pi))
                    elif (i + j) % 2 == 0:
                        g = (2 * C / mp.pi) * (j * r[j] - i * r[i]) / (i ** 2 - j ** 2)
                    else:
                        g = 0
                    S[i - 1, j - 1] = g / mp.sqrt(cHm * (i * mp.pi) ** (-b) * cHm * (j * mp.pi) ** (-b))
            E = mp.eigsy(S, eigvals_only=True)
            hp.append({'H': Hs, 'N': N, 'one_minus_lambda_max': float(1 - max(E)), 'lambda_min': float(min(E))})
            print('hp', Hs, float(1 - max(E)), float(min(E)), flush=True)
    # closed K_lambda against direct quadrature
    maxdiff = 0.0
    for lam in (0.3, 1.0, 5.0, 20.0):
        Kc = K_closed(lam, 9); Kq = K_quad(lam, 9)
        maxdiff = max(maxdiff, float(np.max(np.abs(Kc - Kq)) / np.max(np.abs(Kc))))
    # per-lambda inequality: min over lambda of the smallest eigenvalue of D^{-1/2} S_lam D^{-1/2} - 1/2,
    # S_lam the Schur complement of the constant mode in K_lambda, D = diag(2 lam/(lam^2+omega_k^2)), k >= 1
    worst = np.inf
    for lam in np.geomspace(1e-2, 1e4, 61):
        Kc = K_closed(lam, 400)
        S = Kc[1:, 1:] - np.outer(Kc[1:, 0], Kc[0, 1:]) / Kc[0, 0]
        dd = 2 * lam / (lam ** 2 + (np.arange(1, 400) * np.pi) ** 2)
        ev = np.linalg.eigvalsh(S / np.sqrt(np.outer(dd, dd)))
        worst = min(worst, float(ev[0] - 0.5))
    xs = np.geomspace(1e-3, 1e3, 4001)
    F = (1 - np.exp(-xs)) * ((xs / 2) / np.tanh(xs / 2) - 1) / (xs - 1 + np.exp(-xs))
    out = {'check': 'lower bound G >= (1/2) diag(0, w) with unrestricted a_0, and its proof ingredients',
           'schur_rows': rows_s, 'high_precision': hp,
           'one_minus_F_inf_on_grid': float(np.min(1 - F)), 'one_minus_F_at_x1000': float(1 - F[-1]),
           'K_lambda_formula_vs_quadrature_rel': maxdiff, 'K_lambda_schur_minus_half_min_eig': worst}
    json.dump(out, open(os.path.join(OUTDIR, 'A4_two_sided_bound_checks.json'), 'w'), indent=1)
    print('K formula vs quadrature %.2e; per-lambda min eig - 1/2 = %.3e; inf(1-F) = %.6f' % (maxdiff, worst, out['one_minus_F_inf_on_grid']))


if __name__ == '__main__':
    t0 = time.time(); part_a(); part_b(); part_c(); print('total %.0f s' % (time.time() - t0))
