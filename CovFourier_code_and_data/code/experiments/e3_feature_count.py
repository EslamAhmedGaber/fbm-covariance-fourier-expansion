"""E3 feature_count: measured D(eps) from exact residual energies against the prediction from the weighted
classical cosine tails (CosChaos two-sided inequality), m in {1e3, 1e4, 1e5} (nested prefixes of one design).

Usage: python e3_feature_count.py KIND [H ...]   ->  results/E3_<KIND>_H<H>.json
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import C, np, save_json, Log, HS, WIN, NMODEL, smooth_design   # noqa: E402
from covfourier.gram import CosineGram                                     # noqa: E402

MS = tuple(int(x) for x in os.environ.get('E3_MS', '1000,10000,100000').split(','))
NG = (8, 11, 16, 22, 32, 45, 64, 90, 128, 181, 256, 362, 512, 724, 1024)
EPS = (1.0, 1e-2, 1e-4)
F_NOISE = 0.03
THETA = 1.0 / math.sqrt(2.0)
KTAIL = 8192          # explicit terms of the weighted tails for P and W, analytic remainder beyond


def master(kind, m):
    if kind == 'P':
        rng = np.random.default_rng(101)
        return {'t': rng.uniform(0.01, 1.0, m)}
    if kind == 'W':
        rng = np.random.default_rng(102)
        c = rng.uniform(0.03, 0.97, m); w = rng.uniform(0.005, 0.06, m)
        return {'a': np.clip(c - w / 2, 0, 1), 'b': np.clip(c + w / 2, 0, 1)}
    if kind == 'S':
        rng = np.random.default_rng(103)
        return {'z': rng.uniform(10.0, 500.0, m), 'tl': rng.uniform(1960.0, 2020.0, m)}
    raise ValueError(kind)


def chunk_design(kind, md, s, e):
    if kind == 'P':
        return C.Points(md['t'][s:e])
    if kind == 'W':
        return C.Boxes(md['a'][s:e], md['b'][s:e])
    return smooth_design(md['z'][s:e], md['tl'][s:e])


def weighted_tails(kind, des, H, ngrid):
    """T_n(F_i) = sum_{k>=n} w_k a_k(F_i)^2 for n in ngrid, per observation (m x len(ngrid))."""
    cs = C.c_star(H)
    if kind == 'S':
        A = des.A                                     # (N, m) endpoint-corrected coefficients
        k = np.arange(A.shape[0], dtype=float)
        w = np.zeros_like(k); w[1:] = cs * (k[1:] * math.pi) ** (1 - 2 * H)
        S = np.cumsum((w[:, None] * A ** 2)[::-1], axis=0)[::-1]     # S[k] = sum_{j>=k}
        return np.stack([S[n] for n in ngrid], axis=1)
    k = np.arange(1, KTAIL, dtype=float)
    om = k * math.pi
    w = cs * om ** (1 - 2 * H)
    if kind == 'P':
        t = des.t[:, None]
        a2 = 2.0 * np.sin(om[None, :] * t) ** 2 / om[None, :] ** 2
        # remainder sum_{k>=K} w_k * (2 sin^2/om^2) ~ mean sin^2 = 1/2
        rem = cs * math.pi ** (-1 - 2 * H) * KTAIL ** (-2 * H) / (2 * H)
    else:
        a, b, wid = des.a[:, None], des.b[:, None], des.w[:, None]
        a2 = 2.0 * (np.cos(om[None, :] * a) - np.cos(om[None, :] * b)) ** 2 / (wid ** 2 * om[None, :] ** 4)
        rem = (2.0 / des.w ** 2) * cs * math.pi ** (-3 - 2 * H) * KTAIL ** (-2 - 2 * H) / (2 + 2 * H)
    S = np.cumsum((a2 * w[None, :])[:, ::-1], axis=1)[:, ::-1]       # S[:, j] = sum_{k >= j+1}
    out = np.stack([S[:, n - 1] for n in ngrid], axis=1)
    return out + (rem if np.ndim(rem) == 0 else rem[:, None])


def crossing(ns, rho, eps):
    """Smallest n (log-log interpolation) with rho(n) <= eps; None if never on the grid; ns[0] if already."""
    ns = np.asarray(ns, float); rho = np.asarray(rho, float)
    if rho[0] <= eps:
        return float(ns[0])
    for j in range(1, len(ns)):
        if rho[j] <= eps:
            x0, x1 = math.log(ns[j - 1]), math.log(ns[j])
            y0, y1 = math.log(max(rho[j - 1], 1e-300)), math.log(max(rho[j], 1e-300))
            le = math.log(eps)
            return float(math.exp(x0 + (le - y0) * (x1 - x0) / (y1 - y0)))
    return None


def main(kind, Hs, chunk=1000):
    log = Log('E3_%s.log' % kind)
    mmax = MS[-1]
    md = master(kind, mmax)
    for H in Hs:
        tH = time.perf_counter()
        bases = {n: C.CFBasis(H, n) for n in NG}
        prep = {n: bases[n].setup_seconds for n in NG}
        log('H', H, 'bases ready %.1fs' % (time.perf_counter() - tH))
        sum_e2 = {m: np.zeros(len(NG)) for m in MS}
        sum_T = {m: np.zeros(len(NG)) for m in MS}
        sum_var = {m: 0.0 for m in MS}
        nneg = np.zeros(len(NG), dtype=int)
        tmom = tproj = 0.0
        for s in range(0, mmax, chunk):
            e = min(mmax, s + chunk)
            des = chunk_design(kind, md, s, e)
            t0 = time.perf_counter()
            B = des.moments(H, 4 * NG[-1])
            nrm = des.norms(H)
            tmom += time.perf_counter() - t0
            t0 = time.perf_counter()
            E2 = np.empty((des.m, len(NG)))
            for j, n in enumerate(NG):
                G = bases[n].coords(B)
                e2 = nrm - np.einsum('ij,ij->i', G, G)
                nneg[j] += int(np.sum(e2 < 0))
                E2[:, j] = np.maximum(e2, 0.0)
            tproj += time.perf_counter() - t0
            Tn = weighted_tails(kind, des, H, NG)
            for m in MS:
                if s < m:
                    k = min(e, m) - s
                    sum_e2[m] += E2[:k].sum(axis=0)
                    sum_T[m] += Tn[:k].sum(axis=0)
                    sum_var[m] += float(nrm[:k].sum())
            if (s // chunk) % 10 == 0:
                log('  H %.2f chunk %d/%d  moments %.0fs proj %.0fs' % (H, s // chunk + 1, mmax // chunk, tmom, tproj))
        cells = []
        for m in MS:
            sd = math.sqrt(sum_var[m] / m)
            tau2 = (F_NOISE * sd) ** 2
            rho = sum_e2[m] / tau2
            rho_pred = THETA * sum_T[m] / tau2
            for eps in EPS:
                nm = crossing(NG, rho, eps); npred = crossing(NG, rho_pred, eps)
                cell = {'kind': kind, 'H': H, 'm': m, 'eps': eps, 'tau': math.sqrt(tau2), 'sd_rms': sd,
                        'n_meas': nm, 'n_pred': npred,
                        'D_meas': None if nm is None else nm + 2, 'D_pred': None if npred is None else npred + 2}
                if nm is not None and npred is not None:
                    cell['ratio_D'] = (nm + 2) / (npred + 2)
                cells.append(cell)
        # asymptotic closed forms (P, W): rho ~ A n^{-alpha}
        save_json({'experiment': 'E3', 'kind': kind, 'H': H, 'ngrid': NG, 'theta': THETA, 'f_noise': F_NOISE,
                   'sum_e2': {str(m): sum_e2[m] for m in MS}, 'sum_T': {str(m): sum_T[m] for m in MS},
                   'sum_var': {str(m): sum_var[m] for m in MS}, 'n_negative_e2': nneg, 'cells': cells,
                   'seconds': {'moments': tmom, 'projection': tproj, 'prep': prep, 'total': time.perf_counter() - tH}},
                  'E3_%s_H%.2f.json' % (kind, H))
        log('H', H, 'done %.0fs' % (time.perf_counter() - tH),
            ' '.join('m%d/eps%g: %s vs %s' % (c['m'], c['eps'], c['D_meas'] and round(c['D_meas']),
                                                c['D_pred'] and round(c['D_pred'])) for c in cells))


if __name__ == '__main__':
    kind = sys.argv[1]
    Hs = [float(x) for x in sys.argv[2:]] or list(HS)
    main(kind, Hs)
