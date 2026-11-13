"""E5b: the enclosure of the exact estimate on the informative part of the hyperparameter grid.

When D < m the upper bound grows without bound as kappa -> 0 (log det of Gamma Gamma^T + kappa I), so the
enclosure over kappa in [1e-8, 10] (E5) is the whole H grid.  Here, for every H and rank, the kappa grid is
restricted to rho(H, kappa) = tr R(H) / kappa <= RHO_MAX (known before the data are seen): 31 log-spaced
points in [tr R / RHO_MAX, 10].  On this finite set G the exact profiled evidence, L* and U* (U = best of the
U0, U1 and U2 quadratic lower bounds at every grid point) are computed for the 300 data sets of E5 (same
seeds), ranks D = 64 and 256 (as in E5).  Reported: bracket validity on G, the exact maximiser over G, the
enclosure {H : max_kappa U*(H, kappa) >= max L*}, its width and coverage, agreement of the G-maximiser with the
unrestricted exact estimate of E5, and the fraction of data sets in which H = 1/2 is excluded.

Usage: python e5b_enclosure.py  ->  results/E5b_enclosure.json (+ raw npz)
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import C, np, make_design, save_json, Log, qstats, RESULTS   # noqa: E402
from scipy.linalg import eigh                                            # noqa: E402
import e5_learn_hurst as E5                                              # noqa: E402

RHO_MAX = 10.0
NK = 31
RANKS = (64, 256)


def profiles(lam, UY, perp2, m, kap, shift=0.0):
    r = lam.size
    ld = (m - r) * np.log(kap) + np.log(lam[None, :] + kap[:, None]).sum(axis=1)
    W = 1.0 / (lam[None, :] + kap[:, None] + shift)
    q = perp2[None, :] / (kap[:, None] + shift) + W @ (UY ** 2)
    return ld, q


def prof_val(ld, q, m, extra=None):
    val = -0.5 * (ld[:, None] + m * np.log(np.maximum(q, 1e-300) / m) + m) - 0.5 * m * math.log(2 * math.pi)
    return val if extra is None else val + extra[:, None]


def main():
    log = Log('E5b.log')
    M = E5.M
    des, _ = make_design('W', M)
    data = E5.gen_data(des)
    Y = np.hstack([data[h]['Y'] for h in E5.HTRUE])
    nB = Y.shape[1]
    Hg = np.array(E5.HGRID); G = Hg.size
    EX = {r: np.zeros((G, NK, nB)) for r in RANKS}
    LB = {r: np.zeros((G, NK, nB)) for r in RANKS}
    UB = {r: np.zeros((G, NK, nB)) for r in RANKS}
    KG = {r: np.zeros((G, NK)) for r in RANKS}
    nviol = {r: [0, 0] for r in RANKS}
    t0 = time.perf_counter()
    for g, H in enumerate(Hg):
        tg = time.perf_counter()
        K = des.K(H)
        lam, V = eigh(K); lam = np.maximum(lam, 0.0)
        VY = V.T @ Y
        Bm = des.moments(H, 4 * (RANKS[-1] - 2)); nrm = des.norms(H)
        for r in RANKS:
            bas = C.CFBasis(H, r - 2)
            f = C.cf_features(des, bas, B=Bm, norms=nrm)
            maj = C.CFMajorant(des, bas, f)
            kap = np.logspace(math.log10(f.trR / RHO_MAX), 1.0, NK)
            KG[r][g] = kap
            ldx, qx = profiles(lam, VY, np.zeros(nB), M, kap)
            EX[r][g] = prof_val(ldx, qx, M)
            lr = C.LowRank(f.Gamma)
            UY, perp = lr.proj(Y)
            perp2 = np.einsum('ib,ib->b', perp, perp)
            ld, q = profiles(lr.lam, UY, perp2, M, kap)
            pen = -0.5 * np.array([np.sum(np.log1p(f.e2 / k)) for k in kap])
            LB[r][g] = prof_val(ld, q, M, extra=pen)
            _, q0 = profiles(lr.lam, UY, perp2, M, kap, shift=f.trR)
            qlo = q0.copy()
            e = np.sqrt(f.e2)
            for i, k in enumerate(kap):
                v = lr.solve(Y, k, (UY, perp))
                q1 = q[i] - (np.abs(v).T @ e) ** 2
                bv, _ = maj.beta(v)
                qlo[i] = np.maximum(np.maximum(qlo[i], q1), q[i] - bv)
            UB[r][g] = prof_val(ld, qlo, M)
            tol = 1e-6 + 1e-9 * np.abs(EX[r][g])
            nviol[r][0] += int(np.sum(LB[r][g] > EX[r][g] + tol)); nviol[r][1] += int(np.sum(EX[r][g] > UB[r][g] + tol))
        log('H %.2f done %.1fs (viol %s)' % (H, time.perf_counter() - tg, nviol))
    raw = np.load(os.path.join(RESULTS, 'raw', 'E5_profiles.npz'))
    EXfull = raw['EXACT']                       # (G, nB): exact profile over the full kappa grid of E5
    out = {'experiment': 'E5b', 'rho_max': RHO_MAX, 'nk': NK, 'ranks': list(RANKS), 'violations_L_U': nviol,
           'seconds': time.perf_counter() - t0, 'cells': []}
    for j, Ht in enumerate(E5.HTRUE):
        cols = np.arange(j * E5.NDATA, (j + 1) * E5.NDATA)
        cell = {'H_true': Ht}
        for r in RANKS:
            exG = EX[r].max(axis=1); Lp = LB[r].max(axis=1); Up = UB[r].max(axis=1)
            widths, cover, agree, half, encs = [], [], [], [], []
            for b in cols:
                Lmax = Lp[:, b].max()
                enc = Hg[Up[:, b] >= Lmax]
                widths.append(float(enc.max() - enc.min()) if enc.size else 0.0)
                hx = Hg[int(np.argmax(exG[:, b]))]
                cover.append(bool(np.any(np.abs(enc - hx) < 1e-9)))
                agree.append(bool(abs(hx - Hg[int(np.argmax(EXfull[:, b]))]) < 1e-9))
                half.append(bool(Up[0, b] < Lmax))
                if len(encs) < 10:
                    encs.append([float(enc.min()), float(enc.max()), float(hx)])
            cell['D%d' % r] = {'enclosure_width': qstats(np.array(widths)), 'coverage': float(np.mean(cover)),
                               'G_max_equals_full_exact': float(np.mean(agree)), 'H_half_excluded': float(np.mean(half)),
                               'H_hat_L': qstats(np.array([Hg[int(np.argmax(Lp[:, b]))] for b in cols])),
                               'H_hat_exact_G': qstats(np.array([Hg[int(np.argmax(exG[:, b]))] for b in cols])),
                               'enclosures_first10': encs}
            log('H_true %.2f D %d: width med %.3f q95 %.3f cov %.2f agree %.2f half excluded %.2f' % (
                Ht, r, np.median(widths), np.quantile(widths, 0.95), np.mean(cover), np.mean(agree), np.mean(half)))
        out['cells'].append(cell)
    np.savez_compressed(os.path.join(RESULTS, 'raw', 'E5b_profiles.npz'), HGRID=Hg,
                        **{'EX%d' % r: EX[r].max(axis=1) for r in RANKS}, **{'L%d' % r: LB[r].max(axis=1) for r in RANKS},
                        **{'U%d' % r: UB[r].max(axis=1) for r in RANKS}, **{'kap%d' % r: KG[r] for r in RANKS})
    save_json(out, 'E5b_enclosure.json')


if __name__ == '__main__':
    main()
