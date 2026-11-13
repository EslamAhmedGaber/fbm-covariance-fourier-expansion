"""E5 learn_hurst: learning (H, sigma, tau) with EXACT, CF (lower bound Ld) and IP-VFE (Ld) on window data,
and the enclosure {H : U*(H) >= max_H' Ld*(H')} of the exact (grid) maximiser.

Hyperparameters on grids: H in {0.50, ..., 0.99}; log10 kappa (kappa = tau^2 / sigma^2) on 121 points in
[-8, 1]; sigma^2 profiled in closed form.  The upper bound U = min(U0, U1) on the whole kappa grid and
min(U0, U1, U2) on the 7 grid points around each data set's Ld maximiser (U2 needs the majorant).

Usage: python e5_learn_hurst.py  ->  results/E5_learn_hurst.json (+ raw npz)
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import C, np, make_design, save_json, Log, qstats, RESULTS, SEED0   # noqa: E402
from scipy.linalg import eigh, cholesky                                        # noqa: E402

HTRUE = (0.6, 0.75, 0.9)
HGRID = tuple(np.round(np.arange(0.50, 0.995, 0.01), 2))
RANKS = (16, 64, 256)
LK = np.linspace(-8.0, 1.0, 121)
KAP = 10.0 ** LK
NDATA = 100
F = 0.03
M = 2000
NEAR = 3      # +-3 kappa grid points around the Ld maximiser get U2


def gen_data(des):
    data = {}
    for j, Ht in enumerate(HTRUE):
        K = des.K(Ht)
        sd = float(np.sqrt(np.mean(np.diag(K))))
        tau = F * sd
        rng = np.random.default_rng(SEED0 + 5000 + 10 * j)
        Lc = np.linalg.cholesky(K + tau ** 2 * np.eye(M))
        data[Ht] = {'Y': Lc @ rng.standard_normal((M, NDATA)), 'tau': tau, 'sd': sd}
    return data


def profile_from_spectrum(lam, UY, perp2, m, extra=None, shift=0.0):
    """log N(y; 0, sigma^2 (Q + (kappa + shift) I)) profiled over sigma^2 for all kappa (rows) and data sets
    (cols); the log-determinant always uses kappa (shift only enters the quadratic form, as in U0).
    lam: (r,) eigenvalues of Q, UY: (r, B) projections, perp2: (B,) |y_perp|^2."""
    r = lam.size
    ld = (m - r) * np.log(KAP) + np.log(lam[None, :] + KAP[:, None]).sum(axis=1)          # (G,)
    W = 1.0 / (lam[None, :] + KAP[:, None] + shift)                                         # (G, r)
    q = perp2[None, :] / (KAP[:, None] + shift) + W @ (UY ** 2)                             # (G, B)
    s2 = q / m
    val = -0.5 * (ld[:, None] + m * np.log(s2) + m) - 0.5 * m * math.log(2 * math.pi)
    if extra is not None:
        val = val + extra[:, None]
    return val, s2, ld, q


def refine(xgrid, f):
    """argmax of f over the grid with parabolic refinement (f: (G,) array)."""
    j = int(np.argmax(f))
    if 0 < j < len(f) - 1:
        y0, y1, y2 = f[j - 1], f[j], f[j + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            d = 0.5 * (y0 - y2) / den
            h = xgrid[1] - xgrid[0]
            return float(xgrid[j] + d * h), float(y1 - 0.25 * (y0 - y2) * d), j
    return float(xgrid[j]), float(f[j]), j


def bounds_profiles(feats, Y, majorant, m):
    """Ld(kappa) and U(kappa) profiled over sigma^2 on the kappa grid (arrays (G, B)).
    U = U0 everywhere; min with U1 and U2 (if a majorant is given) at the NEAR grid points around each data
    set's Ld maximiser.  Every entry is a valid bound at its grid point."""
    lr = C.LowRank(feats.Gamma)
    UY, perp = lr.proj(Y)
    perp2 = np.einsum('ib,ib->b', perp, perp)
    e2 = feats.e2
    trR = feats.trR
    pen = -0.5 * np.array([np.sum(np.log1p(e2 / k)) for k in KAP])
    Ld, s2L, ld, q = profile_from_spectrum(lr.lam, UY, perp2, m, extra=pen)
    _, _, _, q0 = profile_from_spectrum(lr.lam, UY, perp2, m, shift=trR)
    qlo = q0.copy()
    e = np.sqrt(e2)
    jmax = np.argmax(Ld, axis=0)
    idx = sorted(set(int(np.clip(j + d, 0, len(KAP) - 1)) for j in jmax for d in range(-NEAR, NEAR + 1)))
    for i in idx:
        cols = np.where(np.abs(jmax - i) <= NEAR)[0]
        if cols.size == 0:
            continue
        v = lr.solve(Y[:, cols], KAP[i], (UY[:, cols], perp[:, cols]))
        q1 = q[i, cols] - (np.abs(v).T @ e) ** 2
        best = np.maximum(qlo[i, cols], q1)
        if majorant is not None:
            bv, _ = majorant.beta(v)
            best = np.maximum(best, q[i, cols] - bv)
        qlo[i, cols] = best
    U = -0.5 * (ld[:, None] + m * np.log(np.maximum(qlo, 1e-300) / m) + m) - 0.5 * m * math.log(2 * math.pi)
    return Ld, U, s2L, jmax


def main():
    log = Log('E5.log')
    des, _ = make_design('W', M)
    data = gen_data(des)
    log('data generated', {h: '%.4f' % d['tau'] for h, d in data.items()})
    Yall = np.hstack([data[h]['Y'] for h in HTRUE])        # (M, 3*NDATA)
    nB = Yall.shape[1]
    G = len(HGRID)
    prof = {'EXACT': np.empty((G, nB)), 'EXACT_grid': np.empty((G, nB))}
    kap_hat = {'EXACT': np.empty((G, nB))}
    s2_hat = {'EXACT': np.empty((G, nB))}
    for r in RANKS:
        for meth in ('CF', 'IP'):
            prof['%s%d_L' % (meth, r)] = np.empty((G, nB)); prof['%s%d_U' % (meth, r)] = np.empty((G, nB))
            prof['%s%d_Lgrid' % (meth, r)] = np.empty((G, nB))
            kap_hat['%s%d' % (meth, r)] = np.empty((G, nB)); s2_hat['%s%d' % (meth, r)] = np.empty((G, nB))
    rho_true = {}
    times = {'exact': 0.0, 'cf': {r: 0.0 for r in RANKS}, 'ip': {r: 0.0 for r in RANKS}}
    for g, H in enumerate(HGRID):
        tg = time.perf_counter()
        K = des.K(H)
        lam, V = eigh(K)
        lam = np.maximum(lam, 0.0)
        VY = V.T @ Yall
        perp2 = np.zeros(nB)
        val, s2, _, _ = profile_from_spectrum(lam, VY, perp2, M)
        prof['EXACT_grid'][g] = val.max(axis=0)
        times['exact'] += time.perf_counter() - tg
        for b in range(nB):
            xk, fk, j = refine(LK, val[:, b])
            prof['EXACT'][g, b] = fk; kap_hat['EXACT'][g, b] = xk; s2_hat['EXACT'][g, b] = s2[j, b]
        Bm = des.moments(H, 4 * (RANKS[-1] - 2))
        nrm = des.norms(H)
        for r in RANKS:
            t0 = time.perf_counter()
            bas = C.CFBasis(H, r - 2)
            f = C.cf_features(des, bas, B=Bm, norms=nrm)
            maj = C.CFMajorant(des, bas, f)
            Ld, U, s2L, jmax = bounds_profiles(f, Yall, maj, M)
            times['cf'][r] += time.perf_counter() - t0
            for b in range(nB):
                xk, fk, j = refine(LK, Ld[:, b])
                prof['CF%d_L' % r][g, b] = fk; kap_hat['CF%d' % r][g, b] = xk; s2_hat['CF%d' % r][g, b] = s2L[j, b]
            prof['CF%d_U' % r][g] = U.max(axis=0)
            prof['CF%d_Lgrid' % r][g] = Ld.max(axis=0)
            if abs(H - 0.6) < 1e-9 or abs(H - 0.75) < 1e-9 or abs(H - 0.9) < 1e-9:
                rho_true[(H, r, 'CF')] = f.trR / (F * math.sqrt(np.mean(nrm))) ** 2
            t0 = time.perf_counter()
            z = np.arange(1, r + 1) / r
            fi = C.ip_features(des, H, z, K_diag=nrm)
            Ldi, Ui, s2i, _ = bounds_profiles(fi, Yall, None, M)
            times['ip'][r] += time.perf_counter() - t0
            for b in range(nB):
                xk, fk, j = refine(LK, Ldi[:, b])
                prof['IP%d_L' % r][g, b] = fk; kap_hat['IP%d' % r][g, b] = xk; s2_hat['IP%d' % r][g, b] = s2i[j, b]
            prof['IP%d_U' % r][g] = Ui.max(axis=0)
            prof['IP%d_Lgrid' % r][g] = Ldi.max(axis=0)
            if abs(H - 0.6) < 1e-9 or abs(H - 0.75) < 1e-9 or abs(H - 0.9) < 1e-9:
                rho_true[(H, r, 'IP')] = fi.trR / (F * math.sqrt(np.mean(nrm))) ** 2
        log('H %.2f done %.1fs' % (H, time.perf_counter() - tg))
    np.savez_compressed(os.path.join(RESULTS, 'raw', 'E5_profiles.npz'), HGRID=np.array(HGRID), LK=LK,
                        **{k: v for k, v in prof.items()}, **{'kap_' + k: v for k, v in kap_hat.items()},
                        **{'s2_' + k: v for k, v in s2_hat.items()})
    # summaries
    Hg = np.array(HGRID)
    out = {'experiment': 'E5', 'kind': 'W', 'm': M, 'ndata': NDATA, 'f': F, 'HGRID': HGRID, 'log10_kappa_grid': [LK[0], LK[-1], len(LK)],
           'times_seconds': times, 'rho_at_true': {'%g_%d_%s' % k: v for k, v in rho_true.items()}, 'cells': []}
    for j, Ht in enumerate(HTRUE):
        cols = np.arange(j * NDATA, (j + 1) * NDATA)
        Hx = np.array([refine(Hg, prof['EXACT'][:, b])[0] for b in cols])
        gx = np.array([int(np.argmax(prof['EXACT'][:, b])) for b in cols])
        gxg = np.array([int(np.argmax(prof['EXACT_grid'][:, b])) for b in cols])
        tau_x = np.sqrt(np.array([10 ** kap_hat['EXACT'][gx[i], b] * s2_hat['EXACT'][gx[i], b] for i, b in enumerate(cols)]))
        sig_x = np.sqrt(np.array([s2_hat['EXACT'][gx[i], b] for i, b in enumerate(cols)]))
        cell = {'H_true': Ht, 'tau_true': data[Ht]['tau'], 'EXACT': {'H': qstats(Hx), 'H_bias_vs_true': qstats(Hx - Ht),
                                                                       'tau': qstats(tau_x), 'sigma': qstats(sig_x)}}
        for r in RANKS:
            for meth in ('CF', 'IP'):
                key = '%s%d' % (meth, r)
                PL, PU, PLg = prof[key + '_L'], prof[key + '_U'], prof[key + '_Lgrid']
                Hm = np.array([refine(Hg, PL[:, b])[0] for b in cols])
                gm = np.array([int(np.argmax(PL[:, b])) for b in cols])
                tau_m = np.sqrt(np.array([10 ** kap_hat[key][gm[i], b] * s2_hat[key][gm[i], b] for i, b in enumerate(cols)]))
                sig_m = np.sqrt(np.array([s2_hat[key][gm[i], b] for i, b in enumerate(cols)]))
                encl_w, cover = [], []
                lo_hi = []
                for i, b in enumerate(cols):
                    thr = PLg[:, b].max()
                    inside = np.where(PU[:, b] >= thr)[0]
                    lo, hi = Hg[inside.min()], Hg[inside.max()]
                    lo_hi.append((float(lo), float(hi)))
                    encl_w.append(hi - lo)
                    cover.append(bool(PU[gxg[i], b] >= thr))
                cell[key] = {'dH_vs_exact': qstats(Hm - Hx), 'abs_dH_le_0.005': float(np.mean(np.abs(Hm - Hx) <= 0.005)),
                             'tau_ratio': qstats(tau_m / tau_x), 'sigma_ratio': qstats(sig_m / sig_x),
                             'enclosure_width': qstats(encl_w), 'coverage': float(np.mean(cover)),
                             'enclosures_first10': lo_hi[:10]}
        out['cells'].append(cell)
        log('H_true', Ht, ' '.join('%s: dH med %.4f |dH|<=.005 %.2f encl %.3f cov %.2f tau_ratio %.3f' % (
            k, v['dH_vs_exact']['median'], v['abs_dH_le_0.005'], v['enclosure_width']['median'], v['coverage'],
            v['tau_ratio']['median']) for k, v in cell.items() if k not in ('H_true', 'tau_true', 'EXACT')))
    save_json(out, 'E5_learn_hurst.json')


if __name__ == '__main__':
    main()
