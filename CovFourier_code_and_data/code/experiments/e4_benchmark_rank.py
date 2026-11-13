"""E4 benchmark_rank (m = 2000, exact available): CF, COS, COS-classical, IP-VFE (grid), IP-opt (optimised
pseudo-inputs, rank <= 32), IP-FITC, ORACLE at equal rank; evidence gaps, KL, held-out RMSE/NLPD, cond(K_uu),
times (one BLAS thread).  Uses the data of E1 (f = 0.03).

Usage: python e4_benchmark_rank.py KIND [H ...]  ->  results/E4_<KIND>_H<H>.json
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import C, np, make_design, joint_K, save_json, Log, qstats, HS, RESULTS   # noqa: E402
from scipy.linalg import cholesky, cho_solve, solve_triangular, eigh                   # noqa: E402
from scipy.optimize import minimize                                                    # noqa: E402

RANKS = (8, 16, 32, 64, 128, 256, 512)
F = 0.03
M_TRAIN, M_HOLD = 2000, 500
OPT_MAX_RANK = 32
OPT_MAXITER = 100


def predictive(feats, Gs, es2, y, tau2, working=False):
    """Collapsed (or working-model) posterior of the feature coordinates and the predictive of held-out
    observations with features Gs and residual energies es2.  Returns mean (h x B), var (h,)."""
    Gam = feats.Gamma
    d = (feats.e2 + tau2) if working else np.full(Gam.shape[0], tau2)
    Gd = Gam / d[:, None]
    A = np.eye(Gam.shape[1]) + Gam.T @ Gd
    La = cholesky(A, lower=True)
    mu = cho_solve((La, True), Gd.T @ y)                 # (D, B)
    mean = Gs @ mu
    W = solve_triangular(La, Gs.T, lower=True)           # (D, h)
    var = np.einsum('ij,ij->j', W, W) + es2 + tau2
    return mean, var


def scores(mean, var, yh):
    err = yh - mean
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    nlpd = np.mean(0.5 * np.log(2 * math.pi * var)[:, None] + 0.5 * err ** 2 / var[:, None], axis=0)
    return rmse, nlpd


def exact_predictive(Kj, y, tau2):
    m = M_TRAIN
    K, Ks, Kss = Kj[:m, :m], Kj[m:, :m], np.diag(Kj)[m:]
    Lc = cholesky(K + tau2 * np.eye(m), lower=True)
    mean = Ks @ cho_solve((Lc, True), y)
    V = solve_triangular(Lc, Ks.T, lower=True)
    var = Kss - np.einsum('ij,ij->j', V, V) + tau2
    return mean, var


def ip_opt_z(des, H, z0, y0, tau2, kd):
    """Optimise pseudo-inputs by maximising the collapsed bound for data set 0 (L-BFGS-B, numerical grad)."""
    def negL(zz):
        zz = np.sort(np.clip(zz, 1e-4, 1.0))
        f = C.ip_features(des, H, zz, K_diag=kd)
        br = C.bracket(f, y0[:, None], tau2, working=False)
        return -float(br['L'][0])
    t0 = time.perf_counter()
    res = minimize(negL, z0, method='L-BFGS-B', bounds=[(1e-4, 1.0)] * len(z0),
                   options={'maxiter': OPT_MAXITER, 'maxfun': OPT_MAXITER * (len(z0) + 2)})
    return np.sort(np.clip(res.x, 1e-4, 1.0)), {'nit': int(res.nit), 'nfev': int(res.nfev),
                                                 'seconds': time.perf_counter() - t0, 'L0': -float(res.fun),
                                                 'L_init': -negL(z0)}


def main(kind, Hs):
    global OPT_MAX_RANK, OPT_MAXITER
    if kind == 'S':
        OPT_MAX_RANK, OPT_MAXITER = 16, 25
    log = Log('E4_%s.log' % kind)
    des, hold = make_design(kind, M_TRAIN, M_HOLD)
    for H in Hs:
        tH = time.perf_counter()
        dat = np.load(os.path.join(RESULTS, 'data', 'E1_%s_H%.2f_f%g.npz' % (kind, H, F)))
        Yall, tau = dat['Y'], float(dat['tau']); tau2 = tau ** 2
        Y, Yh = Yall[:M_TRAIN], Yall[M_TRAIN:]
        Kj = joint_K(des, hold, H)
        K = Kj[:M_TRAIN, :M_TRAIN]
        kd = np.diag(K).copy(); kdh = np.diag(Kj)[M_TRAIN:].copy()
        t0 = time.perf_counter()
        ex = C.exact_evidence(K, Y, tau2)
        t_exact = time.perf_counter() - t0
        mE, vE = exact_predictive(Kj, Y, tau2)
        rmseE, nlpdE = scores(mE, vE, Yh)
        t0 = time.perf_counter()
        evals, evecs = eigh(K)
        t_eig = time.perf_counter() - t0
        Nmax = 4 * (RANKS[-1])
        t0 = time.perf_counter()
        Bm = des.moments(H, Nmax); Bh = hold.moments(H, Nmax)
        t_mom = time.perf_counter() - t0
        nrm, nrmh = des.norms(H), hold.norms(H)
        log('H', H, 'exact %.1fs eig %.1fs moments %.1fs; exact median %.1f' % (t_exact, t_eig, t_mom, np.median(ex)))
        rows = []
        for r in RANKS:
            res = {}
            # --- CF (n = r - 2) and COS (r cosines, no augmentation)
            for name, n, enrich in (('CF', r - 2, True), ('COS', r, False)):
                if n < 2:
                    continue
                t0 = time.perf_counter()
                bas = C.CFBasis(H, n, enrich=enrich)
                t_prep = time.perf_counter() - t0
                t0 = time.perf_counter()
                f = C.cf_features(des, bas, B=Bm, norms=nrm)
                t_feat = time.perf_counter() - t0 + t_mom * (bas.Naux / Nmax)
                maj = C.CFMajorant(des, bas, f)
                t0 = time.perf_counter()
                br = C.bracket(f, Y, tau2, majorant=maj)
                t_br = time.perf_counter() - t0
                Gs = bas.coords(Bh); es2 = np.maximum(nrmh - np.einsum('ij,ij->i', Gs, Gs), 0.0)
                mq, vq = predictive(f, Gs, es2, Y, tau2)
                mw, vw = predictive(f, Gs, es2, Y, tau2, working=True)
                rq, nq = scores(mq, vq, Yh); rw, nw = scores(mw, vw, Yh)
                res[name] = {'trR': f.trR, 'rho': f.trR / tau2, 'gap_L': qstats(ex - br['L']),
                             'gap_Ld': qstats(ex - br['Ld']), 'gap_U': qstats(br['U'] - ex),
                             'gap_U0': qstats(br['U0'] - ex), 'err_W': qstats(np.abs(br['W'] - ex)),
                             'viol': int(np.sum(br['Ld'] - ex > 1e-6 + 1e-9 * np.abs(ex)) +
                                         np.sum(ex - br['U'] > 1e-6 + 1e-9 * np.abs(ex))),
                             'rmse': qstats(rq), 'nlpd': qstats(nq), 'rmse_W': qstats(rw), 'nlpd_W': qstats(nw),
                             'd_nlpd_vs_exact': qstats(nq - nlpdE), 'd_nlpd_W_vs_exact': qstats(nw - nlpdE),
                             'time_prep': t_prep, 'time_features': t_feat, 'time_bound': t_br,
                             'paired_gapL': (ex - br['L']).tolist()}
            # --- COS-classical (invalid ordering)
            if des.kind != 'smooth':
                fcl = C.cos_classical_features(des, H, r)
                brc = C.bracket(fcl, Y, tau2, working=False)
                res['COS-classical'] = {'n_negative_e2': fcl.n_negative, 'min_e2': fcl.min_e2,
                                        'L_minus_exact': qstats(brc['L'] - ex),
                                        'L_above_exact_count': int(np.sum(brc['L'] > ex))}
            # --- IP-VFE on a uniform grid, FITC
            z = np.arange(1, r + 1) / r
            t0 = time.perf_counter()
            fi = C.ip_features(des, H, z, K_diag=kd)
            t_feat = time.perf_counter() - t0
            t0 = time.perf_counter()
            bri = C.bracket(fi, Y, tau2, majorant=C.ExactMajorant(K, fi))
            t_br = time.perf_counter() - t0
            Kuu = C.Points(z).K(H)
            Ksu = hold.cross_points(H, z)
            Lu = cholesky(Kuu + fi.info['jitter_rel'] * np.mean(np.diag(Kuu)) * np.eye(r), lower=True)
            Gs = solve_triangular(Lu, Ksu.T, lower=True).T
            es2 = np.maximum(kdh - np.einsum('ij,ij->i', Gs, Gs), 0.0)
            mq, vq = predictive(fi, Gs, es2, Y, tau2); mw, vw = predictive(fi, Gs, es2, Y, tau2, working=True)
            rq, nq = scores(mq, vq, Yh); rw, nw = scores(mw, vw, Yh)
            res['IP-VFE'] = {'trR': fi.trR, 'rho': fi.trR / tau2, 'gap_L': qstats(ex - bri['L']),
                             'gap_Ld': qstats(ex - bri['Ld']), 'gap_U0': qstats(bri['U0'] - ex),
                             'gap_U_exactmaj': qstats(bri['U'] - ex), 'err_FITC': qstats(np.abs(bri['W'] - ex)),
                             'rmse': qstats(rq), 'nlpd': qstats(nq), 'rmse_FITC': qstats(rw), 'nlpd_FITC': qstats(nw),
                             'd_nlpd_vs_exact': qstats(nq - nlpdE), 'd_nlpd_FITC_vs_exact': qstats(nw - nlpdE),
                             'jitter_rel': fi.info['jitter_rel'], 'cond_Kuu': fi.info['cond_Kuu'],
                             'n_negative_e2': fi.n_negative, 'time_features': t_feat, 'time_bound': t_br,
                             'paired_gapL': (ex - bri['L']).tolist()}
            # --- IP-opt (small rank)
            if r <= OPT_MAX_RANK:
                zo, oinfo = ip_opt_z(des, H, z, Y[:, 0], tau2, kd)
                fo = C.ip_features(des, H, zo, K_diag=kd)
                bro = C.bracket(fo, Y, tau2, majorant=C.ExactMajorant(K, fo))
                res['IP-opt'] = {'trR': fo.trR, 'rho': fo.trR / tau2, 'gap_L': qstats(ex - bro['L']),
                                 'gap_Ld': qstats(ex - bro['Ld']), 'gap_U0': qstats(bro['U0'] - ex),
                                 'cond_Kuu': fo.info['cond_Kuu'], 'jitter_rel': fo.info['jitter_rel'],
                                 'opt': oinfo, 'z': zo.tolist(), 'paired_gapL': (ex - bro['L']).tolist()}
            # --- ORACLE
            fo = C.oracle_features(evals, evecs, r, kd)
            bro = C.bracket(fo, Y, tau2, majorant=C.ExactMajorant(K, fo))
            res['ORACLE'] = {'trR': fo.trR, 'floor': fo.info['floor'], 'gap_L': qstats(ex - bro['L']),
                             'gap_Ld': qstats(ex - bro['Ld']), 'gap_U': qstats(bro['U'] - ex)}
            for nm in ('CF', 'COS', 'IP-VFE'):
                if nm in res:
                    res[nm]['trR_over_floor'] = res[nm]['trR'] / max(res['ORACLE']['floor'], 1e-300)
            rows.append({'rank': r, 'methods': res})
            log('  r %3d  gapL CF %.3g COS %.3g IP %.3g%s ORACLE %.3g | trR/floor CF %.3g IP %.3g | cond Kuu %.2e' % (
                r, res['CF']['gap_L']['median'] if 'CF' in res else float('nan'), res['COS']['gap_L']['median'],
                res['IP-VFE']['gap_L']['median'],
                (' IPopt %.3g' % res['IP-opt']['gap_L']['median']) if 'IP-opt' in res else '',
                res['ORACLE']['gap_L']['median'], res['CF']['trR_over_floor'] if 'CF' in res else float('nan'),
                res['IP-VFE']['trR_over_floor'], res['IP-VFE']['cond_Kuu']))
        save_json({'experiment': 'E4', 'kind': kind, 'H': H, 'f': F, 'tau': tau, 'm': M_TRAIN, 'm_hold': M_HOLD,
                   'ndata': Y.shape[1], 'exact': {'evidence': qstats(ex), 'rmse': qstats(rmseE), 'nlpd': qstats(nlpdE),
                                                  'time_evidence_100': t_exact, 'time_eig': t_eig},
                   'time_moments_all': t_mom, 'rows': rows, 'seconds': time.perf_counter() - tH},
                  'E4_%s_H%.2f.json' % (kind, H))
        log('H', H, 'done %.0fs' % (time.perf_counter() - tH))


if __name__ == '__main__':
    kind = sys.argv[1]
    Hs = [float(x) for x in sys.argv[2:]] or list(HS)
    main(kind, Hs)
