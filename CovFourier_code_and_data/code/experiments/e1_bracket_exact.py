"""E1 bracket_exact (and the data for E2): L <= exact <= U on the full grid of PROTOCOL.md.

Usage: python e1_bracket_exact.py KIND [H ...]
Writes results/E1_<KIND>_H<H>.json (cell summaries) and results/raw/E1_<KIND>_H<H>.npz (per-data-set values),
and results/data/E1_<KIND>_H<H>_f<f>.npz (training + held-out data, reused by E4).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *                                            # noqa: F401,F403
from common import C, np, make_design, joint_K, nuisance, data_seed, save_json, Log, qstats, RESULTS
from common import HS, FLEVELS, NGRID

M_TRAIN, M_HOLD, NDATA = 2000, 500, 100


def main(kind, Hs):
    log = Log('E1_%s.log' % kind)
    os.makedirs(os.path.join(RESULTS, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS, 'data'), exist_ok=True)
    t0 = time.perf_counter()
    des, hold = make_design(kind, M_TRAIN, M_HOLD)
    log('design', kind, 'm =', des.m, 'held-out', hold.m, '%.1fs' % (time.perf_counter() - t0))
    h = nuisance(des)
    for H in Hs:
        iH = HS.index(H)
        tH = time.perf_counter()
        Kj = joint_K(des, hold, H)
        K = Kj[:M_TRAIN, :M_TRAIN]
        sd = float(np.sqrt(np.mean(np.diag(K))))
        log('H', H, 'K built %.1fs, sd_rms %.4f' % (time.perf_counter() - tH, sd))
        tm = time.perf_counter()
        Bmom = des.moments(H, 4 * NGRID[-1])
        nrm = des.norms(H)
        log('moments (%d x %d) %.1fs; max|diag K - norms| %.2e' % (Bmom.shape + (time.perf_counter() - tm,
                                                                                np.max(np.abs(np.diag(K) - nrm)))))
        # data and exact evidence
        data = {}
        for jf, f in enumerate(FLEVELS):
            tau = f * sd
            rng = np.random.default_rng(data_seed(kind, iH, jf))
            Lj = np.linalg.cholesky(Kj + tau ** 2 * np.eye(Kj.shape[0]))
            Yall = Lj @ rng.standard_normal((Kj.shape[0], NDATA))
            beta_g = rng.standard_normal((2, NDATA))
            np.savez_compressed(os.path.join(RESULTS, 'data', 'E1_%s_H%.2f_f%g.npz' % (kind, H, f)),
                                Y=Yall, beta_g=beta_g, tau=tau, sd=sd)
            Y = Yall[:M_TRAIN]
            Yg = Y + h @ beta_g
            Yr = Y + h @ np.array([[0.3], [-0.5]])
            te = time.perf_counter()
            ex = {'none': C.exact_evidence(K, Y, tau ** 2),
                  'gauss': C.exact_evidence(K, Yg, tau ** 2, h=h, Sigma_beta=np.eye(2)),
                  'reml': C.exact_evidence(K, Yr, tau ** 2, h=h, reml=True)}
            data[f] = {'tau': tau, 'Y': {'none': Y, 'gauss': Yg, 'reml': Yr}, 'ex': ex}
            log('  f', f, 'tau %.3e exact evidences %.1fs' % (tau, time.perf_counter() - te))
        cells, raw = [], {}
        for n in NGRID:
            tb = time.perf_counter()
            bas = C.CFBasis(H, n)
            feats = C.cf_features(des, bas, B=Bmom, norms=nrm)
            maj = C.CFMajorant(des, bas, feats)
            tfe = time.perf_counter() - tb
            for jf, f in enumerate(FLEVELS):
                tau = data[f]['tau']
                for mode in ('none', 'gauss', 'reml'):
                    Y = data[f]['Y'][mode]
                    kw = {}
                    if mode == 'gauss':
                        kw = dict(h=h, Sigma_beta=np.eye(2))
                    elif mode == 'reml':
                        kw = dict(h=h, reml=True)
                    br = C.bracket(feats, Y, tau ** 2, majorant=maj, **kw)
                    ex = data[f]['ex'][mode]
                    delta = 1e-6 + 1e-9 * np.abs(ex)
                    vL = int(np.sum(br['L'] - ex > delta)); vU = int(np.sum(ex - br['U'] > delta))
                    vLd = int(np.sum(br['Ld'] - ex > delta))
                    cell = {'kind': kind, 'H': H, 'f': f, 'tau': tau, 'n': n, 'D': bas.D, 'mode': mode,
                            'rho': br['rho'], 'trR': br['trR'], 'n_negative_e2': feats.n_negative,
                            'min_e2': feats.min_e2, 'viol_L': vL, 'viol_Ld': vLd, 'viol_U': vU,
                            'max_L_minus_exact': float(np.max(br['L'] - ex)),
                            'max_exact_minus_U': float(np.max(ex - br['U'])),
                            'gap_L': qstats(ex - br['L']), 'gap_Ld': qstats(ex - br['Ld']), 'gap_U': qstats(br['U'] - ex),
                            'max_Ld_minus_exact': float(np.max(br['Ld'] - ex)),
                            'gap_U0': qstats(br['U0'] - ex), 'gap_U1': qstats(br['U1'] - ex),
                            'gap_U2': qstats(br['U2'] - ex), 'err_W': qstats(np.abs(br['W'] - ex)),
                            'W_minus_exact': qstats(br['W'] - ex),
                            'width': qstats(br['U'] - br['Ld']), 'exact': qstats(ex),
                            'beta_tail_frac': qstats(br['beta_tail'] / np.maximum(br['beta'], 1e-300)),
                            'seconds_features': tfe, 'seconds_bracket': br['seconds'],
                            'prep_seconds': bas.setup_seconds}
                    cells.append(cell)
                    key = 'n%d_f%g_%s' % (n, f, mode)
                    raw[key] = np.vstack([ex, br['L'], br['Ld'], br['U0'], br['U1'], br['U2'], br['U'], br['W']])
            c0 = [c for c in cells if c['n'] == n]
            log('  n %4d D %4d features %.1fs; viol %d/%d; rho(f=0.03) %.3g; median gapL %.3g gapU %.3g' % (
                n, bas.D, tfe, sum(c['viol_L'] + c['viol_Ld'] for c in c0), sum(c['viol_U'] for c in c0),
                [c for c in c0 if c['f'] == 0.03][0]['rho'],
                [c for c in c0 if c['f'] == 0.03 and c['mode'] == 'none'][0]['gap_L']['median'],
                [c for c in c0 if c['f'] == 0.03 and c['mode'] == 'none'][0]['gap_U']['median']))
        np.savez_compressed(os.path.join(RESULTS, 'raw', 'E1_%s_H%.2f.npz' % (kind, H)),
                            rows=np.array(['exact', 'L', 'Ld', 'U0', 'U1', 'U2', 'U', 'W']), **raw)
        save_json({'experiment': 'E1', 'kind': kind, 'H': H, 'm': M_TRAIN, 'ndata': NDATA, 'sd_rms': sd,
                   'cells': cells, 'seconds': time.perf_counter() - tH,
                   'design_info': {'corrected_fraction': (float(np.mean(des.corrected)) if kind == 'S' else None)}},
                  'E1_%s_H%.2f.json' % (kind, H))
        log('H', H, 'done %.1fs' % (time.perf_counter() - tH))


if __name__ == '__main__':
    kind = sys.argv[1]
    Hs = [float(x) for x in sys.argv[2:]] or list(HS)
    main(kind, Hs)
