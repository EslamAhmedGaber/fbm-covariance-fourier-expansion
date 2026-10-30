"""A7.5: certified re-logging design for a real borehole network (posteriors from the real logs, design for 2030).

Network: the selected Canadian sites of A7.1 (real logs, v2 hyperparameters of the best H).  Candidates: re-reading,
in 2030, any original depth <= 300 m of any site (m = sum over sites).  Per site s the latent vector is
x_s = (Z (D), T0, Gamma, delta, g), with g the geological deviation sigma_g W(z) at the site's original depths
(Brownian prior, shared by both logs of the hole) and delta the calibration offset of the new log (sd 0.05 K).
Existing readings define the prior precision P_s; the network target is the mean over sites of
W_s = mean dT over [2011, 2030] minus mean dT over [t_log - 20, t_log].  Sites are independent, so the design uses
covfourier.design.BlockInfoDesign (greedy, block-wise bracket, Frank-Wolfe lower bound).
Baselines: re-log the earliest-logged sites first (all depths, shallow first); all sites at their shallowest depths;
random readings.  Output: results/A7_network_design.json
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from covfourier import heat                              # noqa: E402
from covfourier.design import BlockInfoDesign            # noqa: E402
import borehole_problem as BP                            # noqa: E402
import app7_borehole_realdata as V1                      # noqa: E402
import app7_borehole_realdata_v2 as V2                   # noqa: E402

RES = V1.RES
T_NEW, ZMAX, OFFSET_SD, SB = 2030.0, 300.0, 0.05, (100.0, 0.1)
KS = (5, 10, 20, 50, 100, 200, 400)


def main(n=256, country='Canada', src='A7_borehole_realdata_v2.json', seed=5):
    fit = json.load(open(os.path.join(RES, src)))
    byH = fit['by_H']
    H = max((float(h) for h in byH), key=lambda h: byH[str(h)]['ll_train'])
    rec = byH[str(H)]
    sT, sG = rec['sigma_T'], rec.get('sigma_g', 0.0) or 0.0
    cru = V1.Crutem()
    sites, _, _ = V1.select_sites(cru)
    idx = [i for i, s in enumerate(sites) if s['country'] == country]
    t0 = time.perf_counter()
    prep = BP.Prepared(H, n, N_cand=8192, N_targ=1 << 16)
    D = prep.D
    rows, blk, s2, P_blocks, C_blocks, e2, e2t, meta = [], [], [], [], [], [], [], []
    rows_f, blk_f, s2_f, e2_f, Pi_blocks, forced = [], [], [], [], [], []
    u = heat.midpoints(prep.N_targ)
    Nsite = len(idx)
    for b, i in enumerate(idx):
        s = sites[i]; per = rec['site'][i]
        z, tl = s['z'], s['t_log']
        sE = per['sigma_e']
        J = len(z)
        d = D + 3 + (J if sG > 0 else 0)
        Ze, Ee = prep.readings(z, np.full(J, tl))
        Pi = np.zeros((d, d)); Pi[:D, :D] = np.eye(D)
        Pi[D, D] = 1 / SB[0] ** 2; Pi[D + 1, D + 1] = 1 / SB[1] ** 2; Pi[D + 2, D + 2] = 1 / OFFSET_SD ** 2
        if sG > 0:
            Mg = sG ** 2 * np.minimum.outer(z, z)
            Pi[D + 3:, D + 3:] = np.linalg.inv(Mg)
        def row(Zr, zz, new):
            a = np.zeros((len(zz), d)); a[:, :D] = sT * Zr; a[:, D] = 1.0; a[:, D + 1] = zz
            if new:
                a[:, D + 2] = 1.0
            if sG > 0:
                for r_, zr in enumerate(zz):
                    a[r_, D + 3 + int(np.argmin(np.abs(z - zr)))] = 1.0
            return a
        Ae = row(Ze, z, False)
        P = Pi + (Ae.T / sE ** 2) @ Ae                                   # projected model: residuals dropped (Proposition)
        zc = z[z <= ZMAX]
        Zc, Ec = prep.readings(zc, np.full(len(zc), T_NEW))
        Ac = row(Zc, zc, True)
        F = (heat.epoch_mean_values(2011.0, 2031.0, u, prep.win) - heat.epoch_mean_values(tl - 20, tl, u, prep.win))[None, :]
        Zt, Et, _ = heat.project_samples(prep.bp_t, F)
        c = np.zeros(d); c[:D] = sT * Zt[0] / Nsite
        rows.append(Ac); blk += [b] * len(zc); s2.append(np.full(len(zc), sE ** 2)); e2.append(sT ** 2 * Ec)
        P_blocks.append(P); C_blocks.append(c); e2t.append(sT ** 2 * Et[0] / Nsite ** 2)
        meta.append({'site': s['site'], 't_log': tl, 'n_candidates': int(len(zc)), 'z': zc.tolist(), 'sigma_e': sE})
        # second object for brackets of E u S: prior only, rows = existing + new (existing always selected)
        off_f = sum(len(r_) for r_ in rows_f)
        forced += list(range(off_f, off_f + J))
        rows_f.append(np.vstack([Ae, Ac])); blk_f += [b] * (J + len(zc)); s2_f.append(np.full(J + len(zc), sE ** 2))
        e2_f.append(sT ** 2 * np.r_[Ee, Ec]); Pi_blocks.append(Pi)
    # blocks have different latent dimensions (the geological values at each site's depths): zero-pad every block to
    # the largest dimension; the padding coordinates have prior precision 1, enter no reading and no target
    dmax = max(P.shape[0] for P in P_blocks)
    def pad_rows(R):
        out_ = np.zeros((R.shape[0], dmax)); out_[:, :R.shape[1]] = R; return out_
    def pad_mat(M):
        out_ = np.eye(dmax); out_[:M.shape[0], :M.shape[1]] = M; return out_
    def pad_vec(v):
        out_ = np.zeros(dmax); out_[:len(v)] = v; return out_
    rows = [pad_rows(R) for R in rows]; rows_f = [pad_rows(R) for R in rows_f]
    P_blocks = [pad_mat(P) for P in P_blocks]; Pi_blocks = [pad_mat(P) for P in Pi_blocks]
    C_blocks = [pad_vec(c) for c in C_blocks]
    A = np.vstack(rows); blk = np.array(blk); s2 = np.concatenate(s2); e2 = np.concatenate(e2)
    t_build = time.perf_counter() - t0
    bd = BlockInfoDesign(A, blk, s2, P_blocks, C_blocks, e2=e2, e2_t=np.array(e2t))
    Af = np.vstack(rows_f); blkf = np.array(blk_f)
    full = BlockInfoDesign(Af, blkf, np.concatenate(s2_f), Pi_blocks, C_blocks, e2=np.concatenate(e2_f), e2_t=np.array(e2t))
    pos_new = np.where(~np.isin(np.arange(len(blkf)), forced))[0]          # full-object index of each candidate
    bracket = lambda S: full.value(np.r_[np.array(forced, int), pos_new[np.asarray(S, int)]])
    v0 = bd.value([])['Phi0']
    out = {'config': {'H': H, 'sigma_T': sT, 'sigma_g': sG, 'country': country, 'n_sites': Nsite, 'm': int(len(blk)),
                      't_new': T_NEW, 'zmax': ZMAX, 'offset_sd': OFFSET_SD, 'n': n, 'ks': list(KS)},
           'sd_existing_only': math.sqrt(v0), 'build_s': t_build, 'sites': meta, 'rows': []}
    print('sites', Nsite, 'candidates', len(blk), 'build %.1fs' % t_build, 'sd before re-logging %.4f K' % math.sqrt(v0), flush=True)
    t0 = time.perf_counter(); g = bd.greedy(max(KS)); t_g = time.perf_counter() - t0
    rng = np.random.default_rng(seed)
    order_old = np.argsort([m['t_log'] for m in meta])                 # earliest-logged sites first
    def old_first(k):
        S = []
        for b in order_old:
            cand = np.where(blk == b)[0]
            cand = cand[np.argsort(A[cand, D + 1])]                    # shallow first
            for j in cand:
                if len(S) < k:
                    S.append(int(j))
        return S
    def shallowest_everywhere(k):
        firsts = [int(np.where(blk == b)[0][np.argmin(A[np.where(blk == b)[0], D + 1])]) for b in range(Nsite)]
        firsts = sorted(firsts, key=lambda j: meta[blk[j]]['t_log'])
        return firsts[:k]
    for k in KS:
        S = g['S'][:k]
        b = bracket(S)
        t1 = time.perf_counter(); w0 = np.zeros(bd.m); w0[S] = 1.0
        fw = bd.frank_wolfe(k, iters=60, w0=w0); t_fw = time.perf_counter() - t1
        base = {'earliest sites first': bd.value(old_first(k))['Phi0'],
                'shallowest depth, earliest sites': bd.value(shallowest_everywhere(k))['Phi0'],
                'random (median of 50)': float(np.median([bd.value(rng.choice(bd.m, k, replace=False))['Phi0'] for _ in range(50)]))}
        r = {'k': k, 'sd_D': math.sqrt(b['Phi0']), 'sd_U': math.sqrt(b['Phi_U_post']), 'LB_sd': math.sqrt(max(fw['LB'], 0.0)),
             'certified_rel_gap_var': (b['Phi_U_post'] - fw['LB']) / fw['LB'], 'fw_s': t_fw,
             'n_sites_used': int(len(set(blk[S]))), 'baselines_sd': {kk: math.sqrt(v) for kk, v in base.items()},
             'depths': [float(A[j, D + 1]) for j in S], 'sites': [meta[blk[j]]['site'] for j in S]}
        out['rows'].append(r)
        print('k=%d sd [%.4f, %.4f] LB %.4f gap %.2f%% sites %d | baselines %s' % (k, r['sd_D'], r['sd_U'], r['LB_sd'], 100 * r['certified_rel_gap_var'],
              r['n_sites_used'], {kk: round(v, 4) for kk, v in r['baselines_sd'].items()}), flush=True)
    out['greedy_s'] = t_g
    with open(os.path.join(RES, 'A7_network_design.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
