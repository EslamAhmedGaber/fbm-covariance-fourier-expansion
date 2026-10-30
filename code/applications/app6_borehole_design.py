"""A6.2: certified goal-oriented re-logging design for a borehole (synthetic site, exact-model certificate).

Scenario.  A borehole was logged in 1985 at 20, 30, ..., 600 m.  Between 2021 and 2030 a thermistor chain can be read
at any depth in [20, 600] m (1 m grid) and any quarter (37 times): m = 21,497 candidate readings, noise sd 0.05 K
(instrument plus small-scale geological noise).  Prior: dT = sigma_T B_H on 1500-2030, sigma_T = 1 K; nuisance
(T0, Gamma) with sd (10 K, 50 K/km).  Targets: T1 = mean 1961-90 minus 1850-1900, T2 = mean 1991-2020 minus 1961-90,
T3 = ground heat gain 1960-2020.  Objective Phi = sum_t V_t / V_t(1985 log only).
For each k: greedy design on the D-dimensional objective, its bracket for the exact objective, a Frank-Wolfe lower
bound on the best possible k-design, and baselines (uniform depths at fixed times, the 1985 depth pattern, random).
Exact verification of selected designs with the independent quadrature reference (covfourier.khref).
Output: results/A6_borehole_design.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from covfourier.khref import KHQuad                      # noqa: E402
from covfourier.design import exact_variances            # noqa: E402
from covfourier import heat                              # noqa: E402
import borehole_problem as BP                            # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
SIGMA_T, NOISE, SB = 1.0, 0.05, (10.0, 0.05)
Z_EXIST, T_EXIST = np.arange(20.0, 600.0 + 1e-9, 10.0), 1985.0
KS = (1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60)


def baselines(zc, tc, k, rng, n_random=200):
    out = {}
    zs = np.unique(zc); ts = np.unique(tc)
    def nearest(zt):
        return [int(np.argmin(np.abs(zc - z) + 1e3 * np.abs(tc - t))) for z, t in zt]
    for t in (ts[0], ts[-1]):
        zz = np.linspace(zs[0], zs[-1], k) if k > 1 else np.array([zs[0]])
        out[f'uniform depths at {t:.2f}'] = nearest([(z, t) for z in zz])
    zz = Z_EXIST[:k]
    out['1985 depth pattern at first time'] = nearest([(z, ts[0]) for z in zz])
    out['random'] = [list(rng.choice(len(zc), size=k, replace=False)) for _ in range(n_random)]
    return out


def exact_check(H, prep, zc, tc, designs, n_exist):
    """Exact posterior variances of the targets for the given designs (each includes the 1985 log) by quadrature."""
    win = prep.win
    idx = sorted(set(i for S in designs.values() for i in S))
    pos = {j: r for r, j in enumerate(idx)}
    feats = heat.features(win, (), BP.EPOCH_T1 + BP.EPOCH_T2, BP.HEAT_T3)
    def vals(u):
        u = np.asarray(u, dtype=float)
        return np.vstack([heat.borehole_values(Z_EXIST, T_EXIST, u, win),
                          heat.borehole_values(zc[idx], tc[idx], u, win),
                          BP.target_values(u, win)])
    t0 = time.perf_counter()
    K = KHQuad(H, features=feats, h_max=1 / 128).gram(vals) * SIGMA_T ** 2
    t_ref = time.perf_counter() - t0
    ne, nc = n_exist, len(idx)
    zall = np.r_[Z_EXIST, zc[idx]]
    Hn = np.stack([np.ones(ne + nc), zall], axis=1)
    Sb = np.diag(np.array(SB) ** 2)
    out = {}
    for name, S in designs.items():
        rows = list(range(ne)) + [ne + pos[j] for j in S]
        V = exact_variances(K[np.ix_(rows, rows)], K[np.ix_(rows, list(range(ne + nc, ne + nc + 3)))],
                            np.diag(K)[ne + nc:], np.full(len(rows), NOISE ** 2), Hn[rows], Sb)
        out[name] = V
    return out, t_ref, len(idx)


def main(H=0.75, ns=(64, 128, 256, 512), n_main=256, dz=1.0, dt=0.25, seed=11, verify=True, fw_iters=400):
    rng = np.random.default_rng(seed)
    zc, tc = BP.candidate_grid(dz=dz, dt=dt)
    res = {'config': {'H': H, 'sigma_T': SIGMA_T, 'noise_sd': NOISE, 'nuisance_sd': SB, 'z_exist': Z_EXIST.tolist(),
                      't_exist': T_EXIST, 'candidates': {'dz': dz, 'dt': dt, 'm': int(len(zc)),
                      'z_range': [float(zc.min()), float(zc.max())], 't_range': [float(tc.min()), float(tc.max())]},
                      'targets': BP.TARGET_NAMES, 'ks': list(KS), 'ns': list(ns), 'n_main': n_main, 'seed': seed},
           'refinement': [], 'main': None}
    for n in ns:
        t0 = time.perf_counter()
        prep = BP.Prepared(H, n)
        P = BP.design_problem(prep, Z_EXIST, T_EXIST, zc, tc, sigma_T=SIGMA_T, noise_sd=NOISE, sb=SB)
        cand, full = P['cand'], P['full']
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        g = cand.greedy(max(KS))
        t_greedy = time.perf_counter() - t0
        rows = []
        for k in KS:
            S = g['S'][:k]
            b = full.bracket(np.r_[np.arange(P['n_exist']), P['n_exist'] + np.array(S)])
            rows.append({'k': k, 'Phi_D': b['Phi0'], 'Phi_U': b['Phi_U_post'], 'Phi_U_unif': b['Phi_U_unif'],
                         'rel_width': (b['Phi_U_post'] - b['Phi0']) / b['Phi0'], 'rho': b['rho'],
                         'V_D': b['V0'].tolist(), 'V_U': b['U_post'].tolist()})
        entry = {'n': n, 'D': prep.D, 'build_s': t_build, 'greedy_s': t_greedy, 'timing': P['timing'],
                 'max_rel_E2_cand': float(np.max(P['E2_cand'])), 'E2_targ': P['E2_targ'].tolist(),
                 'V_exist_D': P['V_exist_D'].tolist(), 'greedy_S': [int(i) for i in g['S']], 'rows': rows}
        print(n, 'build %.1fs greedy %.2fs' % (t_build, t_greedy),
              ' '.join('k=%d w=%.1e' % (r['k'], r['rel_width']) for r in rows[::3]), flush=True)
        if n == n_main:
            main_ = {'n': n, 'fw': [], 'baselines': {}, 'greedy_history': g['Phi0_history']}
            for k in KS:
                t0 = time.perf_counter()
                w0 = np.zeros(cand.m); w0[g['S'][:k]] = 1.0
                fw = cand.frank_wolfe(k, iters=fw_iters, w0=w0)
                r = next(r for r in rows if r['k'] == k)
                main_['fw'].append({'k': k, 'LB': fw['LB'], 'relaxed': fw['Phi0_relaxed'], 'iterations': fw['iterations'],
                                    'seconds': time.perf_counter() - t0,
                                    'certified_rel_gap': (r['Phi_U'] - fw['LB']) / fw['LB'],
                                    'approx_part': (r['Phi_U'] - r['Phi_D']) / fw['LB'],
                                    'optim_part': (r['Phi_D'] - fw['LB']) / fw['LB']})
                print('  k=%d FW LB %.6f greedy [%.6f, %.6f] certified gap %.3f%% (%.1fs)' % (
                    k, fw['LB'], r['Phi_D'], r['Phi_U'], 100 * main_['fw'][-1]['certified_rel_gap'], main_['fw'][-1]['seconds']), flush=True)
                bl = baselines(zc, tc, k, rng)
                vals = {}
                for name, S in bl.items():
                    if name == 'random':
                        phis = [full.bracket(np.r_[np.arange(P['n_exist']), P['n_exist'] + np.array(s)])['Phi0'] for s in S]
                        vals[name] = {'median': float(np.median(phis)), 'p05': float(np.percentile(phis, 5)),
                                      'p95': float(np.percentile(phis, 95)), 'min': float(np.min(phis))}
                    else:
                        b = full.bracket(np.r_[np.arange(P['n_exist']), P['n_exist'] + np.array(S)])
                        vals[name] = {'Phi_D': b['Phi0'], 'Phi_U': b['Phi_U_post'], 'S': [int(i) for i in S]}
                main_['baselines'][str(k)] = vals
            # what the greedy chose (depth/time) and posterior history curves for the figure
            main_['greedy_z'] = zc[g['S']].tolist(); main_['greedy_t'] = tc[g['S']].tolist()
            res['main'] = main_
            if verify:
                designs = {f'greedy k={k}': g['S'][:k] for k in (5, 10, 20, 40, 60)}
                bl20 = baselines(zc, tc, 20, np.random.default_rng(seed + 1), n_random=3)
                for name, S in bl20.items():
                    if name == 'random':
                        for i, s in enumerate(S):
                            designs[f'random k=20 #{i}'] = [int(j) for j in s]
                    else:
                        designs[name + ' k=20'] = S
                Vex, t_ref, nint = exact_check(H, prep, zc, tc, designs, P['n_exist'])
                ver = {}
                for name, S in designs.items():
                    b = full.bracket(np.r_[np.arange(P['n_exist']), P['n_exist'] + np.array(S)])
                    V = Vex[name]
                    ver[name] = {'V_exact': V.tolist(), 'V_D': b['V0'].tolist(), 'V_U': b['U_post'].tolist(),
                                 'lower_holds': bool(np.all(b['V0'] <= V * (1 + 1e-9))),
                                 'upper_holds': bool(np.all(V <= b['U_post'] * (1 + 1e-9))),
                                 'Phi_exact': float(P['omega'] @ V), 'Phi_D': b['Phi0'], 'Phi_U': b['Phi_U_post']}
                    print('  verify %-34s Phi exact %.6f in [%.6f, %.6f] %s' % (name, ver[name]['Phi_exact'], b['Phi0'],
                          b['Phi_U_post'], ver[name]['lower_holds'] and ver[name]['upper_holds']), flush=True)
                res['verification'] = {'designs': ver, 'reference_s': t_ref, 'n_integrands': nint + len(Z_EXIST) + 3}
        res['refinement'].append(entry)
        with open(os.path.join(RES, 'A6_borehole_design.json'), 'w') as fh:
            json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
