"""Figures 9 and 10 (certified design and real borehole data); same typography and colours as make_figures.py.

Reads results/A6_*.json(l) and results/A7_*.json; writes figures/fig9.*, figures/fig10.* and figures/data/fig9.json,
fig10.json.  Run:  python figures/src/make_figures_design.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_figures as MF                                   # noqa: E402
from make_figures import plt, Line2D, C_OURS, C_OURS_LIGHT, C_QUAD, C_GREY, TINY, BASE   # noqa: E402

RES = MF.RES


def _load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def fig10():
    d = _load('A6_borehole_design.json')
    main = d['main']; ks = d['config']['ks']
    ref = {r['n']: r for r in d['refinement']}
    nm = d['config']['n_main']
    rows = {r['k']: r for r in ref[nm]['rows']}
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.9), gridspec_kw={'wspace': 0.34, 'hspace': 0.52})
    out = {}
    # (a) objective against budget
    ax = axs[0, 0]
    K = np.array(ks)
    lb = np.array([f['LB'] for f in main['fw']])
    phiD = np.array([rows[k]['Phi_D'] for k in ks]); phiU = np.array([rows[k]['Phi_U'] for k in ks])
    rnd = [main['baselines'][str(k)]['random'] for k in ks]
    ax.fill_between(K, [r['p05'] for r in rnd], [r['p95'] for r in rnd], color='0.88', lw=0, label='random designs (5--95\\%)' if MF.USETEX else 'random designs (5-95%)')
    ax.plot(K, [r['median'] for r in rnd], color='0.6', lw=0.9, label='random designs (median)')
    names = [n for n in main['baselines'][str(ks[-1])] if n != 'random']
    mk = {names[0]: 's', names[1]: 'D', names[2]: '^'}
    lab = {names[0]: 'uniform depths, 2021', names[1]: 'uniform depths, 2030', names[2]: 'the 1985 depths, 2021'}
    for nme in names:
        ax.plot(K, [main['baselines'][str(k)][nme]['Phi_D'] for k in ks], mk[nme], color=C_GREY, mfc='white', ms=3.2, mew=0.7, lw=0, label=lab[nme])
    ax.plot(K, phiU, color=C_OURS, lw=1.3, label='greedy design (bracket)')
    ax.plot(K, lb, 'k--', lw=0.9, label='lower bound, any design')
    if 'verification' in d:
        for nme, v in d['verification']['designs'].items():
            if nme.startswith('greedy'):
                kk = int(nme.split('=')[1])
                ax.plot([kk], [v['Phi_exact']], 'o', color=C_OURS, mfc='white', ms=4.2, mew=0.8)
        ax.plot([], [], 'o', color=C_OURS, mfc='white', ms=4.2, mew=0.8, label='exact (quadrature)')
    ax.set_xscale('log'); ax.set_xlabel('new readings $k$'); ax.set_ylabel('$\\Phi=\\sum_t V_t/V_t^{1985}$')
    ax.set_title('Target variance against budget')
    ax.set_ylim(0.9, 4.15)
    ax.legend(fontsize=TINY - 1.5, loc='upper center', handlelength=1.3, ncol=2, columnspacing=0.8)
    MF.letter(ax, 'a')
    out['a'] = {'k': ks, 'LB': lb.tolist(), 'Phi_D': phiD.tolist(), 'Phi_U': phiU.tolist(), 'random': rnd}
    # (b) where and when: the greedy design, over the value of a single first reading
    ax = axs[0, 1]
    gz = np.array(main['greedy_z']); gt = np.array(main['greedy_t'])
    kmax = 40
    smap = os.path.join(RES, 'A6_design_scoremap.json')
    if os.path.exists(smap):
        sm = _load('A6_design_scoremap.json')
        Zg = np.array(sm['z']); Tg = np.array(sm['t']); Sg = np.array(sm['score_first']).reshape(len(Zg), len(Tg))
        im = ax.pcolormesh(Tg, Zg, Sg / Sg.max(), cmap='Greys', shading='auto', vmin=0, vmax=1, rasterized=True)
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05); cb.set_label('value of one reading (scaled)', fontsize=TINY - 0.5)
        cb.ax.tick_params(labelsize=TINY - 1)
    ax.scatter(gt[:kmax], gz[:kmax], s=13, color=C_OURS, edgecolor='white', linewidth=0.4, zorder=3, label='greedy readings 1--%d' % kmax if MF.USETEX else 'greedy readings 1-%d' % kmax)
    for i in range(3):
        ax.annotate(str(i + 1), (gt[i], gz[i]), textcoords='offset points', xytext=(3, 3), fontsize=TINY - 1, color=C_OURS, zorder=4)
    ax.set_ylim(300, 15); ax.set_xlim(2020.8, 2030.2)
    ax.set_xlabel('reading time [yr]'); ax.set_ylabel('depth [m]')
    ax.set_title('Where and when to read')
    ax.legend(fontsize=TINY - 1, loc='lower right', handlelength=1.0, framealpha=0.9)
    MF.letter(ax, 'b')
    out['b'] = {'z': gz[:kmax].tolist(), 't': gt[:kmax].tolist()}
    # (c) certificate: approximation part vs optimisation part, and the bracket width against D
    ax = axs[1, 0]
    gap_opt = np.array([f['optim_part'] for f in main['fw']]); gap_app = np.array([f['approx_part'] for f in main['fw']])
    ax.loglog(K, np.maximum(gap_opt, 1e-12), 'k-', marker='o', ms=2.8, lw=0.9, label='optimisation: $(\\Phi_D-L)/L$')
    for i, n in enumerate(sorted(ref)):
        w = [r['rel_width'] * r['Phi_D'] / lbk for r, lbk in zip(ref[n]['rows'], lb)]
        ax.loglog(K, w, color=C_OURS, lw=0.6 + 0.35 * i, alpha=0.45 + 0.15 * i,
                  label='approximation: $(\\Phi^+-\\Phi_D)/L$' if i == len(ref) - 1 else None)
        ax.text(K[0] * 0.92, w[0], '$D=%d$' % (n + 2), ha='right', va='center', fontsize=TINY - 1.5, color=C_OURS)
    ax.set_xlim(K[0] * 0.35, K[-1] * 1.3)
    ax.set_xlabel('new readings $k$'); ax.set_ylabel('relative size')
    ax.set_title('The two parts of the certificate')
    ax.legend(fontsize=TINY - 1.5, loc='center right', handlelength=1.4, bbox_to_anchor=(1.0, 0.49))
    MF.letter(ax, 'c')
    out['c'] = {'optim': gap_opt.tolist(), 'approx_by_n': {n: [r['rel_width'] for r in ref[n]['rows']] for n in ref}}
    # (d) cost against the number of candidates
    ax = axs[1, 1]
    path = os.path.join(RES, 'A6_design_scaling.jsonl')
    if os.path.exists(path):
        recs = [json.loads(l) for l in open(path) if l.strip()]
        ex = sorted([r for r in recs if r['method'] == 'expansion'], key=lambda r: r['m'])
        de = sorted([r for r in recs if r['method'] != 'expansion'], key=lambda r: r['m'])
        if ex:
            m = np.array([r['m'] for r in ex])
            ax.loglog(m, [r['build_s'] + r['greedy_s'] + r['fw_100_s'] for r in ex], 'o-', color=C_OURS, ms=3, label='CovFourier: coordinates + greedy + 100 FW')
            ax.loglog(m, [r['greedy_s'] + r['fw_100_s'] for r in ex], 'o:', color=C_OURS, mfc='white', ms=3, label='CovFourier: greedy + 100 FW')
        if de:
            m = np.array([r['m'] for r in de])
            ax.loglog(m, [r['factor_s'] + r['greedy_s'] for r in de], 's-', color=C_QUAD, ms=3, label='product quadrature: dense greedy')
            ch = [(r['m'], r['cholesky_s']) for r in de if r.get('cholesky_s')]
            if ch:
                mm = np.array([c[0] for c in ch]); tt = np.array([c[1] for c in ch])
                ax.loglog(mm, 100 * tt, 'D--', color=C_QUAD, mfc='white', ms=3, label='dense relaxation: 100 steps (measured step)')
                mx = np.logspace(np.log10(mm.max()), np.log10(1.2 * max(r['m'] for r in ex)), 20)
                ax.loglog(mx, 100 * tt[-1] * (mx / mm.max()) ** 3, ':', color=C_QUAD, lw=0.8, label='same, $m^3$ extrapolation')
        import matplotlib.transforms as mtransforms
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for sec, lab in ((3600.0, '1 h'), (86400.0, '1 day')):
            ax.axhline(sec, color='0.8', lw=0.5, zorder=0)
            ax.text(1.01, sec, lab, transform=trans, fontsize=TINY - 2, color='0.45', va='center', ha='left', clip_on=False)
        if ex:
            r = ex[-1]
            ax.annotate('peak %.2f GiB' % r['peak_gib'], (r['m'], r['build_s'] + r['greedy_s'] + r['fw_100_s']), textcoords='offset points',
                        xytext=(-44, -15), fontsize=TINY - 2, color=C_OURS)
        ax.set_xlabel('candidate readings $m$'); ax.set_ylabel('wall time [s]')
        ax.set_ylim(top=3e7)
        ax.legend(fontsize=TINY - 1.5, loc='upper left', handlelength=1.6, framealpha=0.95)
        out['d'] = recs
    ax.set_title('Cost of a certified design')
    MF.letter(ax, 'd')
    MF.save(fig, 'fig10', out)


if __name__ == '__main__':
    MF.style()
    which = sys.argv[1:] or ['fig10']
    for w in which:
        globals()[w]()
# figures renumbered for the revised paper (CovFourier revision)
