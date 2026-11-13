"""Figures of the paper, in the visual language of the companion (CosChaos): one colour per Hurst index
(viridis), our method solid with filled circles, exact/reference wide light grey, competitor dashed with open
squares, STIX serif 8.8 pt at the final width (468 pt).  Reads new_round/results/*.json only.

Usage: python make_figures.py [fig1 fig2 fig3 fig4 fig5]  ->  new_round/figures/Figure_<k>.{pdf,png}
"""
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'figures')
os.makedirs(OUT, exist_ok=True)

BASE, SMALL, TICK = 8.8, 7.6, 8.0
mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': BASE, 'axes.labelsize': BASE, 'axes.titlesize': BASE, 'legend.fontsize': SMALL,
    'xtick.labelsize': TICK, 'ytick.labelsize': TICK, 'lines.linewidth': 1.3, 'lines.markersize': 3.4,
    'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5, 'legend.frameon': False, 'legend.handlelength': 1.8, 'legend.borderaxespad': 0.3,
    'legend.labelspacing': 0.28, 'legend.handletextpad': 0.5, 'legend.columnspacing': 1.1,
    'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': True})
_cm = mpl.colormaps['viridis']
W_ONE = 468 / 72.27
REF = dict(color='0.62', lw=2.6, solid_capstyle='round', zorder=1)
OURS = dict(ls='-', marker='o', ms=3.4, lw=1.3)
COMP = dict(ls=(0, (4, 1.8)), marker='s', ms=3.4, lw=1.0, mfc='white', mew=0.9)
GUIDE = dict(color='0.55', ls=(0, (1, 1.8)), lw=0.9, zorder=0)
KIND_NAME = {'P': 'point values', 'W': 'window averages', 'S': 'heat kernels'}
CHECKS = []


def HC(H):
    H = float(H)
    return '0.15' if abs(H - 0.5) < 1e-9 else _cm(0.08 + 0.62 * (0.95 - H) / 0.45)


def light(c, f):
    c = np.array(mpl.colors.to_rgb(c))
    return tuple(1 - f * (1 - c))


def J(name):
    return json.load(open(os.path.join(RES, name), encoding='utf-8'))


def letter(ax, s, dx=-0.2):
    ax.text(dx, 1.03, '(%s)' % s, transform=ax.transAxes, fontsize=BASE + 0.4, fontweight='bold', va='bottom', ha='left')


def H_key(fig, Hs, y=0.995, x=0.995, extra=()):
    hs = [Line2D([], [], color=HC(H), lw=2.4, solid_capstyle='butt') for H in Hs] + [e[0] for e in extra]
    labs = [r'$H=\frac{1}{2}$' if abs(float(H) - 0.5) < 1e-9 else '$H=%s$' % H for H in Hs] + [e[1] for e in extra]
    fig.legend(hs, labs, loc='upper right', bbox_to_anchor=(x, y), ncol=len(hs), fontsize=SMALL + 0.4,
               handlelength=1.5, columnspacing=1.3, borderaxespad=0)


def logD(ax):
    ax.set_xscale('log')
    ax.set_xticks([10, 30, 100, 300, 1000]); ax.set_xticklabels(['10', '30', '100', '300', '1000'])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300)
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
    over = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    CHECKS.append({'figure': name, 'text_overlaps': over})
    plt.close(fig)


def load_e1(kind):
    cells = []
    for H in (0.55, 0.75, 0.95):
        p = os.path.join(RES, 'E1_%s_H%.2f.json' % (kind, H))
        if os.path.exists(p):
            cells += J(os.path.basename(p))['cells']
    return cells


def four(h=2.25, top=0.9):
    fig, ax = plt.subplots(1, 4, figsize=(W_ONE, h), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.05, rect=(0, 0, 1, top))
    return fig, ax


# ----------------------------------------------------------------------------------------------- Figure 1
def fig1():
    d = J('F1_idea.json')
    fig, ax = four(2.2)
    c = HC(0.75)
    a = ax[0]
    a.plot(d['path_t'], d['path_B'], color='0.35', lw=0.7, zorder=1)
    a.plot(d['points_t'], d['points_B'], ls='none', marker='o', ms=3.6, color=c, zorder=3)
    for lo_, hi_, yv in d['windows_shown']:
        a.plot([lo_, hi_], [yv, yv], color=HC(0.55), lw=3.4, solid_capstyle='butt', zorder=2)
    a.set_xlabel('time $t$'); a.set_ylabel(r'$B_H(t)$, $H=0.75$')
    a.set_title('one path, two views', loc='left', fontsize=BASE)
    hs = [Line2D([], [], ls='none', marker='o', ms=3.6, color=c), Line2D([], [], color=HC(0.55), lw=2.4)]
    a.legend(hs, ['points', 'windows'], loc='lower left', fontsize=SMALL, handlelength=1.0)
    letter(a, 'a', dx=-0.3)
    a = ax[1]
    lk = np.array(d['eig_K']); lq = np.array(d['eig_Q']); k = np.arange(1, lk.size + 1)
    nq = int(np.sum(lq > 1e-14 * lq[0]))
    a.plot(k, lk, **REF)
    a.plot(k[:nq], lq[:nq], color=c, ls='none', marker='o', ms=2.6)
    a.plot(k[:nq], lq[:nq] + d['trR_D34'], color=c, ls=(0, (3, 1.5)), lw=1.0)
    a.set_yscale('log'); a.set_xlabel('index $j$'); a.set_ylabel('eigenvalue')
    a.set_xlim(0, 60)
    a.set_title('covariance sandwich', loc='left', fontsize=BASE)
    hs = [Line2D([], [], **{k_: v for k_, v in REF.items() if k_ != 'zorder'}), Line2D([], [], color=c, ls='none', marker='o', ms=2.6),
          Line2D([], [], color=c, ls=(0, (3, 1.5)), lw=1.0)]
    a.legend(hs, [r'$\lambda_j(K)$', r'$\lambda_j(\Gamma\Gamma^{\top})$', r'$+\,\mathrm{tr}\,R$'], loc='upper right', fontsize=SMALL)
    letter(a, 'b', dx=-0.3)
    a = ax[2]
    w = np.array(d['widths'])
    shades = {'18': light(c, 0.35), '34': light(c, 0.65), '66': c}
    for Dk in ('18', '34', '66'):
        e2 = np.array(d['e2_by_D'][Dk])
        ok = e2 > 0
        a.plot(w[ok], e2[ok], ls='none', marker='o', ms=1.8, color=shades[Dk], label='$D=%s$' % Dk)
    wg = np.array([0.006, 0.06]); e66 = np.array(d['e2_by_D']['66'])
    ref = np.median(e66[(w > 0.02) & (w < 0.03)]) if np.any((w > 0.02) & (w < 0.03)) else e66.max()
    a.plot(wg, ref * (wg / 0.025) ** -2, **GUIDE)
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('window width'); a.set_ylabel(r'residual energy $e_i^2$')
    a.set_xticks([0.01, 0.03]); a.set_xticklabels(['0.01', '0.03']); a.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a.set_title('per observation', loc='left', fontsize=BASE)
    a.legend(loc='lower left', fontsize=SMALL, handletextpad=0.1, markerscale=1.6)
    letter(a, 'c', dx=-0.3)
    a = ax[3]
    br = d['bracket']; D = np.array([r['D'] for r in br])
    lo = np.array([r['L'] for r in br]); up = np.array([r['U'] for r in br])
    a.axhline(d['exact'], **REF)
    a.fill_between(D, lo, up, color=light(c, 0.2), lw=0)
    a.plot(D, lo, color=c, **OURS)
    a.plot(D, up, color=c, ls=(0, (3, 1.5)), lw=1.0, marker='o', ms=3.2, mfc='white', mew=0.8)
    logD(a); a.set_xlim(8, 320); a.set_xlabel('features $D$'); a.set_ylabel('log evidence (nats)')
    a.set_title('two-sided bracket', loc='left', fontsize=BASE)
    hs = [Line2D([], [], color='0.62', lw=2.6), Line2D([], [], color=c, **OURS),
          Line2D([], [], color=c, ls=(0, (3, 1.5)), lw=1.0, marker='o', ms=3.2, mfc='white')]
    a.legend(hs, [r'exact $\ell$', '$L$', '$U$'], loc='lower right', fontsize=SMALL)
    letter(a, 'd', dx=-0.3)
    CHECKS.append({'fig1_bracket_contains_exact': bool(np.all(lo <= d['exact'] + 1e-6) and np.all(up >= d['exact'] - 1e-6)),
                   'fig1_eig_sandwich_ok': bool(np.all(lq[:nq] <= lk[:nq] * (1 + 1e-9)) and np.all(lk[:nq] <= lq[:nq] + d['trR_D34'] + 1e-12))})
    save(fig, 'Figure_1')


# ----------------------------------------------------------------------------------------------- Figure 2
def fig2():
    fig, ax = four(2.2)
    order = {'P': lambda H: 2 * H, 'W': lambda H: 2 * H + 2, 'S': lambda H: 2 * H + 6}
    lab = {'P': r'guides $D^{-2H}$', 'W': r'guides $D^{-2H-2}$', 'S': r'guides $D^{-2H-6}$'}
    for j, kind in enumerate(('P', 'W', 'S')):
        cells = [c for c in load_e1(kind) if c['mode'] == 'none' and abs(c['f'] - 0.03) < 1e-12]
        for H in (0.55, 0.75, 0.95):
            cc = sorted([c for c in cells if c['H'] == H], key=lambda c: c['D'])
            D = np.array([c['D'] for c in cc]); e = np.array([c['trR'] for c in cc]) / 2000
            ok = e > 0
            ax[j].plot(D[ok], e[ok], color=HC(H), **OURS)
            a = order[kind](H)
            Dg = D[ok][-4:]
            ax[j].plot(Dg, e[ok][-1] * (Dg / Dg[-1]) ** (-a), **GUIDE)
        logD(ax[j]); ax[j].set_yscale('log'); ax[j].set_xlabel('features $D$')
        ax[j].set_title(KIND_NAME[kind], loc='left', fontsize=BASE)
        ax[j].text(0.04, 0.04, lab[kind], transform=ax[j].transAxes, ha='left', va='bottom', fontsize=SMALL)
        letter(ax[j], 'abc'[j], dx=-0.3)
    ax[0].set_ylabel(r'mean residual energy $\bar e^{\,2}$')
    a = ax[3]
    mk = {'P': 'o', 'W': 's', 'S': 'D'}
    allr = []
    for kind in ('P', 'W', 'S'):
        for H in (0.55, 0.75, 0.95):
            p = os.path.join(RES, 'E3_%s_H%.2f.json' % (kind, H))
            if not os.path.exists(p):
                continue
            for c in J(os.path.basename(p))['cells']:
                if c['D_meas'] and c['D_pred']:
                    a.plot(c['D_pred'], c['D_meas'], ls='none', marker=mk[kind], ms=3.6, color=HC(H),
                           mfc=HC(H) if kind != 'S' else 'white', mew=0.9)
                    allr.append(c['D_meas'] / c['D_pred'])
    x = np.array([8, 1200])
    a.plot(x, x, **REF); a.plot(x, 2 * x, **GUIDE); a.plot(x, x / 2, **GUIDE)
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(8, 1200); a.set_ylim(8, 1200)
    a.set_xlabel('predicted $D$'); a.set_ylabel('measured $D$')
    a.set_title('feature count', loc='left', fontsize=BASE)
    hs = [Line2D([], [], ls='none', marker=mk[k], color='0.3', mfc='0.3' if k != 'S' else 'white', ms=3.6) for k in mk]
    a.legend(hs, ['points', 'windows', 'kernels'], loc='upper left', fontsize=SMALL, handletextpad=0.2)
    letter(a, 'd', dx=-0.3)
    H_key(fig, ('0.55', '0.75', '0.95'))
    CHECKS.append({'fig2_E3_ratio_within_factor2': float(np.mean([(0.5 <= r <= 2) for r in allr])) if allr else None,
                   'n_cells': len(allr)})
    save(fig, 'Figure_2')


# ----------------------------------------------------------------------------------------------- Figure 3
TRACE = dict(ls=(0, (1, 1.2)), lw=1.1, marker='x', ms=3.0, mew=0.8)


def fig3():
    fig, ax = four(2.3)
    for j, kind in enumerate(('P', 'W', 'S')):
        cells = [c for c in load_e1(kind) if c['mode'] == 'none' and abs(c['f'] - 0.03) < 1e-12]
        for H in (0.55, 0.75, 0.95):
            cc = sorted([c for c in cells if c['H'] == H], key=lambda c: c['D'])
            D = np.array([c['D'] for c in cc])
            lo = np.array([c['gap_Ld']['median'] for c in cc]); up = np.array([c['gap_U']['median'] for c in cc])
            u0 = np.array([c['gap_U0']['median'] for c in cc])
            ax[j].fill_between(D, lo, up, color=light(HC(H), 0.16), lw=0)
            ax[j].plot(D, u0, color=light(HC(H), 0.75), **TRACE)
            ax[j].plot(D, lo, color=HC(H), **OURS)
            ax[j].plot(D, up, color=HC(H), ls=(0, (3, 1.5)), lw=1.0, marker='o', ms=3.2, mfc='white', mew=0.8)
        logD(ax[j]); ax[j].set_yscale('log'); ax[j].set_xlabel('features $D$')
        ax[j].set_title(KIND_NAME[kind], loc='left', fontsize=BASE)
        letter(ax[j], 'abc'[j], dx=-0.3)
        ax[j].axhline(1.0, **GUIDE)
    ax[0].set_ylabel('distance to exact evidence (nats)')
    a = ax[3]
    n_shown = 0
    for kind, mk in (('P', 'o'), ('W', 's'), ('S', 'D')):
        for c in load_e1(kind):
            a.plot(c['rho'], c['width']['median'], ls='none', marker=mk, ms=2.2, color=HC(c['H']), alpha=0.8,
                   mfc=HC(c['H']) if kind != 'S' else 'white', mew=0.6)
            n_shown += 1
    x = np.array([3e-8, 1e6]); a.plot(x, x, **REF)
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel(r'$\rho=\mathrm{tr}\,R/\tau^2$'); a.set_ylabel('bracket width (nats)')
    a.set_xticks([1e-6, 1e-2, 1e2, 1e6]); a.set_yticks([1e-6, 1e-2, 1e2, 1e6])
    a.set_title(r'width against $\rho$', loc='left', fontsize=BASE)
    letter(a, 'd', dx=-0.3)
    CHECKS.append({'fig3_points_in_d': n_shown})
    H_key(fig, ('0.55', '0.75', '0.95'),
          extra=((Line2D([], [], color='0.3', **OURS), r'$\ell-L$'),
                 (Line2D([], [], color='0.3', ls=(0, (3, 1.5)), lw=1.0, marker='o', ms=3.2, mfc='white'), r'$U-\ell$'),
                 (Line2D([], [], color='0.6', **TRACE), r'trace bound $U_0-\ell$')))
    save(fig, 'Figure_3')


# ----------------------------------------------------------------------------------------------- Figure 4
def fig4():
    fig, ax = four(2.25)
    for j, kind in enumerate(('W', 'P')):
        a = ax[j]
        for H in (0.55, 0.75, 0.95):
            p = os.path.join(RES, 'E4_%s_H%.2f.json' % (kind, H))
            if not os.path.exists(p):
                continue
            d = J(os.path.basename(p))
            r = np.array([row['rank'] for row in d['rows']])
            cf = np.array([row['methods']['CF']['gap_Ld']['median'] for row in d['rows']])
            ip = np.array([row['methods']['IP-VFE']['gap_Ld']['median'] for row in d['rows']])
            orc = np.array([row['methods']['ORACLE']['gap_Ld']['median'] for row in d['rows']])
            a.plot(r, orc, color=light(HC(H), 0.45), lw=2.2, zorder=1)
            a.plot(r, cf, color=HC(H), **OURS)
            a.plot(r, ip, color=HC(H), **COMP)
        a.set_xscale('log', base=2); a.set_yscale('log'); a.set_xlabel('rank $D=M$')
        a.set_xticks([8, 32, 128, 512]); a.set_xticklabels(['8', '32', '128', '512'])
        a.set_title(KIND_NAME[kind], loc='left', fontsize=BASE)
        letter(a, 'ab'[j], dx=-0.3)
    ax[0].set_ylabel(r'evidence gap $\ell-L$ (nats)')
    a = ax[2]
    if os.path.exists(os.path.join(RES, 'E5_learn_hurst.json')):
        d = J('E5_learn_hurst.json')
        rk = [16, 64, 256]
        for cell in d['cells']:
            Ht = cell['H_true']
            cf = [max(cell['CF%d' % r]['dH_vs_exact']['median'], 2e-4) for r in rk]
            ip = [max(cell['IP%d' % r]['dH_vs_exact']['median'], 2e-4) for r in rk]
            a.plot(rk, cf, color=HC(Ht), **OURS)
            a.plot(rk, ip, color=HC(Ht), **COMP)
        a.set_xscale('log', base=2); a.set_yscale('log'); a.set_xticks([16, 64, 256]); a.set_xticklabels(['16', '64', '256'])
        a.set_xlabel('rank $D=M$'); a.set_ylabel(r'$\hat H-\hat H_{\rm exact}$')
        a.set_title('learning $H$', loc='left', fontsize=BASE)
        hs = [Line2D([], [], color=HC(h), lw=2.0) for h in (0.6, 0.75, 0.9)]
        a.legend(hs, ['true $H=0.6$', '$0.75$', '$0.9$'], loc='lower left', fontsize=SMALL, handlelength=1.2)
    letter(a, 'c', dx=-0.3)
    a = ax[3]
    if os.path.exists(os.path.join(RES, 'E6_scale.json')):
        d = J('E6_scale.json')
        rows = [r for r in d['rows'] if r['D'] == 258]
        m = np.array([r['m'] for r in rows])
        tt = np.array([r['time_moments_cum'] + r['time_proj_cum'] + r['time_accum_cum'] + r['time_pass2'] for r in rows])
        a.plot(m, tt, color=HC(0.75), **OURS)
        a.plot(m, tt[-1] * m / m[-1], **GUIDE)
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('observations $m$'); a.set_ylabel('time (s)')
        a.set_title('scale, $D=258$', loc='left', fontsize=BASE)
    letter(a, 'd', dx=-0.3)
    H_key(fig, ('0.55', '0.75', '0.95'),
          extra=((Line2D([], [], color='0.3', **OURS), 'CovFourier'), (Line2D([], [], color='0.3', **COMP), 'inducing points'),
                 (Line2D([], [], color='0.75', lw=2.2), 'best rank $D$')))
    save(fig, 'Figure_4')


# ----------------------------------------------------------------------------------------------- Figure 5
def load_e8():
    """E8 results by H, merged from the sequential run and the per-H runs."""
    by = {}
    for fn in sorted(os.listdir(RES)):
        if fn.startswith('E8_real_boreholes') and fn.endswith('.json'):
            by.update(J(fn).get('by_H', {}))
    return {float(k): v for k, v in by.items()}


def fig5():
    fig, ax = plt.subplots(1, 3, figsize=(W_ONE, 2.3), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.06, rect=(0, 0, 1, 0.9))
    c_r, c_b = HC(0.9), HC(0.55)
    if os.path.exists(os.path.join(RES, 'E7c_per_basin_centre_train.json')):
        run = J('E7c_per_basin_centre_train.json')
    else:
        run = J('E7_real_rivers.json')['runs']['main']
    rows = [r for r in run['rows'] if 'Ld_train_gridmax' in r]
    H = np.array([r['H'] for r in rows]); r05 = [r for r in rows if abs(r['H'] - 0.5) < 1e-9][0]
    base = r05['exact_train_gridmax']
    ex = np.array([r['exact_train_gridmax'] for r in rows]) - base
    lo = np.array([r['Ld_train_gridmax'] for r in rows]) - base
    up = np.array([r['U_train_gridmax'] for r in rows]) - base
    a = ax[0]
    a.fill_between(H, lo, up, color=light(c_r, 0.3), lw=0, zorder=2)
    a.plot(H, ex, **REF)
    a.plot(H, lo, color=c_r, lw=0.8, zorder=3); a.plot(H, up, color=c_r, lw=0.8, zorder=3)
    a.set_xlabel('Hurst index $H$'); a.set_ylabel('relative log likelihood (nats)')
    a.set_title('river storage, 10 basins', loc='left', fontsize=BASE)
    letter(a, 'a', dx=-0.3)
    a = ax[1]
    per = run.get('per_basin', {})
    for i, (g, v) in enumerate(sorted(per.items())):
        lo_, hi_ = v['enclosure']
        a.plot([lo_, hi_], [i, i], color=c_r, lw=3.0, solid_capstyle='butt')
        a.plot(v['H_hat_exact'], i, marker='o', color='0.15', ms=3.2, ls='none')
    a.axvline(0.5, **GUIDE)
    a.set_yticks(range(len(per))); a.set_yticklabels(sorted(per), fontsize=TICK - 1)
    a.set_xlim(0.47, 1.0); a.set_xlabel('Hurst index $H$'); a.set_title('enclosure per basin', loc='left', fontsize=BASE)
    letter(a, 'b', dx=-0.45)
    by = load_e8()
    if by:
        Hs = sorted(by); rec = [by[h] for h in Hs]
        base = rec[0]['W_train']
        W_ = np.array([r['W_train'] for r in rec]) - base
        lo = np.array([r['Ld_train_max'] for r in rec]) - base
        up = np.array([r['U_train_max'] for r in rec]) - base
        a = ax[2]
        a.fill_between(Hs, lo, up, color=light(c_b, 0.3), lw=0, zorder=2)
        a.plot(Hs, lo, color=c_b, lw=0.8, marker='o', ms=2.4, zorder=3); a.plot(Hs, up, color=c_b, lw=0.8, marker='o', ms=2.4, mfc='white', zorder=3)
        a.plot(Hs, W_, color='0.25', ls=(0, (1, 1.4)), lw=1.0, zorder=4)
        a.set_xlabel('Hurst index $H$'); a.set_title('boreholes, 345 profiles', loc='left', fontsize=BASE)
        a.set_ylabel('relative log likelihood (nats)')
        letter(a, 'c', dx=-0.3)
    hs = [Line2D([], [], color='0.62', lw=2.6), mpl.patches.Patch(color='0.8'),
          Line2D([], [], marker='o', color='0.15', ls='none', ms=3.2), Line2D([], [], color='0.25', ls=(0, (1, 1.4)), lw=1.0)]
    fig.legend(hs, ['exact', 'bracket $[L^\\ast,U^\\ast]$', 'exact estimate', 'working model'], loc='upper right',
               bbox_to_anchor=(0.995, 0.995), ncol=4, fontsize=SMALL + 0.4)
    save(fig, 'Figure_5')


if __name__ == '__main__':
    which = sys.argv[1:] or ['fig1', 'fig2', 'fig3', 'fig4', 'fig5']
    for w in which:
        globals()[w]()
    with open(os.path.join(OUT, 'figure_checks.json'), 'w') as fh:
        json.dump(CHECKS, fh, indent=1)
    print(json.dumps(CHECKS, indent=0)[:2000])
