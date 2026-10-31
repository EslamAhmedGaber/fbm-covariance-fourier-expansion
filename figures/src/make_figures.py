"""Figures 2-8 of the paper (reads results/ only; writes figures/).

Run from the repository root or from this folder:   python figures/src/make_figures.py
Each figure is written as PDF + SVG (vector) and PNG (300 dpi); the plotted numbers are written to
figures/data/<figure>.json.  Figure 1 (method diagram) is built by build_fig1.py.
Typography: if LaTeX (latex, dvipng) is on PATH, all text is typeset in Latin Modern, as in the manuscript;
otherwise matplotlib's Computer Modern mathtext is used.  Colours: blue is always CovFourier, the covariance-orthonormal Fourier expansion.
"""
import json
import math
import os
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.dirname(HERE)
ROOT = os.path.dirname(FIG)
RES = os.path.join(ROOT, 'results')
DATA = os.path.join(FIG, 'data')
os.makedirs(DATA, exist_ok=True)

BASE, SMALL, TINY = 8.5, 7.5, 7
C_OURS = '#1f5fa8'          # CovFourier (this work), every figure
C_OURS_LIGHT = '#8fb3dd'    # same, variant (fixed correction switch, cosine-only ablation, bands)
C_QUAD = '#d17a00'          # triangle Gauss-Jacobi quadrature
C_GREY = '#6b6b6b'          # cell averages + Toeplitz; Nystrom; exact GP (straightforward build)
C_MC = '#8c510a'            # Monte Carlo
HCOL = {0.5: '#000000', 0.55: '#2b83ba', 0.65: '#5aae61', 0.75: '#7b3294', 0.85: '#d6604d', 0.95: '#b8860b'}
PRETTY = {'1[0,0.3T)': 'indicator, jump $0.3T$', '1[0,0.5T)': 'indicator, jump $0.5T$', '1[0,0.71T)': 'indicator, jump $0.71T$',
          'exp(t)': '$e^{t}$', 'exp(-5t)': '$e^{-5t}$', 't': '$t$', 't2': '$t^2$'}
USETEX = False


def style():
    global USETEX
    # LaTeX typography when latex and dvipng are available; FIGURES_USETEX=0 forces matplotlib's own mathtext
    USETEX = (os.environ.get('FIGURES_USETEX', '1') != '0' and shutil.which('latex') is not None
              and shutil.which('dvipng') is not None)
    rc = {'font.size': BASE, 'axes.titlesize': BASE, 'axes.labelsize': BASE, 'legend.fontsize': TINY,
          'xtick.labelsize': TINY, 'ytick.labelsize': TINY, 'axes.spines.top': False, 'axes.spines.right': False,
          'axes.titlelocation': 'left', 'axes.titleweight': 'normal', 'pdf.fonttype': 42, 'svg.fonttype': 'none',
          'savefig.dpi': 300, 'figure.dpi': 150, 'lines.linewidth': 1.2, 'lines.markersize': 3.5, 'legend.frameon': False,
          'font.family': 'serif'}
    if USETEX:
        rc.update({'text.usetex': True,
                   'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath,amssymb}'})
    else:
        rc.update({'mathtext.fontset': 'cm', 'font.serif': ['cmr10', 'DejaVu Serif'], 'axes.formatter.use_mathtext': True,
                   'axes.unicode_minus': False})
    plt.rcParams.update(rc)


def pct(x):
    return (r'%.1f\%%' if USETEX else r'%.1f%%') % x


def letter(ax, s):
    ax.text(-0.15, 1.07, r'\textbf{%s}' % s if USETEX else s, transform=ax.transAxes, fontsize=BASE + 1,
            fontweight='bold', va='bottom')


def save(fig, name, data):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), bbox_inches='tight')
    with open(os.path.join(DATA, f'{name}.json'), 'w') as fh:
        json.dump(data, fh, indent=1)
    plt.close(fig)


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def have(*names):
    return all(os.path.exists(os.path.join(RES, n)) for n in names)


def _front(rows):
    pts = sorted((r['total_s'], r['rel_fro']) for r in rows)
    fr, best = [], np.inf
    for t, e in pts:
        if e < best:
            fr.append((t, e)); best = e
    return np.array(fr)


def c_H(H):
    return math.gamma(2 * H + 1) * math.sin(math.pi * H)


def fig2():
    A3 = load('A3_weighted_inequality.json'); A5 = load('A5_two_sided_rates_inputs.json')
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={'wspace': 0.32})
    rows = [r for r in A3['rows'] if r['N'] == 2048]
    Hs = [r['H'] for r in rows]
    ax = axs[0]
    ax.axhspan(0.5, 1.0, color='#eef3fa', zorder=0)
    ax.plot(Hs, [r['max_eig_weighted'] for r in rows], 'o-', color=C_OURS, label='largest eigenvalue')
    ax.plot(Hs, [r['min_eig_weighted'] for r in rows], 's-', color=C_QUAD, label='smallest eigenvalue')
    ax.axhline(0.5, color='k', lw=0.7, ls='--'); ax.axhline(1.0, color='k', lw=0.7, ls='--')
    ax.text(0.505, 0.515, 'proved lower bound $1/2$', fontsize=TINY, va='bottom')
    ax.text(0.99, 0.975, 'proved upper bound $1$', fontsize=TINY, va='top', ha='right')
    ax.set_xlabel('Hurst index $H$'); ax.set_ylabel('eigenvalue of $W^{-1/2}G_+W^{-1/2}$')
    ax.set_ylim(0.42, 1.06); ax.set_title('Weighted cosine Gram stays in $[\\tfrac12,1]$')
    ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.62), fontsize=TINY)
    letter(ax, 'a')
    ax = axs[1]
    names = sorted(set(c['f'] for c in A5['cases']))
    mk = ['o', 's', '^', 'v', 'D', 'P', 'X', '*']
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(names)))
    for k, nm in enumerate(names):
        cs = [c for c in A5['cases'] if c['f'] == nm]
        ax.plot([c['J'] for c in cs], [c['ratio_lo'] for c in cs], mk[k % len(mk)], ms=2.6, color=cols[k], alpha=0.8, label=PRETTY.get(nm, nm), ls='none')
    ax.axhline(0.5, color='k', lw=0.7, ls='--'); ax.axhline(1.0, color='k', lw=0.7, ls='--')
    ax.set_xscale('log', base=2); ax.set_ylim(0.42, 1.06)
    ax.set_xlabel('retained cosines $J$'); ax.set_ylabel('$E_J(f)^2$ / weighted coefficient tail')
    ax.set_title('Optimal error equals the weighted tail within factor 2')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.14), ncol=2, fontsize=TINY, handletextpad=0.2, columnspacing=0.6)
    letter(ax, 'b')
    save(fig, 'fig2', {'A3_N2048': rows, 'A5_min_ratio': A5['min_ratio_lo'], 'A5_max_ratio': A5['max_ratio_hi'],
                       'n_cases': len(A5['cases'])})


def fig3():
    B1 = load('B1_augmented_conditioning.json'); S1 = load('S1_scaling_study.json')
    fig, axs = plt.subplots(1, 3, figsize=(7.3, 2.55), gridspec_kw={'wspace': 0.5})
    out = {'B1': [], 'S1': [], 'ablation': {}}
    ax = axs[0]
    for H in (0.55, 0.75, 0.95):
        rr = [r for r in B1['rows'] if r['H'] == H]
        D = np.array([r['D'] for r in rr]); c = np.array([r['cond'] for r in rr])
        ax.loglog(D, c, 'o-', color=HCOL[H], ms=2.6, label=f'$H={H}$')
        ax.loglog(D, c[0] * (D / D[0]) ** (2 * H + 2), ':', color=HCOL[H], lw=0.8)
        out['B1'] += [{'H': H, 'D': int(d), 'cond': float(v)} for d, v in zip(D, c)]
    ax.axhline(1 / np.finfo(float).eps, color='k', lw=0.7, ls='--')
    ax.text(18, 1 / np.finfo(float).eps * 1.8, '$1/\\varepsilon$ (double)', fontsize=TINY)
    ax.set_xlabel('dimension $D$'); ax.set_ylabel('condition number of $A_D$')
    ax.set_title('Raw nested Gram: $\\kappa\\propto D^{2H+2}$')
    ax.legend(loc='lower right', fontsize=TINY); letter(ax, 'a')
    ax = axs[1]
    H = 0.75
    inputs = [('1[0,0.3)', 'indicator', -H), ('t^-0.25', '$t^{-1/4}$', -(H - 0.25)),
              ('|t-1/2|', '$|t-\\tfrac12|$', -(1 + H)), ('exp(t)', '$e^t$', None)]
    cols = ['#2b83ba', '#5aae61', '#d6604d', '#7b3294']
    for (key, lab, slope), col in zip(inputs, cols):
        rr = sorted([r for r in S1['rows'] if r['H'] == H and r['input'] == key], key=lambda r: r['n'])
        D = np.array([r['D'] for r in rr]); e = np.array([r['rel_rms_enriched'] for r in rr])
        ax.loglog(D, e, 'o-', color=col, ms=2.2, label=lab)
        if slope is not None:
            ax.loglog(D, e[1] * (D / D[1]) ** slope, ':', color='k', lw=0.6)
        out['S1'] += [{'H': H, 'input': key, 'D': int(d), 'rel_rms': float(v), 'solver': r['solver']} for d, v, r in zip(D, e, rr)]
    ax.set_xlabel('dimension $D$'); ax.set_ylabel('relative error $e_D/\\|f\\|_H$')
    ax.set_title('Predicted orders to $D=32\\,769$')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.12), fontsize=TINY, handlelength=1.4); letter(ax, 'b')
    ax = axs[2]
    for key in ('exp(t)', '1[0,0.3)', 'sin(140 pi t)'):
        rr = sorted([r for r in S1['rows'] if r['H'] == H and r['input'] == key], key=lambda r: r['n'])
        n_ = np.array([r['n'] for r in rr]); D = n_ + 2
        en = np.array([r['rel_rms_enriched'] for r in rr]); co = np.array([r['rel_rms_cosine_only'] for r in rr])
        out['ablation'][key] = {'D_enriched': D.tolist(), 'enriched': en.tolist(), 'n_cosine_only': n_.tolist(), 'cosine_only': co.tolist(),
                                'ratio_cosine_over_enriched': (co / en).tolist()}
        if key == 'sin(140 pi t)':
            continue
        col = C_OURS if key == 'exp(t)' else C_GREY
        ax.loglog(D, en, 'o-', color=col, ms=2.2, lw=1.2)
        ax.loglog(n_, co, '--', color=col, lw=1.0)
    ex = out['ablation']['exp(t)']; i = ex['D_enriched'].index(2049)
    ratio = ex['cosine_only'][i] / ex['enriched'][i]
    ax.annotate('', xy=(2049, ex['enriched'][i] * 1.5), xytext=(2049, ex['cosine_only'][i] / 1.5),
                arrowprops=dict(arrowstyle='<->', color='k', lw=0.7))
    e10 = int(math.floor(math.log10(ratio)))
    ax.text(2600, math.sqrt(ex['enriched'][i] * ex['cosine_only'][i]), '$%.0f\\times10^{%d}$' % (ratio / 10 ** e10, e10), fontsize=TINY, va='center')
    out['ablation']['exp_ratio_at_D2049'] = ratio
    ax.text(1300, 3e-6, '$e^t$: cosines only,\n$\\propto D^{-1-H}$', color=C_OURS, fontsize=TINY - 0.5, va='bottom')
    ax.text(66, 1.2e-16, '$e^t$: enriched, $\\propto D^{-3-H}$', color=C_OURS, fontsize=TINY - 0.5, va='bottom')
    ax.text(66, 5e-2, 'indicator: no gain', color='0.3', fontsize=TINY - 0.5, va='bottom')
    ax.set_ylim(5e-17, 3)
    ax.set_xlabel('dimension'); ax.set_ylabel('relative error ($H=0.75$)')
    ax.set_title('Polynomial directions raise the rate')
    letter(ax, 'c')
    save(fig, 'fig3', out)


def fig5():
    RR = load('A4_richardson.json')
    arr = np.load(os.path.join(RES, 'A4_eigs_J4096.npz'))
    fig, axs = plt.subplots(1, 3, figsize=(7.3, 2.55), gridspec_kw={'wspace': 0.5})
    out = {}
    ax = axs[0]
    n = np.arange(1, 601); ex = 1 / ((n - 0.5) * np.pi) ** 2
    eg = np.abs(arr['mu_0p5'][:600] - ex) / ex; en = np.abs(arr['nys_0p5_4096'][:600] - ex) / ex
    ax.loglog(n, eg, '-', color=C_OURS, label='Galerkin')
    ax.loglog(n, en, '-', color=C_GREY, label='Nystr\u00f6m')
    k = 99; ratio = en[k] / eg[k]
    ax.annotate('', xy=(100, eg[k] * 1.8), xytext=(100, en[k] / 1.8), arrowprops=dict(arrowstyle='<->', color='k', lw=0.7))
    e10 = int(math.floor(math.log10(ratio)))
    ax.text(118, 1.2e-5, '$%.1f\\times10^{%d}$\nsmaller' % (ratio / 10 ** e10, e10), fontsize=TINY, va='center', ha='left')
    ax.text(0.97, 0.22, '$J=N=4096$', transform=ax.transAxes, fontsize=TINY - 0.5, ha='right', va='bottom', color='0.3')
    ax.set_xlabel('index $n$'); ax.set_ylabel('relative eigenvalue error')
    ax.legend(loc='lower right', fontsize=TINY)
    ax.set_title('Exact spectrum, $H=\\tfrac12$')
    letter(ax, 'a')
    out['bm_rel_err'] = {'galerkin': eg[[0, 9, 49, 99, 299, 599]].tolist(), 'nystrom': en[[0, 9, 49, 99, 299, 599]].tolist(),
                         'n': [1, 10, 50, 100, 300, 600], 'ratio_n100': float(ratio)}
    ax = axs[1]
    out['n_rho_richardson'] = RR
    for H in ('0.55', '0.65', '0.75', '0.85', '0.95'):
        d = RR[H]['n_rho_R']
        ks = [kk for kk in sorted(d, key=lambda s: int(s)) if int(kk) <= 200]
        ax.semilogx([int(kk) for kk in ks], [d[kk] for kk in ks], 'o-', color=HCOL[float(H)], ms=2.4)
        ax.text(int(ks[-1]) * 1.12, d[ks[-1]], f'$H={H}$', color=HCOL[float(H)], fontsize=TINY, va='bottom' if H == '0.55' else 'center')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('index $n$'); ax.set_ylabel('$n\\,(\\nu_n-\\hat\\nu_n)$ (observed)')
    ax.set_xlim(7, 700)
    ax.set_title('Two-term law: remainder $\\approx\\kappa_H/n$')
    letter(ax, 'b')
    ax = axs[2]
    H = 0.75
    from scipy.special import zeta
    cH = c_H(H)
    m = arr['mu_0p75'][:1000]; n = np.arange(1, 1001)
    scale = cH * (n * np.pi) ** (-1 - 2 * H)
    ax.loglog(n, m / scale, '-', color=C_OURS, label='computed $\\mu_n$')
    ax.loglog(n[1:], (n[1:] / (n[1:] - 1)) ** (1 + 2 * H), '--', color='k', lw=0.8, label='upper bound, $n\\ge2$')
    up1 = (1 / (2 * H + 1) - 0.5 * cH * np.pi ** (-1 - 2 * H) * (zeta(2 * H + 1) - 1)) / scale[0]
    ax.plot([1], [up1], marker='v', color='k', ms=4, ls='none', label='upper bound, $n=1$')
    ax.axhline(0.5, color='k', lw=0.8, ls=':', label='lower bound')
    ax.set_ylim(0.35, 12)
    ax.set_xlabel('index $n$'); ax.set_ylabel('$\\lambda_n\\,/\\,c_H(n\\pi)^{-1-2H}$')
    ax.set_title('Explicit bounds, every $n$ ($H=0.75$)')
    ax.legend(loc='upper right', fontsize=TINY - 0.5, handlelength=1.6, title='proved bounds' if False else None); letter(ax, 'c')
    out['normalized_mu_H075_first10'] = (m / scale)[:10].tolist(); out['upper_bound_n1_normalized'] = float(up1)
    save(fig, 'fig5', out)


def fig6():
    A = load('A5_weighted_kl.json')
    MC = load('A5_mc_reference.json') if have('A5_mc_reference.json') else None
    fig, axs = plt.subplots(1, 3, figsize=(7.3, 2.6), gridspec_kw={'wspace': 0.5})
    out = {'bracket': {}, 'mc': {}, 'weyl': {}, 'mc_time_to_match': {}}
    for j, (w, ttl) in enumerate((('inv_t', '$w=1/t$ (no closed form)'), ('one', '$w=1$'))):
        ax = axs[j]
        for H in (0.55, 0.75, 0.95):
            cs = sorted([c for c in A['cases'] if float(c['H']) == H and c['weight'] == w], key=lambda c: c['J'])
            tb = [c['setup_s'] + c['samples_s'] + c['project_s'] + c['eig_s'] for c in cs]
            wb = [c['laplace_bracket']['1.0'][1] - c['laplace_bracket']['1.0'][0] for c in cs]
            ax.loglog(tb, wb, 's-', color=HCOL[H], ms=3.2, lw=1.2)
            out['bracket'][f'{H}_{w}'] = {'J': [c['J'] for c in cs], 'seconds': tb, 'width_L1': wb}
            if MC is not None:
                rr = sorted([r for r in MC['rows'] if r['H'] == H and r['weight'] == w], key=lambda r: r['M'])
                tm = [r['seconds'] for r in rr]; cw = [r['ci95_width'] for r in rr]
                ax.loglog(tm, cw, 'o:', color=HCOL[H], mfc='white', ms=3.2, lw=1.0)
                xs = np.array([tm[-1], 400.0])
                ax.loglog(xs, cw[-1] * (xs / tm[-1]) ** -0.5, '-', color=HCOL[H], lw=0.5, alpha=0.6)
                need = tm[-1] * (cw[-1] / wb[-1]) ** 2
                out['mc'][f'{H}_{w}'] = {'M': [r['M'] for r in rr], 'seconds': tm, 'ci95_width': cw}
                out['mc_time_to_match'][f'{H}_{w}'] = need
        ax.set_xlim(0.4, 400); ax.set_ylim(top=1.0)
        ax.set_xlabel('wall time [s]'); ax.set_ylabel('interval width for $L_w(1)$')
        ax.set_title(ttl)
        letter(ax, 'ab'[j])
        if MC is not None:
            need = out['mc_time_to_match'][f'0.95_{w}']; e10 = int(math.floor(math.log10(need)))
            ax.text(0.97, 0.97, 'MC at the $H=0.95$ width:\n$\\approx10^{%d}$ s (sampling only)' % e10, transform=ax.transAxes, ha='right', va='top', fontsize=TINY - 0.5)
    hl = [Line2D([], [], color='k', marker='s', ms=3.2, lw=1.2, label='bracket'),
          Line2D([], [], color='k', marker='o', mfc='white', ms=3.2, ls=':', lw=1.0, label='MC, 95\\% CI' if USETEX else 'MC, 95% CI')]
    axs[0].legend(handles=hl, loc='lower left', fontsize=TINY - 0.5, handlelength=1.8)
    hh = [Line2D([], [], color=HCOL[H], lw=1.2, label=f'$H={H}$') for H in (0.55, 0.75, 0.95)]
    axs[1].legend(handles=hh, loc='center left', fontsize=TINY - 0.5, handlelength=1.4)
    ax = axs[2]
    for H in (0.55, 0.75, 0.95):
        kap = (1 + 2 * H) / (2 * H); cH = c_H(H)
        for J, alpha in ((256, 0.35), (1024, 1.0)):
            c = [x for x in A['cases'] if float(x['H']) == H and x['weight'] == 'inv_t' and x['J'] == J][0]
            nn = np.arange(1, 51); mu = np.array(c['mu_first50'])
            r = mu / (cH * (nn * np.pi / kap) ** (-1 - 2 * H))
            ks = [kk for kk in (100, 200, 400) if kk <= J]
            nn = np.r_[nn, ks]; rr = np.r_[r, [c['weyl_ratio'][str(kk)] for kk in ks]]
            ax.semilogx(nn, rr, 'o-', color=HCOL[H], alpha=alpha, ms=2.0, lw=1.0, label=(f'$H={H}$' if J == 1024 else None))
            out['weyl'][f'{H}_{J}'] = {'n': nn.tolist(), 'ratio': rr.tolist()}
    ax.axhline(1, color='k', lw=0.6, ls=':')
    ax.set_xlabel('eigenvalue index $n$'); ax.set_ylabel('$\\mu_n\\,/\\,c_H(n\\pi/\\kappa_w)^{-1-2H}$')
    ax.set_ylim(0.6, 1.6)
    ax.set_title('Weighted spectrum, $w=1/t$')
    ax.legend(fontsize=TINY - 0.5, loc='lower left'); letter(ax, 'c')
    ax.text(0.98, 0.97, 'faint: $J=256$\nfull: $J=1024$', transform=ax.transAxes, ha='right', va='top', fontsize=TINY - 0.5)
    save(fig, 'fig6', out)


def fig7():
    V = load('A2_noisy_reconstruction_v2.json'); S = load('A2_v2_summary.json')
    fig, axs = plt.subplots(1, 3, figsize=(7.3, 2.65), gridspec_kw={'wspace': 0.5})
    import matplotlib.transforms as mtransforms
    ax = axs[0]
    cs = V['cases']['0.75']
    m = np.array([c['m'] for c in cs]); gp = np.array([c['GP']['total_s'] for c in cs])
    cf = np.array([[x for x in c['CF'] if x['n'] == 256][0]['total_s'] for c in cs])
    st = [r for r in S['crossover']['strong_gp_build_H075'] if 'gp_strong_s' in r]
    ms_ = np.array([r['m'] for r in st]); gs_ = np.array([r['gp_strong_s'] for r in st])
    g12 = S['crossover']['strong_gp_build_H075'][-1]['gp_strong_s_excl_targets']
    rt = S['crossover']['cf_alone_retimed_H075_n256']; co = S['cf_only_H075_n256']
    mc = np.r_[m, [r['m'] for r in co]]; tc = np.r_[cf, [r['total_s'] for r in co]]
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.axvspan(380, 12800, color='#eeeeee', lw=0, zorder=0)
    ax.text(17000, 0.03, 'CovFourier\nonly', transform=tr, fontsize=TINY - 0.5, va='bottom', ha='left', color='0.35')
    ax.loglog(m, gp, 's-', color=C_GREY, ms=3, lw=1.0, label='GP, straightforward')
    ax.loglog(ms_, gs_, 'D-', color='k', ms=2.8, lw=1.0, label='GP, strengthened')
    ax.loglog([ms_[-1], 12000], [gs_[-1], g12], '-', color='k', lw=1.0)
    ax.plot([12000], [g12], 'D', color='k', mfc='white', ms=2.8)
    ax.loglog(mc, tc, 'o-', color=C_OURS, ms=3, lw=1.6, label='CovFourier, $n=256$')
    for r in rt:
        ax.plot([r['m']] * len(r['cf_total_s_reps']), r['cf_total_s_reps'], 'o', mfc='none', mec=C_OURS, ms=3.0, mew=0.6)
    ax.axvline(14000, ymax=0.62, color='k', lw=0.6, ls='--')
    ax.set_xlabel('number of noisy sensors $m$'); ax.set_ylabel('time at matched accuracy [s]')
    ax.set_ylim(top=1500)
    ax.set_title('Time at matched accuracy')
    ax.legend(loc='upper left', fontsize=TINY - 0.5); letter(ax, 'a')
    out = {'time': {'m': m.tolist(), 'gp_v2': gp.tolist(), 'cf_n256': cf.tolist(), 'gp_strong_m': ms_.tolist(), 'gp_strong': gs_.tolist(),
                    'gp_strong_m12000_excl_targets': g12, 'cf_retimed': rt, 'cf_only': co}}
    ax = axs[1]
    probes = [load('A2_memory_probe_m%d.json' % mm) for mm in (16000, 32000, 64000, 128000) if have('A2_memory_probe_m%d.json' % mm)]
    xs = np.logspace(np.log10(2000), np.log10(1.45e5), 60)
    ax.loglog(xs, 8 * xs ** 2 / 2 ** 30, '-', color=C_GREY, lw=1.1, label='dense $K$ (calculated)')
    mm_ = np.array([p['m'] for p in probes]); pw = np.array([p['peak_working_set_GiB'] for p in probes])
    ax.loglog(mm_, pw, 'o-', color=C_OURS, ms=3.4, lw=1.5, label='CovFourier (measured)')
    ax.axhline(2.0, color='k', lw=0.6, ls='--')
    ax.text(1.5e5, 1.75, '2 GiB budget', fontsize=TINY - 0.5, ha='right', va='top')
    if len(probes) and probes[-1]['m'] == 128000:
        ax.annotate('%.2f GiB measured' % pw[-1], xy=(128000, pw[-1]), xytext=(14000, 0.05), fontsize=TINY - 0.5,
                    arrowprops=dict(arrowstyle='->', lw=0.6, color=C_OURS), color=C_OURS)
        ax.annotate('%.0f GiB (dense $K$)' % (8 * 128000 ** 2 / 2 ** 30), xy=(128000, 8 * 128000 ** 2 / 2 ** 30), xytext=(2300, 250), fontsize=TINY - 0.5,
                    arrowprops=dict(arrowstyle='->', lw=0.6, color=C_GREY), color='0.3', va='center')
    ax.set_ylim(1e-3, 600)
    ax.set_xlabel('number of noisy sensors $m$'); ax.set_ylabel('memory [GiB]')
    ax.set_title('Working memory, $D=258$')
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
    ax.legend(loc='lower right', fontsize=TINY - 0.5); letter(ax, 'b')
    out['memory'] = {'m': mm_.tolist(), 'peak_working_set_GiB': pw.tolist(), 'dense_K_GiB_at_m': [8 * v ** 2 / 2 ** 30 for v in mm_.tolist()]}
    ax = axs[2]
    for n, col, lw in ((128, '#9ecae1', 1.0), (256, C_OURS, 1.6), (512, '#08306b', 1.0)):
        e = [[x for x in c['CF'] if x['n'] == n][0]['smooth_mean_err_sd_max'] for c in cs]
        ax.loglog(m, e, 'o-', color=col, ms=3, lw=lw, label=f'smooth, $n={n}$')
        out[f'smooth_n{n}'] = e
    e = [[x for x in c['CF'] if x['n'] == 512][0]['indicator_mean_err_sd_max'] for c in cs]
    ax.loglog(m, e, 'v-', color='#d6604d', ms=3, lw=1.0, label='indicator, $n=512$')
    out['indicator_n512'] = e
    fl = S['gp_rounding_floor_test_H075_m2000']['one_ulp_perturbation_mean_sd']
    ax.axhspan(fl / 3, fl * 3, color='0.85', lw=0)
    ax.text(m[0], fl * 3.6, 'GP, 1-ulp change of $K$', fontsize=TINY - 0.5, va='bottom')
    ax.axhline(1e-6, color='k', lw=0.6, ls=':'); ax.text(m[-1], 1.4e-6, 'tolerance', fontsize=TINY - 0.5, ha='right', va='bottom')
    ax.set_xlabel('number of noisy sensors $m$'); ax.set_ylabel('$|$mean error$|$ / posterior s.d.' if USETEX else '|mean error| / posterior s.d.')
    tk = [v for v in m if v != 10000]
    ax.set_xticks(tk); ax.set_xticklabels([str(int(v)) for v in tk], fontsize=TINY - 0.5); ax.minorticks_off()
    lo = min(min(out['smooth_n256']), min(out['smooth_n512']))
    ax.set_ylim(10 ** np.floor(np.log10(lo) - 0.3), 300)
    ax.set_title('Agreement with the exact GP')
    ax.legend(loc='upper right', fontsize=TINY - 0.5, ncol=1); letter(ax, 'c')
    out['gp_floor'] = fl
    save(fig, 'fig7', out)


def fig8():
    import matplotlib.transforms as mtransforms
    fig, axs = plt.subplots(1, 3, figsize=(7.3, 2.65), gridspec_kw={'wspace': 0.45, 'width_ratios': [1.3, 1.0, 1.0]})
    out = {}
    ax = axs[0]
    R = load('A2_reconstruction.json'); ex = R['example']
    t = np.array(ex['t']); mu = np.array(ex['mean_cf']); sd = np.array(ex['sd_cf'])
    ax.fill_between(t, mu - 1.959964 * sd, mu + 1.959964 * sd, color=C_OURS_LIGHT, alpha=0.6, lw=0, label='95\\% band' if USETEX else '95% band')
    ax.plot(ex['latent_fine_t'], ex['latent_fine'], color='k', lw=0.55, label='latent path')
    ax.plot(t, mu, color=C_OURS, lw=1.1, label='posterior mean')
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(ex['centres'], np.full(len(ex['centres']), 0.025), '|', color=C_GREY, ms=3.0, mew=0.4, transform=tr)
    ax.text(0.5, 0.075, 'sensor centres', transform=ax.transAxes, fontsize=TINY - 0.5, ha='center', color='0.35')
    ax.set_xlim(0, 1)
    ax.set_xlabel('$t$'); ax.set_ylabel('$B_H(t)$')
    ax.set_title('Reconstruction from $m=%d$ sensors' % R['m'])
    ax.legend(loc='upper left', fontsize=TINY - 0.5, handlelength=1.5)
    ax.text(0.98, 0.17, 'band coverage %s\n(%d paths $\\times$ %d times)' % (pct(100 * R['coverage_cf']), R['R'], len(t)), transform=ax.transAxes, ha='right', va='bottom', fontsize=TINY - 0.5)
    letter(ax, 'a')
    out['reconstruction'] = {k: R[k] for k in R if k != 'example'}
    ax = axs[1]
    Dg = load('A2_residual_diagnostic.json')
    mk = {0.003: 'o', 0.01: 's', 0.03: '^'}
    for r in Dg['rows']:
        y = max(r['rms_projected_smooth_max'], r['rms_projected_indicator_max'])
        ok = r['n'] >= 64
        ax.loglog([r['rho']], [y], mk[r['tau']], color=C_OURS, mfc=C_OURS if ok else 'white', ms=3.6, mew=0.8)
    rr = np.logspace(-9, 5, 200)
    ax.loglog(rr, rr * np.sqrt((1 + rr) / 2), 'k-', lw=0.9, label='bound, Proposition 4')
    hl = [Line2D([], [], color='k', lw=0.9, label='bound, Prop.~4' if USETEX else 'bound, Prop. 4')]
    hl += [Line2D([], [], marker=mk[tv], color=C_OURS, ls='none', ms=3.6, label=f'$\\tau={tv}$') for tv in (0.003, 0.01, 0.03)]
    ax.legend(handles=hl, loc='upper left', fontsize=TINY - 0.5, handlelength=1.4)
    ax.set_xlim(1e-9, 1e5); ax.set_ylim(1e-11, 1e10)
    ax.set_xlabel('$\\rho=\\sum_i e_D(F_i)^2/\\tau^2$'); ax.set_ylabel('RMS mean deviation / $s_Q$')
    ax.set_title('Residual diagnostic')
    letter(ax, 'b')
    out['diagnostic'] = Dg['rows']
    ax = axs[2]
    if have('A2_incremental_reuse.json'):
        I = load('A2_incremental_reuse.json')
        cats = [('initial %d sensors' % I['m0'], I['cf_initial_s'], I['gp_initial_s']),
                ('add %d sensors' % I['added_sensors'], I['cf_add_sensors_s'], I['gp_add_sensors_s']),
                ('add %d targets' % I['targets'], I['cf_targets_s'], I['gp_targets_s'])]
        y = np.arange(len(cats))[::-1]
        ax.barh(y + 0.18, [c[1] for c in cats], height=0.34, color=C_OURS, label='CovFourier (basis reused)')
        ax.barh(y - 0.18, [c[2] for c in cats], height=0.34, color=C_GREY, label='exact GP (block Cholesky)')
        for yy, c in zip(y, cats):
            ax.text(c[1] * 1.15, yy + 0.18, '%.2g s' % c[1], va='center', fontsize=TINY - 0.5)
            ax.text(c[2] * 1.15, yy - 0.18, '%.2g s' % c[2], va='center', fontsize=TINY - 0.5)
        ax.set_yticks([])
        ax.set_xscale('log'); xl = min(min(c[1], c[2]) for c in cats) / 3; ax.set_xlim(left=xl, right=max(max(c[1], c[2]) for c in cats) * 12)
        for yy, c in zip(y, cats):
            ax.text(xl * 1.15, yy + 0.42, c[0], va='bottom', ha='left', fontsize=TINY - 0.5)
        ax.set_ylim(-0.5, len(cats) - 0.15)
        ax.set_xlabel('wall time [s]')
        # direct labels inside the first pair of bars (the caption defines both methods)
        ax.text(xl * 1.3, y[0] + 0.18, 'CovFourier', va='center', ha='left', fontsize=TINY - 0.5, color='white')
        ax.text(xl * 1.3, y[0] - 0.18, 'exact GP', va='center', ha='left', fontsize=TINY - 0.5, color='white')
        out['incremental'] = I
    ax.set_title('Reuse at $m=%s$' % ('8000' if not have('A2_incremental_reuse.json') else str(load('A2_incremental_reuse.json')['m0'])))
    letter(ax, 'c')
    save(fig, 'fig8', out)


def fig4():
    R = load('A1_memory_responses_v2.json')['results']
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.5), sharex=True, sharey=True, gridspec_kw={'wspace': 0.08, 'hspace': 0.24})
    out = {}
    for i, fam in enumerate(('benign', 'stiff')):
        for j, H in enumerate(('0.55', '0.75', '0.95')):
            ax = axs[i, j]; key = f'{fam}_{H}'; rec = R[key]
            series = [('CF_fixed_switch', rec['CF_fixed_switch'], C_OURS_LIGHT, 'o', 'CovFourier, fixed correction switch', 0.9, '--'),
                      ('TGJ', rec['TGJ'], C_QUAD, 's', 'triangle Gauss--Jacobi' if USETEX else 'triangle Gauss-Jacobi', 1.0, '-'),
                      ('CT', rec['CT'], C_GREY, '^', 'cells + Toeplitz', 1.0, '-'),
                      ('CF', rec['CF'], C_OURS, 'o', 'CovFourier, a posteriori endpoint rule', 1.6, '-')]
            out[key] = {}
            for nm, rows, col, mk_, lab, lw, ls in series:
                fr = _front(rows)
                ax.loglog(fr[:, 0], fr[:, 1], mk_, ls=ls, color=col, ms=2.6, lw=lw, label=lab)
                out[key][nm] = fr.tolist()
            ax.axhline(1e-9, color='k', lw=0.5, ls=':')
            ax.set_title('%s   %s, $H=%s$' % ('abcdef'[3 * i + j], 'smooth kernels' if fam == 'benign' else 'stiff modes', H), fontsize=SMALL, loc='left')
            if i == 1:
                ax.set_xlabel('total time [s]')
            if j == 0:
                ax.set_ylabel('rel. Frobenius error of $\\Sigma$')
    hs, ls_ = axs[0, 0].get_legend_handles_labels()
    order = [3, 0, 1, 2]
    fig.legend([hs[k] for k in order], [ls_[k] for k in order], loc='lower center', ncol=4, fontsize=TINY,
               bbox_to_anchor=(0.5, -0.035), handlelength=2.0, columnspacing=1.4)
    axs[1, 1].text(0.97, 0.97, 'lower-left = better', transform=axs[1, 1].transAxes, ha='right', va='top', fontsize=TINY)
    axs[1, 2].text(0.03, 0.03, 'dotted: $10^{-9}$', transform=axs[1, 2].transAxes, ha='left', va='bottom', fontsize=TINY)
    save(fig, 'fig4', out)


if __name__ == '__main__':
    style()
    chosen = [a for a in sys.argv[1:] if a.startswith('fig')] or ['fig2', 'fig3', 'fig4', 'fig5', 'fig6', 'fig7', 'fig8']
    for name in chosen:
        globals()[name]()
    print('figures written to', FIG, '| LaTeX typography:', USETEX)
# figures renumbered for the revised paper (CovFourier revision)
