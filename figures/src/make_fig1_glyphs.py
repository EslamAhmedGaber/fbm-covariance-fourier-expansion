"""Figure 1: the general procedure, abstraction paired with realisation.

Layout (fixed in words before drawing):
  two rows x four columns at final width 16.6 cm;
  top row    = basis, fixed before any input (depends on H, T, D only):
               covariance norm -> raw nested sequence phi -> Gram A_D = C_D C_D^T -> orthonormal v = C_D^{-1} phi;
  bottom row = signal, one pass per input:
               admissible f -> moments b_D(f) -> fixed coefficient rule gamma = C_D^{-1} b_D(f), e_D^2
               -> Wiener integral I_H(f) ~ sum_j gamma_j Z_j;
  three vertical arrows: phi feeds the moments, C_D feeds the coefficient rule, v_j defines Z_j = I_H(v_j).

Every glyph is computed by the covfourier engine (closed Gram, endpoint-corrected DCT, Cholesky) at H = 0.75, T = 1.
Notation follows the paper (phi_j, A_D = C_D C_D^T, v = C_D^{-1} phi, gamma_j(f), e_D(f), Z_j = I_H(v_j)).
This script writes the glyph PDFs (figures/fig1_parts/) and figures/data/fig1_glyphs.json; the TikZ frame
fig1_method.tex includes them at natural size and is compiled by build_fig1.py.
"""
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
FIGROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(FIGROOT)
sys.path.insert(0, os.path.join(ROOT, 'code'))
from covfourier import gram as GR          # noqa: E402
from covfourier import batch as BT         # noqa: E402

PARTS = os.path.join(FIGROOT, 'fig1_parts')
DATA = os.path.join(FIGROOT, 'data')
os.makedirs(PARTS, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

# semantic palette (tikz-paper-figures house style roles): basis = primary, signal = accent
PRIMARY = (62 / 255, 98 / 255, 148 / 255)
PRIMARY_LIGHT = (231 / 255, 238 / 255, 247 / 255)
ACCENT = (176 / 255, 116 / 255, 60 / 255)
ACCENT_LIGHT = (251 / 255, 240 / 255, 226 / 255)
INK = (38 / 255, 42 / 255, 48 / 255)
NEUTRAL = (96 / 255, 102 / 255, 110 / 255)

H, T = 0.75, 1.0
LAM, TAU = 3.0, 0.6          # input: memory kernel f(t) = exp(-LAM (T - t)), truncated input f_tau = f 1_[0,tau)
D = 33                       # 3 + 30 raw functions
D_HEAT = 17
N = 2 ** 14                  # midpoint samples of the Fourier stage
N_SPEC = 2048                # cosine block size for the two-sided band
SEED = 20261029
GLYPH_W_CM, GLYPH_H_CM = 3.45, 1.62


def c_H(H):
    return math.gamma(2 * H + 1) * math.sin(math.pi * H)


def moments_and_norm(F, A, cg):
    """b_D(f) = (<f, phi_j>_H)_{j<D} and ||f||_H^2 for midpoint samples F (q, N): one rule for every input."""
    w = BT._endpoint_weights(8)
    a_g, c, used = BT.endpoint_corrected_dct(F, T, w, 8)
    q = a_g.shape[1]
    pc = np.stack([GR.poly_cos_moments(H, j, N, T) / T ** j for j in (1, 2)])       # (2, N): <(t/T)^j, h_l>_H
    b = np.zeros((D, q)); nrm = np.zeros(q)
    for i in range(q):
        Ga = cg.matvec(a_g[:, i])
        bg = np.empty(D)
        bg[0] = Ga[0]
        bg[1:3] = pc @ a_g[:, i]
        bg[3:] = Ga[1:D - 2]
        bp = A[:, 1] * c[0, i] + A[:, 2] * c[1, i]                                     # <phi_j, p>_H, p = c1 t/T + c2 (t/T)^2
        b[:, i] = bg + bp
        nrm[i] = a_g[:, i] @ Ga + 2 * (c[0, i] * bg[1] + c[1, i] * bg[2]) + c[:, i] @ A[1:3, 1:3] @ c[:, i]
    return b, nrm, used


def compute():
    cg = GR.CosineGram(H, T)
    A = GR.augmented_gram(H, D, T)
    C = np.linalg.cholesky(A)
    out = {'H': H, 'T': T, 'D': D, 'lambda': LAM, 'tau': TAU, 'N_samples': N, 'seed': SEED}
    # top 1: spectral weight c_H |xi|^{1-2H} of the whole-line realisation
    xi = np.logspace(-1, 3, 200)
    out['weight'] = {'xi': xi.tolist(), 'H': [0.5, 0.75, 0.95],
                     'w': [(c_H(h) * xi ** (1 - 2 * h)).tolist() for h in (0.5, 0.75, 0.95)]}
    # top 2: raw nested sequence on a grid
    tg = np.linspace(0, T, 401)
    raw = [np.full_like(tg, T ** -0.5), tg / T, (tg / T) ** 2] + [math.sqrt(2 / T) * np.cos(k * np.pi * tg / T) for k in (1, 2, 3)]
    out['raw'] = {'t': tg.tolist(), 'phi': [r.tolist() for r in raw]}
    # top 3: diagonally scaled Gram (correlation form) for D_HEAT; two-sided band of the cosine Schur block
    Ah = A[:D_HEAT, :D_HEAT]
    dsc = 1 / np.sqrt(np.diag(Ah))
    Rh = dsc[:, None] * Ah * dsc[None, :]
    out['gram_corr'] = Rh.tolist()
    G = cg.dense(N_SPEC + 1)
    ns, lmin, lmax = [], [], []
    for n in [2 ** e for e in range(1, int(round(math.log2(N_SPEC))) + 1)]:
        Gn = G[:n + 1, :n + 1]
        S = Gn[1:, 1:] - np.outer(Gn[1:, 0], Gn[0, 1:]) / Gn[0, 0]       # Schur complement of h0
        k = np.arange(1, n + 1)
        wk = c_H(H) * (k * np.pi / T) ** (1 - 2 * H)
        ev = np.linalg.eigvalsh(S / np.sqrt(np.outer(wk, wk)))
        ns.append(n); lmin.append(float(ev[0])); lmax.append(float(ev[-1]))
    out['band'] = {'n': ns, 'lambda_min': lmin, 'lambda_max': lmax, 'min': min(lmin), 'max': max(lmax)}
    # top 4: first orthonormal functions v = C^{-1} phi on the grid (H-orthonormal by construction)
    Phi = np.vstack([np.full_like(tg, T ** -0.5), tg / T, (tg / T) ** 2] +
                    [math.sqrt(2 / T) * np.cos(kk * np.pi * tg / T) for kk in range(1, D - 2)])
    V = np.linalg.solve(C, Phi)
    out['v'] = {'t': tg.tolist(), 'v': V[:4].tolist()}
    orth = np.linalg.solve(C, np.linalg.solve(C, A).T).T          # C^{-1} A C^{-T} = I
    out['orthonormality_defect'] = float(np.abs(orth - np.eye(D)).max())
    # bottom: input family f_tau on midpoint grid, moments, coordinates, errors, sample paths
    tm = (np.arange(N) + 0.5) / N * T
    taus = np.linspace(0.005, 1.0, 200)
    F = np.exp(-LAM * (T - tm))[None, :] * (tm[None, :] < taus[:, None])
    b, nrm, used = moments_and_norm(F, A, cg)
    gam = np.linalg.solve(C, b)                                     # gamma = C_D^{-1} b_D(f)  (fixed coefficient rule)
    i6 = int(np.argmin(np.abs(taus - TAU)))
    e2 = nrm[i6] - np.cumsum(gam[:, i6] ** 2)                       # e_{j+1}^2 after j+1 coordinates
    out['input'] = {'t': tg.tolist(), 'f': np.exp(-LAM * (T - tg)).tolist(), 'tau_shown': float(taus[i6])}
    out['moments_tau'] = np.abs(b[:, i6]).tolist()
    out['gamma2_tau'] = (gam[:, i6] ** 2).tolist()
    out['e2_tau'] = e2.tolist()
    out['norm2_tau'] = float(nrm[i6])
    out['endpoint_correction_used_fraction'] = float(np.mean(used))
    rng = np.random.default_rng(SEED)
    Z = rng.standard_normal(D)
    paths = {}
    for Dk in (5, 9, 33):
        paths[str(Dk)] = (gam[:Dk].T @ Z[:Dk]).tolist()
    eD = np.sqrt(np.maximum(nrm - np.sum(gam ** 2, axis=0), 0))
    out['paths'] = {'tau': taus.tolist(), 'paths': paths, 'eD_33': eD.tolist(), 'sd': np.sqrt(nrm).tolist()}
    # independent check of ||f_tau||_H^2 at tau = 1 (closed form for the exponential kernel)
    out['norm2_tau1'] = float(nrm[-1])
    return out


def _style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
        'axes.unicode_minus': False, 'axes.formatter.use_mathtext': True,
        'font.size': 6.5, 'axes.labelsize': 6.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2.0, 'ytick.major.size': 2.0, 'xtick.minor.size': 1.0, 'ytick.minor.size': 1.0,
        'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5, 'axes.labelpad': 1.0,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'lines.linewidth': 0.9})


def _xlab(ax, text, xfrac):
    """x label set inline with the tick labels, between two ticks (no collision, no extra height)."""
    ax.text(xfrac, -0.045, text, transform=ax.transAxes, ha='center', va='top', fontsize=6.2, color=INK)


def _glyph(rect=(0.2, 0.2, 0.77, 0.74)):
    fig = plt.figure(figsize=(GLYPH_W_CM / 2.54, GLYPH_H_CM / 2.54))
    ax = fig.add_axes(rect)
    return fig, ax


def _save(fig, name):
    fig.savefig(os.path.join(PARTS, name + '.pdf'))
    fig.savefig(os.path.join(PARTS, name + '.png'), dpi=600)
    plt.close(fig)


def render(d):
    _style()
    # T1 weight
    fig, ax = _glyph((0.2, 0.2, 0.74, 0.74))
    xi = np.array(d['weight']['xi'])
    for h, wv, lw, a in zip(d['weight']['H'], d['weight']['w'], (0.7, 1.1, 0.7), (0.45, 1.0, 0.7)):
        ax.loglog(xi, wv, color=PRIMARY, lw=lw, alpha=a, ls='-' if h == 0.75 else (':' if h == 0.5 else '--'))
    ax.text(2.0, 1.3, r'$H=\frac{1}{2}$', fontsize=5.8, color=NEUTRAL, va='bottom', ha='left')
    ax.text(700, 0.115, r'$0.75$', fontsize=5.8, color=PRIMARY, va='bottom', ha='right')
    ax.text(30, 0.026, r'$0.95$', fontsize=5.8, color=PRIMARY, va='bottom', ha='left')
    ax.text(700, 18, r'$c_H|\xi|^{1-2H}$', fontsize=6.2, color=INK, va='top', ha='right')
    ax.set_xlim(0.1, 1000); ax.set_ylim(0.02, 20)
    ax.set_xticks([0.1, 1000]); ax.set_xticklabels([r'$10^{-1}$', r'$10^{3}$']); ax.set_yticks([0.1, 10])
    ax.minorticks_off(); _xlab(ax, r'$|\xi|$', 0.5)
    _save(fig, 'g1_weight')
    # T2 raw sequence
    fig, ax = _glyph()
    tg = np.array(d['raw']['t'])
    styles = [dict(color=PRIMARY, lw=1.1), dict(color=PRIMARY, lw=1.1, ls='--'), dict(color=PRIMARY, lw=1.1, ls=':'),
              dict(color=NEUTRAL, lw=0.6, alpha=0.9), dict(color=NEUTRAL, lw=0.6, alpha=0.65), dict(color=NEUTRAL, lw=0.6, alpha=0.45)]
    for r, st in zip(d['raw']['phi'], styles):
        ax.plot(tg, r, **st)
    ax.text(0.02, 2.55, r'$\phi_0,\phi_1,\phi_2$', fontsize=5.8, color=PRIMARY, va='top', ha='left')
    ax.text(0.98, 2.55, r'$h_1,h_2,h_3$', fontsize=5.8, color=NEUTRAL, va='top', ha='right')
    ax.set_xlim(0, 1); ax.set_ylim(-1.6, 2.6); ax.set_xticks([0, 1]); ax.set_yticks([-1, 0, 1])
    _xlab(ax, r'$t/T$', 0.5)
    _save(fig, 'g2_raw')
    # T3 Gram (correlation form, D_HEAT) + two-sided band of the cosine Schur block
    fig = plt.figure(figsize=(GLYPH_W_CM / 2.54, GLYPH_H_CM / 2.54))
    axh = fig.add_axes((0.04, 0.08, 0.38, 0.86))
    R = np.abs(np.array(d['gram_corr']))
    cmap = LinearSegmentedColormap.from_list('p', [(1, 1, 1), PRIMARY_LIGHT, PRIMARY, (0.12, 0.2, 0.33)])
    axh.imshow(np.log10(np.maximum(R, 1e-3)), cmap=cmap, vmin=-3, vmax=0, interpolation='nearest')
    axh.set_xticks([]); axh.set_yticks([])
    for s in axh.spines.values():
        s.set_visible(True); s.set_linewidth(0.4); s.set_color(NEUTRAL)
    axh.axhline(2.5, color=ACCENT, lw=0.5); axh.axvline(2.5, color=ACCENT, lw=0.5)
    axb = fig.add_axes((0.58, 0.2, 0.39, 0.74))
    nb = np.array(d['band']['n'])
    axb.axhspan(0.5, 1.0, color=PRIMARY_LIGHT, lw=0)
    axb.plot(nb, d['band']['lambda_max'], color=PRIMARY, lw=1.0, marker='o', ms=1.4)
    axb.plot(nb, d['band']['lambda_min'], color=PRIMARY, lw=1.0, marker='o', ms=1.4)
    axb.set_xscale('log', base=2); axb.minorticks_off()
    axb.set_ylim(0.4, 1.08); axb.set_xlim(nb[0] / 1.4, nb[-1] * 1.4); axb.set_yticks([0.5, 1.0]); axb.set_yticklabels([r'$\frac{1}{2}$', '1'])
    axb.set_xticks([nb[0], nb[-1]]); axb.set_xticklabels([r'$%d$' % nb[0], r'$2^{%d}$' % int(round(math.log2(nb[-1])))])
    _xlab(axb, r'$n$', 0.5)
    _save(fig, 'g3_gram')
    # T4 orthonormal functions
    fig, ax = _glyph()
    tv = np.array(d['v']['t'])
    for j, (vv, lw, a) in enumerate(zip(d['v']['v'], (1.1, 1.0, 0.9, 0.8), (1.0, 0.85, 0.7, 0.55))):
        ax.plot(tv, vv, color=PRIMARY, lw=lw, alpha=a)
    vmax = max(np.abs(np.array(d['v']['v'])).max(), 1)
    ax.set_xlim(0, 1); ax.set_xticks([0, 1]); _xlab(ax, r'$t/T$', 0.5)
    ax.set_ylim(-1.08 * vmax, 1.08 * vmax)
    _save(fig, 'g4_basis')
    # B1 input
    fig, ax = _glyph()
    ti = np.array(d['input']['t']); fi = np.array(d['input']['f']); ts = d['input']['tau_shown']
    ax.fill_between(ti, 0, np.where(ti < ts, fi, 0), color=ACCENT_LIGHT, lw=0, step=None)
    ax.plot(ti, fi, color=NEUTRAL, lw=0.6, ls='--')
    ax.plot(ti[ti < ts], fi[ti < ts], color=ACCENT, lw=1.2)
    ax.plot([ts, ts], [0, np.exp(-LAM * (T - ts))], color=ACCENT, lw=1.2)
    ax.text(ts + 0.03, 0.9, r'$f_\tau=f\,\mathbf{1}_{[0,\tau)}$', fontsize=5.8, color=ACCENT, va='top')
    ax.text(0.98, 0.04, r'$f$', fontsize=5.8, color=NEUTRAL, ha='right', va='bottom')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.set_xticks([0, ts, 1]); ax.set_xticklabels(['0', r'$\tau$', '1'])
    ax.set_yticks([0, 1]); _xlab(ax, r'$t/T$', 0.3)
    _save(fig, 'g5_input')
    # B2 moments
    fig, ax = _glyph()
    m = np.array(d['moments_tau']); j = np.arange(m.size)
    ax.vlines(j, 1e-5, m, color=ACCENT, lw=0.8)
    ax.plot(j, m, 'o', ms=1.3, color=ACCENT)
    ax.set_yscale('log'); ax.set_ylim(1e-4, 1); ax.set_xlim(-0.8, D - 0.2)
    ax.set_xticks([0, 16, 32]); ax.set_yticks([1e-3, 1]); ax.minorticks_off()
    _xlab(ax, r'$j$', 0.25)
    ax.text(D - 1, 0.6, r'$|b_j(f_\tau)|$', fontsize=6.2, color=ACCENT, ha='right', va='top')
    _save(fig, 'g6_moments')
    # B3 coordinates and error
    fig, ax = _glyph()
    g2 = np.array(d['gamma2_tau']); e2 = np.array(d['e2_tau'])
    ax.bar(np.arange(g2.size), g2, width=0.7, color=ACCENT, alpha=0.55, lw=0)
    ax.plot(np.arange(1, e2.size + 1) - 0.5, e2, color=INK, lw=1.0)
    ax.set_yscale('log'); ax.set_ylim(1e-6, 1); ax.set_xlim(-0.8, D - 0.2)
    ax.set_xticks([0, 16, 32]); ax.set_yticks([1e-5, 1e-3, 1e-1]); ax.minorticks_off()
    ax.text(D - 1, 0.6, r'$e_D(f_\tau)^2$', fontsize=6.2, color=INK, ha='right', va='top')
    ax.text(0.5, 0.6, r'$\gamma_j(f_\tau)^2$', fontsize=6.2, color=ACCENT, ha='left', va='top')
    _xlab(ax, r'$j,\ D$', 0.25)
    _save(fig, 'g7_coeffs')
    # B4 sample path tau -> I_H(f_tau) with one shared Z
    fig, ax = _glyph()
    tau = np.array(d['paths']['tau']); p = d['paths']['paths']; eD = np.array(d['paths']['eD_33'])
    p33 = np.array(p['33'])
    ax.plot(tau, p['5'], color=ACCENT, lw=0.6, alpha=0.45)
    ax.plot(tau, p['9'], color=ACCENT, lw=0.7, alpha=0.7)
    ax.plot(tau, p33, color=ACCENT, lw=1.1)
    ax.set_xlim(0, 1); ax.set_xticks([0, 1]); _xlab(ax, r'$\tau$', 0.5)
    allp = np.concatenate([p33, p['5'], p['9']])
    lo, hi = allp.min(), allp.max()
    pad = 0.08 * (hi - lo); ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.text(0.03, lo, r'$D=5,9,33$', fontsize=5.8, color=ACCENT, va='bottom', ha='left')
    _save(fig, 'g8_paths')


if __name__ == '__main__':
    d = compute()
    with open(os.path.join(DATA, 'fig1_glyphs.json'), 'w') as fh:
        json.dump(d, fh)
    render(d)
    print(json.dumps({k: d[k] for k in ('orthonormality_defect', 'norm2_tau', 'norm2_tau1', 'endpoint_correction_used_fraction')}))
    print('band min %.4f max %.6f over n=%s' % (d['band']['min'], d['band']['max'], d['band']['n']))
    print('e2 at D=33 for tau=0.6: %.3e ; gamma2[0..4] %s' % (d['e2_tau'][-1], np.round(d['gamma2_tau'][:5], 5)))
