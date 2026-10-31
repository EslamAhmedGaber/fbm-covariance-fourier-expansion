"""Figure 10 (real borehole data); same typography and colours as make_figures.py.

Reads results/A7_*.json; writes figures/fig10.* and figures/data/fig10.json.
Run:  python figures/src/make_figure_realdata.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_figures as MF                                   # noqa: E402
from make_figures import plt, C_OURS, C_OURS_LIGHT, C_QUAD, C_GREY, TINY   # noqa: E402

RES = MF.RES


def _load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def _pct():
    return r'\%' if MF.USETEX else '%'


def history_panel(ax, fname, letter, show_legend):
    h = _load(fname)
    dec = np.array(h['decades']) + 5.0
    mu = np.array(h['mu']); sd = np.array(h['sd'])
    ax.fill_between(dec, mu - 1.96 * sd, mu + 1.96 * sd, color=C_OURS_LIGHT, alpha=0.6, lw=0,
                    label='boreholes only: 95' + _pct() + ' band')
    ax.plot(dec, mu, color=C_OURS, lw=1.1, label='boreholes only: decadal mean')
    y = np.array(h['crutem_years']); v = np.array(h['crutem_mean'])
    ax.plot(y, v, color='0.45', lw=0.45, alpha=0.9, label='CRUTEM5, annual (sites\' cells)')
    dd = np.arange(1850, 2010, 10)
    vm = np.array([v[(y >= d0) & (y < d0 + 10)].mean() if np.any((y >= d0) & (y < d0 + 10)) else np.nan for d0 in dd])
    ax.plot(dd + 5, vm, 'ks', ms=2.6, mfc='white', mew=0.7, label='CRUTEM5, decadal mean')
    lt = np.array(h['log_years'])
    ax.plot(lt, np.full(len(lt), ax.get_ylim()[0] if False else -1.45), '|', color=C_GREY, ms=3, mew=0.4)
    ax.set_xlim(1800, 2012); ax.set_ylim(-1.5, 2.0)
    ax.set_xlabel('year'); ax.set_ylabel(r'$\Delta T$ rel. to 1901--1930 [K]' if MF.USETEX else 'dT rel. to 1901-1930 [K]')
    name = h['country'].replace('United States Of America', 'United States')
    ax.set_title('%s: %d boreholes' % (name, h['n_sites']))
    if show_legend:
        ax.legend(fontsize=TINY - 1.5, loc='upper left', handlelength=1.3)
    ok = np.isfinite(vm)
    idx = [int(np.where(np.array(h['decades']) == d0)[0][0]) for d0 in dd[ok]]
    z = (vm[ok] - mu[idx]) / sd[idx]
    ax.text(0.98, 0.04, 'decades in 95' + _pct() + ' band: %d of %d' % (int(np.sum(np.abs(z) <= 1.96)), len(z)),
            transform=ax.transAxes, ha='right', va='bottom', fontsize=TINY - 1.5)
    MF.letter(ax, letter)
    return {'country': h['country'], 'n_sites': h['n_sites'], 'z_decades': z.tolist(), 'decades': dd[ok].tolist()}


def fig9():
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.9), gridspec_kw={'wspace': 0.32, 'hspace': 0.55})
    out = {}
    out['a'] = history_panel(axs[0, 0], 'A7_regional_history_Canada.json', 'a', True)
    out['b'] = history_panel(axs[0, 1], 'A7_regional_history_United_States_Of_America.json', 'b', False)
    # (c) country means of W: regional model vs CRUTEM5, database reconstructions for comparison
    ax = axs[1, 0]
    R = _load('A7_borehole_regional.json')
    H = max(R['by_H'], key=lambda h: R['by_H'][h]['ll'])
    C = R['by_H'][H]['countries']
    vs = _load('A7_validation_summary_v2.json') if os.path.exists(os.path.join(RES, 'A7_validation_summary_v2.json')) else _load('A7_validation_summary_v2_partial.json')
    hu = vs['by_H'][list(vs['by_H'])[0]]['countries']
    names = sorted(C, key=lambda c: C[c]['W_inst'])
    TE = _load('A7_target_error.json')                      # error of the instrumental target (station archive only)
    x = np.array([C[c]['W_inst'] for c in names]); xe = np.array([TE['countries_target'][c]['sd_target'] for c in names])
    y = np.array([C[c]['mu'] for c in names]); ye = np.array([C[c]['sd'] for c in names])
    hh = np.array([hu[c]['huang'] if c in hu else np.nan for c in names])
    lo = min((x - 2 * xe).min(), (y - 2 * ye).min(), np.nanmin(hh)) - 0.1; hi = max((x + 2 * xe).max(), (y + 2 * ye).max(), np.nanmax(hh)) + 0.1
    ax.plot([lo, hi], [lo, hi], color='0.6', lw=0.6)
    ax.errorbar(x, y, xerr=1.96 * xe, yerr=1.96 * ye, fmt='o', color=C_OURS, ms=3.2, lw=0.7, capsize=0,
                label='regional model (95' + _pct() + ')')
    ax.plot(x, hh, 's', color=C_GREY, mfc='white', ms=3.0, mew=0.7, label='database reconstructions')
    for c, xi, yi in zip(names, x, y):
        if c in ('Japan', 'China'):
            ax.annotate(c + (' (urban sites)' if c == 'Japan' else ''), (xi, yi), textcoords='offset points',
                        xytext=(-5, 4) if c == 'China' else (8, -11), ha='right' if c == 'China' else 'left', fontsize=TINY - 2, color=C_OURS)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('CRUTEM5 warming $W$ [K]'); ax.set_ylabel('borehole warming $W$ [K]')
    ax.set_title('Country means, %d countries' % len(names))
    ax.legend(fontsize=TINY - 1.5, loc='upper left', handlelength=1.2)
    MF.letter(ax, 'c')
    out['c'] = {c: C[c] | {'sd_target': TE['countries_target'][c]['sd_target'],
                           'z_target': TE['models']['regional']['countries'][c]['z']} for c in names}
    # (d) re-logging forecasts: all five pairs, error of the predicted change profile
    ax = axs[1, 1]
    if os.path.exists(os.path.join(RES, 'A7_relog_forecast.json')):
        rf = _load('A7_relog_forecast.json')
        pairs = rf['pairs']
        yy = np.arange(len(pairs))[::-1]
        priors = list(pairs[0]['by_prior'])
        mk = {priors[0]: ('o', C_QUAD), priors[1]: ('o', C_OURS)}
        for y0, p in zip(yy, pairs):
            for nm in priors:
                r = p['by_prior'][nm]
                ax.plot(r['rmse_change_centred'], y0 + (0.12 if nm == priors[1] else -0.12), mk[nm][0], color=mk[nm][1], ms=3.6,
                        mfc=mk[nm][1] if r['chi2'] / r['dof'] <= 2 else 'white', mew=0.8)
            ax.plot(p['no_change']['rmse_change_centred'], y0, 'x', color='k', ms=3.6, mew=0.8)
        stems = {'US-nha64': 'US nha', 'US-nos65': 'US nos', 'US-sar64': 'US sar', 'US-HAYSPR': 'US HAY', 'CA-442-18': 'CA 442'}
        ax.set_yticks(yy)
        ax.set_yticklabels(['%s %d--%d' % (stems.get(p['pair'][0], p['pair'][0]), int(p['t1']), int(p['t2'])) if MF.USETEX else
                            '%s %d-%d' % (stems.get(p['pair'][0], p['pair'][0]), int(p['t1']), int(p['t2'])) for p in pairs], fontsize=TINY - 1)
        ax.set_xscale('log'); ax.set_ylim(-0.6, len(pairs) + 0.9)
        ax.set_xlabel('RMS error of the centred change profile [K]')
        ax.set_title('Forecast of the later log')
        hl = [plt.Line2D([], [], marker='o', color=mk[nm][1], ls='none', ms=3.6, label=nm) for nm in priors[::-1]]
        hl += [plt.Line2D([], [], marker='x', color='k', ls='none', ms=3.6, label='no change'),
               plt.Line2D([], [], marker='o', color='0.4', mfc='white', ls='none', ms=3.6, label='$\\chi^2/$dof$>2$')]
        ax.legend(handles=hl, fontsize=TINY - 1.5, loc='upper center', handlelength=1.0, ncol=4, columnspacing=0.6,
                  handletextpad=0.2)
        out['d'] = [{'pair': p['pair'], 't1': p['t1'], 't2': p['t2'], 'no_change': p['no_change']['rmse_change_centred'],
                     'by_prior': {k: {kk: vv for kk, vv in v.items() if kk in ('lpd_contrast', 'chi2', 'dof', 'rmse_change_centred')}
                                  for k, v in p['by_prior'].items()}} for p in pairs]
    MF.letter(ax, 'd')
    MF.save(fig, 'fig9', out)


if __name__ == '__main__':
    MF.style()
    fig9()
# figures renumbered for the revised paper (CovFourier revision)
