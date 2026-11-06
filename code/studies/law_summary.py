"""Summary of the law study: reference values (exact law at H = 1/2; Richardson-extrapolated discretized law
otherwise), containment checks, Monte Carlo agreement, naive errors, test sizes, Monte Carlo cost to match the
bracket, and the LaTeX tables (main Table 4 and Supplementary Table S4) generated from the JSON results.
Writes results/L8_summary.json, ../manuscript/table_law_main.tex, ../manuscript/table_law_supp.tex."""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(HERE, '..', 'covfourier')):          # inside the repository: code/studies/
    REPO = os.path.dirname(os.path.dirname(HERE)); ROOT = REPO
    RES = os.path.join(REPO, 'results'); TABLES = os.path.join(RES, 'law_tables')
else:                                                                # outside the repository: set COVFOURIER_REPO to its root
    ROOT = os.path.dirname(HERE); REPO = os.environ.get('COVFOURIER_REPO', os.path.dirname(ROOT))
    RES = os.path.join(ROOT, 'results'); TABLES = os.path.join(ROOT, 'manuscript')
os.makedirs(TABLES, exist_ok=True)
L = lambda f: json.load(open(os.path.join(RES, f)))
B = {}
for f in ('L2_brackets_one.json', 'L2_brackets_inv.json'):
    B.update(L(f))
EX = L('L3_exact_bm.json'); DI = L('L4_discretized.json'); MC = L('L5_monte_carlo.json')
PS = ['0.5', '0.9', '0.95', '0.99', '0.999']
CASES = [('one', 0.5, 'one_0.5', 'one_0.5_1024'), ('one', 0.75, 'one_0.75', 'one_0.75_1024'), ('one', 0.95, 'one_0.95', 'one_0.95_1024'),
         ('inv_t', 0.55, 'inv_0.55_16384', 'inv_t_0.55_4096'), ('inv_t', 0.75, 'inv_0.75_16384', 'inv_t_0.75_4096'),
         ('inv_t', 0.95, 'inv_0.95_16384', 'inv_t_0.95_4096')]


def richardson(w, H, p):
    q = [DI[f'{w}_{H}_{N}']['quantiles'][p] for N in (2048, 4096, 8192)]
    d1, d2 = q[1] - q[0], q[2] - q[1]
    r = d1 / d2                                   # observed contraction per doubling
    return q[2] + d2 / (r - 1), r


out = {}
for w, H, bk, mk in CASES:
    r = B[bk]; rows = {}
    for p in PS:
        lo, hi = r['quantiles'][p]['lo'], r['quantiles'][p]['hi']
        if w == 'one' and H == 0.5:
            ref, ratio, kind = EX[p]['exact'], None, 'exact'
        else:
            ref, ratio = richardson(w, H, p); kind = 'extrapolated discretized law'
        m = MC[mk]['quantiles'][p]
        disc_mc_grid = DI.get(f'{w}_{H}_{1024 if w == "one" else 4096}', {}).get('quantiles', {}).get(p) if not (w == 'one' and H == 0.5) else None
        dens = r['quantiles'][p]['density']; pp = float(p)
        # Monte Carlo paths for a 95% half-width equal to half the bracket width (sampling only)
        M_match = (1.959964 * math.sqrt(pp * (1 - pp)) / (dens * 0.5 * (hi - lo))) ** 2
        rows[p] = {'lo': lo, 'hi': hi, 'width': hi - lo, 'rel_width': (hi - lo) / lo, 'ref': ref, 'ref_kind': kind,
                   'richardson_ratio': ratio, 'ref_inside': bool(lo <= ref <= hi), 'ref_minus_lo': ref - lo,
                   'mc': m['q'], 'mc_ci': m['ci'], 'mc_rel_halfwidth': 0.5 * (m['ci'][1] - m['ci'][0]) / m['q'],
                   'disc_on_mc_grid': disc_mc_grid,
                   'mc_contains_disc_on_grid': (None if disc_mc_grid is None else bool(m['ci'][0] <= disc_mc_grid <= m['ci'][1])),
                   'mc_contains_bracket_mid': bool(m['ci'][0] <= 0.5 * (lo + hi) <= m['ci'][1]),
                   'gauss_rel_err': r['naive'][p]['gauss'] / lo - 1, 'chi2_rel_err': r['naive'][p]['chi2_two_moment'] / lo - 1,
                   'mc_paths_to_match': M_match}
    out[bk] = {'w': w, 'H': H, 'Delta': r['Delta'], 'P_gauss_below_0': r['P_gauss_below_0'], 'events': r['events'],
               'inv_check_imhof': r['inversion_check_imhof_max_abs'], 'inv_check_talbot': r['inversion_check_talbot_M24_M32'],
               'rows': rows}
    if w == 'inv_t':
        rr = B[bk.replace('16384', '32768')]['quantiles']
        out[bk]['refinement_N32768'] = {p: {'lo_shift': rr[p]['lo'] - r['quantiles'][p]['lo'], 'hi_shift': rr[p]['hi'] - r['quantiles'][p]['hi'],
                                            'inside': bool(r['quantiles'][p]['lo'] <= rr[p]['lo'] and rr[p]['hi'] <= r['quantiles'][p]['hi'])} for p in PS}
if os.path.exists(os.path.join(RES, 'L7_inversion_check.json')):
    out['dehoog'] = L('L7_inversion_check.json')
json.dump(out, open(os.path.join(RES, 'L8_summary.json'), 'w'), indent=1)

# ---------------------------------------------------------------- console summary
for bk, v in out.items():
    if bk == 'dehoog':
        continue
    print(bk, 'Delta %.2e' % v['Delta'], 'P(G<0) %.3f' % v['P_gauss_below_0'],
          'size1%%(G) %.4f-%.4f' % tuple(v['events']['gauss_0.01']['P_exceed']), 'size0.1%%(G) %.4f-%.4f' % tuple(v['events']['gauss_0.001']['P_exceed']),
          'size1%%(chi2) %.4f-%.4f' % tuple(v['events']['chi2_two_moment_0.01']['P_exceed']))
    for p, x in v['rows'].items():
        print('   p=%s [%.9f, %.9f] rel %.1e | ref %.9f (%s) in=%s ref-lo=%.1e | MC %.5f [%.5f,%.5f] contains disc=%s mid=%s | G %+.1f%% chi2 %+.2f%% | M_match %.1e'
              % (p, x['lo'], x['hi'], x['rel_width'], x['ref'], x['ref_kind'][:5], x['ref_inside'], x['ref_minus_lo'], x['mc'], *x['mc_ci'],
                 x['mc_contains_disc_on_grid'], x['mc_contains_bracket_mid'], 100 * x['gauss_rel_err'], 100 * x['chi2_rel_err'], x['mc_paths_to_match']))
    if 'refinement_N32768' in v:
        print('   refinement inside:', all(z['inside'] for z in v['refinement_N32768'].values()),
              'max |lo shift| %.1e max |hi shift| %.1e' % (max(abs(z['lo_shift']) for z in v['refinement_N32768'].values()),
                                                         max(abs(z['hi_shift']) for z in v['refinement_N32768'].values())))


# ---------------------------------------------------------------- LaTeX tables
def fx(x, d, mode='near'):
    """Fixed-point text; bracket ends are rounded outward (mode 'down' / 'up') so that printed brackets stay valid."""
    if mode == 'down':
        x = math.floor(x * 10 ** d) / 10 ** d
    elif mode == 'up':
        x = math.ceil(x * 10 ** d) / 10 ** d
    return ('%.' + str(d) + 'f') % x


def sci(x):
    e = int(math.floor(math.log10(abs(x)))); m = round(x / 10 ** e, 1)
    if m >= 10:
        m, e = m / 10, e + 1
    return r'%.1f\times10^{%d}' % (m, e)


def wlab(w):
    return '$1$' if w == 'one' else '$1/t$'


main = [r'\begin{tabular}{llcccccc}\toprule',
        r'$w$ & $H$ & $\Delta$ & $q_{0.99}$: CovFourier bracket & reference & MC, $10^6$ paths & Gaussian & scaled $\chi^2$\\\midrule']
for w, H, bk, mk in CASES:
    v = out[bk]; x = v['rows']['0.99']
    d = 7 if v['Delta'] < 1e-6 else (6 if v['Delta'] < 1e-4 else 5)
    ref = fx(x['ref'], d) + ('$^{\\rm e}$' if x['ref_kind'] == 'exact' else '')
    main.append(r'%s & %s & $%s$ & $[%s,\ %s]$ & %s & %s & $%+.1f\%%$ & $%+.1f\%%$\\' % (
        wlab(w), H, sci(v['Delta']), fx(x['lo'], d, 'down'), fx(x['hi'], d, 'up'), ref,
        r'%.3f $\pm$ %.3f' % (x['mc'], 0.5 * (x['mc_ci'][1] - x['mc_ci'][0])), 100 * x['gauss_rel_err'], 100 * x['chi2_rel_err']))
main.append(r'\bottomrule\end{tabular}')
open(os.path.join(TABLES, 'table_law_main.tex'), 'w').write('\n'.join(main) + '\n')

supp = [r'\begin{tabular}{lllccc}\toprule', r'case & $p$ & CovFourier bracket for $q_p$ & reference & MC, $10^6$ paths (95\% CI) & MC paths to match\\\midrule']
for w, H, bk, mk in CASES:
    v = out[bk]
    d = 8 if v['Delta'] < 1e-6 else (7 if v['Delta'] < 1e-4 else 6)
    for i, p in enumerate(PS):
        x = v['rows'][p]
        case = (r'$w=%s$, $H=%s$' % ('1' if w == 'one' else '1/t', H)) if i == 0 else ''
        supp.append(r'%s & %s & $[%s,\ %s]$ & %s%s & %.4f\ [%.4f, %.4f] & $%s$\\' % (
            case, p, fx(x['lo'], d, 'down'), fx(x['hi'], d, 'up'), fx(x['ref'], d), '$^{\\rm e}$' if x['ref_kind'] == 'exact' else '',
            x['mc'], x['mc_ci'][0], x['mc_ci'][1], sci(x['mc_paths_to_match'])))
    supp.append(r'\addlinespace')
supp[-1] = r'\bottomrule\end{tabular}'
open(os.path.join(TABLES, 'table_law_supp.tex'), 'w').write('\n'.join(supp) + '\n')
print('tables written')
