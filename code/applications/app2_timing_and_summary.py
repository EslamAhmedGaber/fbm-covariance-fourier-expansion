"""A2 supplement: CF re-timing, the m = 12000 paired run, the exact-GP rounding-floor test and the summary.

Commands (run from this directory):
  python app2_timing_and_summary.py retime     -> results/A2_cf_retime_H075.json   (CF pipeline alone, 2 repeats)
  python app2_timing_and_summary.py m12000     -> results/A2_m12000_H075.json      (needs ~1.2 GiB free RAM)
  python app2_timing_and_summary.py summary [out.json] -> results/A2_v2_summary.json (rounding-floor test + consolidation)
Timings are machine dependent (+-20% on the test machine); accuracy numbers are deterministic.
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import app2_noisy_reconstruction_v2 as V2
import app2_gp_baseline_strong as S
from covfourier.engine import CovarianceFourier
from covfourier.batch import BatchProjector
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr
RES = V2.RES


def _cf_once(m, c, s, y):
    t0 = time.perf_counter(); cf = CovarianceFourier(0.75, 256, enrich=True, Naux=2048); bp = BatchProjector(cf)
    Z, E2s = V2.cf_coords(bp, c[:m], s[:m]); Wt, E2t = V2.cf_targets(bp, c[m:], s[m:]); V2.cf_posterior(Z, E2s, y, Wt, E2t)
    return time.perf_counter() - t0


def retime():
    out = []
    for m in (4000, 8000, 10000):
        rng = np.random.default_rng(20261029 + m)
        c, s = V2.draw_windows(rng, m + V2.N_NEW)
        y = np.random.default_rng(5).standard_normal(m)
        reps = [_cf_once(m, c, s, y) for _ in range(2)]
        out.append({'m': m, 'n': 256, 'cf_total_s_reps': reps, 'cf_total_s_min': min(reps)})
        print(json.dumps(out[-1]), flush=True)
    json.dump(out, open(os.path.join(RES, 'A2_cf_retime_H075.json'), 'w'), indent=1)


def m12000():
    m = 12000
    rng = np.random.default_rng(20261029 + m)
    c, s = V2.draw_windows(rng, m + V2.N_NEW)
    rec = {'m': m, 'H': 0.75, 'avail_GiB_before': V2.avail_gib()}
    try:
        fa = S.FasterAbsMoment(1.5)
        t0 = time.perf_counter(); K = S.lower_K(0.75, c, s, m, fa); tb = time.perf_counter() - t0
        K.flat[::m + 1] += V2.TAU ** 2
        t0 = time.perf_counter(); L = cho_factor(K, lower=True, overwrite_a=True, check_finite=False); tf = time.perf_counter() - t0
        rec.update({'gp_strong_build_s': tb, 'gp_strong_factor_s': tf, 'gp_strong_total_s_excl_targets': tb + tf})
        del K, L
    except MemoryError:
        rec['gp_strong_error'] = 'MemoryError'
    y = np.random.default_rng(5).standard_normal(m)
    rec['cf_total_s_reps'] = [_cf_once(m, c, s, y) for _ in range(2)]
    print(json.dumps(rec), flush=True)
    json.dump(rec, open(os.path.join(RES, 'A2_m12000_H075.json'), 'w'), indent=1)


def rounding_floor(H=0.75, m=2000):
    """Change of the exact GP posterior under a one-ulp entrywise perturbation of its covariance."""
    rng = np.random.default_rng(20261029 + m)
    c, s = V2.draw_windows(rng, m + V2.N_NEW)
    fa = S.FasterAbsMoment(2 * H)
    K, Kt, vt = V2.gp_covariances(H, c, s, m, fa)
    K.flat[::m + 1] += V2.TAU ** 2
    L = cho_factor(K.copy(), lower=True)
    y = np.tril(L[0]) @ rng.standard_normal(m)

    def post(Kmat):
        Lf = cho_factor(Kmat, lower=True)
        return Kt @ cho_solve(Lf, y), vt - np.einsum('ij,ji->i', Kt, cho_solve(Lf, Kt.T))
    m0, v0 = post(K.copy())
    E = np.random.default_rng(9).standard_normal((m, m)); E = (E + E.T) / 2
    m1, v1 = post(K + np.finfo(float).eps * np.abs(K) * E)
    ev = np.linalg.eigvalsh(K)
    nt = 2 + len(V2.TSTARS); smooth = np.r_[0, 1, np.arange(nt, nt + V2.N_NEW)]
    sd = np.sqrt(v0)
    return {'kappa_K_plus_tau2I': float(ev[-1] / ev[0]),
            'one_ulp_perturbation_mean_sd': float(np.max(np.abs(m1 - m0)[smooth] / sd[smooth])),
            'one_ulp_perturbation_var_rel': float(np.max(np.abs(v1 / v0 - 1)[smooth]))}


def summary(out_path=None):
    a2v = json.load(open(os.path.join(RES, 'A2_noisy_reconstruction_v2.json')))
    strong = json.load(open(os.path.join(RES, 'A2_gp_baseline_strong.json')))
    ret = json.load(open(os.path.join(RES, 'A2_cf_retime_H075.json')))
    m12 = json.load(open(os.path.join(RES, 'A2_m12000_H075.json')))
    fl = rounding_floor()
    c2 = [cc for cc in a2v['cases']['0.75'] if cc['m'] == 2000][0]
    r256 = [r for r in c2['CF'] if r['n'] == 256][0]
    fl.update({'cf_vs_gp_mean_sd_n256': r256['smooth_mean_err_sd_max'], 'cf_vs_gp_var_rel_n256': r256['smooth_var_rel_err_max']})
    v2b = [{'H': H, 'm': cc['m'], 'gp_s': cc['GP']['total_s'], 'cf_s': cc['matched'].get('cf_total_s'), 'n': cc['matched'].get('n'),
            'gp_over_cf': cc['matched'].get('gp_over_cf')} for H in a2v['cases'] for cc in a2v['cases'][H]]
    strong_rows = [{'m': cc['m'], 'gp_strong_s': cc['total_s'], 'gp_v2_s': cc['v2_gp_total_s']} for cc in strong['cases']]
    cf_rt = ret + [{'m': 12000, 'n': 256, 'cf_total_s_reps': m12['cf_total_s_reps'], 'cf_total_s_min': min(m12['cf_total_s_reps'])}]

    def ratio_range(m):
        r = [x['gp_over_cf'] for x in v2b if x['m'] == m and x['gp_over_cf'] is not None]
        return '%.2f-%.2f' % (min(r), max(r)) if r else 'n/a'
    reading = ('straightforward GP build: GP/CF time ratio %s at m=2000 and %s at m=4000 (all H). '
               'Strengthened GP build: %s; CF alone (two repeats): %s; m=12000: GP %.1f s (excluding targets), CF %.1f-%.1f s. '
               'Timing variability about +-20%%.' % (
                   ratio_range(2000), ratio_range(4000),
                   ', '.join('m=%d %.1f s' % (r['m'], r['gp_strong_s']) for r in strong_rows),
                   ', '.join('m=%d %.1f-%.1f s' % (r['m'], min(r['cf_total_s_reps']), max(r['cf_total_s_reps'])) for r in ret),
                   m12['gp_strong_total_s_excl_targets'], min(m12['cf_total_s_reps']), max(m12['cf_total_s_reps'])))
    out = {
        'what': 'A2 consolidated evidence. Sources: A2_noisy_reconstruction_v2.json (sweep), A2_gp_baseline_strong.json, '
                'A2_cf_retime_H075.json, A2_m12000_H075.json, and the rounding-floor test computed here.',
        'alignment': {'max_outside_mass_all_cases': max(cc['max_outside_mass'] for H in a2v['cases'] for cc in a2v['cases'][H]),
                      'bound': '2*Phi(-10) = %.2e' % (2 * float(ndtr(-10.0)))},
        'kernel_checks': a2v['kernel_checks'], 'kernel_speed_v2_table_vs_scipy_2e6': a2v['kernel_speed_2e6'],
        'strong_kernel': strong['kernel_2e6_z_0_30'],
        'gp_rounding_floor_test_H075_m2000': fl,
        'matched_tolerance': a2v['tol'],
        'crossover': {'v2_gp_build': v2b,
                      'strong_gp_build_H075': strong_rows + [{'m': 12000, 'gp_strong_s_excl_targets': m12['gp_strong_total_s_excl_targets'],
                                                              'avail_GiB_before': m12['avail_GiB_before']}],
                      'cf_alone_retimed_H075_n256': cf_rt, 'reading': reading},
        'cf_only_H075_n256': a2v['cf_only'], 'gp_fit_EXTRAPOLATION_only': a2v['gp_fit_H075'],
        'diagonal_residual_model': 'approximation; its size is reported as sum_i E_i^2/tau^2 per case'}
    json.dump(out, open(out_path or os.path.join(RES, 'A2_v2_summary.json'), 'w'), indent=1)
    print(json.dumps(fl), '\n', reading)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'summary'
    {'retime': retime, 'm12000': m12000}.get(cmd, lambda: summary(sys.argv[2] if len(sys.argv) > 2 else None))()
