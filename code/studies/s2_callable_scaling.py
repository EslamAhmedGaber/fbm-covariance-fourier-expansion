"""S2: end-to-end GENERIC-CALLABLE scaling with white coordinates .

S1 used analytic classical coefficients through the moment interface, and its PCG route returned no
white coordinates. Here every input is only a vectorized callable, sampled at N midpoints. All inputs
go through the same steps: the a-posteriori endpoint-correction rule, the batched Gram mat-vec, and
white coordinates from covfourier.large.LargeProjector (Schur complement + Chebyshev x^{-1/2}).
Reported per (H, n, input):
  E2_callable   squared residual of the enriched projection, callable route, finite auxiliary space N
  E2_moment     the same quantity from S1 (exact classical coefficients, its own Naux), when available
  energy check  ||z||^2 + E2 = ||f||^2 (Pythagoras inside the auxiliary space)
  timings       setup (once per H, n), Fourier stage, projection + white coordinates
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from covfourier.large import LargeProjector        # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')

INPUTS = {
    'exp(t)': lambda t: np.exp(t),
    'sin(140 pi t)': lambda t: np.sin(140 * np.pi * t),
    'sin(1000 pi t)': lambda t: np.sin(1000 * np.pi * t),
    '|t-1/2|': lambda t: np.abs(t - 0.5),
    '1[0,0.3)': lambda t: (t < 0.3).astype(float),
    't^-0.25': lambda t: t ** -0.25,
}


def run(Hs=(0.55, 0.75, 0.95), ns=(1023, 4095, 16383, 32767), out='S2_callable_scaling.json'):
    s1 = json.load(open(os.path.join(RES, 'S1_scaling_study.json')))
    ref = {}
    for r in s1['rows']:
        ref.setdefault((r['H'], r['n'], r['input']), r)
    data = {'note': __doc__, 'rows': [], 'setups': []}
    T0 = time.perf_counter()
    for H in Hs:
        for n in ns:
            N = min(32 * (n + 1), 1 << 20)
            t0 = time.perf_counter()
            lp = LargeProjector(H, n, N)
            ts = time.perf_counter() - t0
            data['setups'].append({'H': H, 'n': n, 'D': n + 2, 'N': N, 'setup_s': ts, 'cheb_degree': lp.sc.cheb_degree,
                                   'enrich_solves': lp.enrich_solve_info, 'orthogonality_defect': lp.orthogonality_defect})
            t = (np.arange(N) + 0.5) / N
            for name, f in INPUTS.items():
                t1 = time.perf_counter()
                a, c = lp.fourier_stage(f(t)[None, :])
                tf = time.perf_counter() - t1
                lp.sc.matvecs = 0
                t1 = time.perf_counter()
                r = lp.project(a, c, explicit=True)
                tp = time.perf_counter() - t1
                row = {'H': H, 'n': n, 'D': n + 2, 'N': N, 'input': name, 'corrected': bool(lp.last_corrected[0]),
                       'norm2': float(r['norm2'][0]), 'energy': float(r['energy'][0]), 'E2_callable': float(r['error2'][0]),
                       'eta': float(r['eta'][0]), 'fourier_s': tf, 'project_white_s': tp, 'white_matvecs': lp.sc.matvecs,
                       'E2_aux_explicit': float(r['E2_aux_explicit'][0]), 'tau2_poly_tail_bound': float(np.atleast_1d(r['tau2_poly_tail_bound'])[0]),
                       'explicit_pcg_iterations': r['explicit_pcg_iterations']}
                s = ref.get((H, n, name))
                if s is not None:
                    row['E2_moment_S1'] = s['error2_enriched']; row['norm2_exact'] = s['norm2_exact']
                    row['Naux_S1'] = s['Naux']
                    row['rel_diff_E2'] = abs(row['E2_callable'] - s['error2_enriched']) / max(abs(s['error2_enriched']), 1e-300)
                    row['rel_diff_norm2'] = abs(row['norm2'] - s['norm2_exact']) / s['norm2_exact']
                data['rows'].append(row)
                print('H %.2f n %5d N %7d %-15s E2 %.3e (S1 %s) dnorm %.1e  corr %s  t %.1f+%.1fs' % (
                    H, n, N, name, row['E2_callable'], ('%.3e' % row['E2_moment_S1']) if s else '-', row.get('rel_diff_norm2', float('nan')),
                    row['corrected'], tf, tp), flush=True)
            del lp
            with open(os.path.join(RES, out), 'w') as fh:
                json.dump(data, fh, indent=1)
    data['seconds'] = time.perf_counter() - T0
    with open(os.path.join(RES, out), 'w') as fh:
        json.dump(data, fh, indent=1)
    return data


if __name__ == '__main__':
    run()
