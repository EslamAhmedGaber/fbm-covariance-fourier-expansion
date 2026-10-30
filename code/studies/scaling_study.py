"""S1 scaling study: same procedure, dimensions far beyond the 65/257-term trials.

Exact classical cosine coefficients and exact norms are used as *inputs* (moment
interface) so that the reported errors are projection errors of the fixed space,
not sampling errors.  Errors are the direct quadratic form of the residual vector.
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from covfourier import engine as EN, gram as G, moments as MO   # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')


def cusp_coeffs(N):
    k = np.arange(N); a = np.zeros(N); a[0] = 0.25
    ev = (k % 2 == 0) & (k > 0)
    a[ev] = math.sqrt(2) * 2 * (1 - np.cos(k[ev] * np.pi / 2)) / (k[ev] * np.pi) ** 2
    return a


def cusp_norm2(H):
    J = lambda p: (p / 2 + 2 ** (-p - 1)) / ((p + 1) * (p + 2))
    return H * J(2 * H - 1) - J(2 * H) + 2 ** (-2 * H - 1) / (2 * H + 2)


def inputs(H):
    return {
        'exp(t)': (lambda N: MO.Exponential(1.0).cosine(N), MO.Exponential(1.0).norm2(H)),
        'sin(140 pi t)': (lambda N: MO.Sine(140 * np.pi).cosine(N), MO.Sine(140 * np.pi).norm2(H)),
        'sin(1000 pi t)': (lambda N: MO.Sine(1000 * np.pi).cosine(N), MO.Sine(1000 * np.pi).norm2(H)),
        '1[0,0.3)': (lambda N: MO.Indicator(0.3).cosine(N), 0.3 ** (2 * H)),
        '|t-1/2|': (cusp_coeffs, cusp_norm2(H)),
        't^-0.25': (lambda N: G.power_cosine_moments(-0.25, N), G.poly_poly(H, -0.25, -0.25)),
    }


def run(Hs=(0.55, 0.75, 0.95), dense_n=(63, 127, 255, 511, 1023, 2047, 4095),
        pcg_n=(8191, 16383, 32767), out='S1_scaling_study.json'):
    path = os.path.join(RES, out)
    data = {'study': 'S1 scaling of the stable double-precision covariance-Fourier engine',
            'note': 'D = n + 2 (n cosines + t, t^2). Inputs via exact classical coefficients; errors via e^T G e.',
            'rows': [], 'setups': []}
    t_start = time.perf_counter()
    for H in Hs:
        cache = {}
        for n, solver in [(n, 'cholesky') for n in dense_n] + [(n, 'pcg') for n in pcg_n]:
            Naux = max(1 << 18, 1 << int(math.ceil(math.log2(64 * (n + 1)))))
            Naux = min(Naux, 1 << 21)
            t0 = time.perf_counter()
            eng = EN.CovarianceFourier(H, n, 1.0, enrich=True, Naux=Naux, solver=solver)
            eng.record_cosine_only = True
            setup = time.perf_counter() - t0
            srow = {'H': H, 'n': n, 'D': n + 2, 'solver': solver, 'Naux': Naux, 'setup_seconds': setup,
                    'cond_cosine_gram': eng.cond_cosine, 'schur_tail_S': eng.S.tolist(),
                    'schur_subtractive': eng.schur_subtractive.tolist(),
                    'moment_truncation_error': eng.moment_truncation_error}
            if solver == 'pcg':
                srow['pcg_iterations_setup'] = list(eng._pcg.iterations)
            data['setups'].append(srow)
            for name, (af, n2) in inputs(H).items():
                key = (name, Naux)
                if key not in cache:
                    cache[key] = af(Naux)
                a = cache[key]
                t1 = time.perf_counter(); r = eng.project(a); tp = time.perf_counter() - t1
                row = {'H': H, 'n': n, 'D': n + 2, 'solver': solver, 'Naux': Naux, 'input': name,
                       'norm2_exact': n2, 'norm2_aux': r['norm2_aux'],
                       'error2_enriched': r['error2'], 'error2_cosine_only': r['error2_cosine_only'],
                       'error2_subtractive': n2 - r['energy'],
                       'rel_rms_enriched': math.sqrt(max(r['error2'], 0) / n2),
                       'rel_rms_cosine_only': math.sqrt(max(r['error2_cosine_only'], 0) / n2),
                       'project_seconds': tp}
                if solver == 'pcg':
                    row['pcg_iterations_last'] = eng._pcg.iterations[-2:]
                data['rows'].append(row)
            data['seconds'] = time.perf_counter() - t_start
            with open(path, 'w') as fh:
                json.dump(data, fh, indent=1)
            print('H', H, 'n', n, solver, 'setup %.2fs' % setup, 'elapsed %.0fs' % data['seconds'], flush=True)
            del eng
    data['status'] = 'PASS'
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=1)
    return data


if __name__ == '__main__':
    run()
