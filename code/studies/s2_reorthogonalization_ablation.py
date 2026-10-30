"""S2 supplement: ablation of the re-orthogonalization step for the two residual directions (Section 5, step 2).

LargeProjector(H, n, N, reorth=False) skips the second solve; the orthogonality defect max |(G d)[:n]| of the normalized
residual directions is recorded with and without it, for H in {0.55, 0.75, 0.95}, n in {4095, 32767} and
N = min(32 (n + 1), 2^20) as in S2_callable_scaling.  Writes results/S2_reorthogonalization_ablation.json (about 3 minutes).
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from covfourier.large import LargeProjector             # noqa: E402
RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')


def run(Hs=(0.55, 0.75, 0.95), ns=(4095, 32767), out='S2_reorthogonalization_ablation.json'):
    data = {'note': __doc__, 'rows': []}
    for H in Hs:
        for n in ns:
            N = min(32 * (n + 1), 1 << 20)
            row = {'H': H, 'n': n, 'D': n + 2, 'N': N}
            for flag in (False, True):
                t0 = time.perf_counter(); lp = LargeProjector(H, n, N, reorth=flag)
                row['defect_reorth' if flag else 'defect_single_solve'] = lp.orthogonality_defect
                row['setup_s_reorth' if flag else 'setup_s_single_solve'] = time.perf_counter() - t0
            data['rows'].append(row)
            print('H %.2f n %5d defect single solve %.2e  with re-orthogonalization %.2e' % (H, n, row['defect_single_solve'], row['defect_reorth']), flush=True)
            with open(os.path.join(RES, out), 'w') as fh:
                json.dump(data, fh, indent=1)
    return data


if __name__ == '__main__':
    run()
