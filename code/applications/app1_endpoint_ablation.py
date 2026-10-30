"""A1 supplement: ablation of the endpoint correction (Section 5, step 4) on the smooth multi-output family.

Benign family of app1_memory_responses (80 kernels), H in {0.55, 0.75, 0.95}, auxiliary lengths N in {1024, 4096, 16384}.
For each N the best relative Frobenius covariance error over n is recorded (a) without the correction (rule='ratio',
max_ratio=0: never applied, i.e. a plain midpoint discrete cosine transform) and (b) with the a-posteriori rule (default).
Writes results/A1_endpoint_ablation.json (about a minute).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import app1_memory_responses as A1        # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')


def run(Hs=(0.55, 0.75, 0.95), Ns=(1 << 10, 1 << 12, 1 << 14), out='A1_endpoint_ablation.json'):
    fam = A1.family('benign')
    data = {'note': __doc__, 'rows': []}
    for H in Hs:
        Kex = fam.exact_covariance(H, dps=30)
        A1.warm_tables(H)
        for N in Ns:
            ns = tuple(n for n in (16, 32, 64, 128, 256, 512, 1024, 2048) if N >= 8 * n)
            plain = A1.run_cf(fam, H, Kex, ns=ns, Nauxs=(N,), rule='ratio', max_ratio=0.0)
            corr = A1.run_cf(fam, H, Kex, ns=ns, Nauxs=(N,))
            row = {'H': H, 'N': N, 'best_rel_fro_plain': min(r['rel_fro'] for r in plain),
                   'best_rel_fro_corrected': min(r['rel_fro'] for r in corr),
                   'plain': [{'n': r['n'], 'rel_fro': r['rel_fro']} for r in plain],
                   'corrected': [{'n': r['n'], 'rel_fro': r['rel_fro']} for r in corr]}
            data['rows'].append(row)
            print('H %.2f N %6d best plain %.2e corrected %.2e' % (H, N, row['best_rel_fro_plain'], row['best_rel_fro_corrected']), flush=True)
            with open(os.path.join(RES, out), 'w') as fh:
                json.dump(data, fh, indent=1)
    return data


if __name__ == '__main__':
    run()
