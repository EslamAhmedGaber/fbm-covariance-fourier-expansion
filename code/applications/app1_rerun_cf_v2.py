"""A1: re-run of the covariance-Fourier (CF) curves with the a-posteriori endpoint-correction rule.

Engine change against the first A1 run (batch.py): the a-posteriori endpoint-correction rule (tail-energy test and
|p| <= 1e3 |f|) replaces the fixed max_ratio = 100 switch; the first CF curves are kept as CF_fixed_switch. TGJ and CT
results are unchanged and are copied from A1_memory_responses.json. Output: A1_memory_responses_v2.json."""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import app1_memory_responses as A1        # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')


def run():
    old = json.load(open(os.path.join(RES, 'A1_memory_responses.json')))
    new = {'note': __doc__, 'results': {}}
    for key, rec in old['results'].items():
        name, H = key.split('_'); H = float(H)
        fam = A1.family(name)
        Kex = fam.exact_covariance(H, dps=30)
        A1.warm_tables(H)                      # untimed one-time per-H tables, see A1.warm_tables
        # default endpoint rule (a-posteriori, max_ratio = 1e3); the fixed switch is rule='ratio', max_ratio=100
        if name == 'stiff':
            cf = A1.run_cf(fam, H, Kex, ns=(64, 128, 256, 512, 1024, 2048), Nauxs=(1 << 14, 1 << 16))
        else:
            cf = A1.run_cf(fam, H, Kex)
        rr = dict(rec); rr['CF_fixed_switch'] = rec['CF']; rr['CF'] = cf
        rr['matched'] = {'CF': A1.matched(cf), 'CF_fixed_switch': A1.matched(rec['CF']), 'TGJ': rec['matched']['TGJ'], 'CT': rec['matched']['CT']}
        new['results'][key] = rr
        with open(os.path.join(RES, 'A1_memory_responses_v2.json'), 'w') as fh:
            json.dump(new, fh, indent=1)
    return new


if __name__ == '__main__':
    run()
