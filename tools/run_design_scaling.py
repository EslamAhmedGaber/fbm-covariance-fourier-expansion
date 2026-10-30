"""Portable replacement for the development-machine scaling batch; appends results."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPANSION = [(10, 1), (5, 1), (2, .5), (1, .25), (.5, .25), (.5, .125)]
DENSE = [(10, 1, 1600), (5, 1, 1600), (2, .5, 1600)]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    env = dict(os.environ)
    env.setdefault('OPENBLAS_NUM_THREADS', '4')
    env.setdefault('OMP_NUM_THREADS', '4')
    app = ROOT / 'code/applications/app6_design_scaling.py'
    for mode, configs in [('expansion', EXPANSION), ('dense', DENSE)]:
        for cfg in configs:
            cmd = [sys.executable, '-u', str(app), mode, *map(str, cfg)]
            print(mode, *cfg, flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=app.parent, env=env, check=True)

if __name__ == '__main__':
    main()
