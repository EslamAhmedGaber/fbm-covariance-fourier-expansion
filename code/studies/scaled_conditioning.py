"""Condition number of the scaled Schur block M = W^{-1/2} S_+ W^{-1/2} (Corollary 5) at the dimensions
of the raw-Gram study B1 (D = n + 2 = 17 ... 2049), H = 0.55, 0.75, 0.95, T = 1.  Dense eigenvalues."""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(HERE, '..', 'covfourier')):          # inside the repository: code/studies/
    REPO = os.path.dirname(os.path.dirname(HERE)); ROOT = REPO
    RES = os.path.join(REPO, 'results'); TABLES = os.path.join(RES, 'law_tables')
else:                                                                # outside the repository: set COVFOURIER_REPO to its root
    ROOT = os.path.dirname(HERE); REPO = os.environ.get('COVFOURIER_REPO', os.path.dirname(ROOT))
    RES = os.path.join(ROOT, 'results'); TABLES = os.path.join(ROOT, 'manuscript')
sys.path.insert(0, os.path.join(REPO, 'code'))
from covfourier.gram import CosineGram
rows = []
for H in (0.55, 0.75, 0.95):
    cg = CosineGram(H, 1.0)
    G = cg.dense(2048)
    cHH = math.gamma(2 * H + 1) * math.sin(math.pi * H)
    for D in (17, 33, 65, 129, 257, 513, 1025, 2049):
        n = D - 2
        Gp = G[1:n + 1, 1:n + 1]; g = G[0, 1:n + 1]
        S = Gp - np.outer(g, g) / G[0, 0]
        w = cHH * (np.arange(1, n + 1) * np.pi) ** (1 - 2 * H)
        Mm = S / np.sqrt(np.outer(w, w))
        ev = np.linalg.eigvalsh(Mm)
        rows.append({'H': H, 'D': D, 'n': n, 'min_eig': float(ev[0]), 'max_eig': float(ev[-1]), 'cond': float(ev[-1] / ev[0])})
        print(rows[-1], flush=True)
json.dump({'study': __doc__, 'rows': rows}, open(os.path.join(RES, 'L6_scaled_conditioning.json'), 'w'), indent=1)
