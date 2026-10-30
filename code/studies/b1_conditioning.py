"""Study B1: conditioning of the polynomial-augmented raw Gram matrix A_D (double precision).

A_D is the covariance Gram of phi_0 = T^-1/2, phi_1 = t/T, phi_2 = (t/T)^2, phi_{k+2} = h_k, from closed forms.
Eigenvalues by LAPACK in double precision; condition numbers approaching 1/eps are indicative only.
Output: results/B1_augmented_conditioning.json  (optional argument: other output path)
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
OUT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None
from scipy.linalg import cholesky, solve_triangular
from covfourier import gram as G

rows = []
for H in [0.55, 0.75, 0.95]:
    for D in [17, 33, 65, 129, 257, 513, 1025, 2049]:
        A = G.augmented_gram(H, D, 1.0)
        ev = np.linalg.eigvalsh(A)
        try:
            C = cholesky(A, lower=True); ok = True
            Ci = solve_triangular(C, np.eye(D), lower=True)
            defect = float(np.linalg.norm(Ci @ A @ Ci.T - np.eye(D), 2)); diagC = np.diag(C); invnorm = float(np.linalg.norm(Ci, 2))
        except Exception:
            ok = False; defect = None; diagC = np.array([np.nan]); invnorm = None
        rows.append({'H': H, 'D': D, 'eig_min': float(ev[0]), 'eig_max': float(ev[-1]),
                     'cond': float(ev[-1] / ev[0]) if ev[0] > 0 else None, 'chol_ok': ok, 'self_defect': defect,
                     'min_diag_C': float(np.nanmin(diagC)), 'norm_Cinv': invnorm})
        print(H, D, 'cond %.3e' % (ev[-1] / max(ev[0], 1e-300)), flush=True)
json.dump({'study': 'conditioning of augmented nested Gram A_D (double precision)', 'rows': rows},
          open(OUT or os.path.join(RES, 'B1_augmented_conditioning.json'), 'w'), indent=1)
