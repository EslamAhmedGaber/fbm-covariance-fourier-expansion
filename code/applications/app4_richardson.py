"""A4 supplement: Richardson extrapolation in J of the Galerkin KL eigenvalues (J = 1024, 2048, 4096).

Observed order p from successive differences; extrapolated mu_R = mu_4096 + (mu_4096 - mu_2048)/(2^3 - 1);
nu_n = (c_H/mu_R)^(1/(2H+1)) against the two-term law nu_hat_n (Chigansky-Kleptsyna); reports n (nu_n - nu_hat_n).
Needs results/A4_eigs_J4096.npz from app4_analysis.py.  Output: results/A4_richardson.json
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
OUT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None
import app4_kl_spectrum as A4
import app4_analysis as A4X

arr = np.load(os.path.join(RES, 'A4_eigs_J4096.npz'))
rich = {}
n = np.arange(1, 1001, dtype=float)
for H in (0.5, 0.55, 0.65, 0.75, 0.85, 0.95):
    m1, _ = A4X.galerkin_eigs(H, 1024); m2, _ = A4X.galerkin_eigs(H, 2048)
    m4 = arr['mu_' + str(H).replace('.', 'p')]
    i1 = (m2[:1000] - m1[:1000]); i2 = (m4[:1000] - m2[:1000])
    p = np.log2(i1 / i2)
    mR = m4[:1000] + i2 / (2 ** 3 - 1)
    nu = (A4X.cH(H) / mR) ** (1 / (2 * H + 1)); rho = nu - A4.ck_nu(H, n)
    rich[H] = {'p_at': {k: float(p[k - 1]) for k in (50, 100, 300, 1000)},
               'n_rho_R': {k: float(rho[k - 1] * k) for k in (10, 40, 100, 200, 300, 500, 700, 1000)}}
    if H == 0.5:
        ex = 1 / ((n - 0.5) * np.pi) ** 2
        rich[H]['bm_err_R'] = {k: float(abs(mR[k - 1] - ex[k - 1]) / ex[k - 1]) for k in (100, 300, 500, 1000)}
    print(H, {k: round(v, 4) for k, v in rich[H]['n_rho_R'].items()}, flush=True)
json.dump({str(k): v for k, v in rich.items()}, open(OUT or os.path.join(RES, 'A4_richardson.json'), 'w'), indent=1)
