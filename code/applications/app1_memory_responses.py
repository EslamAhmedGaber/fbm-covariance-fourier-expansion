"""A1: shared-noise multi-output linear memory responses (stochastic convolution at the terminal time).

Outputs  Y_i = int_0^1 k_i(1 - s) dB_H(s),  i = 1..q,  driven by ONE fBm path (shared noise).
Families
  benign : 40 damped oscillators k(tau) = exp(-zeta w tau) sin(w_d tau)/w_d, w in [1, 200],
           zeta cycling (0.05, 0.2, 0.5);  40 relaxations k(tau) = exp(-lam tau), lam in [0.1, 100]
           (terminal values of fractional Ornstein-Uhlenbeck-type filters).
  stiff  : 40 modes of the stochastic heat equation u_t = kappa u_xx + Bdot_H(t) (Dirichlet),
           u_k(1) = int exp(-lam_k (1-s)) dB_H(s), lam_k = kappa (k pi)^2, kappa = 0.1 (lam up to 1579).
Exact reference: closed-form bilinear exponential moments (comparators.ExpFamily, 30 digits).
Methods (all receive only vectorized callables / values; none uses the closed form):
  CF   covariance-Fourier engine (this work): DCT stage + covariance-orthonormal projection,
       enriched with t, t^2.  Shared coordinates Z (q x D):  Sigma_hat = Z Z^T, samples Y = Z xi.
       Certified: Sigma - Sigma_hat = R R^T >= 0 and |Sigma - Sigma_hat|_ij <= E_i E_j.
  TGJ  two-triangle Gauss-Jacobi product quadrature of the double integral (pairwise).
  CT   cell averages on N uniform cells + exact Toeplitz increment covariance (FFT).
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from covfourier.engine import CovarianceFourier                                   # noqa: E402
from covfourier.batch import BatchProjector                                       # noqa: E402
from covfourier.comparators import ExpFamily, triangle_gj_covariance, cell_toeplitz_covariance  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')


def family(name):
    if name == 'benign':
        w = np.geomspace(1.0, 200.0, 40)
        z = np.resize([0.05, 0.2, 0.5], 40)
        wd = w * np.sqrt(1 - z ** 2)
        a = np.r_[z * w - 1j * wd, np.geomspace(0.1, 100.0, 40)]
        c = np.r_[-1j / wd, np.ones(40)]
    elif name == 'stiff':
        k = np.arange(1, 41)
        a = 0.1 * (k * np.pi) ** 2 + 0j
        c = np.ones(40, dtype=complex)
    else:
        raise ValueError(name)
    return ExpFamily(a, c, s0=1.0)


def rel_err(K, Kex):
    d = np.sqrt(np.diag(Kex))
    return (float(np.linalg.norm(K - Kex) / np.linalg.norm(Kex)),
            float(np.max(np.abs(K - Kex) / np.outer(d, d))))


def warm_tables(H):
    # One-time, per-H table construction (30-digit Chebyshev tables of the Kummer function in covfourier.kummer,
    # about 1 s per Hurst index in a fresh process).  It is shared by all n, N and inputs and is excluded from the
    # CF timings, as in the reported runs (which were executed after these tables had been built).
    CovarianceFourier(H, 16, enrich=True, Naux=1 << 10)


def run_cf(fam, H, Kex, ns=(16, 32, 64, 128, 256, 512, 1024, 2048), Nauxs=(1 << 10, 1 << 12, 1 << 14),
           rule='a_posteriori', max_ratio=1e3):
    """Covariance-Fourier with the batched, endpoint-corrected Fourier stage (covfourier.batch)."""
    # rule / max_ratio select the endpoint-correction rule (covfourier.batch.endpoint_corrected_dct).  The first
    # A1 run (run() below) used the fixed switch rule='ratio', max_ratio=100; app1_rerun_cf_v2.py uses the default.
    out = []
    q = len(fam)
    for Naux in Nauxs:
        t = (np.arange(Naux) + 0.5) / Naux
        t0 = time.perf_counter()
        F = fam.values(t)                                  # (q, Naux) midpoint samples, once per Naux
        t_samp_vals = time.perf_counter() - t0
        for n in ns:
            if Naux < 8 * n:
                continue
            t0 = time.perf_counter()
            cf = CovarianceFourier(H, n, enrich=True, Naux=Naux)
            bp = BatchProjector(cf, max_ratio=max_ratio, rule=rule)
            t_prep = time.perf_counter() - t0
            t0 = time.perf_counter()
            a_g, c = bp.fourier_stage(F)
            t_dct = time.perf_counter() - t0 + t_samp_vals
            t0 = time.perf_counter()
            r = bp.project(a_g, c, norms=True)
            t_proj = time.perf_counter() - t0
            Z = r['coords'].T
            E2 = np.clip(r['error2'], 0.0, None) + r['eta']
            K = Z @ Z.T
            fro, mx = rel_err(K, Kex)
            Dm = Kex - K
            E = np.sqrt(E2)
            viol = float(np.max(np.abs(Dm) - np.outer(E, E)))
            lam_min = float(np.linalg.eigvalsh(0.5 * (Dm + Dm.T)).min())
            # joint sampling cost: 10^4 draws of all q outputs
            xi = np.random.default_rng(0).standard_normal((cf.D, 10_000))
            t0 = time.perf_counter(); Y = Z @ xi; t_samp = time.perf_counter() - t0
            out.append({'n': n, 'D': cf.D, 'Naux': Naux, 'rel_fro': fro, 'max_corr_err': mx,
                        'prep_s': t_prep, 'dct_s': t_dct, 'project_s': t_proj,
                        'per_output_s': (t_dct + t_proj) / q, 'total_s': t_prep + t_dct + t_proj,
                        'sample_1e4_s': t_samp,
                        'certificate': {'max(|dK| - E_i E_j)': viol, 'lambda_min(K - Khat)': lam_min,
                                        'max E_i^2 / K_ii': float(np.max(E2 / np.diag(Kex)))}})
            print('  CF n %5d Naux %6d fro %.2e prep %.2fs proj %.2fs viol %.1e lam_min %.1e' %
                  (n, Naux, fro, t_prep, t_proj, viol, lam_min), flush=True)
            del cf, Z, Y
    return out


def run_tgj(fam, H, Kex, ns=(8, 16, 32, 64, 128, 256, 512, 1024)):
    out = []
    for n in ns:
        K, t = triangle_gj_covariance(fam.values, H, n)
        fro, mx = rel_err(K, Kex)
        t0 = time.perf_counter()
        w, V = np.linalg.eigh(K)
        keep = w > 1e-14 * w.max()
        F = V[:, keep] * np.sqrt(w[keep])
        xi = np.random.default_rng(0).standard_normal((F.shape[1], 10_000))
        Y = F @ xi
        t_samp = time.perf_counter() - t0
        out.append({'n': n, 'rel_fro': fro, 'max_corr_err': mx, 'total_s': t, 'per_output_s': t / len(fam),
                    'sample_1e4_s_incl_factor': t_samp, 'min_eig_Khat': float(w.min())})
        print('  TGJ n %5d fro %.2e time %.3fs min eig %.1e' % (n, fro, t, w.min()), flush=True)
    return out


def run_ct(fam, H, Kex, Ns=tuple(1 << k for k in range(6, 17))):
    out = []
    for N in Ns:
        K, t = cell_toeplitz_covariance(fam.values, H, N)
        fro, mx = rel_err(K, Kex)
        out.append({'N': N, 'rel_fro': fro, 'max_corr_err': mx, 'total_s': t, 'per_output_s': t / len(fam)})
        print('  CT N %6d fro %.2e time %.3fs' % (N, fro, t), flush=True)
    return out


def matched(rows, key='total_s', targets=(1e-3, 1e-6, 1e-9)):
    res = {}
    for tau in targets:
        ok = [r for r in rows if r['rel_fro'] <= tau]
        res['%.0e' % tau] = min(ok, key=lambda r: r[key]) if ok else None
    return res


def run(Hs=(0.55, 0.75, 0.95), fams=('benign', 'stiff'), out='A1_memory_responses.json'):
    data = {'note': __doc__, 'results': {}}
    T0 = time.perf_counter()
    for name in fams:
        fam = family(name)
        for H in Hs:
            t0 = time.perf_counter()
            Kex = fam.exact_covariance(H, dps=30)
            t_ex = time.perf_counter() - t0
            print(name, H, 'exact reference %.1fs' % t_ex, 'cond %.2e' % np.linalg.cond(Kex), flush=True)
            warm_tables(H)                                     # untimed, see warm_tables
            if name == 'stiff':
                cf = run_cf(fam, H, Kex, ns=(64, 128, 256, 512, 1024, 2048), Nauxs=(1 << 14, 1 << 16),
                            rule='ratio', max_ratio=100.0)
                tgj = run_tgj(fam, H, Kex, ns=(32, 64, 128, 256, 512, 1024))
                ct = run_ct(fam, H, Kex, Ns=tuple(1 << k for k in range(8, 18)))
            else:
                cf = run_cf(fam, H, Kex, rule='ratio', max_ratio=100.0)
                tgj = run_tgj(fam, H, Kex); ct = run_ct(fam, H, Kex)
            data['results'][f'{name}_{H}'] = {
                'q': len(fam), 'exact_reference_s': t_ex, 'CF': cf, 'TGJ': tgj, 'CT': ct,
                'matched': {'CF': matched(cf), 'TGJ': matched(tgj), 'CT': matched(ct)}}
            with open(os.path.join(RES, out), 'w') as fh:
                json.dump(data, fh, indent=1)
    data['seconds_total'] = time.perf_counter() - T0
    with open(os.path.join(RES, out), 'w') as fh:
        json.dump(data, fh, indent=1)
    return data


if __name__ == '__main__':
    run()
