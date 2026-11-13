"""E6 scale: the evidence bracket streamed over chunks of box-window observations with exact moments,
m in {1.6e4, 6.4e4, 1.28e5, 5.12e5, 1e6} (nested prefixes of one design), D in {66, 130, 258}.

Memory: O(chunk * N + D^2 + m) (the per-observation coordinates are written to a scratch folder between the
two passes; the second pass gives v = C^{-1} y and the per-vector upper bounds U1, U2).
Data: one fBm path per data set by the Davies-Harte method on a 2^21 grid; window averages of the piecewise
linear interpolant; white noise tau = 0.03 sd_rms.

Usage: python e6_scale.py SCRATCH_DIR [H]  ->  results/E6_scale.json
"""
import ctypes
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import C, np, save_json, Log                                    # noqa: E402
from covfourier.batch import gram_matmat                                    # noqa: E402
from scipy.linalg import cholesky, cho_solve                                # noqa: E402

MS = (16000, 64000, 128000, 512000, 1000000)
NS = (64, 128, 256)
CHUNK = 4000
NDATA = 3
F = 0.03
NGRID_PATH = 1 << 21


def peak_memory_gib():
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t)]
    k32 = ctypes.windll.kernel32; k32.GetCurrentProcess.restype = wintypes.HANDLE
    ps = ctypes.windll.psapi
    ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    ps.GetProcessMemoryInfo.restype = wintypes.BOOL
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2 ** 30, c.WorkingSetSize / 2 ** 30


def davies_harte_path(H, n, rng):
    """B_H at t_j = j/n, j = 0..n (exact joint law on the grid)."""
    k = np.arange(n + 1, dtype=float)
    g = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    c = np.r_[g, g[-2:0:-1]]                               # circulant first row, length 2n
    lam = np.fft.fft(c).real
    if lam.min() < -1e-10 * lam.max():
        raise RuntimeError('circulant embedding not PSD')
    lam = np.maximum(lam, 0.0)
    N2 = 2 * n
    Z = rng.standard_normal(N2) + 1j * rng.standard_normal(N2)
    X = np.fft.fft(np.sqrt(lam / N2) * Z)[:n].real          # fGn with unit-spacing variance 1
    B = np.r_[0.0, np.cumsum(X)] * (1.0 / n) ** H
    return B


def window_averages(B, a, b):
    """(1/w) int_a^b of the piecewise-linear interpolant of the grid path B (grid spacing 1/n)."""
    n = B.size - 1
    h = 1.0 / n
    I = np.r_[0.0, np.cumsum(0.5 * (B[1:] + B[:-1]) * h)]   # int_0^{t_j}

    def prim(x):
        j = np.minimum((x * n).astype(np.int64), n - 1)
        dx = x - j * h
        return I[j] + B[j] * dx + 0.5 * (B[j + 1] - B[j]) * dx ** 2 / h

    return (prim(b) - prim(a)) / (b - a)


def main(scratch, H=0.75):
    log = Log('E6.log')
    os.makedirs(scratch, exist_ok=True)
    mmax = MS[-1]
    rng = np.random.default_rng(106)
    c = rng.uniform(0.03, 0.97, mmax); w = rng.uniform(0.005, 0.06, mmax)
    a = np.clip(c - w / 2, 0, 1); b = np.clip(c + w / 2, 0, 1)
    des_all = C.Boxes(a, b)
    sd = math.sqrt(float(np.mean(des_all.norms(H)[:MS[0]])))
    tau = F * sd; tau2 = tau ** 2
    t0 = time.perf_counter()
    Y = np.empty((mmax, NDATA))
    for j in range(NDATA):
        B = davies_harte_path(H, NGRID_PATH, np.random.default_rng(1060 + j))
        Y[:, j] = window_averages(B, a, b)
        del B
    Y += tau * np.random.default_rng(1069).standard_normal(Y.shape)
    t_sim = time.perf_counter() - t0
    log('simulated %d windows x %d data sets in %.1fs; tau %.4e' % (mmax, NDATA, t_sim, tau))
    t0 = time.perf_counter()
    bases = {n: C.CFBasis(H, n) for n in NS}
    t_prep = {n: bases[n].setup_seconds for n in NS}
    Nmom = 4 * NS[-1]
    acc = {n: {'GG': np.zeros((n + 2, n + 2)), 'GY': np.zeros((n + 2, NDATA)), 'trR': 0.0, 'l1p': 0.0, 'nneg': 0}
           for n in NS}
    yy = np.zeros(NDATA)
    snap = {}
    tim = {'moments': 0.0, 'proj': {n: 0.0 for n in NS}, 'accum': {n: 0.0 for n in NS}}
    for s in range(0, mmax, CHUNK):
        e = min(mmax, s + CHUNK)
        des = des_all.subset(np.arange(s, e))
        tt = time.perf_counter()
        Bm = des.moments(H, Nmom)
        nrm = des.norms(H)
        tim['moments'] += time.perf_counter() - tt
        yc = Y[s:e]
        yy += np.einsum('ib,ib->b', yc, yc)
        for n in NS:
            tt = time.perf_counter()
            G = bases[n].coords(Bm)
            e2 = nrm - np.einsum('ij,ij->i', G, G)
            tim['proj'][n] += time.perf_counter() - tt
            tt = time.perf_counter()
            A = acc[n]
            A['nneg'] += int(np.sum(e2 < 0)); e2 = np.maximum(e2, 0.0)
            A['GG'] += G.T @ G; A['GY'] += G.T @ yc
            A['trR'] += float(e2.sum()); A['l1p'] += float(np.sum(np.log1p(e2 / tau2)))
            np.save(os.path.join(scratch, 'G_n%d_%07d.npy' % (n, s)), G)
            np.save(os.path.join(scratch, 'e2_n%d_%07d.npy' % (n, s)), e2)
            tim['accum'][n] += time.perf_counter() - tt
        if e in MS:
            snap[e] = {'yy': yy.copy(), 'time_moments': tim['moments'],
                       'time_proj': dict(tim['proj']), 'time_accum': dict(tim['accum']),
                       'acc': {n: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in acc[n].items()} for n in NS},
                       'mem_GiB': peak_memory_gib()}
            log('  pass 1 reached m = %d: moments %.0fs, peak working set %.3f GiB' % (e, tim['moments'], snap[e]['mem_GiB'][0]))
    t_pass1 = time.perf_counter() - t0
    results = []
    for m in MS:
        sn = snap[m]
        for n in NS:
            Dd = n + 2
            A = sn['acc'][n]
            tt = time.perf_counter()
            Mx = tau2 * np.eye(Dd) + A['GG']
            Lm = cholesky(Mx, lower=True)
            ldM = 2 * float(np.sum(np.log(np.diag(Lm))))
            logdetC = (m - Dd) * math.log(tau2) + ldM
            wv = cho_solve((Lm, True), A['GY'])
            q = (sn['yy'] - np.einsum('jb,jb->b', A['GY'], wv)) / tau2
            Mt = (tau2 + A['trR']) * np.eye(Dd) + A['GG']
            wt = np.linalg.solve(Mt, A['GY'])
            q0 = (sn['yy'] - np.einsum('jb,jb->b', A['GY'], wt)) / (tau2 + A['trR'])
            nconst = 0.5 * m * math.log(2 * math.pi)
            L = -0.5 * logdetC - 0.5 * q - nconst - A['trR'] / (2 * tau2)
            Ld = -0.5 * logdetC - 0.5 * q - nconst - 0.5 * A['l1p']
            U0 = -0.5 * logdetC - 0.5 * q0 - nconst
            t_alg = time.perf_counter() - tt
            # pass 2: v = (y - G w)/tau2 ; |v|^T e ; Av ; G^T v ; l2 quadratic form
            tt = time.perf_counter()
            bas = bases[n]
            Nmaj = bas.Naux
            Av = np.zeros((Nmaj, NDATA)); Gv = np.zeros((Dd, NDATA)); ve = np.zeros(NDATA)
            V = np.empty((m, NDATA))
            for s in range(0, m, CHUNK):
                G = np.load(os.path.join(scratch, 'G_n%d_%07d.npy' % (n, s)))
                e2 = np.load(os.path.join(scratch, 'e2_n%d_%07d.npy' % (n, s)))
                k = min(m, s + CHUNK) - s
                G = G[:k]; e2 = e2[:k]
                vc = (Y[s:s + k] - G @ wv) / tau2
                V[s:s + k] = vc
                ve += np.abs(vc).T @ np.sqrt(e2)
                Gv += G.T @ vc
                Ac = des_all.subset(np.arange(s, s + k)).cosine(Nmaj)       # (k, N)
                Av += Ac.T @ vc
            Pc = bas.phi_coeffs(Gv.T).T
            E = Av.copy(); E[:Pc.shape[0]] -= Pc
            head = np.maximum(np.einsum('kb,kb->b', E, gram_matmat(bas.G, E)), 0.0)
            l2 = des_all.subset(np.arange(m)).l2_quad(V)
            tail = C.Lambda_J(H, Nmaj) * np.maximum(l2 - np.einsum('kb,kb->b', Av, Av), 0.0)
            beta = (np.sqrt(head) + np.sqrt(tail)) ** 2
            U1 = -0.5 * logdetC - 0.5 * (q - ve ** 2) - nconst
            U2 = -0.5 * logdetC - 0.5 * (q - beta) - nconst
            Ub = -0.5 * logdetC - 0.5 * np.maximum.reduce([q0, q - ve ** 2, q - beta]) - nconst
            t_pass2 = time.perf_counter() - tt
            del V
            row = {'m': m, 'n': n, 'D': Dd, 'H': H, 'tau': tau, 'rho': A['trR'] / tau2, 'n_negative_e2': A['nneg'],
                   'L': L.tolist(), 'Ld': Ld.tolist(), 'U0': U0.tolist(), 'U1': U1.tolist(), 'U2': U2.tolist(), 'U': Ub.tolist(),
                   'width_U_minus_Ld': (Ub - Ld).tolist(), 'width_U0_minus_Ld': (U0 - Ld).tolist(),
                   'tail_fraction': (tail / np.maximum(beta, 1e-300)).tolist(),
                   'time_moments_cum': sn['time_moments'], 'time_proj_cum': sn['time_proj'][n],
                   'time_accum_cum': sn['time_accum'][n], 'time_algebra': t_alg, 'time_pass2': t_pass2,
                   'prep_seconds': t_prep[n], 'peak_working_set_GiB_at_pass1': sn['mem_GiB'][0],
                   'dense_K_GiB': 8.0 * m * m / 2 ** 30}
            results.append(row)
            log('m %7d D %3d rho %.3g width(U-Ld) %.3g width(U0-Ld) %.3g | moments %.0fs proj %.1fs pass2 %.1fs' % (
                m, Dd, row['rho'], float(np.median(Ub - Ld)), float(np.median(U0 - Ld)), sn['time_moments'],
                sn['time_proj'][n], t_pass2))
    pk = peak_memory_gib()
    save_json({'experiment': 'E6', 'H': H, 'f': F, 'chunk': CHUNK, 'ndata': NDATA, 'path_grid': NGRID_PATH,
               'time_simulation': t_sim, 'time_pass1_total': t_pass1, 'rows': results,
               'peak_working_set_GiB_total': pk[0]}, 'E6_scale.json')
    log('done; peak working set %.3f GiB' % pk[0])


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.75)
