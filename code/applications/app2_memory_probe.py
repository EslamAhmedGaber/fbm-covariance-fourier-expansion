"""A2 supplement: peak memory of the covariance-Fourier posterior at m = 128 000 sensors (H = 0.75, n = 256).

Repeats the CF-only branch of app2_noisy_reconstruction_v2.run() for one m with the same seed and settings and
records the peak working set of the process (Windows: GetProcessMemoryInfo; elsewhere: getrusage ru_maxrss) and the
tracemalloc peak.  Writes results/A2_memory_probe_m<m>.json.  Usage: python app2_memory_probe.py [m]   (about 4 min)
"""
import json
import os
import sys
import time
import tracemalloc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')

import numpy as np                                                    # noqa: E402
import app2_noisy_reconstruction_v2 as A                              # noqa: E402
from covfourier.engine import CovarianceFourier                       # noqa: E402
from covfourier.batch import BatchProjector                           # noqa: E402


def peak_memory_gib():
    """Peak resident memory of this process in GiB (working set on Windows, ru_maxrss elsewhere)."""
    if os.name == 'nt':
        import ctypes
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
        if not ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            raise OSError('GetProcessMemoryInfo failed')
        return c.PeakWorkingSetSize / 2 ** 30, c.PeakPagefileUsage / 2 ** 30
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == 'darwin' else 1024                 # bytes on macOS, KiB on Linux
    return r * scale / 2 ** 30, None


def main(m=128000):
    base = peak_memory_gib()[0]
    tracemalloc.start()
    rng = np.random.default_rng(7 + m)                              # same seed as the CF-only branch of the A2 driver
    c, s = A.draw_windows(rng, m + A.N_NEW)
    t0 = time.perf_counter(); cf = CovarianceFourier(0.75, 256, enrich=True, Naux=8 * 256); bp = BatchProjector(cf); tp = time.perf_counter() - t0
    t0 = time.perf_counter(); Z, E2s = A.cf_coords(bp, c[:m], s[:m]); tc = time.perf_counter() - t0
    t0 = time.perf_counter(); Wt, E2t = A.cf_targets(bp, c[m:], s[m:]); tt = time.perf_counter() - t0
    y = np.random.default_rng(3).standard_normal(m) * 0.5
    t0 = time.perf_counter(); A.cf_posterior(Z, E2s, y, Wt, E2t); tu = time.perf_counter() - t0
    _, tm_peak = tracemalloc.get_traced_memory()
    pw, pp = peak_memory_gib()
    rec = {'m': m, 'n': 256, 'D': int(Z.shape[1]), 'N_aux_grid': int(bp.N), 'total_s': tp + tc + tt + tu, 'coords_s': tc,
           'peak_working_set_GiB': pw, 'peak_commit_GiB': pp, 'baseline_working_set_GiB_after_imports': base,
           'tracemalloc_peak_GiB': tm_peak / 2 ** 30, 'Z_array_GiB': Z.nbytes / 2 ** 30, 'dense_K_GiB': 8 * m * m / 2 ** 30,
           'sum_E2_over_tau2': float(E2s.sum() / A.TAU ** 2), 'python': sys.version.split()[0], 'numpy': np.__version__,
           'platform': sys.platform}
    with open(os.path.join(RES, 'A2_memory_probe_m%d.json' % m), 'w') as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 128000)
