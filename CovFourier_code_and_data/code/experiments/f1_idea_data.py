"""Data for Figure 1 (the idea): one path and its observations, the covariance sandwich, residual energies,
and the evidence bracket against D for one data set.  Writes results/F1_idea.json."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import C, np, save_json                      # noqa: E402
import e6_scale as E6                                    # noqa: E402


def main():
    H = 0.75
    rng = np.random.default_rng(2026)
    B = E6.davies_harte_path(H, 1 << 12, rng)
    tg = np.linspace(0, 1, B.size)
    # observations: 300 windows (for the sandwich and the bracket) and a few shown on the path
    m = 300
    c = rng.uniform(0.03, 0.97, m); w = rng.uniform(0.005, 0.06, m)
    a = np.clip(c - w / 2, 0, 1); b = np.clip(c + w / 2, 0, 1)
    des = C.Boxes(a, b)
    K = des.K(H)
    sd = float(np.sqrt(np.mean(np.diag(K)))); tau = 0.03 * sd
    y = E6.window_averages(B, a, b) + tau * rng.standard_normal(m)
    lamK = np.sort(np.linalg.eigvalsh(K))[::-1]
    bas = C.CFBasis(H, 32)
    f = C.cf_features(des, bas)
    mu = np.sort(np.linalg.eigvalsh(f.Gamma @ f.Gamma.T))[::-1]
    e2_by_n = {}
    for n in (16, 32, 64):
        fb = C.cf_features(des, C.CFBasis(H, n))
        e2_by_n[str(n + 2)] = fb.e2.tolist()
    ex = float(C.exact_evidence(K, y[:, None], tau ** 2)[0])
    br_rows = []
    for n in (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
        bs = C.CFBasis(H, n); fb = C.cf_features(des, bs)
        br = C.bracket(fb, y[:, None], tau ** 2, majorant=C.CFMajorant(des, bs, fb))
        br_rows.append({'D': bs.D, 'L': float(br['Ld'][0]), 'U': float(br['U'][0]), 'rho': br['rho']})
    pts = np.sort(rng.uniform(0.05, 0.95, 6))
    shown = np.argsort(c)[::50][:6]
    save_json({'H': H, 'path_t': tg[::8].tolist(), 'path_B': B[::8].tolist(),
               'points_t': pts.tolist(), 'points_B': np.interp(pts, tg, B).tolist(),
               'windows_shown': [[float(a[i]), float(b[i]), float(y[i])] for i in shown],
               'eig_K': lamK[:120].tolist(), 'eig_Q': mu[:120].tolist(), 'trR_D34': f.trR,
               'widths': w.tolist(), 'e2_by_D': e2_by_n, 'exact': ex, 'bracket': br_rows, 'tau': tau}, 'F1_idea.json')


if __name__ == '__main__':
    main()
