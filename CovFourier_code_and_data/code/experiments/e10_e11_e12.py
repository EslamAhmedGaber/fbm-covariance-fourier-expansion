"""E10 stress, E11 precision, E12 cost model (PROTOCOL.md).

Usage: python e10_e11_e12.py E10|E11|E12  ->  results/E10_stress.json, E11_precision.json, E12_cost_model.json
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import C, np, make_design, save_json, Log, qstats, RESULTS, SEED0     # noqa: E402
from scipy.linalg import cholesky, eigh                                           # noqa: E402


class Increments:
    """Interval increments Y_i = B_H(b_i) - B_H(a_i) (integrand 1_[a_i, b_i)); exact through point formulas."""
    kind = 'increment'

    def __init__(self, a, b):
        self.a, self.b = np.asarray(a, float), np.asarray(b, float)
        self.m = self.a.size; self.T = 1.0
        self.Pa, self.Pb = C.Points(self.a), C.Points(self.b)

    def loc(self):
        return 0.5 * (self.a + self.b)

    def norms(self, H):
        return (self.b - self.a) ** (2 * H)

    def K(self, H):
        r = lambda s, t: 0.5 * (s[:, None] ** (2 * H) + t[None, :] ** (2 * H) - np.abs(s[:, None] - t[None, :]) ** (2 * H))
        return r(self.b, self.b) - r(self.b, self.a) - r(self.a, self.b) + r(self.a, self.a)

    def moments(self, H, N):
        return self.Pb.moments(H, N) - self.Pa.moments(H, N)

    def cosine(self, N):
        return self.Pb.cosine(N) - self.Pa.cosine(N)

    def l2_quad(self, V):
        V = np.asarray(V, float)
        if V.ndim == 1:
            V = V[:, None]
        p = np.unique(np.r_[0.0, 1.0, self.a, self.b])
        # g(u) = sum_i v_i 1[a_i <= u < b_i] is piecewise constant between breakpoints
        mid = 0.5 * (p[:-1] + p[1:])
        oa, ob = np.argsort(self.a), np.argsort(self.b)
        ca = np.vstack([np.zeros((1, V.shape[1])), np.cumsum(V[oa], 0)])
        cb = np.vstack([np.zeros((1, V.shape[1])), np.cumsum(V[ob], 0)])
        g = ca[np.searchsorted(self.a[oa], mid, side='right')] - cb[np.searchsorted(self.b[ob], mid, side='right')]
        return (np.diff(p)[:, None] * g ** 2).sum(0)


def rl_path(H, n, rng):
    """Riemann-Liouville process X(t) = int_0^t (t-s)^{H-1/2} dW(s) on t_j = j/n (left-point convolution)."""
    dW = rng.standard_normal(n) / math.sqrt(n)
    k = np.arange(1, n + 1) / n
    ker = k ** (H - 0.5)
    X = np.convolve(dW, ker)[:n]
    return np.r_[0.0, X]


def window_avg_grid(B, a, b):
    n = B.size - 1
    I = np.r_[0.0, np.cumsum(0.5 * (B[1:] + B[:-1]) / n)]
    t = np.arange(n + 1) / n
    return (np.interp(b, t, I) - np.interp(a, t, I)) / (b - a)


def run_bracket_cells(des, H, Ytau, tau, ns, label, log):
    K = des.K(H)
    ex = C.exact_evidence(K, Ytau, tau ** 2)
    rows = []
    for n in ns:
        bas = C.CFBasis(H, n)
        f = C.cf_features(des, bas)
        maj = C.CFMajorant(des, bas, f)
        br = C.bracket(f, Ytau, tau ** 2, majorant=maj)
        d = 1e-6 + 1e-9 * np.abs(ex)
        rows.append({'case': label, 'H': H, 'n': n, 'D': bas.D, 'rho': br['rho'], 'n_negative_e2': f.n_negative,
                     'viol': int(np.sum(br['Ld'] - ex > d) + np.sum(ex - br['U'] > d)),
                     'gap_Ld': qstats(ex - br['Ld']), 'gap_U': qstats(br['U'] - ex), 'err_W': qstats(np.abs(br['W'] - ex))})
        log('  %s H %.2f D %4d rho %.3g viol %d gapLd %.3g gapU %.3g' % (label, H, bas.D, br['rho'], rows[-1]['viol'],
                                                                      rows[-1]['gap_Ld']['median'], rows[-1]['gap_U']['median']))
    return rows


def e10():
    log = Log('E10.log')
    m, ndata = 1000, 50
    out = {'experiment': 'E10', 'm': m, 'ndata': ndata, 'rows': [], 'learning': []}
    desW, _ = make_design('W', m)
    ns = (16, 64, 256, 1024)
    # 1. small noise
    for H in (0.75,):
        K = desW.K(H); sd = math.sqrt(np.mean(np.diag(K))); tau = 0.001 * sd
        Y = np.linalg.cholesky(K + tau ** 2 * np.eye(m)) @ np.random.default_rng(SEED0 + 9001).standard_normal((m, ndata))
        out['rows'] += run_bracket_cells(desW, H, Y, tau, ns, 'W small noise f=0.001', log)
    # 2. H near 1/2 and near 1
    for H in (0.51, 0.99):
        K = desW.K(H); sd = math.sqrt(np.mean(np.diag(K))); tau = 0.03 * sd
        Y = np.linalg.cholesky(K + tau ** 2 * np.eye(m)) @ np.random.default_rng(SEED0 + 9002).standard_normal((m, ndata))
        out['rows'] += run_bracket_cells(desW, H, Y, tau, ns, 'W extreme H', log)
    # 3. discontinuous integrands: interval increments
    rng = np.random.default_rng(14)
    c = rng.uniform(0.05, 0.95, m); w = rng.uniform(0.005, 0.06, m)
    inc = Increments(np.clip(c - w / 2, 0, 1), np.clip(c + w / 2, 0, 1))
    for H in (0.75,):
        K = inc.K(H); sd = math.sqrt(np.mean(np.diag(K))); tau = 0.03 * sd
        Y = np.linalg.cholesky(K + tau ** 2 * np.eye(m)) @ np.random.default_rng(SEED0 + 9003).standard_normal((m, ndata))
        out['rows'] += run_bracket_cells(inc, H, Y, tau, ns, 'increments', log)
    # 4. misspecified data: Riemann-Liouville paths and an H switch, learned with exact and CF lower bound
    Hg = np.round(np.arange(0.50, 0.995, 0.01), 2)
    LK = np.linspace(-8, 1, 91); KAP = 10 ** LK
    for label in ('Riemann-Liouville H=0.75', 'switch 0.6->0.9 at t=1/2'):
        Ys = []
        for j in range(20):
            r = np.random.default_rng(SEED0 + 9100 + j)
            if label.startswith('Riemann'):
                B = rl_path(0.75, 1 << 16, r)
            else:
                import e6_scale as E6
                B1 = E6.davies_harte_path(0.6, 1 << 15, r); B2 = E6.davies_harte_path(0.9, 1 << 15, r)
                B = np.r_[B1 * (0.5 ** 0.6), B1[-1] * (0.5 ** 0.6) + B2[1:] * (0.5 ** 0.9)]
            y = window_avg_grid(B, desW.a, desW.b)
            Ys.append(y)
        Y = np.column_stack(Ys)
        sd = float(np.std(Y)); Y = Y + 0.03 * sd * np.random.default_rng(SEED0 + 9200).standard_normal(Y.shape)
        prof_ex, prof_L, prof_U = np.empty((len(Hg), Y.shape[1])), np.empty((len(Hg), Y.shape[1])), np.empty((len(Hg), Y.shape[1]))
        for g, H in enumerate(Hg):
            K = desW.K(H); lam, V = eigh(K); lam = np.maximum(lam, 0); VY2 = (V.T @ Y) ** 2
            best = -np.inf * np.ones(Y.shape[1])
            for k in KAP:
                q = (VY2 / (lam + k)[:, None]).sum(0)
                val = -0.5 * (np.sum(np.log(lam + k)) + m * np.log(q / m) + m)
                best = np.maximum(best, val)
            prof_ex[g] = best
            bas = C.CFBasis(H, 254); f = C.cf_features(desW, bas)
            lr = C.LowRank(f.Gamma); UY, perp = lr.proj(Y); p2 = np.einsum('ib,ib->b', perp, perp)
            bL = -np.inf * np.ones(Y.shape[1]); bU = -np.inf * np.ones(Y.shape[1])
            for k in KAP:
                q = p2 / k + ((UY ** 2) / (lr.lam + k)[:, None]).sum(0)
                ld = (m - lr.r) * math.log(k) + float(np.sum(np.log(lr.lam + k)))
                Lv = -0.5 * (ld + m * np.log(q / m) + m) - 0.5 * float(np.sum(np.log1p(f.e2 / k)))
                q0 = p2 / (k + f.trR) + ((UY ** 2) / (lr.lam + k + f.trR)[:, None]).sum(0)
                v = lr.solve(Y, k, (UY, perp)); q1 = q - (np.abs(v).T @ np.sqrt(f.e2)) ** 2
                Uv = -0.5 * (ld + m * np.log(np.maximum(np.maximum(q0, q1), 1e-300) / m) + m)
                bL = np.maximum(bL, Lv); bU = np.maximum(bU, Uv)
            prof_L[g], prof_U[g] = bL, bU
        Hx = Hg[np.argmax(prof_ex, 0)]; HL = Hg[np.argmax(prof_L, 0)]
        thr = prof_L.max(0)
        encl = [(float(Hg[np.where(prof_U[:, b] >= thr[b])[0]].min()), float(Hg[np.where(prof_U[:, b] >= thr[b])[0]].max()))
                for b in range(Y.shape[1])]
        cover = [bool(prof_U[np.argmax(prof_ex[:, b]), b] >= thr[b]) for b in range(Y.shape[1])]
        out['learning'].append({'case': label, 'H_exact': qstats(Hx), 'H_CF': qstats(HL), 'dH': qstats(HL - Hx),
                                'enclosure_width': qstats([e[1] - e[0] for e in encl]), 'coverage': float(np.mean(cover)),
                                'enclosures_first5': encl[:5]})
        log('  learning %s: H_exact med %.3f H_CF med %.3f coverage %.2f' % (label, np.median(Hx), np.median(HL), np.mean(cover)))
    save_json(out, 'E10_stress.json')


def e11():
    import mpmath as mp
    log = Log('E11.log')
    mp.mp.dps = 50
    out = {'experiment': 'E11', 'kummer': {}, 'moments': {}, 'orthonormality': {}, 'bracket_terms': [], 'exact_two_algorithms': []}
    rng = np.random.default_rng(1111)
    # (a) Kummer function M_a(x) = 1F1(a; a+1; i x)
    for H in (0.55, 0.75, 0.95):
        errs = []
        for a in (2 * H, 2 * H + 1):
            xs = np.sign(rng.uniform(-1, 1, 300)) * 10 ** rng.uniform(-3, math.log10(4096 * math.pi), 300)
            vd = C.KM(a, xs)
            for x, v in zip(xs, vd):
                ref = mp.hyp1f1(a, a + 1, 1j * mp.mpf(float(x)))
                errs.append(abs(complex(v) - complex(ref)) / max(abs(complex(ref)), 1e-300))
        out['kummer'][str(H)] = {'max_rel': float(np.max(errs)), 'median_rel': float(np.median(errs)), 'n': len(errs)}
        log('  Kummer H %.2f max rel %.2e' % (H, np.max(errs)))
    # (b) moments of points and windows against the same closed forms in 50 digits
    def mp_point(H, t, k):
        b = 2 * H - 1; T = 1
        if k == 0:
            return 0.5 * (mp.mpf(t) ** (2 * H) + 1 - (1 - mp.mpf(t)) ** (2 * H))
        w = k * mp.pi
        e = mp.exp(1j * w * t)
        p1 = e * mp.mpf(t) ** (b + 1) * mp.hyp1f1(b + 1, b + 2, -1j * w * t) / (b + 1)
        p2 = e * (1 - mp.mpf(t)) ** (b + 1) * mp.hyp1f1(b + 1, b + 2, 1j * w * (1 - t)) / (b + 1)
        Pu = mp.sqrt(2) * mp.re(mp.hyp1f1(b + 1, b + 2, 1j * w)) / (b + 1)
        return H * (Pu + mp.sqrt(2) * mp.re(p1 - p2))

    def mp_box(H, a, bb, k):
        be = 2 * H - 1
        wid = mp.mpf(bb) - mp.mpf(a)
        if k == 0:
            p = 2 * H + 1
            return 0.5 * ((mp.mpf(bb) ** p - mp.mpf(a) ** p) / (p * wid) + 1 + ((1 - mp.mpf(bb)) ** p - (1 - mp.mpf(a)) ** p) / (p * wid))
        w = k * mp.pi
        J = lambda c: mp.exp(1j * w * c) * mp.mpf(c) ** (be + 2) * mp.hyp1f1(be + 2, be + 3, -1j * w * c) / (be + 2)
        J2 = lambda c: mp.exp(1j * w * c) * (1 - mp.mpf(c)) ** (be + 2) * mp.hyp1f1(be + 2, be + 3, 1j * w * (1 - c)) / (be + 2)
        Pu = mp.sqrt(2) * mp.re(mp.hyp1f1(be + 1, be + 2, 1j * w)) / (be + 1)
        S = mp.re(J(bb) - J(a) + J2(bb) - J2(a)) / ((be + 1) * wid)
        return H * (Pu + mp.sqrt(2) * S)
    ks = (0, 1, 7, 64, 511, 2047, 4095)
    for H in (0.55, 0.75, 0.95):
        tp = rng.uniform(0.01, 1, 6); P = C.Points(tp); bP = P.moments(H, 4096)
        ab = rng.uniform(0.03, 0.9, 6); wb = rng.uniform(0.005, 0.06, 6); Bx = C.Boxes(ab, ab + wb); bB = Bx.moments(H, 4096)
        eP = max(abs(bP[i, k] - float(mp_point(H, float(tp[i]), k))) for i in range(6) for k in ks)
        eB = max(abs(bB[i, k] - float(mp_box(H, float(ab[i]), float(ab[i] + wb[i]), k))) for i in range(6) for k in ks)
        out['moments'][str(H)] = {'points_max_abs': eP, 'boxes_max_abs': eB}
        log('  moments H %.2f points %.2e boxes %.2e' % (H, eP, eB))
        out['orthonormality'][str(H)] = {str(n): C.CFBasis(H, n).orthonormality_defect() for n in (8, 64, 512, 1024)}
    # (c) bracket terms in 50 digits given the double inputs (m = 120)
    def mp_logdet_quad(A, y):
        Am = mp.matrix(A.tolist()); L = mp.cholesky(Am)
        ld = 2 * mp.fsum(mp.log(L[i, i]) for i in range(A.shape[0]))
        x = mp.cholesky_solve(Am, mp.matrix(y.tolist()))
        q = mp.fsum(x[i] * y[i] for i in range(len(y)))
        return ld, q
    for kind in ('P', 'W', 'S'):
        des, _ = make_design(kind, 2000)
        sub = des.subset(np.arange(0, 2000, 2000 // 120)[:120])
        for H in (0.55, 0.75, 0.95):
            K = sub.K(H); sd = math.sqrt(np.mean(np.diag(K))); tau2 = (0.01 * sd) ** 2
            y = np.linalg.cholesky(K + tau2 * np.eye(sub.m)) @ np.random.default_rng(11).standard_normal(sub.m)
            ex_d = float(C.exact_evidence(K, y[:, None], tau2)[0])
            ld, q = mp_logdet_quad(K + tau2 * np.eye(sub.m), y)
            ex_mp = float(-0.5 * ld - 0.5 * q - 0.5 * sub.m * mp.log(2 * mp.pi))
            for n in (8, 64, 512):
                bas = C.CFBasis(H, n); f = C.cf_features(sub, bas)
                Cm = f.Gamma @ f.Gamma.T + tau2 * np.eye(sub.m)
                br = C.bracket(f, y[:, None], tau2, majorant=C.CFMajorant(sub, bas, f))
                ld2, q2 = mp_logdet_quad(Cm, y)
                Lmp = float(-0.5 * ld2 - 0.5 * q2 - 0.5 * sub.m * mp.log(2 * mp.pi) - mp.fsum(f.e2.tolist()) / (2 * tau2))
                out['bracket_terms'].append({'kind': kind, 'H': H, 'n': n, 'exact_double_minus_mp': ex_d - ex_mp,
                                             'L_double_minus_mp': float(br['L'][0]) - Lmp})
            log('  terms %s H %.2f exact err %.2e L err %.2e' % (kind, H, ex_d - ex_mp, out['bracket_terms'][-1]['L_double_minus_mp']))
    # (d) two double-precision algorithms for the exact evidence at m = 2000, f = 0.01 (E1 data)
    for kind in ('P', 'W', 'S'):
        des, _ = make_design(kind, 2000)
        for H in (0.55, 0.95):
            dat = np.load(os.path.join(RESULTS, 'data', 'E1_%s_H%.2f_f%g.npz' % (kind, H, 0.01)))
            Y, tau = dat['Y'][:2000], float(dat['tau'])
            K = des.K(H)
            a1 = C.exact_evidence(K, Y, tau ** 2)
            lam, V = eigh(K + tau ** 2 * np.eye(2000))
            VY = V.T @ Y
            a2 = -0.5 * np.sum(np.log(lam)) - 0.5 * (VY ** 2 / lam[:, None]).sum(0) - 0.5 * 2000 * math.log(2 * math.pi)
            out['exact_two_algorithms'].append({'kind': kind, 'H': H, 'max_abs_diff': float(np.max(np.abs(a1 - a2))),
                                                'cond': float(lam[-1] / lam[0])})
            log('  two algorithms %s H %.2f max |diff| %.2e cond %.2e' % (kind, H, np.max(np.abs(a1 - a2)), lam[-1] / lam[0]))
    save_json(out, 'E11_precision.json')


def e12():
    log = Log('E12.log')
    des, _ = make_design('W', 2000)
    out = {'experiment': 'E12', 'kind': 'W', 'm': 2000, 'threads': os.environ.get('MKL_NUM_THREADS'), 'per_candidate': []}
    dat = np.load(os.path.join(RESULTS, 'data', 'E1_W_H0.75_f0.03.npz'))
    Y, tau = dat['Y'][:2000], float(dat['tau'])
    Acos_t0 = time.perf_counter(); A = des.cosine(1016); t_cos = time.perf_counter() - Acos_t0
    for H in (0.6, 0.75, 0.9):
        rec = {'H': H}
        t0 = time.perf_counter(); K = des.K(H); rec['exact_K'] = time.perf_counter() - t0
        t0 = time.perf_counter(); Lc = cholesky(K + tau ** 2 * np.eye(2000), lower=True); rec['exact_chol'] = time.perf_counter() - t0
        for r in (64, 256):
            t0 = time.perf_counter(); bas = C.CFBasis(H, r - 2); tp = time.perf_counter() - t0
            t0 = time.perf_counter(); Bm = des.moments(H, bas.Naux); tm = time.perf_counter() - t0
            t0 = time.perf_counter(); f = C.cf_features(des, bas, B=Bm); tc = time.perf_counter() - t0
            maj = C.CFMajorant(des, bas, f)
            t0 = time.perf_counter(); br = C.bracket(f, Y[:, :1], tau ** 2, majorant=maj); tb = time.perf_counter() - t0
            t0 = time.perf_counter(); br2 = C.bracket(f, Y[:, :1], tau ** 2, majorant=None, working=False); tl = time.perf_counter() - t0
            z = np.arange(1, r + 1) / r
            t0 = time.perf_counter(); fi = C.ip_features(des, H, z); ti = time.perf_counter() - t0
            t0 = time.perf_counter(); bri = C.bracket(fi, Y[:, :1], tau ** 2, working=False); tbi = time.perf_counter() - t0
            rec['r%d' % r] = {'CF_prep': tp, 'CF_moments': tm, 'CF_coords': tc, 'CF_bracket_with_U2': tb,
                              'CF_lower_only': tl, 'IP_features': ti, 'IP_bound': tbi,
                              'CF_total_per_candidate': tp + tm + tc + tb, 'IP_total_per_candidate': ti + tbi}
        out['per_candidate'].append(rec)
        log('  H %.2f exact K %.2fs chol %.2fs | r256 CF total %.2fs (moments %.2fs) IP %.2fs' % (
            H, rec['exact_K'], rec['exact_chol'], rec['r256']['CF_total_per_candidate'], rec['r256']['CF_moments'],
            rec['r256']['IP_total_per_candidate']))
    out['cached_across_H'] = {'classical_cosine_coefficients_seconds': t_cos,
                              'note': 'classical cosine coefficients and design arrays are H-free; moments, basis and '
                                      'coordinates are recomputed for every candidate H; IP refactors K_uu and K_fu'}
    save_json(out, 'E12_cost_model.json')


if __name__ == '__main__':
    {'E10': e10, 'E11': e11, 'E12': e12}[sys.argv[1]]()
