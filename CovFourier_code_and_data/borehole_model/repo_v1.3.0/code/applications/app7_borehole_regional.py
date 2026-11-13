"""A7.6: regional model for real borehole profiles: shared climate history plus local ground-surface deviations.

Why: with one history per site (A7.1 v1/v2) the site posteriors are overconfident against CRUTEM5 while the
ensemble mean tracks the instrumental warming; local ground-surface changes (land use, snow, vegetation, near-surface
disturbance) decouple each hole from the regional air temperature.  Borehole climatology pools sites for this reason.
Model for the sites s of one country:
    dT_s = sigma_C B_H^C + sigma_L B_H^(s)        (shared climate path, independent local paths, same H),
    y_s  = T0_s + Gamma_s z + sigma_C I_H^C(f) + sigma_L I_H^(s)(f) + sigma_g W_s(z) + eps_s,
flat priors on (T0_s, Gamma_s) (REML projectors), sigma_eps_s from A7.1 v2.  Every reading and target is a coordinate
row of ONE basis; the shared climate block is solved in D dimensions after eliminating each site in data space:
    P_C = I + sigma_C^2 sum_s Gamma_s^T Pi_s Gamma_s,   b_C = sigma_C sum_s Gamma_s^T Pi_s y_s.
Hyperparameters (sigma_C, sigma_L, sigma_g) pooled over all countries by REML on borehole data only; CRUTEM5 is used
only for validation.  Targets: W_s (mean dT over [t_log-30, t_log] minus 1901-1930) at each site (climate + local),
its climate part, and the country mean of the climate parts.
Output: results/A7_borehole_regional.json
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import app7_borehole_realdata as V1                      # noqa: E402
import app7_borehole_realdata_v2 as V2                   # noqa: E402

RES = V1.RES
SC_GRID = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)
SL_GRID = (0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)
SG_GRID = (0.002, 0.004, 0.008)
MIN_SITES = 10


class SiteR:
    def __init__(self, s, Z, E2, g, e2g, sE):
        self.y, self.z = s['T'], s['z']
        self.m = len(self.y)
        self.GG = Z @ Z.T + np.diag(E2)
        self.M = np.minimum.outer(self.z, self.z)
        self.Hn = np.stack([np.ones(self.m), self.z], axis=1)
        self.Z, self.g, self.e2g, self.sE = Z, g, e2g, sE

    def local(self, sL, sG):
        """REML projector of the site given the shared climate: Pi = K^-1 - K^-1 H (H^T K^-1 H)^-1 H^T K^-1."""
        K = sL ** 2 * self.GG + sG ** 2 * self.M
        K[np.diag_indices_from(K)] += self.sE ** 2
        cf = cho_factor(K, lower=True, check_finite=False)
        KiH = cho_solve(cf, self.Hn, check_finite=False)
        A = self.Hn.T @ KiH
        cA = cho_factor(A, lower=True, check_finite=False)
        def Pi(V):
            KiV = cho_solve(cf, V, check_finite=False)
            return KiV - KiH @ cho_solve(cA, self.Hn.T @ KiV, check_finite=False)
        ld = 2 * np.sum(np.log(np.diag(cf[0]))) + 2 * np.sum(np.log(np.diag(cA[0])))
        return Pi, ld


def country_fit(sites, sC, sL, sG, predict=False):
    D = sites[0].Z.shape[1]
    P = np.eye(D); b = np.zeros(D)
    ll_sites, store = 0.0, []
    for st in sites:
        Pi, ld = st.local(sL, sG)
        PiZ = Pi(st.Z); Piy = Pi(st.y)
        P += sC ** 2 * (st.Z.T @ PiZ)
        b += sC * (st.Z.T @ Piy)
        ll_sites += -0.5 * (st.y @ Piy + ld + (st.m - 2) * math.log(2 * math.pi))
        if predict:
            store.append((Pi, PiZ, Piy))
    cP = cho_factor(P, lower=True, check_finite=False)
    mu = cho_solve(cP, b, check_finite=False)
    ll = ll_sites + 0.5 * b @ mu - 0.5 * 2 * np.sum(np.log(np.diag(cP[0])))
    if not predict:
        return ll
    Sig = cho_solve(cP, np.eye(D), check_finite=False)
    res = []
    cbar = np.zeros(D)
    for st, (Pi, PiZ, Piy) in zip(sites, store):
        c = sC * st.g                                            # climate part of W_s:  c^T Z_C
        kl = sL ** 2 * (st.Z @ st.g)                             # Cov(local part, y_s)
        vl = sL ** 2 * (st.g @ st.g + st.e2g)
        Pikl = Pi(kl)
        a = c - sC * (st.Z.T @ Pikl)                             # coefficient of Z_C in E[W_s | Z_C, y_s]
        m_s = float(a @ mu + kl @ Piy)
        v_s = float(a @ Sig @ a + vl - kl @ Pikl)
        res.append({'mu': m_s, 'sd': math.sqrt(max(v_s, 0.0)), 'mu_climate': float(c @ mu),
                    'sd_climate': math.sqrt(max(float(c @ Sig @ c), 0.0))})
        cbar += c / len(sites)
    reg = {'mu': float(cbar @ mu), 'sd': math.sqrt(max(float(cbar @ Sig @ cbar), 0.0))}
    return ll, res, reg


def main(H_list=None, n=256, src='A7_borehole_realdata_v2.json', out_name='A7_borehole_regional.json'):
    fit = json.load(open(os.path.join(RES, src)))
    cru = V1.Crutem()
    sites, _, _ = V1.select_sites(cru)
    countries = {}
    for i, s in enumerate(sites):
        countries.setdefault(s['country'], []).append(i)
    countries = {c: v for c, v in countries.items() if len(v) >= MIN_SITES}
    H_list = H_list or [h for h in fit['by_H']]
    out = {'config': {'n': n, 'countries': {c: len(v) for c, v in countries.items()}, 'sC_grid': SC_GRID,
                      'sL_grid': SL_GRID, 'sG_grid': SG_GRID, 'min_sites': MIN_SITES,
                      'model': 'sigma_C B_H^C (shared) + sigma_L B_H^(s) (local) + sigma_g W(z) + eps; flat (T0, Gamma)'},
           'by_H': {}}
    for H in H_list:
        t0 = time.perf_counter()
        Z, E2, Zt, Et, off, tim = V2.cached_coordinates(float(H), n, sites)
        sE = [p['sigma_e'] for p in fit['by_H'][str(H)]['site']]
        S = {c: [SiteR(sites[i], Z[off[i]:off[i + 1]], E2[off[i]:off[i + 1]], Zt[i], Et[i], sE[i]) for i in idx]
             for c, idx in countries.items()}
        prof = {}
        for sC in SC_GRID:
            for sL in SL_GRID:
                for sG in SG_GRID:
                    prof[(sC, sL, sG)] = sum(country_fit(S[c], sC, sL, sG) for c in S)
        sC, sL, sG = max(prof, key=prof.get)
        rec = {'sigma_C': sC, 'sigma_L': sL, 'sigma_g': sG, 'll': float(prof[(sC, sL, sG)]), 'countries': {}, 'sites': {}}
        for c, idx in countries.items():
            _, res, reg = country_fit(S[c], sC, sL, sG, predict=True)
            W = [sites[i]['inst']['W'] for i in idx]; sdi = [sites[i]['inst']['sd'] for i in idx]
            rec['countries'][c] = reg | {'n': len(idx), 'W_inst': float(np.mean(W)),
                                         'sd_inst': float(math.sqrt(np.sum(np.square(sdi))) / len(idx))}
            for i, r in zip(idx, res):
                rec['sites'][sites[i]['file']] = r | {'W_inst': sites[i]['inst']['W'], 'sd_inst': sites[i]['inst']['sd'], 'country': c}
        rec['seconds'] = time.perf_counter() - t0
        out['by_H'][str(H)] = rec
        print('H', H, 'sigma_C %.2f sigma_L %.2f sigma_g %.3f ll %.1f (%.0fs)' % (sC, sL, sG, rec['ll'], rec['seconds']), flush=True)
        with open(os.path.join(RES, out_name), 'w') as fh:
            json.dump(out, fh, indent=0)


if __name__ == '__main__':
    main(H_list=sys.argv[1:] or None)


def country_history(country='Canada', H=None, n=256, src='A7_borehole_realdata_v2.json', reg='A7_borehole_regional.json'):
    """Posterior of the shared climate history (decadal means, relative to 1901-1930) for one country, with the mean of
    the CRUTEM5 cell series of its sites.  Uses the fitted (sigma_C, sigma_L, sigma_g) of the regional model."""
    from covfourier import heat
    import borehole_problem as BP
    fit = json.load(open(os.path.join(RES, src))); R = json.load(open(os.path.join(RES, reg)))
    H = H or max(R['by_H'], key=lambda h: R['by_H'][h]['ll'])
    rr = R['by_H'][str(H)]
    sC, sL, sG = rr['sigma_C'], rr['sigma_L'], rr['sigma_g']
    cru = V1.Crutem()
    sites, _, _ = V1.select_sites(cru)
    idx = [i for i, s in enumerate(sites) if s['country'] == country]
    Z, E2, Zt, Et, off, _ = V2.cached_coordinates(float(H), n, sites)
    sE = [p['sigma_e'] for p in fit['by_H'][str(H)]['site']]
    S = [SiteR(sites[i], Z[off[i]:off[i + 1]], E2[off[i]:off[i + 1]], Zt[i], Et[i], sE[i]) for i in idx]
    D = Z.shape[1]
    P = np.eye(D); b = np.zeros(D)
    for st in S:
        Pi, _ = st.local(sL, sG)
        P += sC ** 2 * (st.Z.T @ Pi(st.Z)); b += sC * (st.Z.T @ Pi(st.y))
    cP = cho_factor(P, lower=True); mu = cho_solve(cP, b); Sig = cho_solve(cP, np.eye(D))
    prep = BP.Prepared(float(H), n, N_cand=8192, N_targ=1 << 16)
    decs = np.arange(1500, 2010, 10)
    u = heat.midpoints(prep.N_targ)
    F = np.vstack([heat.epoch_mean_values(d, d + 10, u, prep.win) - heat.epoch_mean_values(1901, 1931, u, prep.win) for d in decs])
    G = []
    for i0 in range(0, len(F), 10):
        a, _, _ = heat.project_samples(prep.bp_t, F[i0:i0 + 10]); G.append(a)
    G = sC * np.vstack(G)
    m_d = G @ mu; s_d = np.sqrt(np.clip(np.einsum('ij,jk,ik->i', G, Sig, G), 0, None))
    years = {}
    for i in idx:
        a = cru.annual(sites[i]['lat'], sites[i]['lon'])
        ref = np.mean([a[y] for y in range(1901, 1931) if y in a])
        for y, v in a.items():
            years.setdefault(y, []).append(v - ref)
    yrs = sorted(y for y in years if len(years[y]) >= max(3, len(idx) // 4))
    out = {'country': country, 'H': H, 'n_sites': len(idx), 'sigma_C': sC, 'sigma_L': sL, 'sigma_g': sG,
           'decades': decs.tolist(), 'mu': m_d.tolist(), 'sd': s_d.tolist(),
           'crutem_years': yrs, 'crutem_mean': [float(np.mean(years[y])) for y in yrs],
           'log_years': [sites[i]['t_log'] for i in idx]}
    with open(os.path.join(RES, f'A7_regional_history_{country.replace(" ", "_")}.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(country, len(idx), 'sites; climate 1990s vs 1901-30: %.2f +- %.2f K' % (m_d[-2], s_d[-2]))
    return out
