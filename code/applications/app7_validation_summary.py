"""A7.3: calibration of the borehole posteriors against CRUTEM5 (sites and country aggregates).

Reads results/A7_borehole_realdata.json.  For each prior (H, pooled sigma_T) and each site: z = (W_inst - mu) /
sqrt(sd^2 + sd_inst^2), where sd_inst is the sampling sd of the instrumental window difference.  Reports coverage of
central 50 % and 90 % intervals, mean and sd of z, correlation of mu with W_inst, the mean squared error against the
prior-only forecast (mu = 0) and against the database's own century-trend reconstruction (Huang et al.), the
continuous ranked probability score (Gaussian CRPS), separately for the training and the held-out halves, and
country aggregates (countries with >= 10 sites; posterior errors treated as independent across sites).
Output: results/A7_validation_summary.json
"""
import json
import math
import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')


def crps_gauss(x, mu, sd):
    z = (x - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))


def summarize(W, sdi, mu, sd, mask):
    W, sdi, mu, sd = W[mask], sdi[mask], mu[mask], sd[mask]
    tot = np.sqrt(sd ** 2 + sdi ** 2)
    z = (W - mu) / tot
    return {'n': int(mask.sum()), 'cover50': float(np.mean(np.abs(z) <= norm.ppf(0.75))),
            'cover90': float(np.mean(np.abs(z) <= norm.ppf(0.95))), 'z_mean': float(np.mean(z)), 'z_sd': float(np.std(z, ddof=1)),
            'corr_mu_W': float(np.corrcoef(mu, W)[0, 1]), 'mse': float(np.mean((W - mu) ** 2)),
            'mse_prior_only': float(np.mean(W ** 2)), 'crps': float(np.mean(crps_gauss(W, mu, tot))),
            'median_sd': float(np.median(sd)), 'mean_mu': float(np.mean(mu)), 'mean_W': float(np.mean(W))}


def main(src='A7_borehole_realdata.json', dst='A7_validation_summary.json'):
    d = json.load(open(os.path.join(RES, src)))
    sites = d['sites']; tr = np.array(d['is_train'])
    W = np.array([s['inst']['W'] for s in sites]); sdi = np.array([s['inst']['sd'] for s in sites])
    hw = np.array([np.nan if s['huang_W'] is None else s['huang_W'] for s in sites])
    country = np.array([s['country'] for s in sites])
    out = {'n_sites': len(sites), 'huang_reconstruction': {}, 'by_H': {}}
    ok = np.isfinite(hw)
    out['huang_reconstruction'] = {'corr': float(np.corrcoef(hw[ok], W[ok])[0, 1]), 'mse': float(np.mean((W[ok] - hw[ok]) ** 2)),
                                   'mse_prior_only': float(np.mean(W[ok] ** 2)), 'mean': float(np.mean(hw[ok]))}
    for H, rec in d['by_H'].items():
        mu = np.array([p['mu'] for p in rec['site']]); sd = np.array([p['sd'] for p in rec['site']])
        entry = {'sigma_T': rec['sigma_T'], 'sigma_g': rec.get('sigma_g'), 'll_train': rec['ll_train'], 'll_test': rec['ll_test'],
                 'train': summarize(W, sdi, mu, sd, tr), 'test': summarize(W, sdi, mu, sd, ~tr),
                 'all': summarize(W, sdi, mu, sd, np.ones(len(W), bool))}
        agg = {}
        for c in sorted(set(country)):
            msk = country == c
            if msk.sum() < 10:
                continue
            m_mu = float(np.mean(mu[msk])); m_sd = float(math.sqrt(np.sum(sd[msk] ** 2)) / msk.sum())
            m_W = float(np.mean(W[msk])); m_sdi = float(math.sqrt(np.sum(sdi[msk] ** 2)) / msk.sum())
            agg[c] = {'n': int(msk.sum()), 'mu': m_mu, 'sd': m_sd, 'W_inst': m_W, 'sd_inst': m_sdi,
                      'z': (m_W - m_mu) / math.sqrt(m_sd ** 2 + m_sdi ** 2),
                      'huang': float(np.nanmean(hw[msk]))}
        entry['countries'] = agg
        zc = np.array([v['z'] for v in agg.values()])
        entry['countries_summary'] = {'n': len(zc), 'cover90': float(np.mean(np.abs(zc) <= 1.645)),
                                      'corr_mu_W': float(np.corrcoef([v['mu'] for v in agg.values()], [v['W_inst'] for v in agg.values()])[0, 1]),
                                      'corr_huang_W': float(np.corrcoef([v['huang'] for v in agg.values()], [v['W_inst'] for v in agg.values()])[0, 1])}
        out['by_H'][H] = entry
        t = entry['test']
        print('H %s sigma_T %.2f | test: cover50 %.2f cover90 %.2f z %.2f+-%.2f corr %.2f mse %.3f (prior %.3f) crps %.3f | countries cover90 %.2f corr %.2f (huang %.2f)' % (
            H, rec['sigma_T'], t['cover50'], t['cover90'], t['z_mean'], t['z_sd'], t['corr_mu_W'], t['mse'], t['mse_prior_only'], t['crps'],
            entry['countries_summary']['cover90'], entry['countries_summary']['corr_mu_W'], entry['countries_summary']['corr_huang_W']))
    print('Huang reconstruction: corr %.3f mse %.3f prior-only %.3f' % (out['huang_reconstruction']['corr'], out['huang_reconstruction']['mse'],
                                                                         out['huang_reconstruction']['mse_prior_only']))
    with open(os.path.join(RES, dst), 'w') as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == '__main__':
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        main()
