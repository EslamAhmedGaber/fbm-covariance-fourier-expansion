"""Shared settings for the experiments (see experiments/PROTOCOL.md)."""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
RESULTS = os.path.join(ROOT, 'results')
LOGS = os.path.join(ROOT, 'logs')
for p in (CODE,):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                   # noqa: E402
from cfgp import core as C                           # noqa: E402
from covfourier.heat import Window, borehole_values, midpoints   # noqa: E402

KINDS = ('P', 'W', 'S')
HS = (0.55, 0.75, 0.95)
FLEVELS = (0.1, 0.03, 0.01)
NGRID = (8, 16, 32, 64, 128, 256, 512, 1024)
WIN = Window(1500.0, 2030.0)
NMODEL = 8192
SEED0 = 20261112


def data_seed(kind, iH, inoise, offset=0):
    return SEED0 + 1000 * KINDS.index(kind) + 100 * iH + 10 * inoise + offset


def make_design(kind, m, m_hold=0, seed_offset=0):
    """Training design of size m and (optionally) a held-out design of size m_hold, as in PROTOCOL 0.2."""
    if kind == 'P':
        rng = np.random.default_rng(11 + seed_offset)
        t = rng.uniform(0.01, 1.0, m + m_hold)
        return C.Points(t[:m]), (C.Points(t[m:]) if m_hold else None)
    if kind == 'W':
        rng = np.random.default_rng(12 + seed_offset)
        c = rng.uniform(0.03, 0.97, m + m_hold)
        w = rng.uniform(0.005, 0.06, m + m_hold)
        a = np.clip(c - w / 2, 0.0, 1.0); b = np.clip(c + w / 2, 0.0, 1.0)
        return C.Boxes(a[:m], b[:m]), (C.Boxes(a[m:], b[m:]) if m_hold else None)
    if kind == 'S':
        nz = m // 2
        z = np.tile(np.linspace(10.0, 500.0, nz), 2)
        tl = np.repeat([1985.0, 2015.0], nz)
        tr = smooth_design(z, tl)
        ho = None
        if m_hold:
            rng = np.random.default_rng(13 + seed_offset)
            zh = rng.uniform(10.0, 500.0, m_hold)
            th = np.where(np.arange(m_hold) % 2 == 0, 1985.0, 2015.0)
            ho = smooth_design(zh, th)
        return tr, ho
    raise ValueError(kind)


def smooth_design(z, tl, N=NMODEL, chunk=500):
    u = midpoints(N)
    parts_A, parts_C, corr = [], [], []
    for s in range(0, len(z), chunk):
        F = borehole_values(z[s:s + chunk], tl[s:s + chunk], u, WIN)
        d = C.Smooth(F)
        parts_A.append(d.A); parts_C.append(d.C); corr.append(np.asarray(d.corrected))
    d = C.Smooth.__new__(C.Smooth)
    d.T, d.N = 1.0, N
    d.A = np.ascontiguousarray(np.hstack(parts_A)); d.C = np.ascontiguousarray(np.hstack(parts_C))
    d.m = d.A.shape[1]; d.corrected = np.concatenate(corr)
    d._loc = (np.asarray(z) - 10.0) / 490.0
    d.meta = {'z': np.asarray(z), 't_log': np.asarray(tl)}
    d._cache = {}
    return d


def joint_K(design, hold, H):
    """Covariance of training and held-out observations (training first)."""
    if hold is None:
        return design.K(H)
    if design.kind == 'smooth':
        j = C.Smooth.__new__(C.Smooth)
        j.T, j.N = 1.0, design.N
        j.A = np.ascontiguousarray(np.hstack([design.A, hold.A])); j.C = np.ascontiguousarray(np.hstack([design.C, hold.C]))
        j.m = j.A.shape[1]; j._cache = {}; j._loc = None; j.meta = {}; j.corrected = None
        return j.K(H)
    if design.kind == 'point':
        return C.Points(np.r_[design.t, hold.t]).K(H)
    return C.Boxes(np.r_[design.a, hold.a], np.r_[design.b, hold.b]).K(H)


def nuisance(design):
    x = np.asarray(design.loc(), dtype=float)
    return np.c_[np.ones(design.m), x - 0.5]


def save_json(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, default=_default)
    os.replace(path + '.tmp', path)
    return path


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


class Log:
    def __init__(self, name):
        os.makedirs(LOGS, exist_ok=True)
        self.path = os.path.join(LOGS, name)
        self.t0 = time.perf_counter()
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write('==== start %s\n' % time.strftime('%Y-%m-%d %H:%M:%S'))

    def __call__(self, *msg):
        s = '[%8.1fs] ' % (time.perf_counter() - self.t0) + ' '.join(str(x) for x in msg)
        print(s, flush=True)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(s + '\n')


def qstats(x):
    x = np.asarray(x, dtype=float)
    return {'median': float(np.median(x)), 'q05': float(np.quantile(x, 0.05)), 'q95': float(np.quantile(x, 0.95)),
            'max': float(np.max(x)), 'min': float(np.min(x))}
