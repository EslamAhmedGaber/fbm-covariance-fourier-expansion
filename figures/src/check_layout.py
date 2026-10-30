"""Layout check for the figures of the paper.

Rebuilds the requested figures with make_figures.py and reports collisions between
text (annotations, legend entries, in-view tick labels, axis labels, titles) and
data (lines, markers, bars), and between pairs of text boxes. Exits with status 1
if any collision is found.

Usage:  python check_layout.py            # figures 2-8
        python check_layout.py fig5 fig7  # selected figures
Shaded backgrounds (axvspan, fill_between) and white labels placed inside bars
are not treated as collisions.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.text import Text
from matplotlib.container import BarContainer
from matplotlib.collections import PathCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figures as mf  # noqa: E402


def in_view_ticklabels(ax):
    out = []
    for axis in (ax.xaxis, ax.yaxis):
        tr = axis.get_transform()
        lo, hi = sorted(tr.transform(np.asarray(axis.get_view_interval(), float)))
        eps = 1e-9 * abs(hi - lo)
        locs = axis.get_majorticklocs()
        ticks = axis.get_major_ticks(len(locs))
        tl = tr.transform(np.asarray(locs, float)) if len(locs) else []
        for tick, loc in zip(ticks, tl):
            for lab in (tick.label1, tick.label2):
                if np.isfinite(loc) and lo - eps <= loc <= hi + eps and lab.get_visible() and lab.get_text().strip():
                    out.append(lab)
    return out


def text_boxes(fig, r):
    out = []
    for i, ax in enumerate(fig.axes):
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                out.append(dict(kind='text', ax=i, desc='ax%d text "%s"' % (i, t.get_text()[:28].replace('\n', ' ')),
                                bb=Text.get_window_extent(t, r), white=t.get_color() in ('white', 'w', '#ffffff')))
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            for t in leg.get_texts():
                out.append(dict(kind='legend', ax=i, desc='ax%d legend "%s"' % (i, t.get_text()[:28]), bb=t.get_window_extent(r), white=False))
            for h in getattr(leg, 'legend_handles', []):
                try:
                    out.append(dict(kind='legend-handle', ax=i, desc='ax%d legend handle' % i, bb=h.get_window_extent(r), white=False))
                except Exception:
                    pass
        for t in in_view_ticklabels(ax):
            bb = t.get_window_extent(r)
            if bb.width > 0:
                out.append(dict(kind='tick', ax=i, desc='ax%d tick "%s"' % (i, t.get_text()[-14:]), bb=bb, white=False))
        for lab in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if lab.get_visible() and lab.get_text().strip():
                out.append(dict(kind='label', ax=i, desc='ax%d label "%s"' % (i, lab.get_text()[:28]), bb=lab.get_window_extent(r), white=False))
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            out.append(dict(kind='figtext', ax=-1, desc='fig text "%s"' % t.get_text()[:28], bb=t.get_window_extent(r), white=False))
    for leg in fig.legends:
        for t in leg.get_texts():
            out.append(dict(kind='legend', ax=-1, desc='fig legend "%s"' % t.get_text()[:28], bb=t.get_window_extent(r), white=False))
    return out


def data_points(fig, dpi):
    pts = []
    for i, ax in enumerate(fig.axes):
        for ln in ax.get_lines():
            if not ln.get_visible():
                continue
            xy = np.asarray(ln.get_xydata(), float)
            if xy.size == 0:
                continue
            P = ln.get_transform().transform(xy)
            P = P[np.isfinite(P).all(1)]
            if len(P) == 0:
                continue
            has_marker = ln.get_marker() not in (None, 'None', '', ' ')
            rad = ln.get_markersize() / 2 * dpi / 72 if has_marker else 0.0
            if ln.get_linestyle() not in ('None', '', ' ') and len(P) > 1:
                seg = [a + (b - a) * np.linspace(0, 1, int(max(2, np.hypot(*(b - a)))))[:, None] for a, b in zip(P[:-1], P[1:])]
                pts.append((i, 'line "%s"' % ln.get_label()[:24], np.vstack(seg), ln.get_linewidth() / 2 * dpi / 72))
            if has_marker:
                pts.append((i, 'markers "%s"' % ln.get_label()[:24], P, rad))
        for c in ax.collections:
            if isinstance(c, PathCollection) and c.get_visible() and len(c.get_offsets()):
                pts.append((i, 'scatter', c.get_offset_transform().transform(c.get_offsets()), 2.0))
    return pts


def bars(fig, r):
    return [(i, p.get_window_extent(r)) for i, ax in enumerate(fig.axes) for c in ax.containers
            if isinstance(c, BarContainer) for p in c.patches if p.get_visible()]


def n_inside(P, bb, rad, shrink=1.0):
    return int(((P[:, 0] + rad > bb.x0 + shrink) & (P[:, 0] - rad < bb.x1 - shrink) &
                (P[:, 1] + rad > bb.y0 + shrink) & (P[:, 1] - rad < bb.y1 - shrink)).sum())


def check(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    T, D, B, issues = text_boxes(fig, r), data_points(fig, fig.dpi), bars(fig, r), []
    for t in T:
        if t['kind'] in ('tick', 'label'):
            continue
        for i, desc, P, rad in D:
            if i == t['ax'] or t['ax'] == -1:
                k = n_inside(P, t['bb'], rad)
                if k:
                    issues.append('%s  x  %s (%d px)' % (t['desc'], desc, k))
        if not t['white']:
            for i, bb in B:
                ox = min(t['bb'].x1, bb.x1) - max(t['bb'].x0, bb.x0)
                oy = min(t['bb'].y1, bb.y1) - max(t['bb'].y0, bb.y0)
                if i == t['ax'] and ox > 1 and oy > 1:
                    issues.append('%s  x  bar (%.0fx%.0f px)' % (t['desc'], ox, oy))
    for a_i in range(len(T)):
        for b_i in range(a_i + 1, len(T)):
            a, b = T[a_i], T[b_i]
            kinds = {a['kind'], b['kind']}
            if kinds == {'legend-handle'} or (kinds == {'legend', 'legend-handle'} and a['ax'] == b['ax']):
                continue
            ox = min(a['bb'].x1, b['bb'].x1) - max(a['bb'].x0, b['bb'].x0)
            oy = min(a['bb'].y1, b['bb'].y1) - max(a['bb'].y0, b['bb'].y0)
            if ox > 1.5 and oy > 1.5:
                issues.append('%s  x  %s (%.0fx%.0f px)' % (a['desc'], b['desc'], ox, oy))
    return issues


def main(names):
    mf.style()
    report, save = {}, mf.save

    def save_checked(fig, name, data):
        report[name] = check(fig)
        save(fig, name, data)

    mf.save = save_checked
    try:
        for name in names:
            getattr(mf, name)()
    finally:
        mf.save = save
    bad = 0
    for name in names:
        issues = report.get(name, [])
        bad += len(issues)
        print('%s: %s' % (name, 'no collisions' if not issues else '%d collision(s)' % len(issues)))
        for x in issues:
            print('   ', x)
    return 1 if bad else 0


if __name__ == '__main__':
    chosen = [a for a in sys.argv[1:] if a.startswith('fig')] or ['fig2', 'fig3', 'fig4', 'fig5', 'fig6', 'fig7', 'fig8']
    sys.exit(main(chosen))
