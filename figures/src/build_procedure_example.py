"""Build the worked procedure example (former Figure 1): compile fig1_method.tex, export PDF / SVG / 600-dpi PNG.

Run after make_fig1_glyphs.py, in an environment that provides PyMuPDF (fitz) and either
tectonic (preferred, self-contained) or pdflatex.
Outputs: figures/extra/procedure_example.{pdf,svg,png}, figures/data/procedure_example_build.json.
Usage:   python build_procedure_example.py [TECTONIC_CACHE_DIR]
"""
import json
import os
import shutil
import subprocess
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.dirname(HERE)
NAME = 'procedure_example'
SRC = 'fig1_method'
OUTDIR = os.path.join(FIG, 'extra')


def _ensure_known_folders():
    # Windows only: tectonic resolves LocalAppData through SHGetKnownFolderPath; create it if the profile lacks it
    if os.name != 'nt':
        return
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [('Data1', wintypes.DWORD), ('Data2', wintypes.WORD), ('Data3', wintypes.WORD), ('Data4', ctypes.c_ubyte * 8)]
    for gid in ('{F1B32785-6FBA-4FCF-9D55-7B8E7F157091}', '{3EB685DB-65F9-4CF6-A03A-E3EF65729F3D}'):
        u = uuid.UUID(gid); g = GUID(); g.Data1, g.Data2, g.Data3 = u.time_low, u.time_mid, u.time_hi_version
        for i, b in enumerate(u.bytes[8:]):
            g.Data4[i] = b
        p = ctypes.c_wchar_p()
        if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(g), 0x00004000, None, ctypes.byref(p)) == 0 and p.value:
            os.makedirs(p.value, exist_ok=True)


def _compile(cache_dir):
    env = dict(os.environ)
    if cache_dir:
        env['TECTONIC_CACHE_DIR'] = cache_dir
    # tectonic is self-contained; keep a TeX Live installation on PATH from shadowing its helpers
    path_tec = os.pathsep.join(p for p in env.get('PATH', '').split(os.pathsep) if 'texlive' not in p.lower())
    exe = shutil.which('tectonic', path=path_tec)
    if exe:
        _ensure_known_folders()
        env['PATH'] = path_tec
        cmd = [exe, '-X', 'compile', '--keep-logs', SRC + '.tex']
    elif shutil.which('pdflatex'):
        cmd = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', SRC + '.tex']
    else:
        sys.exit('neither tectonic nor pdflatex found on PATH')
    res = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-3000:], res.stderr[-3000:]); sys.exit(res.returncode)


def main(cache_dir=None):
    import fitz
    _compile(cache_dir)
    log = open(os.path.join(HERE, SRC + '.log'), encoding='utf-8', errors='replace').read()
    overfull = [l for l in log.splitlines() if 'Overfull' in l or 'Underfull' in l]
    shutil.copy2(os.path.join(HERE, SRC + '.pdf'), os.path.join(OUTDIR, f'{NAME}.pdf'))
    doc = fitz.open(os.path.join(OUTDIR, f'{NAME}.pdf'))
    page = doc[0]
    w_cm, h_cm = page.rect.width / 72 * 2.54, page.rect.height / 72 * 2.54
    page.get_pixmap(dpi=600).save(os.path.join(OUTDIR, f'{NAME}.png'))
    open(os.path.join(OUTDIR, f'{NAME}.svg'), 'w', encoding='utf-8').write(page.get_svg_image(text_as_path=False))
    fonts = sorted({(f[3], f[1]) for f in page.get_fonts(full=True)})
    info = {'width_cm': round(w_cm, 3), 'height_cm': round(h_cm, 3), 'overfull_or_underfull': overfull,
            'fonts': [{'name': n, 'type': t} for n, t in fonts], 'type3_fonts': [n for n, t in fonts if t == 'Type3']}
    os.makedirs(os.path.join(FIG, 'data'), exist_ok=True)
    json.dump(info, open(os.path.join(FIG, 'data', f'{NAME}_build.json'), 'w'), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
