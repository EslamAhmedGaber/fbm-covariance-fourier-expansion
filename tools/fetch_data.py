"""Fetch exact published dataset snapshots using the tracked SHA-256 manifests."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {'noaa': 'Borehole_NOAA_Huang', 'crutem5': 'CRUTEM5'}

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['all', *DATASETS], default='all')
    p.add_argument('--limit', type=int, help='check/download only the first N entries per dataset')
    p.add_argument('--verify-only', action='store_true', help='do not access the network')
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    total = 0
    for key in DATASETS if args.dataset == 'all' else [args.dataset]:
        base = ROOT / '04_DATA' / DATASETS[key]
        manifest = json.loads((base / 'DATA_MANIFEST.json').read_text(encoding='utf-8'))
        for entry in manifest['files'][:args.limit]:
            folder = base / 'huang2016' if key == 'noaa' else base
            dest = (folder / entry['name']).resolve()
            if not dest.is_relative_to(folder.resolve()):
                raise ValueError('Manifest destination escapes dataset directory')
            if dest.exists():
                if sha256(dest) != entry['sha256']:
                    raise ValueError('Existing file has wrong SHA-256: ' + str(dest))
            elif args.verify_only:
                raise FileNotFoundError(dest)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                temp = dest.with_name(dest.name + '.part')
                try:
                    req = urllib.request.Request(entry['url'], headers={'User-Agent': 'CovarianceFourierResearch/1.1.0'})
                    with urllib.request.urlopen(req, timeout=180) as response, temp.open('wb') as f:
                        while chunk := response.read(1024 * 1024):
                            f.write(chunk)
                    if sha256(temp) != entry['sha256']:
                        raise ValueError('Provider file differs from recorded snapshot: ' + entry['url'])
                    temp.replace(dest)
                finally:
                    if temp.exists():
                        temp.unlink()
            total += 1
            print('verified', key, entry['name'], flush=True)
    print(f'{total} files verified')

if __name__ == '__main__':
    main()
