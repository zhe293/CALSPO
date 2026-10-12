"""Restore large review assets from unchanged binary parts using Python's standard library."""
from pathlib import Path
import argparse, hashlib, json, os
root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--verify-only', action='store_true', help='Check all parts without writing output files')
args = parser.parse_args()
folder = root / 'review_downloads'
manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf8'))
for entry in manifest['files']:
    output = (root / entry['path']).resolve()
    if not output.is_relative_to(root.resolve()):
        raise ValueError('Destination escapes repository')
    digest = hashlib.sha256()
    total = 0
    temp = output.with_name(output.name + '.review-restore.tmp')
    if not args.verify_only:
        output.parent.mkdir(parents=True, exist_ok=True)
    handle = None if args.verify_only else temp.open('wb')
    try:
        for part in entry['parts']:
            path = (folder / part['path']).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError('Part escapes download directory')
            data = path.read_bytes()
            if len(data) != part['size'] or hashlib.sha256(data).hexdigest() != part['sha256']:
                raise ValueError(f'Invalid part: {part["path"]}')
            header = bytes.fromhex(manifest['binary_header_hex'])
            if not data.startswith(header):
                raise ValueError('Invalid binary part header')
            data = data[len(header):]
            digest.update(data)
            total += len(data)
            if handle:
                handle.write(data)
        if total != entry['size'] or digest.hexdigest() != entry['sha256']:
            raise ValueError(f'Restored asset mismatch: {entry["path"]}')
    finally:
        if handle:
            handle.close()
    if not args.verify_only:
        os.replace(temp, output)
    print('VERIFIED' if args.verify_only else 'RESTORED', entry['path'], total)
