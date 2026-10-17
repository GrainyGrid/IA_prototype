"""Download pinned RetinaFace R34 / ArcFace R50 models and verify SHA256."""
import hashlib
import json
from pathlib import Path
import ssl
import tempfile
from urllib.request import urlopen
import zipfile

import certifi

ROOT = Path(__file__).resolve().parent / 'models'
TLS = ssl.create_default_context(cafile=certifi.where())


def verified(path, spec):
    if not path.exists() or path.stat().st_size != spec['size']:
        return False
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest() == spec['sha256']


def main():
    ROOT.mkdir(exist_ok=True)
    specs = json.loads((ROOT / 'manifest.json').read_text())
    for name, spec in specs.items():
        target = ROOT / name
        if verified(target, spec):
            print(f'Already verified: {name}')
            continue
        print(f'Downloading {name}', flush=True)
        temporary = target.with_suffix('.part')
        try:
            with tempfile.TemporaryFile() as downloaded:
                with urlopen(spec['url'], timeout=120, context=TLS) as response:
                    while chunk := response.read(1024 * 1024):
                        downloaded.write(chunk)
                downloaded.seek(0)
                if 'member' in spec:
                    with zipfile.ZipFile(downloaded) as archive:
                        with archive.open(spec['member']) as source, temporary.open('wb') as output:
                            while chunk := source.read(1024 * 1024):
                                output.write(chunk)
                else:
                    with temporary.open('wb') as output:
                        while chunk := downloaded.read(1024 * 1024):
                            output.write(chunk)
            if not verified(temporary, spec):
                raise RuntimeError(f'Checksum or size mismatch: {name}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Verified: {name}')
    (ROOT / 'checksums.json').write_text(json.dumps({name: spec['sha256'] for name, spec in specs.items()}, indent=2)+'\n')


if __name__ == '__main__':
    main()
