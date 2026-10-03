"""Acquire the author public v1 archives; preserve original bytes and audit inventory."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

from PIL import Image

ROOT = Path('/datasets/belmadoui-arabic-sign-language')


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'acquisition-manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for release in previous['releases']:
            assert sha(ROOT / release['archive']) == release['archive_sha256']
            for entry in release['files']:
                assert sha(ROOT / entry['path']) == entry['sha256']
        print(json.dumps({'already_verified': True, 'manifest_sha256': sha(manifest_path)}))
        return
    releases = []
    for name in ['augmented', 'unaugmented']:
        url = ('https://www.kaggle.com/api/v1/datasets/download/sabribelmadoui/'
               f'arabic-sign-language-{name}-dataset?datasetVersionNumber=1')
        archive = ROOT / f'{name}-v1.zip'
        if not archive.exists():
            request = urllib.request.Request(url, headers={'User-Agent': 'REPRO-SIGN research acquisition'})
            temporary = archive.with_suffix('.part')
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            assert zipfile.is_zipfile(temporary), 'Response was not a public dataset ZIP'
            temporary.replace(archive)
        destination = ROOT / 'original' / name
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                target = (destination / member.filename).resolve()
                assert target.is_relative_to(destination.resolve()), 'Unsafe archive path'
                assert (member.external_attr >> 16) & 0o170000 != 0o120000, 'Archive symlink'
            zf.extractall(destination)
        files = []
        suffixes = collections.Counter()
        sizes = collections.Counter()
        image_hashes = collections.Counter()
        annotation_examples = []
        for path in sorted(destination.rglob('*')):
            if not path.is_file():
                continue
            digest = sha(path)
            record = {'path': str(path.relative_to(ROOT)), 'size_bytes': path.stat().st_size, 'sha256': digest}
            suffixes[path.suffix.lower()] += 1
            if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                with Image.open(path) as image:
                    image.load()
                    record['image_size'] = list(image.size)
                    record['image_mode'] = image.mode
                    sizes[str(image.size)] += 1
                image_hashes[digest] += 1
            elif path.suffix.lower() in {'.yaml', '.yml', '.json', '.csv', '.xml', '.txt'} and len(annotation_examples) < 12:
                annotation_examples.append({'path': record['path'], 'text_excerpt': path.read_text(errors='replace')[:4000]})
            files.append(record)
        release = {'name': name, 'version': 1, 'source_url': url,
                   'archive': archive.name, 'archive_sha256': sha(archive),
                   'archive_size_bytes': archive.stat().st_size, 'files': files,
                   'suffix_counts': dict(suffixes), 'image_size_counts': dict(sizes),
                   'unique_image_byte_hashes': len(image_hashes),
                   'duplicate_image_byte_occurrences': sum(v - 1 for v in image_hashes.values()),
                   'annotation_examples': annotation_examples}
        releases.append(release)
        print(json.dumps({k: v for k, v in release.items() if k != 'files'}), flush=True)
    result = {'acquired_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'license': 'Unknown; user-authorized internal research exception, no redistribution',
              'releases': releases}
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(manifest_path)
    print(json.dumps({'manifest_sha256': sha(manifest_path)}), flush=True)


if __name__ == '__main__':
    main()
