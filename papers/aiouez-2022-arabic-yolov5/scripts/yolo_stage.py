"""Stage hash-verified archive entries locally for the unchanged YOLO loader."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import yaml

ROOT = Path('/datasets/belmadoui-arabic-sign-language')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    payload = (ROOT / 'manifest.json').read_bytes()
    assert hashlib.sha256(payload).hexdigest() == args.manifest_sha256
    manifest = json.loads(payload)
    splits = manifest['preflight_splits' if args.preflight else 'splits']
    destination = Path('/tmp/yolo-data')
    for directory in ['images', 'labels']:
        (destination / directory).mkdir(parents=True, exist_ok=True)
    archive = Path('/tmp/yolo-augmented-v1.zip')
    shutil.copyfile(ROOT / 'augmented-v1.zip', archive)
    hasher = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    assert hasher.hexdigest() == manifest['archive_sha256']
    with zipfile.ZipFile(archive) as zf:
        for split, ids in splits.items():
            paths = []
            for index in ids:
                record = manifest['records'][index]
                assert record['id'] == index and record['split'] == split
                image = zf.read(record['image'].removeprefix('original/augmented/'))
                label = zf.read(record['label'].removeprefix('original/augmented/'))
                assert hashlib.sha256(image).hexdigest() == record['image_sha256']
                assert hashlib.sha256(label).hexdigest() == record['label_sha256']
                image_path = destination / 'images' / f'{index:06d}.jpg'
                label_path = destination / 'labels' / f'{index:06d}.txt'
                image_path.write_bytes(image)
                label_path.write_bytes(label)
                paths.append(str(image_path))
            (destination / f'{split}.txt').write_text('\n'.join(paths) + '\n')
    config = {'path': str(destination), 'nc': 28, 'names': manifest['classes']}
    config.update({k: str(destination / f'{k}.txt') for k in splits})
    (destination / 'data.yaml').write_text(yaml.safe_dump(config))
    scope = 'preflight' if args.preflight else 'full'
    coco = ROOT / 'coco' / scope / 'test.json'
    assert hashlib.sha256(coco.read_bytes()).hexdigest() == manifest['coco_files'][str(coco.relative_to(ROOT))]
    shutil.copyfile(coco, destination / 'coco-test.json')
    print(json.dumps({'staged_counts': {k: len(v) for k, v in splits.items()},
                      'manifest_sha256': args.manifest_sha256, 'image_id_rule': 'Numeric stem is immutable canonical record.id; labels unchanged.'}), flush=True)


if __name__ == '__main__':
    main()
