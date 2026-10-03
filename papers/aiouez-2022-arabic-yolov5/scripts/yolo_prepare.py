"""Validate original YOLO labels and emit one shared seeded split and COCO view."""
import collections
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import yaml

ROOT = Path('/datasets/belmadoui-arabic-sign-language')


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    acquisition = json.loads((ROOT / 'acquisition-manifest.json').read_text())
    release = next(r for r in acquisition['releases'] if r['name'] == 'augmented')
    assert release['archive_sha256'] == 'd29fac5a6afdc107d02815689f06deb4a1e861690164c28c43b9f7f235fa9421'
    local_archive = Path('/tmp/belmadoui-augmented-v1.zip')
    shutil.copyfile(ROOT / release['archive'], local_archive)
    hasher = hashlib.sha256()
    with local_archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    assert hasher.hexdigest() == release['archive_sha256']
    files = {f['path']: f for f in release['files']}
    prefix = 'original/augmented/'
    with zipfile.ZipFile(local_archive) as zf:
        config_path = next(p for p in files if p.endswith('/data.yaml'))
        config = yaml.safe_load(zf.read(config_path.removeprefix(prefix)))
        classes = config['names']
        assert config['nc'] == len(classes) == len(set(classes)) == 28
        images = [f for f in release['files'] if f['path'].lower().endswith('.jpg')]
        records, missing, unusable = [], [], []
        seen_labels = set()
        for image in images:
            label_path = image['path'].replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
            if label_path not in files:
                missing.append(image)
                continue
            label_bytes = zf.read(label_path.removeprefix(prefix))
            assert digest(label_bytes) == files[label_path]['sha256']
            boxes = []
            for line in label_bytes.decode().splitlines():
                parts = [float(v) for v in line.split()]
                assert len(parts) == 5 and all(np.isfinite(parts))
                cls, x, y, w, h = parts
                assert cls == int(cls) and 0 <= cls < 28
                assert 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1
                boxes.append({'class_id': int(cls), 'xywh_normalized': [x, y, w, h]})
            assert boxes, 'Unexpected empty hand annotation'
            assert image['image_size'] == [416, 416]
            seen_labels.add(label_path)
            if any(a['xywh_normalized'][2] == 0 or a['xywh_normalized'][3] == 0 for a in boxes):
                assert len(boxes) == 1
                assert files[label_path]['sha256'] == 'd3b182f79df356f113ef1e33b3a4b8c91f0dc94547c393e43ef2d3ebb6f058ab'
                unusable.append({'image': image['path'], 'image_sha256': image['sha256'],
                                 'label': label_path, 'label_sha256': files[label_path]['sha256'],
                                 'original_label': label_bytes.decode(),
                                 'reason': 'Sole annotation has zero area; exclude this one unusable example without inventing a box or treating it as background.'})
                continue
            records.append({'id': len(records), 'image': image['path'], 'label': label_path,
                            'image_sha256': image['sha256'], 'label_sha256': files[label_path]['sha256'],
                            'width': 416, 'height': 416, 'annotations': boxes,
                            'released_split': image['path'].split('/')[-3],
                            'source_family_heuristic': Path(image['path']).stem.split('.rf.')[0]})
    assert seen_labels == {p for p in files if p.endswith('.txt')}, 'Unmatched label file'
    labeled_hashes = collections.defaultdict(list)
    for record in records:
        labeled_hashes[record['image_sha256']].append(record)
    exclusions = []
    for image in missing:
        matches = labeled_hashes[image['sha256']]
        assert matches, 'Unlabeled image is not an exact-byte duplicate; do not invent annotations'
        assert len({r['label_sha256'] for r in matches}) == 1, 'Duplicate annotation disagreement'
        exclusions.append({'image': image['path'], 'sha256': image['sha256'],
                           'identical_labeled_images': [r['image'] for r in matches],
                           'reason': 'Redundant exact-byte duplicate with no matching label; valid labeled original retained.'})
    assert len(images) == 15088 and len(records) == 15085 and len(exclusions) == 2 and len(unusable) == 1
    observed_classes = sorted({a['class_id'] for r in records for a in r['annotations']})
    assert observed_classes == [i for i in range(28) if i != 24]
    permutation = np.random.default_rng(42).permutation(len(records)).tolist()
    train_end = int(.8 * len(records))
    val_end = train_end + (len(records) - train_end) // 2
    splits = {'train': permutation[:train_end], 'val': permutation[train_end:val_end], 'test': permutation[val_end:]}
    preflight = {}
    for split, ids in splits.items():
        for index in ids:
            records[index]['split'] = split
        selected = set()
        per_class = 4 if split == 'train' else 1
        for cls in observed_classes:
            matches = [i for i in ids if any(a['class_id'] == cls for a in records[i]['annotations'])]
            assert len(matches) >= per_class
            selected.update(matches[:per_class])
        minimum = 112 if split == 'train' else 28
        for index in ids:
            if len(selected) >= minimum:
                break
            selected.add(index)
        preflight[split] = [i for i in ids if i in selected]
    family_splits = collections.defaultdict(set)
    byte_splits = collections.defaultdict(set)
    for record in records:
        family_splits[record['source_family_heuristic']].add(record['split'])
        byte_splits[record['image_sha256']].add(record['split'])
    unaugmented = next(r for r in acquisition['releases'] if r['name'] == 'unaugmented')
    unaug_archive = Path('/tmp/belmadoui-unaugmented-v1.zip')
    shutil.copyfile(ROOT / unaugmented['archive'], unaug_archive)
    assert digest(unaug_archive.read_bytes()) == unaugmented['archive_sha256']
    unaug_counts, unaug_invalid = collections.Counter(), []
    with zipfile.ZipFile(unaug_archive) as zf:
        for entry in unaugmented['files']:
            if not entry['path'].endswith('.txt'):
                continue
            payload = zf.read(entry['path'].removeprefix('original/unaugmented/'))
            assert digest(payload) == entry['sha256']
            for line in payload.decode().splitlines():
                try:
                    values = [float(v) for v in line.split()]
                    assert len(values) == 5 and all(np.isfinite(values)) and values[0] == int(values[0])
                    unaug_counts[int(values[0])] += 1
                except (ValueError, AssertionError):
                    unaug_invalid.append({'label': entry['path'], 'line': line})
    output_files = {}
    for scope, membership in [('full', splits), ('preflight', preflight)]:
        for split, ids in membership.items():
            document = {'info': {'description': 'Unchanged author YOLO boxes converted to COCO; seeded reconstruction'},
                        'licenses': [], 'images': [], 'annotations': [],
                        'categories': [{'id': i + 1, 'name': name} for i, name in enumerate(classes)]}
            for index in ids:
                record = records[index]
                document['images'].append({'id': index, 'file_name': record['image'], 'width': 416, 'height': 416})
                for box in record['annotations']:
                    x, y, w, h = box['xywh_normalized']
                    document['annotations'].append({'id': len(document['annotations']) + 1, 'image_id': index,
                                                    'category_id': box['class_id'] + 1,
                                                    'bbox': [(x - w / 2) * 416, (y - h / 2) * 416, w * 416, h * 416],
                                                    'area': w * h * 416 * 416, 'iscrowd': 0})
            path = ROOT / 'coco' / scope / f'{split}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = (json.dumps(document) + '\n').encode()
            if path.exists():
                assert path.read_bytes() == payload, 'Existing COCO view differs; refusing overwrite'
            else:
                temporary = path.with_suffix('.tmp')
                temporary.write_bytes(payload)
                temporary.replace(path)
            output_files[str(path.relative_to(ROOT))] = digest(payload)
    manifest = {'version': 'author-v1-seed42-80-10-10', 'classes': classes,
                'acquisition_manifest_sha256': digest((ROOT / 'acquisition-manifest.json').read_bytes()),
                'archive_sha256': release['archive_sha256'], 'records': records,
                'splits': splits, 'preflight_splits': preflight, 'excluded_duplicate_artifacts': exclusions,
                'excluded_zero_area_examples': unusable, 'observed_class_ids': observed_classes,
                'missing_class_ids': [24],
                'split_rule': 'Sorted acquired image paths; NumPy default_rng42 permutation; floor80percent train, remaining half val and half test. No class stratification or source-family grouping.',
                'split_counts': {k: len(v) for k, v in splits.items()},
                'released_split_counts': dict(collections.Counter(r['released_split'] for r in records)),
                'box_count': sum(len(r['annotations']) for r in records),
                'source_families_crossing_splits': sum(len(v) > 1 for v in family_splits.values()),
                'identical_image_hashes_crossing_splits': sum(len(v) > 1 for v in byte_splits.values()),
                'source_family_note': 'Filename prefix before .rf. is a heuristic for augmentation relatives, not proven signer/original provenance. Original images and transformed relatives can cross the paper-style post-augmentation random split.',
                'unaugmented_label_support': {'class_object_counts': dict(unaug_counts),
                                             'malformed_rows': unaug_invalid,
                                             'archive_sha256': unaugmented['archive_sha256']},
                'coco_files': output_files}
    payload = (json.dumps(manifest, indent=2) + '\n').encode()
    path = ROOT / 'manifest.json'
    if path.exists():
        assert path.read_bytes() == payload, 'Existing canonical manifest differs; refusing overwrite'
    else:
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(payload)
        temporary.replace(path)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['records', 'splits', 'preflight_splits']}), flush=True)
    print(json.dumps({'manifest_sha256': digest(payload)}), flush=True)


if __name__ == '__main__':
    main()
