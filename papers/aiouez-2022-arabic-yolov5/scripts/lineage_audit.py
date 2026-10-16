"""Read-only comparison of pinned author releases; never repairs annotations."""
import collections
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import yaml

ROOT = Path('/datasets/belmadoui-arabic-sign-language')
ACQUISITION_SHA = 'ee043a72a3bf6eb7e204c01d415563b2c9e50d3889a05c016bb46069113b2090'
DERIVED_SHA = 'bdb2b2a87d2b93af07d977dafac14cf5f99c2e3333e97350cada17da8475356e'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def family(path):
    return Path(path).stem.split('.rf.')[0]


def main(output):
    assert sha(ROOT / 'acquisition-manifest.json') == ACQUISITION_SHA
    assert sha(ROOT / 'manifest.json') == DERIVED_SHA
    acquisition = json.loads((ROOT / 'acquisition-manifest.json').read_text())
    derived = json.loads((ROOT / 'manifest.json').read_text())
    releases, configs = {}, []
    for release in acquisition['releases']:
        name = release['name']
        archive = Path('/tmp') / ('lineage-' + release['archive'])
        shutil.copyfile(ROOT / release['archive'], archive)
        assert sha(archive) == release['archive_sha256']
        files = {entry['path']: entry for entry in release['files']}
        labels = {}
        with zipfile.ZipFile(archive) as source:
            for path, entry in files.items():
                if not (path.endswith('/data.yaml') or path.endswith('.txt')):
                    continue
                payload = source.read(path.removeprefix('original/' + name + '/'))
                assert hashlib.sha256(payload).hexdigest() == entry['sha256']
                if path.endswith('/data.yaml'):
                    config = yaml.safe_load(payload)
                    assert config['nc'] == 28 and len(config['names']) == 28
                    assert isinstance(config['names'], list) and all(isinstance(n, str) and n for n in config['names'])
                    assert len(set(config['names'])) == 28
                    configs.append({'release': name, 'path': path, 'sha256': entry['sha256'],
                                    'names': config['names'], 'equals_derived_names': config['names'] == derived['classes']})
                else:
                    rows = [[float(value) for value in line.split()]
                            for line in payload.decode().splitlines() if line.strip()]
                    assert all(len(row) == 5 and row[0] == int(row[0]) and 0 <= row[0] < 28
                               for row in rows)
                    labels[path] = rows
        releases[name] = {'files': files, 'labels': labels,
                          'archive_sha256': release['archive_sha256']}
    assert {row['release'] for row in configs} == {'augmented', 'unaugmented'}
    augmented = derived['records']
    for row in augmented:
        parsed = releases['augmented']['labels'][row['label']]
        assert parsed == [[a['class_id'], *a['xywh_normalized']] for a in row['annotations']]
    unaugmented = []
    missing_labels = []
    for path, image in releases['unaugmented']['files'].items():
        if 'image_size' not in image:
            continue
        label = path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        record = {'image': path, 'image_sha256': image['sha256'], 'label': label,
                  'rows': releases['unaugmented']['labels'].get(label)}
        unaugmented.append(record)
        if record['rows'] is None:
            missing_labels.append(path)
    by_hash = collections.defaultdict(list)
    un_families, aug_families = collections.defaultdict(list), collections.defaultdict(list)
    for row in unaugmented:
        by_hash[row['image_sha256']].append(row)
        un_families[family(row['image'])].append(row)
    pairs, transitions, pair_summary = [], collections.Counter(), collections.Counter()
    for row in augmented:
        aug_families[family(row['image'])].append(row)
        aug_rows = releases['augmented']['labels'][row['label']]
        for old in by_hash[row['image_sha256']]:
            old_rows = old['rows']
            if old_rows is None:
                kind = 'missing_unaugmented_label'
            elif old_rows == aug_rows:
                kind = 'same_parsed_annotations'
            elif [r[0] for r in old_rows] != [r[0] for r in aug_rows]:
                kind = 'different_class_sequence'
            else:
                kind = 'same_class_sequence_different_coordinates'
            pair_summary[kind] += 1
            if old_rows is not None:
                transitions[str(([int(r[0]) for r in old_rows], [int(r[0]) for r in aug_rows]))] += 1
            pairs.append({'image_sha256': row['image_sha256'], 'augmented_id': row['id'],
                          'augmented_label': row['label'], 'unaugmented_label': old['label'],
                          'augmented_label_sha256': row['label_sha256'],
                          'unaugmented_label_sha256': releases['unaugmented']['files'].get(old['label'], {}).get('sha256'),
                          'augmented_rows': aug_rows, 'unaugmented_rows': old_rows, 'comparison': kind})
    families, family_transitions = [], collections.Counter()
    for key in sorted(set(un_families) & set(aug_families)):
        old = un_families[key]
        new = aug_families[key]
        old_classes = sorted({int(a[0]) for r in old if r['rows'] is not None for a in r['rows']})
        new_classes = sorted({a['class_id'] for r in new for a in r['annotations']})
        family_transitions[str((old_classes, new_classes))] += 1
        families.append({'family': key, 'unaugmented_classes': old_classes,
                         'augmented_classes': new_classes, 'augmented_ids': [r['id'] for r in new],
                         'unaugmented_image_hashes': sorted({r['image_sha256'] for r in old}),
                         'exact_image_overlap': bool({r['image_sha256'] for r in old} &
                                                     {r['image_sha256'] for r in new})})
    summary = {'class_name_arrays_equal': all(row['equals_derived_names'] for row in configs),
               'verified_config_count': len(configs),
               'exact_image_pair_count': len(pairs), 'exact_pair_comparisons': dict(pair_summary),
               'exact_class_transitions': dict(sorted(transitions.items())),
               'shared_filename_families': len(families),
               'heuristic_family_class_transitions': dict(sorted(family_transitions.items())),
               'unaugmented_missing_label_images': len(missing_labels)}
    report = {'summary': summary, 'acquisition_manifest_sha256': ACQUISITION_SHA,
              'derived_manifest_sha256': DERIVED_SHA,
              'archive_sha256': {name: value['archive_sha256'] for name, value in releases.items()},
              'interpretation_limit': 'Exact-image evidence establishes annotation differences for identical bytes. Numeric class transitions require matching class-name arrays to imply semantic changes; actual arrays are preserved. Filename-family evidence is heuristic lineage only, not proof for transformed images. No labels were modified; raw annotations remain internal.',
              'verified_configs': configs, 'exact_image_pairs': pairs,
              'heuristic_families': families, 'unaugmented_missing_label_images': missing_labels}
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
