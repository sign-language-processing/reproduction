"""Read-only audit of label coordinates before choosing any correction."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

ROOT = Path('/datasets/belmadoui-arabic-sign-language')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'acquisition-manifest.json').read_text())
    release = next(r for r in manifest['releases'] if r['name'] == 'augmented')
    archive = Path('/tmp/label-audit-augmented.zip')
    shutil.copyfile(ROOT / release['archive'], archive)
    h = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    assert h.hexdigest() == release['archive_sha256']
    invalid, empty = [], []
    objects = 0
    class_counts = collections.Counter()
    minima, maxima = [math.inf] * 4, [-math.inf] * 4
    with zipfile.ZipFile(archive) as zf:
        for entry in release['files']:
            if not entry['path'].endswith('.txt'):
                continue
            data = zf.read(entry['path'].removeprefix('original/augmented/'))
            assert hashlib.sha256(data).hexdigest() == entry['sha256']
            if not data.strip():
                empty.append(entry['path'])
            for number, line in enumerate(data.decode().splitlines(), 1):
                parts = [float(v) for v in line.split()]
                reasons = []
                if len(parts) != 5 or not all(math.isfinite(v) for v in parts):
                    reasons.append('malformed_or_nonfinite')
                else:
                    cls, x, y, w, h = parts
                    class_counts[str(cls)] += 1
                    objects += 1
                    for index, value in enumerate(parts[1:]):
                        minima[index] = min(minima[index], value)
                        maxima[index] = max(maxima[index], value)
                    if cls != int(cls) or not 0 <= cls < 28:
                        reasons.append('invalid_class')
                    if not all(0 <= v <= 1 for v in [x, y, w, h]):
                        reasons.append('normalized_coordinate_outside_0_1')
                    if w <= 0 or h <= 0:
                        reasons.append('nonpositive_extent')
                    if x - w / 2 < 0 or y - h / 2 < 0 or x + w / 2 > 1 or y + h / 2 > 1:
                        reasons.append('box_crosses_image_edge')
                if reasons:
                    invalid.append({'label': entry['path'], 'label_sha256': entry['sha256'],
                                    'line_number': number, 'original_line': line, 'reasons': reasons})
    result = {'archive_sha256': release['archive_sha256'], 'objects': objects,
              'class_object_counts': dict(class_counts), 'empty_labels': empty,
              'coordinate_minima_xywh': minima, 'coordinate_maxima_xywh': maxima,
              'flagged_rows': invalid, 'reason_counts': dict(collections.Counter(
                  reason for row in invalid for reason in row['reasons']))}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'flagged_rows'}), flush=True)


if __name__ == '__main__':
    main()
