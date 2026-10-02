#!/usr/bin/env bash
# Controlled acquisition of author-released CSVs; no raw video decoding.
set -euo pipefail
while read -r directory filename file_id expected; do
  mkdir -p "/datasets/$directory/siformer"
  destination="/datasets/$directory/siformer/$filename"
  if ! echo "$expected  $destination" | sha256sum --check --status 2>/dev/null; then
    curl --fail --location --max-time 600 --retry 3 --user-agent 'Mozilla/5.0' "https://drive.usercontent.google.com/download?id=$file_id&export=download&confirm=t" -o "$destination.partial"
    echo "$expected  $destination.partial" | sha256sum --check
    mv "$destination.partial" "$destination"
  fi
done <<'FILES'
lsa64 LSA64_60fps.csv 1hOjX7_JhWN5oCb27cPl7j52aNhjyjdAW 52a169473cf199cc5432eab988eaacfab5b527fcb339a24952bc1202b2e247dd
WLASL WLASL100_train_25fps.csv 1HayRIcBeh7GZjhbRsKEjCBLla_c8TeMu 3027464fe8e53afafaf3f7d859a43df11529a146af3f15a09b90abdd3b21716b
WLASL WLASL100_val_25fps.csv 1-OWsPdUsGWLiEvD9X3GJCQ3evE5A3qI7 19829dcb1bacef8e57fc3c85dbdd86437d495825c90bb7313c6b90bfd48a15fd
FILES
python3 - <<'PY'
import collections, csv, hashlib, json
from pathlib import Path
for dataset, license in [('lsa64','CC BY-NC-SA 4.0'),('WLASL','C-UDA 1.0; academic computational use')]:
    root=Path('/datasets')/dataset/'siformer'
    manifest=root/'manifest.json'
    files=[]
    for path in sorted(root.glob('*.csv')):
        with path.open() as source:
            counts=collections.Counter(row['labels'] for row in csv.DictReader(source))
        files.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sample_count=sum(counts.values()),class_counts=dict(counts)))
    # Preserve existing source provenance; never silently replace a retained manifest.
    if not manifest.exists():
        manifest.write_text(json.dumps(dict(files=files,license=license,source='https://github.com/mpuu00001/Siformer',accessed_at='2026-10-01',processing='Author skeletal CSVs; rectification provenance unresolved'),indent=2))
    print(json.dumps(dict(dataset=dataset,files=files)))
PY
