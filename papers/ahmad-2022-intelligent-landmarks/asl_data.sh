#!/usr/bin/env bash
set -euo pipefail

# Populate the complete user-authorized ASL Alphabet release. The paper says
# "28 gestures"; evaluation selects A-Z, SPACE, DELETE and excludes NOTHING.
dataset_root="${1:-/datasets/asl-alphabet}"
source_url="https://www.kaggle.com/api/v1/datasets/download/grassknoted/asl-alphabet"
source_bytes=1100887034

if [[ -f "${dataset_root}/manifest.json" ]]; then
  python3 - "${dataset_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
selected = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DELETE"]
if manifest.get("training_classes") != selected or manifest.get("excluded_classes") != ["NOTHING"]:
    raise SystemExit("existing ASL manifest does not match the documented 28-class training choice")
if manifest.get("stored_training_image_count") != 87000 or manifest.get("selected_training_image_count") != 84000:
    raise SystemExit("existing ASL manifest has unexpected training-image counts")
if not (root / manifest.get("training_root", "")).is_dir():
    raise SystemExit("existing ASL manifest's training root is absent")
print(json.dumps(manifest, sort_keys=True))
PY
  exit 0
fi

if [[ -e "${dataset_root}" ]]; then
  echo "refusing to populate nonempty unmanifested path: ${dataset_root}" >&2
  exit 1
fi

staging_root="${dataset_root}.staging-$$"
trap 'rm -rf -- "${staging_root}"' EXIT
mkdir -p "${staging_root}"

python3 - "${staging_root}" "${source_url}" "${source_bytes}" <<'PY'
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1])
url = sys.argv[2]
expected_bytes = int(sys.argv[3])
archive = root / "source.zip"
all_classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DELETE", "NOTHING"]
selected_classes = all_classes[:-1]

urllib.request.urlretrieve(url, archive)
if archive.stat().st_size != expected_bytes:
    raise ValueError(f"unexpected source archive size: {archive.stat().st_size}")
digest = hashlib.sha256()
with archive.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
archive_sha256 = digest.hexdigest()
with zipfile.ZipFile(archive) as zipped:
    for member in zipped.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive entry: {member.filename}")
    zipped.extractall(root / "source")

# Kaggle's bundle may contain its train/test archives as nested ZIPs. Unpack
# those too, after the same path-safety check, so the complete release is
# retained as files rather than an opaque inner archive.
for nested in sorted((root / "source").rglob("*.zip")):
    with zipfile.ZipFile(nested) as zipped:
        for member in zipped.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe nested archive entry: {member.filename}")
        zipped.extractall(nested.parent)
    nested.unlink()

candidate_roots = [
    path for path in (root / "source").rglob("*")
    if path.is_dir() and {child.name for child in path.iterdir() if child.is_dir()} == set(all_classes)
]
if len(candidate_roots) != 1:
    directories = sorted(str(path.relative_to(root)) for path in (root / "source").rglob("*") if path.is_dir())
    raise ValueError(f"could not identify exactly one 29-class training directory: {candidate_roots}; directories={directories[:100]}")
training_root = candidate_roots[0]
counts = {name: sum(1 for path in (training_root / name).rglob("*") if path.is_file()) for name in all_classes}
if any(count != 3000 for count in counts.values()):
    raise ValueError(f"unexpected training class counts: {counts}")
manifest = {
    "source": {
        "dataset": "grassknoted/asl-alphabet",
        "version": 1,
        "url": url,
        "archive_size_bytes": expected_bytes,
        "archive_sha256": archive_sha256,
    },
    "training_root": str(training_root.relative_to(root)),
    "stored_classes": all_classes,
    "training_classes": selected_classes,
    "excluded_classes": ["NOTHING"],
    "stored_images_per_class": counts,
    "stored_training_image_count": sum(counts.values()),
    "selected_training_image_count": sum(counts[name] for name in selected_classes),
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
archive.unlink()
PY

mv "${staging_root}" "${dataset_root}"
trap - EXIT
