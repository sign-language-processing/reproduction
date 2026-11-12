#!/usr/bin/env bash
set -euo pipefail

# Populate the complete Kaggle "American Sign Language Dataset" (ayuraj/asl-dataset,
# version 1, CC0). The archive stores every image twice (asl_dataset/ and the nested
# asl_dataset/asl_dataset/); both copies are kept because the paper's Table 3.2
# support (5030) equals the full archive file count.
dataset_root="${1:-/datasets/asl-dataset}"
source_url="https://www.kaggle.com/api/v1/datasets/download/ayuraj/asl-dataset?datasetVersionNumber=1"
expected_tree_sha256="eac4c06624e1fd8c91f7f500839002024662ecb104186cd205648ede38400c84"

if [[ -f "${dataset_root}/manifest.json" ]]; then
  grep -q "\"tree_sha256\": \"${expected_tree_sha256}\"" "${dataset_root}/manifest.json" || {
    echo "existing manifest does not match the expected file tree" >&2
    exit 1
  }
  cat "${dataset_root}/manifest.json"
  exit 0
fi

if [[ -e "${dataset_root}" ]]; then
  echo "refusing to populate nonempty unmanifested path: ${dataset_root}" >&2
  exit 1
fi

staging_root="${dataset_root}.staging-$$"
trap 'rm -rf -- "${staging_root}"' EXIT
mkdir -p "${staging_root}"

python3 - "${staging_root}" "${source_url}" "${expected_tree_sha256}" <<'PY'
import collections
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1])
url = sys.argv[2]
expected_tree = sys.argv[3]
archive = root / "source.zip"

urllib.request.urlretrieve(url, archive)
archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
tree = hashlib.sha256()
content = set()
counts = collections.defaultdict(collections.Counter)
with zipfile.ZipFile(archive) as zipped:
    names = sorted(name for name in zipped.namelist() if not name.endswith("/"))
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive entry: {name}")
        digest = hashlib.sha256(zipped.read(name)).hexdigest()
        tree.update(f"{name} {digest}\n".encode())
        content.add(digest)
        counts[str(path.parent.parent)][path.parent.name] += 1
    if tree.hexdigest() != expected_tree:
        raise ValueError(f"unexpected file tree digest: {tree.hexdigest()}")
    zipped.extractall(root / "files")

manifest = {
    "source": {
        "dataset": "ayuraj/asl-dataset",
        "version": 1,
        "license": "CC0: Public Domain",
        "url": url,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha256,
    },
    "tree_sha256": expected_tree,
    "tree_sha256_definition": "SHA-256 over sorted lines '<archive path> <file sha256>\\n'",
    "file_count": len(names),
    "unique_file_contents": len(content),
    "images_per_class": {parent: dict(sorted(classes.items())) for parent, classes in sorted(counts.items())},
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
archive.unlink()
PY

mv "${staging_root}" "${dataset_root}"
trap - EXIT
cat "${dataset_root}/manifest.json"
