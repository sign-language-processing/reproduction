#!/usr/bin/env bash
set -euo pipefail

# Populate only the paper-cited ISL-HS source, pinned to this Git revision.
# The user authorized project-cloud use on 2026-08-25.  The Table III
# evaluation protocol remains separately unresolved.
dataset_root="${1:-/datasets/isl-hs}"
source_revision="d1d50bb65540b904e3e0a6ffe0997872c4e9e645"

if [[ -f "${dataset_root}/manifest.json" ]]; then
  python3 - "${dataset_root}" "${source_revision}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("source_revision") != sys.argv[2] or manifest.get("video_count") != 468:
    raise SystemExit("existing ISL-HS manifest does not match the pinned 468-video source")
if len(list((root / "videos").rglob("*.mov"))) != 468:
    raise SystemExit("existing ISL-HS video count does not match its manifest")
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
mkdir -p "${staging_root}/archives" "${staging_root}/videos"

python3 - "${staging_root}" "${source_revision}" <<'PY'
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1])
revision = sys.argv[2]
archive_dir = root / "archives"
videos_dir = root / "videos"
archives = []

for person in range(1, 7):
    relative = f"Videos/Person{person}.zip"
    url = f"https://raw.githubusercontent.com/marlondcu/ISL/{revision}/{relative}"
    archive = archive_dir / f"Person{person}.zip"
    urllib.request.urlretrieve(url, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        for member in members:
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe archive entry: {member.filename}")
        video_members = [member for member in members if not member.is_dir()]
        if len(video_members) != 78 or any(not member.filename.endswith(".mov") for member in video_members):
            raise ValueError(f"Person{person} archive is not the expected 78-video release")
        zipped.extractall(videos_dir)
    archives.append({"path": relative, "url": url, "sha256": digest, "size_bytes": archive.stat().st_size})
    archive.unlink()

videos = sorted(videos_dir.rglob("*.mov"))
if len(videos) != 468:
    raise ValueError(f"expected 468 ISL-HS videos, found {len(videos)}")
paths = [str(path.relative_to(root)) for path in videos]
manifest = {
    "source_repository": "https://github.com/marlondcu/ISL",
    "source_revision": revision,
    "source_tree_sha256": "f10a08e008a71025696686c9413ce98c79e78785df74d083df40143355841046",
    "archive_count": len(archives),
    "archives": archives,
    "video_count": len(videos),
    "relative_video_paths_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest(),
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

mv "${staging_root}" "${dataset_root}"
trap - EXIT
