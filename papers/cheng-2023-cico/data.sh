#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from pathlib import Path
import json, hashlib, zipfile, gdown, datetime

for file_id, root, filename in [
    (
        "1Vb-HFZd-rhjN49sB5WwLRpIbyhiC6xTy",
        Path("/datasets/cico-features"),
        "sign_features.zip",
    ),
    (
        "1Hpcn5obCcG5JHa3nLvHqX9pfrp7g6wDu",
        Path("/outputs/released-weights"),
        "final_models.zip",
    ),
]:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    archive = root / filename
    expected = {
        "sign_features.zip": "9ba1956cf416df9a31ae3d1a71a3fa9a2d1e2b3724670288b608c8d4eb895c51",
        "final_models.zip": "f02ea0b2a64123b2c8386ccb467c00143454ee6405ed4c07e074dcc712224c6c",
    }
    if manifest.exists() and archive.exists():
        m = json.loads(manifest.read_text())
        assert (
            hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
            == expected[filename]
        ), "Retained archive checksum mismatch"
        if all(
            (root / e["path"]).is_file()
            and (root / e["path"]).stat().st_size == e["bytes"]
            for e in m["entries"]
        ):
            print({k: v for k, v in m.items() if k != "entries"})
            continue
    if not archive.exists():
        partial = root / (filename + ".partial")
        gdown.download(id=file_id, output=str(partial), quiet=False, resume=True)
        partial.rename(archive)
    sha = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    assert sha == expected[filename], "Archive changed; inspect source before using"
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            dest = (root / item.filename).resolve()
            if not dest.is_relative_to(root.resolve()):
                raise ValueError("Unsafe archive path")
        z.extractall(root)
        entries = [
            {"path": i.filename, "bytes": i.file_size, "crc32": f"{i.CRC:08x}"}
            for i in z.infolist()
            if not i.is_dir()
        ]
    data = {
        "url": f"https://drive.google.com/file/d/{file_id}/view",
        "archive": filename,
        "sha256": sha,
        "bytes": archive.stat().st_size,
        "retrieved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "entries": entries,
    }
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: v for k, v in data.items() if k != "entries"}))
    print("entries", len(entries))
    print("top paths", entries[:10])
PY
