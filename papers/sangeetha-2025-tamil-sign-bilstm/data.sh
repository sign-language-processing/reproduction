#!/usr/bin/env bash
# Idempotently populate /datasets/tlfs23 in the shared Modal Volume `datasets`
# with class folders 1-13 of TLFS23 v2 (Mendeley Data 39kzs5pxmk, CC BY 4.0):
# the 12 Tamil vowels plus the aytham, the 13 classes in the paper's Figure 2.
# Each file is verified against the SHA-256 that Mendeley publishes for it.
# The other 235 TLFS23 classes are not stored.
set -euo pipefail

WRAPPER="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh"

if "$WRAPPER" volume ls datasets tlfs23/MANIFEST.sha256 >/dev/null 2>&1; then
  echo "datasets/tlfs23 already present; skipping population." >&2
  exit 0
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
STAGE="$WORKDIR/tlfs23"

python3 - "$STAGE" <<'EOF'
import hashlib, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://data.mendeley.com/public-api/datasets/39kzs5pxmk"
stage = Path(sys.argv[1])


def get(url):
    return subprocess.run(["curl", "-sSfL", "--retry", "5", "--retry-all-errors", "--retry-delay", "2", "-A", "Mozilla/5.0", url], check=True, capture_output=True).stdout


folders = {f["name"]: f["id"] for f in json.loads(get(f"{API}/folders/2"))}
jobs = []
for name in map(str, range(1, 14)):
    files = json.loads(get(f"{API}/files?folder_id={folders[name]}&version=2"))
    assert len(files) == 1000, (name, len(files))
    jobs += [(stage / name / f["filename"], f["content_details"]) for f in files]


def fetch(job):
    path, details = job
    data = get(details["download_url"])
    if hashlib.sha256(data).hexdigest() != details["sha256_hash"]:
        raise ValueError(f"checksum mismatch: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return f"{details['sha256_hash']}  {path.relative_to(stage)}"


with ThreadPoolExecutor(6) as pool:
    lines = sorted(pool.map(fetch, jobs), key=lambda line: line.split("  ", 1)[1])
(stage / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
print(f"downloaded and verified {len(lines)} files", file=sys.stderr)
EOF

cat > "$STAGE/PROVENANCE.md" <<EOF
# TLFS23 (class folders 1-13 only)

Source: Mendeley Data, "TLFS23 - Tamil Language Finger Spelling Image Dataset",
version 2, https://data.mendeley.com/datasets/39kzs5pxmk/2 (CC BY 4.0).
Data paper: TLFS23 Tamil language fingerspelling dataset, Data in Brief 2024 (PMC10790027).
Content: folders 1-13 (அ ஆ இ ஈ உ ஊ எ ஏ ஐ ஒ ஓ ஔ ஃ), 1,000 JPEG images each, 640x480.
Every file was checked against the SHA-256 published by Mendeley; MANIFEST.sha256 lists them.
Folders 14-248, Background and the reference images are not stored.
Populated for papers/sangeetha-2025-tamil-sign-bilstm on $(date -u +%F).
EOF

"$WRAPPER" volume put datasets "$STAGE" tlfs23
echo "Populated datasets/tlfs23." >&2
