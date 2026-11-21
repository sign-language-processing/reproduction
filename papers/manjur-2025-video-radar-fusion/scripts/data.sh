#!/usr/bin/env bash
# Populate the restricted MultiMeDaLIS Track-2 copy from the Kaggle competition.
#
# Runs inside Modal (scripts/modal_app.py::data) in Restricted Environment
# $MODAL_ENV, with Secret kaggle-credentials (KAGGLE_API_TOKEN) and that
# environment's Volume `datasets` mounted read-write at /datasets.
#
#   populate  (default) download, extract, verify, write manifest and provenance;
#             if a verified copy already exists, only re-verify it
#   verify    re-check counts and every file checksum against MANIFEST.sha256
#
# Expected counts: train 126 class folders x 100 samples = 12,600; val 4,410
# (35/class); test 7,560 (60/class; the paper's "70 per class" is contradicted
# by the organisers' summary and the authors' own prediction files). Val and
# test folders are class-agnostic, so only their totals can be checked.
#
# The kaggle CLI reads the token from the environment. This script never prints
# the environment, never enables xtrace, and writes raw CLI output only to a
# temporary file that is not copied into the Volume.
set -euo pipefail
set +x
export LC_ALL=C

COMPETITION=iccv-mslr-2025-track-2
ROOT=/datasets/multimedalis-iccv-mslr-2025-track-2
RAW="$ROOT/raw"
MANIFEST="$ROOT/MANIFEST.sha256"
PROVENANCE="$ROOT/PROVENANCE.json"
INVENTORY="$ROOT/INVENTORY.json"

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Count splits, check the expected totals, and write INVENTORY.json.
inventory() {
  python3 - "$RAW" "$INVENTORY" <<'EOF'
import collections, json, re, sys
from pathlib import Path

raw, out = Path(sys.argv[1]), Path(sys.argv[2])
sample = re.compile(r"^SAMPLE_\d+$")
def dirs(name):
    return sorted(p for p in raw.rglob(name) if p.is_dir() and "__MACOSX" not in p.parts)
train_dirs, val_dirs, test_dirs = dirs("train"), dirs("val"), dirs("test")
assert len(train_dirs) == 1 and len(val_dirs) == 1 and len(test_dirs) == 1, (train_dirs, val_dirs, test_dirs)
train, val, test = train_dirs[0], val_dirs[0], test_dirs[0]

classes = sorted((p for p in train.iterdir() if p.is_dir()), key=lambda p: int(p.name.split("_")[0]))
per_class = {c.name: sum(1 for s in c.iterdir() if s.is_dir() and sample.match(s.name)) for c in classes}
val_samples = [s for s in val.iterdir() if s.is_dir() and sample.match(s.name)]
test_samples = [s for s in test.iterdir() if s.is_dir() and sample.match(s.name)]

def file_kinds(samples):
    kinds = collections.Counter()
    for s in samples:
        names = sorted(f.name for f in s.iterdir() if f.is_file())
        kinds[" + ".join(sorted(collections.Counter(
            ("RDM" if "RDM" in n.upper() else "other") + Path(n).suffix.lower() for n in names
        ).elements()))] += 1
    return dict(kinds)

train_samples = [s for c in classes for s in c.iterdir() if s.is_dir() and sample.match(s.name)]
ids = [int(s.name.split("_")[1]) for s in train_samples + val_samples + test_samples]
report = {
    "split_dirs": {k: str(v.relative_to(raw)) for k, v in (("train", train), ("val", val), ("test", test))},
    "classes": [c.name for c in classes],
    "train_per_class": per_class,
    "counts": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
    "unique_sample_ids": len(set(ids)),
    "sample_file_kinds": {"train": file_kinds(train_samples), "val": file_kinds(val_samples), "test": file_kinds(test_samples)},
}
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: report[k] for k in ("split_dirs", "counts", "unique_sample_ids", "sample_file_kinds")}, indent=2))
problems = []
if len(classes) != 126 or sorted(int(c.name.split("_")[0]) for c in classes) != list(range(126)):
    problems.append(f"expected class folders 0..125, found {len(classes)}")
bad = {k: v for k, v in per_class.items() if v != 100}
if bad:
    problems.append(f"train classes without 100 samples: {bad}")
for split, n, expected in (("val", len(val_samples), 4410), ("test", len(test_samples), 7560)):
    if n != expected:
        problems.append(f"{split}: expected {expected} samples, found {n}")
if len(set(ids)) != len(ids):
    problems.append("sample ids are not unique across splits")
if problems:
    sys.exit("inventory check failed: " + "; ".join(problems))
print("inventory check passed")
EOF
}

write_manifest() {
  (cd "$ROOT" && find raw -type f -print0 | xargs -0 -P 8 -n 256 sha256sum | sort -k2) > "$MANIFEST.tmp"
  mv "$MANIFEST.tmp" "$MANIFEST"
}

verify() {
  test -s "$MANIFEST" && test -s "$PROVENANCE" || { echo "no verified copy at $ROOT" >&2; return 1; }
  inventory
  local listed actual
  listed=$(wc -l < "$MANIFEST")
  actual=$(cd "$ROOT" && find raw -type f | wc -l)
  [[ "$listed" == "$actual" ]] || { echo "manifest lists $listed files, found $actual" >&2; return 1; }
  (cd "$ROOT" && sha256sum --quiet --check "$MANIFEST")
  echo "verified $listed files against $(sha256sum "$MANIFEST" | cut -d' ' -f1)"
}

populate() {
  if [[ -s "$PROVENANCE" ]]; then
    echo "existing copy found; re-verifying"
    verify
    return
  fi
  local started download archive log origin
  started=$(now)
  download="$ROOT/.download"
  archive="$download/$COMPETITION.zip"
  mkdir -p "$download"
  # An archive staged here by the operator (modal volume put) is used if intact.
  if [[ -s "$archive" ]] && unzip -tq "$archive" >/dev/null 2>&1; then
    origin="staged archive (operator upload of the Kaggle competition download)"
    echo "using staged archive $archive"
  else
    : "${KAGGLE_API_TOKEN:?Secret kaggle-credentials is not mounted}"
    rm -f "$archive"
    origin="kaggle competitions download inside Modal"
    log=$(mktemp)
    echo "downloading $COMPETITION archive"
    if ! kaggle competitions download -c "$COMPETITION" -p "$download" >"$log" 2>&1; then
      echo "kaggle download failed; last lines:" >&2
      tail -n 5 "$log" >&2
      return 1
    fi
    rm -f "$log"
  fi
  pip freeze > "$ROOT/PIP_FREEZE.txt"
  test -s "$archive"
  local archive_bytes archive_sha
  archive_bytes=$(stat -c %s "$archive")
  archive_sha=$(sha256sum "$archive" | cut -d' ' -f1)
  echo "archive $archive_bytes bytes, sha256 $archive_sha"

  rm -rf "$RAW.tmp"
  mkdir -p "$RAW.tmp"
  unzip -q "$archive" -d "$RAW.tmp"
  rm -rf "$RAW"
  mv "$RAW.tmp" "$RAW"
  inventory
  write_manifest

  python3 - "$PROVENANCE" "$COMPETITION" "$started" "$(now)" "$archive_bytes" "$archive_sha" \
      "$(wc -l < "$MANIFEST")" "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" \
      "$(sha256sum "$INVENTORY" | cut -d' ' -f1)" "$origin" <<'EOF'
import json, sys
(path, competition, started, finished, size, sha, files, manifest_sha, inventory_sha, origin) = sys.argv[1:]
json.dump({
    "dataset": "MultiMeDaLIS, ICCV MSLR 2025 Track 2 release",
    "source": f"https://www.kaggle.com/competitions/{competition}/data",
    "competition_id": 102067,
    "access": "Kaggle account that joined the competition and accepted its rules (Modal Secret kaggle-credentials in the paper's Restricted Environment)",
    "license": "CC BY-NC-SA 4.0 (Kaggle competition license field); competition rules apply",
    "download_command": f"kaggle competitions download -c {competition}",
    "archive_origin": origin,
    "kaggle_cli": "kaggle==2.2.4, kagglesdk==0.1.38",
    "started_at_utc": started,
    "finished_at_utc": finished,
    "archive": {"name": f"{competition}.zip", "bytes": int(size), "sha256": sha,
                "kept": False, "note": "deleted from the Volume after verified extraction"},
    "extracted_root": "raw/",
    "manifest": {"path": "MANIFEST.sha256", "files": int(files), "sha256": manifest_sha},
    "inventory": {"path": "INVENTORY.json", "sha256": inventory_sha},
    "environment": "PIP_FREEZE.txt",
}, open(path, "w"), indent=2)
EOF
  rm -rf "$download"
  echo "populated $ROOT; provenance:"
  cat "$PROVENANCE"
}

case "${1:-populate}" in
  populate) populate ;;
  verify) verify ;;
  *) echo "usage: data.sh [populate|verify]" >&2; exit 2 ;;
esac
