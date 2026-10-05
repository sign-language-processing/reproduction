#!/usr/bin/env bash
# Idempotently populate /datasets/dvs-sign-v2e in the shared Modal Volume
# `datasets` from the pinned `master` branch of najie1314/DVS, which the
# paper's Data Availability Statement names as the DVS_Sign_v2e dataset
# (15 classes, 40 train + 10 test event-stream CSVs per class).
#
# Permission basis: the CC BY 4.0 paper states the dataset "has been open
# source" at this URL. The repository itself carries no LICENSE file and the
# events derive from LSA64 (non-commercial terms), so the data is kept only in
# the private project Volume and is never redistributed.
#
# The `main` branch (named as the DAVIS346 "DVS_Sign" dataset) is NOT used: its
# 150 files are byte-identical to classes 0-2 of `master`.
set -euo pipefail

REPO_URL="https://github.com/najie1314/DVS.git"
REPO_COMMIT="439682135dc126337cd1d7f60e50764ec771ad2f"
WRAPPER="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh"

if "$WRAPPER" volume ls datasets dvs-sign-v2e/MANIFEST.sha256 >/dev/null 2>&1; then
  echo "datasets/dvs-sign-v2e already present; skipping population." >&2
  exit 0
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

git clone --quiet --branch master --single-branch "$REPO_URL" "$WORKDIR/repo"
[[ "$(git -C "$WORKDIR/repo" rev-parse HEAD)" == "$REPO_COMMIT" ]] || { echo "unexpected upstream revision" >&2; exit 1; }

STAGE="$WORKDIR/dvs-sign-v2e"
mkdir "$STAGE"
cp -R "$WORKDIR/repo/train" "$WORKDIR/repo/test" "$STAGE/"
[[ "$(find "$STAGE/train" -name '*.csv' | wc -l)" -eq 600 && "$(find "$STAGE/test" -name '*.csv' | wc -l)" -eq 150 ]] \
  || { echo "expected 600 train / 150 test CSV files" >&2; exit 1; }
(cd "$STAGE" && find train test -name '*.csv' | LC_ALL=C sort | xargs shasum -a 256 > MANIFEST.sha256)

cat > "$STAGE/PROVENANCE.md" <<EOF
# DVS_Sign_v2e

Source: $REPO_URL @ $REPO_COMMIT (branch \`master\`, directories train/ and test/)
Paper: Chen et al., Electronics 2023, 12, 786, https://doi.org/10.3390/electronics12040786
Content: 15 classes x (40 train + 10 test) v2e event streams derived from LSA64;
CSV rows are \`timestamp_seconds,x,y,polarity\` on a 128x128 grid.
License: none declared in the repository; the paper's Data Availability
Statement calls the dataset open source. Do not redistribute.
Populated for papers/chen-2023-event-camera-snn-sign; MANIFEST.sha256 lists
the SHA-256 of all 750 files.
EOF

"$WRAPPER" volume put datasets "$STAGE" dvs-sign-v2e
echo "Populated datasets/dvs-sign-v2e." >&2
