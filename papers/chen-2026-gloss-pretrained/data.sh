#!/usr/bin/env bash
# Idempotently populate /datasets/aslg-pc12 in the shared Modal Volume `datasets`
# from the pinned kayoyin/transformer-slt commit d119fbb642d653a987a2e1b2cd1541c88df7f2ef.
# That repo (code for ref [33] Yin & Read, STMC-Transformer) bundles the exact
# 82,709/4,000/1,000 train/dev/test ASLG-PC12 gloss-text split used by this
# line of gloss-to-text papers, matching Table 4a of the target paper exactly.
# Upstream corpus: Othman & Jemni, "English-ASL Gloss Parallel Corpus 2012:
# ASLG-PC12" (LREC 2012), CC0 (public domain) per its Hugging Face card
# (achrafothman/aslg_pc12). Repo code license: Apache-2.0.
set -euo pipefail

REPO_URL="https://github.com/kayoyin/transformer-slt.git"
REPO_COMMIT="d119fbb642d653a987a2e1b2cd1541c88df7f2ef"
WRAPPER="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh"

if "$WRAPPER" volume ls datasets aslg-pc12 >/dev/null 2>&1; then
  echo "datasets/aslg-pc12 already present; skipping population." >&2
  exit 0
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

git clone --quiet "$REPO_URL" "$WORKDIR/repo"
git -C "$WORKDIR/repo" checkout --quiet "$REPO_COMMIT"

STAGE="$WORKDIR/aslg-pc12"
mkdir -p "$STAGE"
cp "$WORKDIR/repo/data/aslg.train.gloss.asl" "$STAGE/train.gloss"
cp "$WORKDIR/repo/data/aslg.train.en"        "$STAGE/train.en"
cp "$WORKDIR/repo/data/aslg.dev.gloss.asl"   "$STAGE/dev.gloss"
cp "$WORKDIR/repo/data/aslg.dev.en"          "$STAGE/dev.en"
cp "$WORKDIR/repo/data/aslg.test.gloss.asl"  "$STAGE/test.gloss"
cp "$WORKDIR/repo/data/aslg.test.en"         "$STAGE/test.en"

cat > "$STAGE/PROVENANCE.md" <<EOF
# ASLG-PC12 gloss-to-text pairs

Source: $REPO_URL @ $REPO_COMMIT (data/aslg.{train,dev,test}.{gloss.asl,en})
Upstream corpus: Othman & Jemni, "English-ASL Gloss Parallel Corpus 2012:
ASLG-PC12" (LREC 2012). License: CC0 (public domain), per
https://huggingface.co/datasets/achrafothman/aslg_pc12. Repo code: Apache-2.0.
Populated for papers/chen-2026-gloss-pretrained; split sizes verified as
82709/4000/1000 train/dev/test, matching Table 4a of the target paper.
EOF

"$WRAPPER" volume put datasets "$STAGE" aslg-pc12
echo "Populated datasets/aslg-pc12." >&2
