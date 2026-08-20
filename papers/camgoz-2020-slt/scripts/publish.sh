#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(git -C "${paper_dir}" rev-parse --show-toplevel)"
wrapper="${repo_root}/.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh"
publish_root="$(mktemp -d /tmp/neccam-slt-publish.XXXXXX)"
trap 'rm -rf -- "${publish_root}"' EXIT

checkpoint_listing="$("${wrapper}" volume ls --json \
  neccam-slt-results neccam-slt/pami0-seed-42/)"
checkpoint_path="$(jq -r \
  '.[] | select(.type == "file" and (.filename | test("/[0-9]+[.]ckpt$"))) | .filename' \
  <<<"${checkpoint_listing}")"
if [[ -z "${checkpoint_path}" || "${checkpoint_path}" == *$'\n'* ]]; then
  echo "Expected exactly one retained checkpoint, found: ${checkpoint_path}" >&2
  exit 1
fi

"${wrapper}" volume get neccam-slt-results \
  "${checkpoint_path}" "${publish_root}/model.ckpt"
"${wrapper}" volume get neccam-slt-results \
  neccam-slt/pami0-seed-42/config.yaml \
  "${publish_root}/config.yaml"
"${wrapper}" volume get neccam-slt-results \
  neccam-slt/pami0-seed-42/modal-run.json \
  "${publish_root}/modal-run.json"

mkdir -p "${publish_root}/upload"
cp "${publish_root}/model.ckpt" "${publish_root}/upload/model.ckpt"
cp "${publish_root}/config.yaml" "${publish_root}/upload/config.yaml"
cp "${paper_dir}/README.md" "${publish_root}/upload/README.md"
cp "${paper_dir}/reproduction.json" "${publish_root}/upload/reproduction.json"
cp "${publish_root}/modal-run.json" "${publish_root}/upload/modal-run.json"
(cd "${publish_root}/upload" && shasum -a 256 \
  model.ckpt config.yaml reproduction.json modal-run.json > SHA256SUMS)

hf repos create repro-sign/neccam-slt --exist-ok --public
hf upload repro-sign/neccam-slt "${publish_root}/upload" . \
  --commit-message "Publish reproduced neccam/slt seed-42 checkpoint"
