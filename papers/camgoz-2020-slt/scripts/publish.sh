#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(git -C "${paper_dir}" rev-parse --show-toplevel)"
wrapper="${repo_root}/.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh"
publish_root="$(mktemp -d /tmp/neccam-slt-publish.XXXXXX)"
trap 'rm -rf -- "${publish_root}"' EXIT
upload_dir="${publish_root}/upload"
mkdir -p "${upload_dir}"

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
  "${checkpoint_path}" "${upload_dir}/model.ckpt"
"${wrapper}" volume get neccam-slt-results \
  neccam-slt/pami0-seed-42/config.yaml \
  "${upload_dir}/config.yaml"

cp "${paper_dir}/README.md" "${upload_dir}/README.md"
cp "${paper_dir}/reproduction.json" "${upload_dir}/reproduction.json"
(cd "${upload_dir}" && shasum -a 256 \
  model.ckpt config.yaml reproduction.json > SHA256SUMS)

hf repos create repro-sign/neccam-slt --exist-ok --public
hf upload repro-sign/neccam-slt "${upload_dir}" . \
  --commit-message "Publish reproduced neccam/slt seed-42 checkpoint"
