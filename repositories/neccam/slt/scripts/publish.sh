#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(git -C "${paper_dir}" rev-parse --show-toplevel)"
wrapper="${repo_root}/.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh"
publish_root="$(mktemp -d /tmp/neccam-slt-publish.XXXXXX)"
trap 'rm -rf -- "${publish_root}"' EXIT

"${wrapper}" volume get \
  neccam-slt-results \
  neccam-slt/full-seed-42/ \
  "${publish_root}/full-seed-42"

mkdir -p "${publish_root}/upload"
best_checkpoint="${publish_root}/full-seed-42/best.ckpt"
if [[ -L "${best_checkpoint}" ]]; then
  checkpoint_target="$(readlink "${best_checkpoint}")"
  cp "${publish_root}/full-seed-42/${checkpoint_target}" \
    "${publish_root}/upload/model.ckpt"
elif [[ -f "${best_checkpoint}" ]]; then
  cp "${best_checkpoint}" "${publish_root}/upload/model.ckpt"
else
  echo "Missing persisted best checkpoint: ${best_checkpoint}" >&2
  exit 1
fi

cp "${paper_dir}/configs/sign.yaml" "${publish_root}/upload/config.yaml"
cp "${paper_dir}/model-card.md" "${publish_root}/upload/README.md"
cp "${paper_dir}/metrics.json" "${publish_root}/upload/metrics.json"
cp "${publish_root}/full-seed-42/modal-result.json" \
  "${publish_root}/upload/modal-result.json"
(cd "${publish_root}/upload" && shasum -a 256 \
  model.ckpt config.yaml metrics.json modal-result.json > SHA256SUMS)

hf repos create repro-sign/neccam-slt --exist-ok --public
hf upload repro-sign/neccam-slt "${publish_root}/upload" . \
  --commit-message "Publish reproduced neccam/slt seed-42 checkpoint"
