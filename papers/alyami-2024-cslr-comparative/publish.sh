#!/usr/bin/env bash
# Publish the retained (evaluated) checkpoint of each of the 55 full runs to
# https://huggingface.co/repro-sign/alyami-2024-cslr-comparative under CC BY-NC-SA 4.0,
# with this paper's README.md as the model card.
#
# Uploads only: README.md, SHA256SUMS and checkpoints/<run_id>-epoch<NN>.pt.
# No dataset, prediction, log, gloss dictionary or reproduction.json is uploaded.
#
# usage: publish.sh STAGING_DIR [--stage-only]
#   Downloads each checkpoint listed in reproduction.json (kind "checkpoint") from the
#   results Volume into STAGING_DIR/checkpoints and checks its recorded sha256 (files that
#   already match are kept), then creates the repo and uploads. Needs `hf auth login`
#   with write access to the repro-sign organization; --stage-only stops before that.
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "${paper_dir}" rev-parse --show-toplevel)"
wrapper="${repo_root}/.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh"
hf_repo="repro-sign/alyami-2024-cslr-comparative"
stage="${1:?usage: publish.sh STAGING_DIR [--stage-only]}"
mkdir -p "${stage}/checkpoints"

# artifact_id, volume path, published file name, sha256 for every retained checkpoint
jq -r '.artifacts[] | select(.kind == "checkpoint")
  | [.artifact_id,
     (.uri | sub("^modal://repro-sign/alyami-2024-cslr-results/"; "")),
     ((.artifact_id | sub("-checkpoint-"; "-")) + ".pt"),
     .sha256] | @tsv' "${paper_dir}/reproduction.json" > "${stage}/checkpoints.tsv"
[[ "$(wc -l < "${stage}/checkpoints.tsv")" -eq 55 ]] || { echo "expected 55 checkpoints" >&2; exit 1; }

: > "${stage}/SHA256SUMS.tmp"
while IFS=$'\t' read -r artifact_id volume_path name sha; do
  target="${stage}/checkpoints/${name}"
  if [[ ! -f "${target}" ]] || [[ "$(shasum -a 256 "${target}" | cut -d' ' -f1)" != "${sha}" ]]; then
    "${wrapper}" volume get --force alyami-2024-cslr-results "${volume_path}" "${target}" > /dev/null
  fi
  actual="$(shasum -a 256 "${target}" | cut -d' ' -f1)"
  [[ "${actual}" == "${sha}" ]] || { echo "${artifact_id}: sha256 ${actual}, recorded ${sha}" >&2; exit 1; }
  echo "${sha}  checkpoints/${name}" >> "${stage}/SHA256SUMS.tmp"
done < "${stage}/checkpoints.tsv"
mv "${stage}/SHA256SUMS.tmp" "${stage}/SHA256SUMS"
cp "${paper_dir}/README.md" "${stage}/README.md"
echo "staged 55 checkpoints, all sha256 verified"
[[ "${2:-}" == "--stage-only" ]] && exit 0

hf repos create "${hf_repo}" --exist-ok --public
hf upload "${hf_repo}" "${stage}" . \
  --include "README.md" --include "SHA256SUMS" --include "checkpoints/*.pt" \
  --commit-message "Publish the 55 retained full-run checkpoints of the Alyami 2024 CSLR reproduction"
python3 -c "from huggingface_hub import HfApi; print(HfApi().repo_info('${hf_repo}').sha)"
