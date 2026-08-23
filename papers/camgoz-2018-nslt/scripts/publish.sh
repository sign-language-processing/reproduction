#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(git -C "${paper_dir}" rev-parse --show-toplevel)"
wrapper="${repo_root}/.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh"
remote_root="camgoz-nslt/luong-seed-285"
checkpoint_dir="${remote_root}/best_bleu"
publish_root="$(mktemp -d /tmp/camgoz-nslt-publish.XXXXXX)"
trap 'rm -rf -- "${publish_root}"' EXIT
upload_dir="${publish_root}/upload"
mkdir -p "${upload_dir}"

"${wrapper}" volume get camgoz-nslt-results \
  "${checkpoint_dir}/checkpoint" "${publish_root}/checkpoint"
checkpoint_path="$(sed -n 's/^model_checkpoint_path: "\([^"]*\)"/\1/p' \
  "${publish_root}/checkpoint")"
prefix="${checkpoint_path##*/}"
if [[ -z "${checkpoint_path}" || -z "${prefix}" ]]; then
  echo "Could not resolve the selected checkpoint from ${checkpoint_dir}/checkpoint" >&2
  exit 1
fi

printf 'model_checkpoint_path: "%s"\nall_model_checkpoint_paths: "%s"\n' \
  "${prefix}" "${prefix}" >"${upload_dir}/checkpoint"
for suffix in data-00000-of-00001 index meta; do
  "${wrapper}" volume get camgoz-nslt-results \
    "${checkpoint_dir}/${prefix}.${suffix}" "${upload_dir}/${prefix}.${suffix}"
done
for artifact in hparams output_dev output_test; do
  "${wrapper}" volume get camgoz-nslt-results \
    "${remote_root}/${artifact}" "${upload_dir}/${artifact}"
done

cp "${paper_dir}/README.md" "${upload_dir}/README.md"
cp "${paper_dir}/reproduction.json" "${upload_dir}/reproduction.json"
(cd "${upload_dir}" && shasum -a 256 * >SHA256SUMS)

hf repos create repro-sign/camgoz-nslt --exist-ok --public
hf upload repro-sign/camgoz-nslt "${upload_dir}" . \
  --commit-message "Publish reproduced camgoz/nslt Luong checkpoint"
