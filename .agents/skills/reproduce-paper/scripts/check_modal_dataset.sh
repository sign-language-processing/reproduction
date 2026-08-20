#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 DATASET_SLUG [MANIFEST_RELATIVE_PATH]" >&2
  exit 2
fi

dataset_slug="$1"
manifest_path="${2:-}"

if [[ ! "${dataset_slug}" =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
  echo "Invalid dataset slug '${dataset_slug}'. Use lowercase letters, digits, dots, underscores, or hyphens." >&2
  exit 2
fi

if [[ -n "${manifest_path}" && ("${manifest_path}" == /* || "${manifest_path}" == *".."*) ]]; then
  echo "Manifest path must be relative to the dataset directory and cannot contain '..'." >&2
  exit 2
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
modal_wrapper="${script_directory}/modal_repro_sign.sh"
dataset_path="${dataset_slug}/"

if ! "${modal_wrapper}" volume ls huggingface-cache / --json >/dev/null; then
  echo "Required shared Volume 'huggingface-cache' is absent." >&2
  exit 1
fi

if ! listing="$("${modal_wrapper}" volume ls datasets "${dataset_path}" --json)"; then
  echo "Requested dataset '${dataset_slug}' is absent from Modal Volume 'datasets'." >&2
  exit 1
fi

if [[ "${listing}" == "[]" ]]; then
  echo "Requested dataset '${dataset_slug}' exists but is empty in Modal Volume 'datasets'." >&2
  exit 1
fi

if [[ -n "${manifest_path}" ]]; then
  if ! "${modal_wrapper}" volume ls datasets "${dataset_slug}/${manifest_path}" --json >/dev/null; then
    echo "Dataset '${dataset_slug}' is missing required manifest '${manifest_path}'." >&2
    exit 1
  fi
fi

echo "Confirmed dataset '${dataset_slug}' in Modal Volume 'datasets' at '${dataset_path}'."
