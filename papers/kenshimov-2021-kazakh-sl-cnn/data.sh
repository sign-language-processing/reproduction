#!/usr/bin/env bash
# Populate /datasets/ksl-fingerspelling-kenshimov-2021 on the shared Modal Volume
# from the authors' archive resized_gray_dataset.7z. Safe to re-run.
set -euo pipefail

archive="${1:?Usage: data.sh /path/to/resized_gray_dataset.7z}"
expected="baf1f1436f2d827b03d88c6b42db783d4e9ca0b84b7bb82fc8ee2f133e494ec2"
slug="ksl-fingerspelling-kenshimov-2021"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
modal="${here}/../../.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh"

actual="$(shasum -a 256 "${archive}" | cut -d' ' -f1)"
if [[ "${actual}" != "${expected}" ]]; then
  echo "Archive SHA-256 ${actual} does not match ${expected}." >&2
  exit 1
fi

if ! "${modal}" volume ls datasets "${slug}/source/resized_gray_dataset.7z" >/dev/null 2>&1; then
  "${modal}" volume put datasets "${archive}" "${slug}/source/resized_gray_dataset.7z"
fi
"${modal}" run "${here}/modal_app.py::populate"
