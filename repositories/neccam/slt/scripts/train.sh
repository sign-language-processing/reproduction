#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(git -C "${paper_dir}" rev-parse --show-toplevel)"

"${repo_root}/.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh" \
  run --detach "${paper_dir}/modal_app.py::train"
