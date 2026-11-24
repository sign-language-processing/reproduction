#!/usr/bin/env bash
# Populate /datasets/bosphorussign22k on the shared Modal Volume from the dataset
# owners' Google Drive folder. Safe to re-run. Pass --depth to include depth.tar.gz. Needs the Modal Secret
# 'gdrive-rclone' holding a read-only rclone login for that folder.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${here}/../../.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh" run --detach "${here}/modal_app.py::populate" "$@"
