#!/usr/bin/env bash
# Populate the shared Modal Volume: three public hand datasets from their official
# pages, and /datasets/bosphorussign22k from the dataset owners' Google Drive folder. Safe to re-run. Pass --depth to include depth.tar.gz. Needs the Modal Secret
# 'gdrive-rclone' holding a read-only rclone login for that folder.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
modal="${here}/../../.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh"
# Public hand-segmentation datasets: hgr2a, hgr2b, rendered-handpose-dataset.
"${modal}" run --detach "${here}/modal_app.py::populate_public"
"${modal}" run --detach "${here}/modal_app.py::populate" "$@"
