#!/usr/bin/env bash
# Populate the shared Modal Volume: three public hand datasets from their official
# pages, and /datasets/bosphorussign22k from a local copy of the dataset owners'
# Google Drive folder. Safe to re-run: files already uploaded are skipped.
#
# Usage: data.sh FILE...
#   FILE...  the zips Google Drive produces when the folder's items are downloaded
#            through its web interface, the two CSV index files, the licence
#            agreement, and drive-md5.txt, the folder's MD5 list from
#            `rclone md5sum REMOTE: --fast-list --exclude /depth.tar.gz | sort -k2`.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
modal="${here}/../../.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh"

# Public hand-segmentation datasets: hgr2a, hgr2b, rendered-handpose-dataset.
"${modal}" run --detach "${here}/modal_app.py::populate_public"

present="$("${modal}" volume ls datasets bosphorussign22k/source/ 2>/dev/null || true)"
for file in "$@"; do
  name="$(basename -- "${file}")"
  if ! grep -qxF "bosphorussign22k/source/${name}" <<<"${present}"; then
    "${modal}" volume put datasets "${file}" "bosphorussign22k/source/${name}"
  fi
done
"${modal}" run --detach "${here}/modal_app.py::populate"
