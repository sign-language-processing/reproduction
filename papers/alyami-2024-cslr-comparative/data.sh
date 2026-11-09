#!/usr/bin/env bash
# Populates the shared Modal Volume `datasets` for this paper. Idempotent:
# files already present and verified are kept. Run from the repository root.
#   /datasets/gsl-continuous      Zenodo 3941811 RGB zips (CC BY 4.0) + Drive split files
#   /datasets/arab-sign/RGB/videos  the six RGB 7z archives extracted to mp4
# Phoenix2014 is already on the Volume (rwth-phoenix-2014) and is only checked.
set -euo pipefail

W=.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh
C=.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh
A=papers/alyami-2024-cslr-comparative/modal_app.py

$W run "$A::launch_populate_gsl"
$W run "$A::launch_extract_arabsign"

$C rwth-phoenix-2014 raw/phoenix2014-release/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv
$C gsl-continuous manifest.json
$C arab-sign RGB/videos/manifest.json
