#!/usr/bin/env bash
# One-time agent tooling setup for this repo. Safe to re-run.
set -euo pipefail

# Simplicity bias for agents: https://github.com/DietrichGebert/ponytail
if command -v claude >/dev/null; then
  claude plugin marketplace add DietrichGebert/ponytail
  claude plugin install ponytail@ponytail
else
  echo "Claude CLI not found; skipping optional ponytail plugin." >&2
fi

# Modal's own generic agent skill + docs; we do not maintain our own copy.
if ! command -v modal >/dev/null; then
  python3 -m pip install modal
  echo "Modal is installed. Run 'modal setup', select workspace 'repro-sign', then re-run ./setup.sh." >&2
  exit 1
fi

.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh skills install -y

docker pull ghcr.io/sign-language-processing/reproduction:latest

echo "Done. Read AGENTS.md, then use the reproduce-paper skill."
