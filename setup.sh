#!/usr/bin/env bash
# One-time agent tooling setup for this repo. Safe to re-run.
set -euo pipefail

# Simplicity bias for agents: https://github.com/DietrichGebert/ponytail
claude plugin marketplace add DietrichGebert/ponytail
claude plugin install ponytail@ponytail

# Modal's own agent skills + docs; we do not maintain our own copy.
if command -v modal >/dev/null; then
  modal skills install --claude -y
else
  echo "modal CLI not found: pip install modal, then re-run to get its skills" >&2
fi

docker pull ghcr.io/sign-language-processing/reproduction:latest

echo "Done. Read CLAUDE.md, then use the reproduce-paper skill."
