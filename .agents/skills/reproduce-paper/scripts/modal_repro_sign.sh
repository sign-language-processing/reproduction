#!/usr/bin/env bash
set -euo pipefail

required_profile="repro-sign"
install_command="python3 -m pip install modal"
setup_command="modal setup"

if ! command -v modal >/dev/null 2>&1; then
  echo "Modal CLI is unavailable; installing it with: ${install_command}" >&2
  if ! python3 -m pip install modal; then
    echo "Modal installation failed." >&2
    exit 2
  fi
  hash -r
  echo "Modal is installed. Ask the user to run '${setup_command}' and select workspace '${required_profile}', then retry." >&2
  exit 2
fi

export MODAL_PROFILE="${required_profile}"

if ! current_profile="$(modal profile current 2>/dev/null)"; then
  echo "Modal profile '${required_profile}' is unavailable. Ask the user to run '${setup_command}' and select that workspace." >&2
  exit 2
fi

if [[ "${current_profile}" != "${required_profile}" ]]; then
  echo "Refusing Modal operation: expected profile '${required_profile}', got '${current_profile}'." >&2
  echo "Ask the user to run '${setup_command}' and select workspace '${required_profile}'." >&2
  exit 2
fi

if ! token_info="$(modal token info 2>/dev/null)"; then
  echo "Modal credentials for '${required_profile}' are invalid or expired." >&2
  echo "Ask the user to run '${setup_command}' and select that workspace." >&2
  exit 2
fi

if [[ "${token_info}" != *"Workspace: ${required_profile} ("* ]]; then
  echo "Refusing Modal operation: profile '${required_profile}' is not authenticated to workspace '${required_profile}'." >&2
  echo "Ask the user to run '${setup_command}' and select workspace '${required_profile}'." >&2
  exit 2
fi
unset token_info

if [[ $# -eq 0 ]]; then
  echo "Verified Modal profile and workspace '${required_profile}'."
  exit 0
fi

if [[ "$1" == "token" || "$1" == "config" || "$1" == "secret" ]]; then
  echo "Refusing credential-sensitive Modal command through the reproduction wrapper." >&2
  exit 2
fi

exec modal "$@"
