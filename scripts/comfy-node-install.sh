#!/usr/bin/env bash
# comfy-node-install: install custom ComfyUI nodes and fail with non-zero
# exit code if any of them cannot be installed.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: comfy-node-install <node1> [<node2> ...]" >&2
  exit 64
fi

log=$(mktemp)

set +e
comfy node install --mode=remote "$@" 2>&1 | tee "$log"
cli_status=$?
set -e

failed_nodes=$(grep -oP "(?<=An error occurred while installing ')[^']+" "$log" | sort -u || true)

if [[ -z "$failed_nodes" ]]; then
  failed_nodes=$(grep -oP "(?<=Node ')[^@']+" "$log" | sort -u || true)
fi

if [[ -n "$failed_nodes" ]]; then
  echo "Comfy node installation failed for the following nodes:" >&2
  echo "$failed_nodes" | while read -r n; do echo "  * $n" >&2; done
  echo "Please verify the node names at https://registry.comfy.org/ and try again." >&2
  exit 1
fi

if [[ $cli_status -ne 0 ]]; then
  echo "Warning: comfy node install exited with status $cli_status but no node errors were detected." >&2
fi
