#!/usr/bin/env bash
# One command: prepare the node-local Enroot runtime, then launch training.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/runtime/run.sh" "$@"
