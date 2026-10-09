#!/usr/bin/env bash
# Same eight-A40 final settings as node A; node B has healthy P2P.
# Hardware/image equivalence confirmed by the deployment author on 2026-10-09.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NCCL_P2P_DISABLE=0
export PORT="${PORT:-8000}"
export CONTAINER_NAME="${CONTAINER_NAME:-deepseek-v41-a40-p2p}"
exec bash "$SCRIPT_DIR/start-dsv41-a40-final.sh"
