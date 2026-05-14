#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REFERENCE_FRAME_PATH="${REFERENCE_FRAME_PATH:-none}"
export FOLLOW_MODE="${FOLLOW_MODE:-person_yolo}"
export ROBOT_TYPE="${ROBOT_TYPE:-MIRA3}"
export INPUT_SOURCE="${INPUT_SOURCE:-ros}"

exec "${SCRIPT_DIR}/run_scene_template_chain.sh" "$@"
