#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE_DIR="$(date +%Y%m%d)"
TIME_DIR="$(date +%H%M%S)"
SESSION_DIR="${ROOT_DIR}/data/pbvs_eval/${DATE_DIR}/${TIME_DIR}"
mkdir -p "${SESSION_DIR}"
printf '%s\n' "${SESSION_DIR}"
