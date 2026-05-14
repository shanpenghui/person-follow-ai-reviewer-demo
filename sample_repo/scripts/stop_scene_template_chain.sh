#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${ROOT_DIR}/logs/scene_template_chain"

if [ ! -d "${LOG_ROOT}" ]; then
  echo "[INFO] no scene_template_chain log dir"
  exit 0
fi

LATEST_SESSION="${1:-$(find "${LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)}"
if [ -z "${LATEST_SESSION}" ] || [ ! -d "${LATEST_SESSION}" ]; then
  echo "[INFO] no session to stop"
  exit 0
fi

PIDS_ENV="${LATEST_SESSION}/pids.env"
if [ ! -f "${PIDS_ENV}" ]; then
  echo "[WARN] ${PIDS_ENV} not found"
  exit 0
fi

# shellcheck disable=SC1090
source "${PIDS_ENV}"

for pid_var in SERVO_PID CHASSIS_PID BRIDGE_PID; do
  pid="${!pid_var:-}"
  if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    echo "[INFO] stopped ${pid_var}=${pid}"
  fi
done

sleep 1

for pid_var in SERVO_PID CHASSIS_PID BRIDGE_PID; do
  pid="${!pid_var:-}"
  if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}" 2>/dev/null || true
    echo "[WARN] force-killed ${pid_var}=${pid}"
  fi
done

# kill any leftover servo executable from the unified chain
pkill -f '/install/lib/person_follow/scene_servo_node' 2>/dev/null || true
pkill -f '/install/lib/person_follow/scene_template_servo_node' 2>/dev/null || true
pkill -f '/install/lib/person_follow/person_yolo_servo_node' 2>/dev/null || true
