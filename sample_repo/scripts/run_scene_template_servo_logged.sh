#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_TYPE="${ROBOT_TYPE:-MIRA3}"
case "${ROBOT_TYPE^^}" in
  MIRA3) DEFAULT_ROS_DOMAIN_ID=56 ;;
  MIRA2) DEFAULT_ROS_DOMAIN_ID=55 ;;
  *) DEFAULT_ROS_DOMAIN_ID=55 ;;
esac
WS_DIR="${WS_DIR:-}"
if [ -z "${WS_DIR}" ]; then
  SEARCH_DIR="${ROOT_DIR}"
  while [ "${SEARCH_DIR}" != "/" ]; do
    if [ -f "${SEARCH_DIR}/install/setup.bash" ]; then
      WS_DIR="${SEARCH_DIR}"
      break
    fi
    SEARCH_DIR="$(dirname "${SEARCH_DIR}")"
  done
fi
WS_DIR="${WS_DIR:-$(cd "${ROOT_DIR}/../.." && pwd)}"
LOG_DIR="${ROOT_DIR}/logs/scene_template_servo"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${STAMP}.log}"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/config/person_follow_all.yaml}"
TEMPLATE_PATH="${TEMPLATE_PATH:-}"
INPUT_SOURCE="${INPUT_SOURCE:-}"
FOLLOW_MODE="${FOLLOW_MODE:-}"
SERVO_EXECUTABLE="${SERVO_EXECUTABLE:-}"

if [ -z "${SERVO_EXECUTABLE}" ]; then
  case "${FOLLOW_MODE}" in
    scene_template) SERVO_EXECUTABLE="scene_template_servo_node" ;;
    person_yolo) SERVO_EXECUTABLE="person_yolo_servo_node" ;;
    *) SERVO_EXECUTABLE="scene_servo_node" ;;
  esac
fi

cd "${ROOT_DIR}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-${DEFAULT_ROS_DOMAIN_ID}}"
set +u
source /opt/ros/humble/setup.bash
source "${WS_DIR}/install/setup.bash"
set -u

CMD=(ros2 run person_follow "${SERVO_EXECUTABLE}" --ros-args --params-file "${CONFIG_FILE}")
if [ -n "${TEMPLATE_PATH}" ]; then
  CMD+=(-p "template_path:=${TEMPLATE_PATH}")
fi
if [ -n "${INPUT_SOURCE}" ]; then
  CMD+=(-p "input_source:=${INPUT_SOURCE}")
fi
if [ -n "${FOLLOW_MODE}" ]; then
  CMD+=(-p "follow_mode:=${FOLLOW_MODE}")
fi
if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

exec > >(tee "${LOG_PATH}") 2>&1
echo "# SCENE_TEMPLATE_SERVO"
echo "# START_TIME: $(date --iso-8601=seconds)"
echo "# CWD: $(pwd)"
echo "# CONFIG_FILE: ${CONFIG_FILE}"
echo "# TEMPLATE_PATH: ${TEMPLATE_PATH}"
echo "# ROBOT_TYPE: ${ROBOT_TYPE}"
echo "# INPUT_SOURCE: ${INPUT_SOURCE}"
echo "# FOLLOW_MODE: ${FOLLOW_MODE}"
echo "# SERVO_EXECUTABLE: ${SERVO_EXECUTABLE}"
echo "# CMD: ${CMD[*]}"
echo "------------------------------------------------------------"
exec "${CMD[@]}"
