#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROBOT_TYPE="${ROBOT_TYPE:-MIRA3}"
case "${ROBOT_TYPE^^}" in
  MIRA3)
    DEFAULT_ROS_DOMAIN_ID=56
    DEFAULT_INPUT_SOURCE=ros
    ;;
  MIRA2)
    DEFAULT_ROS_DOMAIN_ID=55
    DEFAULT_INPUT_SOURCE=realsense
    ;;
  *)
    DEFAULT_ROS_DOMAIN_ID=55
    DEFAULT_INPUT_SOURCE=realsense
    ;;
esac
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-${DEFAULT_ROS_DOMAIN_ID}}"
LOG_ROOT="${ROOT_DIR}/logs/scene_template_chain"
mkdir -p "${LOG_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_DIR="${LOG_ROOT}/${STAMP}"
mkdir -p "${SESSION_DIR}"

REFERENCE_FRAME_PATH="${REFERENCE_FRAME_PATH:-${TEMPLATE_PATH:-}}"
INPUT_SOURCE="${INPUT_SOURCE:-${DEFAULT_INPUT_SOURCE}}"
if [ -z "${REFERENCE_FRAME_PATH}" ]; then
  echo "[ERROR] REFERENCE_FRAME_PATH is empty"
  exit 1
fi

FOLLOW_MODE="${FOLLOW_MODE:-}"
if [ -z "${FOLLOW_MODE}" ]; then
  if [ "${REFERENCE_FRAME_PATH}" = "none" ]; then
    FOLLOW_MODE="person_yolo"
  else
    FOLLOW_MODE="scene_template"
  fi
fi

case "${FOLLOW_MODE}" in
  scene_template) SERVO_EXECUTABLE="${SERVO_EXECUTABLE:-scene_template_servo_node}" ;;
  person_yolo) SERVO_EXECUTABLE="${SERVO_EXECUTABLE:-person_yolo_servo_node}" ;;
  *) SERVO_EXECUTABLE="${SERVO_EXECUTABLE:-scene_servo_node}" ;;
esac

# WorldPilot data collection (optional)
# Set WORLDPILOT_RECORD=1 to start the trajectory logger alongside scene_servo.
# Logger output goes to WORLDPILOT_OUT_DIR (default: ~/worldpilot/data/trajectories/traj_<timestamp>)
WORLDPILOT_RECORD="${WORLDPILOT_RECORD:-0}"
WORLDPILOT_GOAL_ID="${WORLDPILOT_GOAL_ID:-kitchen_table_d455_2026-04-15_v1}"
WORLDPILOT_OUT_DIR="${WORLDPILOT_OUT_DIR:-}"
WORLDPILOT_SCRIPT="${WORLDPILOT_SCRIPT:-${HOME}/worldpilot/scripts/p0_record_trajectory.py}"
WORLDPILOT_IMAGE_TOPIC="${WORLDPILOT_IMAGE_TOPIC:-/cam_chest/d455/color/image_raw}"
WORLDPILOT_CMD_TOPIC="${WORLDPILOT_CMD_TOPIC:-/smooth_cmd_vel}"

SERVO_LOG="${SESSION_DIR}/servo.log"
PIDS_ENV="${SESSION_DIR}/pids.env"

echo "[INFO] session_dir=${SESSION_DIR}"
echo "[INFO] reference_frame=${REFERENCE_FRAME_PATH}"
echo "[INFO] follow_mode=${FOLLOW_MODE}"
echo "[INFO] servo_executable=${SERVO_EXECUTABLE}"
echo "[INFO] robot_type=${ROBOT_TYPE}"
echo "[INFO] input_source=${INPUT_SOURCE}"
echo "[INFO] ros_domain_id=${ROS_DOMAIN_ID}"
if [ "${WORLDPILOT_RECORD}" = "1" ]; then
  echo "[INFO] worldpilot: ENABLED goal_id=${WORLDPILOT_GOAL_ID}"
fi

cd "${ROOT_DIR}"

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

cleanup() {
  local code=$?
  trap - INT TERM EXIT
  echo "[INFO] stopping scene_template_chain..."
  # First let child processes handle SIGTERM gracefully.
  # The servo node uses this window to publish stop and send a home-position goal.
  if [ -f "${PIDS_ENV}" ]; then
    for pid in $(cat "${PIDS_ENV}"); do
      kill -TERM "${pid}" 2>/dev/null || true
    done
  fi
  # Match the node-side shutdown home wait (default 2.5s) with a little margin.
  sleep 3.0
  # Kill entire process group as a fallback for anything nested that ignored TERM.
  kill -- -$$ 2>/dev/null || true
  sleep 0.5
  # Force kill anything remaining
  if [ -f "${PIDS_ENV}" ]; then
    for pid in $(cat "${PIDS_ENV}"); do
      kill -KILL "${pid}" 2>/dev/null || true
    done
  fi
  wait 2>/dev/null || true
  exit "${code}"
}

trap cleanup INT TERM EXIT

ACTION_SERVER_LOG="${SESSION_DIR}/action_server.log"

echo "[INFO] starting follow_action_server..."
LOG_PATH="${ACTION_SERVER_LOG}" ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  bash -lc "set +u; source /opt/ros/humble/setup.bash; source '${WS_DIR}/install/setup.bash'; set -u; exec python3 '${ROOT_DIR}/person_follow/follow_action_server_node.py'" &
ACTION_PID=$!

echo "[INFO] starting ${SERVO_EXECUTABLE}..."
LOG_PATH="${SERVO_LOG}" ROBOT_TYPE="${ROBOT_TYPE}" ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  TEMPLATE_PATH="${REFERENCE_FRAME_PATH}" INPUT_SOURCE="${INPUT_SOURCE}" FOLLOW_MODE="${FOLLOW_MODE}" \
  SERVO_EXECUTABLE="${SERVO_EXECUTABLE}" \
  bash ./scripts/run_scene_template_servo_logged.sh &
SERVO_PID=$!

LOGGER_PID=""
if [ "${WORLDPILOT_RECORD}" = "1" ]; then
  if [ -z "${WORLDPILOT_OUT_DIR}" ]; then
    WP_STAMP="$(date +%Y%m%d_%H%M%S)"
    WORLDPILOT_OUT_DIR="${HOME}/worldpilot/data/trajectories/traj_${WP_STAMP}"
  fi
  LOGGER_LOG="${SESSION_DIR}/worldpilot_logger.log"
  echo "[INFO] starting worldpilot logger... out=${WORLDPILOT_OUT_DIR}"
  python3 "${WORLDPILOT_SCRIPT}" \
    --out-dir "${WORLDPILOT_OUT_DIR}" \
    --goal-id "${WORLDPILOT_GOAL_ID}" \
    --image-topic "${WORLDPILOT_IMAGE_TOPIC}" \
    --cmd-topic "${WORLDPILOT_CMD_TOPIC}" \
    --cmd-msg-type TwistStamped \
    --sample-period-sec 0.1 \
    --source-robot mira3 \
    --camera d455_chest \
    --teacher auto_servo_continuous \
    --stop-event-topic /person_follow/gesture_stop_event \
    > "${LOGGER_LOG}" 2>&1 &
  LOGGER_PID=$!
  echo "[INFO] logger_pid=${LOGGER_PID} log=${LOGGER_LOG}"
fi

echo "${SERVO_PID} ${ACTION_PID} ${LOGGER_PID}" | tr -s ' ' '\n' | grep -v '^$' > "${PIDS_ENV}"

echo "[INFO] servo_pid=${SERVO_PID} action_pid=${ACTION_PID}"
if [ -n "${LOGGER_PID}" ]; then
  echo "[INFO] logger_pid=${LOGGER_PID}"
fi
echo "[INFO] log: ${SERVO_LOG}"

# wait for any to exit
if [ -n "${LOGGER_PID}" ]; then
  wait -n "${SERVO_PID}" "${ACTION_PID}" "${LOGGER_PID}" 2>/dev/null || wait
else
  wait -n "${SERVO_PID}" "${ACTION_PID}" 2>/dev/null || wait
fi
