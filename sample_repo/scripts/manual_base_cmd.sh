#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROBOT_TYPE="${ROBOT_TYPE:-MIRA3}"
case "${ROBOT_TYPE^^}" in
  MIRA3)
    DEFAULT_ROS_DOMAIN_ID=56
    DEFAULT_LINEAR_SPEED=0.20
    DEFAULT_ANGULAR_SPEED=0.80
    ;;
  MIRA2)
    DEFAULT_ROS_DOMAIN_ID=55
    DEFAULT_LINEAR_SPEED=0.08
    DEFAULT_ANGULAR_SPEED=0.20
    ;;
  *)
    DEFAULT_ROS_DOMAIN_ID=55
    DEFAULT_LINEAR_SPEED=0.08
    DEFAULT_ANGULAR_SPEED=0.20
    ;;
esac

ACTION="${1:-}"
AMOUNT="${2:-}"

if [ -z "${ACTION}" ]; then
  cat <<'USAGE'
Usage:
  ./scripts/manual_base_cmd.sh <action> [distance_m|angle_deg]

Actions:
  forward   [distance_m=0.30]
  back      [distance_m=0.30]
  left      [angle_deg=15]
  right     [angle_deg=15]
  stop
USAGE
  exit 1
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-${DEFAULT_ROS_DOMAIN_ID}}"
set +u
source /opt/ros/humble/setup.bash
if [ -f "${ROOT_DIR}/../../install/setup.bash" ]; then
  source "${ROOT_DIR}/../../install/setup.bash"
fi
set -u

TOPIC="${CMD_TOPIC:-/smooth_cmd_vel}"
FRAME_ID="${CMD_FRAME_ID:-base_link}"
RATE="${CMD_RATE:-10}"
LINEAR_SPEED="${LINEAR_SPEED:-${DEFAULT_LINEAR_SPEED}}"
ANGULAR_SPEED="${ANGULAR_SPEED:-${DEFAULT_ANGULAR_SPEED}}"
STOP_DURATION="${STOP_DURATION:-0.3}"

vx="0.0"
wz="0.0"
duration="${STOP_DURATION}"
unit=""

case "${ACTION}" in
  forward)
    AMOUNT="${AMOUNT:-0.30}"
    vx="${LINEAR_SPEED}"
    duration="$(awk -v d="${AMOUNT}" -v v="${LINEAR_SPEED}" 'BEGIN { if (v <= 0) exit 1; printf "%.3f", d / v }')"
    unit="m"
    ;;
  back)
    AMOUNT="${AMOUNT:-0.30}"
    vx="-${LINEAR_SPEED}"
    duration="$(awk -v d="${AMOUNT}" -v v="${LINEAR_SPEED}" 'BEGIN { if (v <= 0) exit 1; printf "%.3f", d / v }')"
    unit="m"
    ;;
  left)
    AMOUNT="${AMOUNT:-15}"
    wz="${ANGULAR_SPEED}"
    duration="$(awk -v a="${AMOUNT}" -v w="${ANGULAR_SPEED}" 'BEGIN { if (w <= 0) exit 1; printf "%.3f", (a * 3.141592653589793 / 180.0) / w }')"
    unit="deg"
    ;;
  right)
    AMOUNT="${AMOUNT:-15}"
    wz="-${ANGULAR_SPEED}"
    duration="$(awk -v a="${AMOUNT}" -v w="${ANGULAR_SPEED}" 'BEGIN { if (w <= 0) exit 1; printf "%.3f", (a * 3.141592653589793 / 180.0) / w }')"
    unit="deg"
    ;;
  stop)
    AMOUNT="0"
    unit=""
    ;;
  *)
    echo "[ERROR] unknown action: ${ACTION}" >&2
    exit 1
    ;;
esac

echo "[INFO] robot_type=${ROBOT_TYPE} topic=${TOPIC} frame_id=${FRAME_ID} action=${ACTION} amount=${AMOUNT}${unit} vx=${vx} wz=${wz} duration=${duration}s rate=${RATE}Hz ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"

python3 - "$TOPIC" "$FRAME_ID" "$vx" "$wz" "$duration" "$RATE" <<'PY'
import sys, time
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node

_, topic, frame_id, vx_s, wz_s, duration_s, rate_s = sys.argv
vx = float(vx_s)
wz = float(wz_s)
duration = max(0.0, float(duration_s))
rate = max(1.0, float(rate_s))
period = 1.0 / rate

rclpy.init()
node = Node('manual_base_cmd_once')
pub = node.create_publisher(TwistStamped, topic, 10)
start = time.time()
next_t = start
try:
    while time.time() - start < duration:
        now = time.time()
        if now < next_t:
            time.sleep(min(next_t - now, 0.01))
            continue
        msg = TwistStamped()
        stamp_ns = node.get_clock().now().nanoseconds
        msg.header.stamp.sec = stamp_ns // 1_000_000_000
        msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
        msg.header.frame_id = frame_id
        msg.twist.linear.x = vx
        msg.twist.angular.z = wz
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.0)
        next_t += period

    stop = TwistStamped()
    stamp_ns = node.get_clock().now().nanoseconds
    stop.header.stamp.sec = stamp_ns // 1_000_000_000
    stop.header.stamp.nanosec = stamp_ns % 1_000_000_000
    stop.header.frame_id = frame_id
    pub.publish(stop)
    rclpy.spin_once(node, timeout_sec=0.05)
finally:
    node.destroy_node()
    rclpy.shutdown()
PY
