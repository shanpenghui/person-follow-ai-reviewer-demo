#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Optional, Tuple

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Int32


class GestureDetectorNode(Node):
    """Detect hand gesture 1/2/3 and publish Int32 gesture_id.

    gesture_id:
      - 0: no valid gesture
      - 1: index finger up
      - 2: index + middle fingers up
      - 3: switch gesture (OK gesture OR index+middle+ring)
    """

    def __init__(self) -> None:
        super().__init__('gesture_detector_node')

        self.declare_parameter('image_topic', '/cam_head/d435/color/image_raw')
        self.declare_parameter('gesture_topic', '/person_follow/gesture_id')
        self.declare_parameter('debug_gesture_topic', '/person_follow/gesture_debug')
        self.declare_parameter('process_hz', 8.0)
        self.declare_parameter('max_hands', 1)
        self.declare_parameter('min_detection_confidence', 0.6)
        self.declare_parameter('min_tracking_confidence', 0.5)
        self.declare_parameter('mirror_image', False)
        self.declare_parameter('gesture_hold_frames', 2)
        self.declare_parameter('enable_ok_as_gesture3', True)
        # anti-false-trigger gating
        self.declare_parameter('gesture_emit_cooldown_sec', 1.2)
        self.declare_parameter('ok_min_middle_up', True)
        self.declare_parameter('ok_min_ring_up', True)
        self.declare_parameter('ok_thumb_index_ratio', 0.10)
        self.declare_parameter('min_hand_scale', 0.06)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.gesture_topic = str(self.get_parameter('gesture_topic').value)
        self.debug_gesture_topic = str(self.get_parameter('debug_gesture_topic').value)
        self.process_hz = max(1.0, float(self.get_parameter('process_hz').value))
        self.max_hands = max(1, int(self.get_parameter('max_hands').value))
        self.min_det_conf = float(self.get_parameter('min_detection_confidence').value)
        self.min_trk_conf = float(self.get_parameter('min_tracking_confidence').value)
        self.mirror_image = bool(self.get_parameter('mirror_image').value)
        self.gesture_hold_frames = max(1, int(self.get_parameter('gesture_hold_frames').value))
        self.enable_ok_as_gesture3 = bool(self.get_parameter('enable_ok_as_gesture3').value)
        self.gesture_emit_cooldown_sec = max(0.0, float(self.get_parameter('gesture_emit_cooldown_sec').value))
        self.ok_min_middle_up = bool(self.get_parameter('ok_min_middle_up').value)
        self.ok_min_ring_up = bool(self.get_parameter('ok_min_ring_up').value)
        self.ok_thumb_index_ratio = max(0.03, float(self.get_parameter('ok_thumb_index_ratio').value))
        self.min_hand_scale = max(0.01, float(self.get_parameter('min_hand_scale').value))

        self.bridge = CvBridge()
        self.latest_image: Optional[Image] = None
        self.latest_stamp_ns: int = 0
        self.last_proc_stamp_ns: int = 0

        self.last_candidate = 0
        self.hold_count = 0
        self.last_pub = -1
        self.last_log_time = 0.0
        self.last_emit_time = 0.0
        self.emit_latched = False

        self.gesture_pub = self.create_publisher(Int32, self.gesture_topic, 10)
        self.debug_pub = self.create_publisher(Int32, self.debug_gesture_topic, 10)
        # camera image topics are usually best-effort sensor QoS
        self.create_subscription(Image, self.image_topic, self._on_image, qos_profile_sensor_data)

        self.mp_ok = False
        self.hands = None
        try:
            import mediapipe as mp  # type: ignore

            self._mp = mp
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.min_det_conf,
                min_tracking_confidence=self.min_trk_conf,
            )
            self.mp_ok = True
        except Exception as e:
            self.get_logger().error(
                'mediapipe import/init failed: %s. '
                'Please install manually: python3 -m pip install --user mediapipe==0.10.14' % str(e)
            )

        self.timer = self.create_timer(1.0 / self.process_hz, self._tick)
        self.get_logger().info(
            f'gesture_detector_node started. mp_ok={self.mp_ok} image={self.image_topic} '
            f'gesture_topic={self.gesture_topic} debug_topic={self.debug_gesture_topic} '
            f'hold={self.gesture_hold_frames} ok_as_3={self.enable_ok_as_gesture3} '
            f'cooldown={self.gesture_emit_cooldown_sec:.2f}s'
        )

    def _on_image(self, msg: Image) -> None:
        self.latest_image = msg
        self.latest_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    @staticmethod
    def _is_up(landmarks, tip_idx: int, pip_idx: int, margin: float = 0.006) -> bool:
        return landmarks[tip_idx].y < (landmarks[pip_idx].y - margin)

    @staticmethod
    def _dist2(a, b) -> float:
        dx = float(a.x) - float(b.x)
        dy = float(a.y) - float(b.y)
        return dx * dx + dy * dy

    def _is_ok_gesture(self, lm) -> bool:
        # thumb tip(4) close to index tip(8)
        d_thumb_index = self._dist2(lm[4], lm[8])
        hand_scale = self._dist2(lm[0], lm[9])  # wrist to middle MCP as normalization
        if hand_scale <= 1e-6 or hand_scale < self.min_hand_scale:
            return False

        close_enough = d_thumb_index < (self.ok_thumb_index_ratio * hand_scale)
        middle_up = self._is_up(lm, 12, 10, margin=0.004)
        ring_up = self._is_up(lm, 16, 14, margin=0.004)

        if self.ok_min_middle_up and (not middle_up):
            return False
        if self.ok_min_ring_up and (not ring_up):
            return False

        return close_enough

    def _classify_hand(self, hand_landmarks) -> Tuple[int, int]:
        lm = hand_landmarks.landmark

        index_up = self._is_up(lm, 8, 6)
        middle_up = self._is_up(lm, 12, 10)
        ring_up = self._is_up(lm, 16, 14)
        pinky_up = self._is_up(lm, 20, 18)

        # bitmask for debug: 1=index,2=middle,4=ring,8=pinky
        mask = (1 if index_up else 0) | (2 if middle_up else 0) | (4 if ring_up else 0) | (8 if pinky_up else 0)

        # Gesture 3 preferred: OK sign
        if self.enable_ok_as_gesture3 and self._is_ok_gesture(lm):
            return 3, mask

        # Gesture 1: only index finger up
        if index_up and (not middle_up) and (not ring_up) and (not pinky_up):
            return 1, mask

        # Gesture 2: index + middle up, ring/pinky down
        if index_up and middle_up and (not ring_up) and (not pinky_up):
            return 2, mask

        # Gesture 3 fallback: index + middle + ring up (pinky allowed)
        # keep only as explicit fallback when OK is disabled, to reduce office false positives.
        if (not self.enable_ok_as_gesture3) and index_up and middle_up and ring_up:
            return 3, mask

        return 0, mask

    def _detect_gesture(self, frame_bgr) -> Tuple[int, int]:
        if self.hands is None:
            return 0, 0

        if self.mirror_image:
            frame_bgr = cv2.flip(frame_bgr, 1)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return 0, 0

        last_mask = 0
        for hand_lm in results.multi_hand_landmarks:
            gid, mask = self._classify_hand(hand_lm)
            last_mask = mask
            if gid in (1, 2, 3):
                return gid, mask

        return 0, last_mask

    def _publish(self, gid: int) -> None:
        msg = Int32()
        msg.data = int(gid)
        self.gesture_pub.publish(msg)

    def _publish_debug(self, candidate: int, mask: int) -> None:
        msg = Int32()
        msg.data = int(100 * candidate + mask)
        self.debug_pub.publish(msg)

    def _tick(self) -> None:
        if self.latest_image is None:
            return

        if self.latest_stamp_ns == self.last_proc_stamp_ns:
            return

        self.last_proc_stamp_ns = self.latest_stamp_ns

        if not self.mp_ok:
            now = time.time()
            if now - self.last_log_time > 5.0:
                self.last_log_time = now
                self.get_logger().warn('mediapipe unavailable, publishing gesture_id=0 only.')
            self._publish(0)
            self._publish_debug(0, 0)
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
            candidate, finger_mask = self._detect_gesture(frame)
        except Exception as e:
            self.get_logger().warn(f'gesture detect exception: {e}')
            candidate, finger_mask = 0, 0

        if candidate == self.last_candidate and candidate in (1, 2, 3):
            self.hold_count += 1
        else:
            self.last_candidate = candidate
            self.hold_count = 1 if candidate in (1, 2, 3) else 0

        out = candidate if (candidate in (1, 2, 3) and self.hold_count >= self.gesture_hold_frames) else 0

        now = time.time()
        # anti-repeat: require gesture drop to 0 before next non-zero emit
        if out == 0:
            self.emit_latched = False
        if out != 0:
            if self.emit_latched:
                out = 0
            elif (now - self.last_emit_time) < self.gesture_emit_cooldown_sec:
                out = 0
            else:
                self.emit_latched = True
                self.last_emit_time = now

        self._publish(out)
        self._publish_debug(candidate, finger_mask)

        if now - self.last_log_time > 1.0:
            self.last_log_time = now
            if out != self.last_pub:
                self.get_logger().info(
                    f'gesture_id={out} (candidate={candidate}, hold={self.hold_count}, mask={finger_mask})'
                )
                self.last_pub = out


def main(args=None):
    rclpy.init(args=args)
    node = GestureDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
