#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Person association node (Phase1+).

目标：
- 基于几何 + 时序融合 D435 / D455 目标
- 支持“overlap 信息可用则收紧门控，否则回退原逻辑”
- 不依赖 ReID，不订阅原始图像，低算力开销

输入：
- /person_follow/target, /person_follow/target_valid
- /person_follow/target_d455, /person_follow/target_valid_d455
- /person_follow/d435_bbox, /person_follow/d455_bbox  (Float32MultiArray)
  data = [x1,y1,x2,y2,conf,img_w,img_h,stamp_sec]

输出：
- /person_follow/target_fused (PointStamped: x=distance, y=bearing, z=confidence)
- /person_follow/target_fused_valid (Bool)
"""

import math
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, Float32MultiArray


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def ang_diff(a: float, b: float) -> float:
    return wrap_pi(a - b)


class PersonAssociationNode(Node):

    def __init__(self) -> None:
        super().__init__('person_association_node')

        # ---- topics ----
        self.declare_parameter('d435_target_topic', '/person_follow/target')
        self.declare_parameter('d435_valid_topic', '/person_follow/target_valid')
        self.declare_parameter('d455_target_topic', '/person_follow/target_d455')
        self.declare_parameter('d455_valid_topic', '/person_follow/target_valid_d455')

        self.declare_parameter('d435_bbox_topic', '/person_follow/d435_bbox')
        self.declare_parameter('d455_bbox_topic', '/person_follow/d455_bbox')

        self.declare_parameter('fused_target_topic', '/person_follow/target_fused')
        self.declare_parameter('fused_valid_topic', '/person_follow/target_fused_valid')

        # ---- timing ----
        self.declare_parameter('association_hz', 20.0)
        self.declare_parameter('max_input_age_sec', 0.45)
        self.declare_parameter('valid_hold_sec', 0.25)
        self.declare_parameter('bbox_timeout_sec', 0.60)
        self.declare_parameter('bbox_time_slop_sec', 0.10)

        # ---- geometry gates ----
        self.declare_parameter('gate_bearing_deg', 15.0)
        self.declare_parameter('gate_bearing_deg_overlap', 10.0)
        self.declare_parameter('gate_range_m', 0.80)
        self.declare_parameter('gate_range_m_overlap', 0.55)

        # ---- fallback bearing from bbox (for D435 no-depth mode) ----
        self.declare_parameter('use_bbox_bearing_fallback', True)
        self.declare_parameter('bbox_target_x_ratio', 0.5)
        self.declare_parameter('d435_fov_h_deg', 69.0)

        # ---- overlap windows (normalized, in each camera image) ----
        # default: fairly宽松的中部区域
        self.declare_parameter('enable_overlap_gating', True)
        self.declare_parameter('d435_overlap_x1', 0.10)
        self.declare_parameter('d435_overlap_y1', 0.10)
        self.declare_parameter('d435_overlap_x2', 0.90)
        self.declare_parameter('d435_overlap_y2', 0.98)
        self.declare_parameter('d455_overlap_x1', 0.10)
        self.declare_parameter('d455_overlap_y1', 0.05)
        self.declare_parameter('d455_overlap_x2', 0.90)
        self.declare_parameter('d455_overlap_y2', 0.98)

        # ---- fusion ----
        self.declare_parameter('bearing_mix_alpha_d455', 0.75)   # overlap/gate通过时 d455 bearing 占比
        self.declare_parameter('output_alpha', 0.55)             # fused输出平滑
        self.declare_parameter('use_d455_as_range_master', True)

        # ---- state machine ----
        self.declare_parameter('confirm_hits', 2)
        self.declare_parameter('miss_tolerance', 8)
        self.declare_parameter('coast_timeout_sec', 0.8)
        self.declare_parameter('stop_on_lost', True)

        # ---- output safety ----
        self.declare_parameter('max_output_jump_m', 0.50)
        self.declare_parameter('max_output_jump_deg', 15.0)

        self.declare_parameter('log_period_sec', 1.0)

        # ---- get params ----
        self.d435_target_topic = str(self.get_parameter('d435_target_topic').value)
        self.d435_valid_topic = str(self.get_parameter('d435_valid_topic').value)
        self.d455_target_topic = str(self.get_parameter('d455_target_topic').value)
        self.d455_valid_topic = str(self.get_parameter('d455_valid_topic').value)
        self.d435_bbox_topic = str(self.get_parameter('d435_bbox_topic').value)
        self.d455_bbox_topic = str(self.get_parameter('d455_bbox_topic').value)
        self.fused_target_topic = str(self.get_parameter('fused_target_topic').value)
        self.fused_valid_topic = str(self.get_parameter('fused_valid_topic').value)

        self.association_hz = float(self.get_parameter('association_hz').value)
        self.max_input_age_sec = float(self.get_parameter('max_input_age_sec').value)
        self.valid_hold_sec = float(self.get_parameter('valid_hold_sec').value)
        self.bbox_timeout_sec = float(self.get_parameter('bbox_timeout_sec').value)
        self.bbox_time_slop_sec = float(self.get_parameter('bbox_time_slop_sec').value)

        self.gate_bearing_deg = float(self.get_parameter('gate_bearing_deg').value)
        self.gate_bearing_deg_overlap = float(self.get_parameter('gate_bearing_deg_overlap').value)
        self.gate_range_m = float(self.get_parameter('gate_range_m').value)
        self.gate_range_m_overlap = float(self.get_parameter('gate_range_m_overlap').value)

        self.use_bbox_bearing_fallback = bool(self.get_parameter('use_bbox_bearing_fallback').value)
        self.bbox_target_x_ratio = float(self.get_parameter('bbox_target_x_ratio').value)
        self.d435_fov_h_deg = float(self.get_parameter('d435_fov_h_deg').value)

        self.enable_overlap_gating = bool(self.get_parameter('enable_overlap_gating').value)
        self.d435_overlap = (
            float(self.get_parameter('d435_overlap_x1').value),
            float(self.get_parameter('d435_overlap_y1').value),
            float(self.get_parameter('d435_overlap_x2').value),
            float(self.get_parameter('d435_overlap_y2').value),
        )
        self.d455_overlap = (
            float(self.get_parameter('d455_overlap_x1').value),
            float(self.get_parameter('d455_overlap_y1').value),
            float(self.get_parameter('d455_overlap_x2').value),
            float(self.get_parameter('d455_overlap_y2').value),
        )

        self.bearing_mix_alpha_d455 = clamp(float(self.get_parameter('bearing_mix_alpha_d455').value), 0.0, 1.0)
        self.output_alpha = clamp(float(self.get_parameter('output_alpha').value), 0.01, 1.0)
        self.use_d455_as_range_master = bool(self.get_parameter('use_d455_as_range_master').value)

        self.confirm_hits = max(1, int(self.get_parameter('confirm_hits').value))
        self.miss_tolerance = max(1, int(self.get_parameter('miss_tolerance').value))
        self.coast_timeout_sec = float(self.get_parameter('coast_timeout_sec').value)
        self.stop_on_lost = bool(self.get_parameter('stop_on_lost').value)

        self.max_output_jump_m = float(self.get_parameter('max_output_jump_m').value)
        self.max_output_jump_deg = float(self.get_parameter('max_output_jump_deg').value)

        self.log_period_sec = float(self.get_parameter('log_period_sec').value)

        # ---- state ----
        self.last_d435_target: Optional[PointStamped] = None
        self.last_d435_target_time: float = 0.0
        self.last_d455_target: Optional[PointStamped] = None
        self.last_d455_target_time: float = 0.0

        self.d435_valid: bool = False
        self.d435_valid_true_time: float = 0.0
        self.d455_valid: bool = False
        self.d455_valid_true_time: float = 0.0

        self.last_d435_bbox: Optional[Tuple[float, float, float, float, float, float, float, float]] = None
        self.last_d455_bbox: Optional[Tuple[float, float, float, float, float, float, float, float]] = None
        self.last_d435_bbox_time: float = 0.0
        self.last_d455_bbox_time: float = 0.0

        self.state: str = 'UNLOCKED'
        self.hit_count: int = 0
        self.miss_count: int = 0

        self.last_fused_distance: Optional[float] = None
        self.last_fused_bearing: Optional[float] = None
        self.last_fused_conf: float = 0.0
        self.last_fused_time: float = 0.0

        self.last_pub_valid: bool = False
        self.last_pub_valid_time: float = 0.0

        self.last_log_time: float = 0.0
        self.diag_tick = 0
        self.diag_overlap_on = 0
        self.diag_gate_reject = 0
        self.diag_jump_reject = 0
        self.diag_pub_ok = 0
        self.diag_lost = 0

        # ---- io ----
        self.create_subscription(PointStamped, self.d435_target_topic, self._on_d435_target, 10)
        self.create_subscription(Bool, self.d435_valid_topic, self._on_d435_valid, 10)
        self.create_subscription(PointStamped, self.d455_target_topic, self._on_d455_target, 10)
        self.create_subscription(Bool, self.d455_valid_topic, self._on_d455_valid, 10)
        self.create_subscription(Float32MultiArray, self.d435_bbox_topic, self._on_d435_bbox, 10)
        self.create_subscription(Float32MultiArray, self.d455_bbox_topic, self._on_d455_bbox, 10)

        self.fused_pub = self.create_publisher(PointStamped, self.fused_target_topic, 10)
        self.fused_valid_pub = self.create_publisher(Bool, self.fused_valid_topic, 10)

        dt = 1.0 / max(1.0, self.association_hz)
        self.timer = self.create_timer(dt, self._tick)

        self.get_logger().info(
            f'person_association_node started. in=({self.d435_target_topic},{self.d455_target_topic}) '
            f'bbox=({self.d435_bbox_topic},{self.d455_bbox_topic}) out={self.fused_target_topic} '
            f'overlap_gate={self.enable_overlap_gating} hz={self.association_hz}'
        )

    # ---- callbacks ----
    def _on_d435_target(self, msg: PointStamped) -> None:
        self.last_d435_target = msg
        self.last_d435_target_time = time.time()

    def _on_d435_valid(self, msg: Bool) -> None:
        self.d435_valid = bool(msg.data)
        if self.d435_valid:
            self.d435_valid_true_time = time.time()

    def _on_d455_target(self, msg: PointStamped) -> None:
        self.last_d455_target = msg
        self.last_d455_target_time = time.time()

    def _on_d455_valid(self, msg: Bool) -> None:
        self.d455_valid = bool(msg.data)
        if self.d455_valid:
            self.d455_valid_true_time = time.time()

    def _on_d435_bbox(self, msg: Float32MultiArray) -> None:
        if msg.data is None or len(msg.data) < 8:
            return
        d = msg.data
        self.last_d435_bbox = (float(d[0]), float(d[1]), float(d[2]), float(d[3]),
                              float(d[4]), float(d[5]), float(d[6]), float(d[7]))
        self.last_d435_bbox_time = time.time()

    def _on_d455_bbox(self, msg: Float32MultiArray) -> None:
        if msg.data is None or len(msg.data) < 8:
            return
        d = msg.data
        self.last_d455_bbox = (float(d[0]), float(d[1]), float(d[2]), float(d[3]),
                              float(d[4]), float(d[5]), float(d[6]), float(d[7]))
        self.last_d455_bbox_time = time.time()

    # ---- helpers ----
    def _is_target_valid(self, is_valid: bool, last_true_t: float, now: float) -> bool:
        if is_valid:
            return True
        return (now - last_true_t) <= self.valid_hold_sec

    def _is_fresh(self, t: float, now: float, timeout: Optional[float] = None) -> bool:
        if t <= 0.0:
            return False
        lim = self.max_input_age_sec if timeout is None else timeout
        return (now - t) <= lim

    def _bbox_center_norm(self, b: Tuple[float, float, float, float, float, float, float, float]) -> Optional[Tuple[float, float]]:
        x1, y1, x2, y2, _conf, iw, ih, _ts = b
        if iw <= 1.0 or ih <= 1.0:
            return None
        bw = max(1.0, x2 - x1)
        px = x1 + self.bbox_target_x_ratio * bw
        py = y1 + 0.5 * (y2 - y1)
        nx = px / iw
        ny = py / ih
        return nx, ny

    @staticmethod
    def _inside_rect(nx: float, ny: float, rect: Tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = rect
        return (nx >= x1) and (nx <= x2) and (ny >= y1) and (ny <= y2)

    def _overlap_available(self, now: float) -> bool:
        if not self.enable_overlap_gating:
            return False
        if self.last_d435_bbox is None or self.last_d455_bbox is None:
            return False
        if not self._is_fresh(self.last_d435_bbox_time, now, self.bbox_timeout_sec):
            return False
        if not self._is_fresh(self.last_d455_bbox_time, now, self.bbox_timeout_sec):
            return False

        c1 = self._bbox_center_norm(self.last_d435_bbox)
        c2 = self._bbox_center_norm(self.last_d455_bbox)
        if c1 is None or c2 is None:
            return False

        ok1 = self._inside_rect(c1[0], c1[1], self.d435_overlap)
        ok2 = self._inside_rect(c2[0], c2[1], self.d455_overlap)

        t1 = self.last_d435_bbox[7]
        t2 = self.last_d455_bbox[7]
        t_ok = abs(t1 - t2) <= self.bbox_time_slop_sec

        return ok1 and ok2 and t_ok

    def _d435_obs(self, now: float) -> Tuple[Optional[float], Optional[float], float]:
        """Return: (range_m or None, bearing_rad or None, conf)."""
        rng = None
        bearing = None
        conf = 0.0

        if self.last_d435_target is not None and self._is_fresh(self.last_d435_target_time, now):
            if self._is_target_valid(self.d435_valid, self.d435_valid_true_time, now):
                rng = float(self.last_d435_target.point.x)
                bearing = float(self.last_d435_target.point.y)
                conf = clamp(float(self.last_d435_target.point.z), 0.0, 1.0)

        if (bearing is None) and self.use_bbox_bearing_fallback and self.last_d435_bbox is not None:
            if self._is_fresh(self.last_d435_bbox_time, now, self.bbox_timeout_sec):
                c = self._bbox_center_norm(self.last_d435_bbox)
                if c is not None:
                    nx, _ny = c
                    ex = (nx - 0.5) / 0.5
                    bearing = math.radians(-ex * (self.d435_fov_h_deg * 0.5))
                    conf = max(conf, clamp(self.last_d435_bbox[4], 0.0, 1.0) * 0.7)

        return rng, bearing, conf

    def _d455_obs(self, now: float) -> Tuple[Optional[float], Optional[float], float]:
        if self.last_d455_target is None:
            return None, None, 0.0
        if not self._is_fresh(self.last_d455_target_time, now):
            return None, None, 0.0
        if not self._is_target_valid(self.d455_valid, self.d455_valid_true_time, now):
            return None, None, 0.0
        return (
            float(self.last_d455_target.point.x),
            float(self.last_d455_target.point.y),
            clamp(float(self.last_d455_target.point.z), 0.0, 1.0)
        )

    def _publish_valid(self, valid: bool) -> None:
        now = time.time()
        if valid != self.last_pub_valid or (now - self.last_pub_valid_time) > self.log_period_sec:
            b = Bool()
            b.data = bool(valid)
            self.fused_valid_pub.publish(b)
            self.last_pub_valid = valid
            self.last_pub_valid_time = now

    def _publish_target(self, distance_m: float, bearing_rad: float, confidence: float) -> None:
        m = PointStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'base_link'
        m.point.x = float(max(0.0, distance_m))
        m.point.y = float(wrap_pi(bearing_rad))
        m.point.z = float(clamp(confidence, 0.0, 1.0))
        self.fused_pub.publish(m)

    def _maybe_log(self) -> None:
        now = time.time()
        if (now - self.last_log_time) < max(0.5, self.log_period_sec):
            return
        self.last_log_time = now
        self.get_logger().info(
            f'state={self.state} hit={self.hit_count} miss={self.miss_count} '
            f'overlap={self.diag_overlap_on} gate_rej={self.diag_gate_reject} jump_rej={self.diag_jump_reject} '
            f'pub_ok={self.diag_pub_ok} lost={self.diag_lost} '
            f'fused=({self.last_fused_distance if self.last_fused_distance is not None else -1:.2f}m,'
            f'{math.degrees(self.last_fused_bearing) if self.last_fused_bearing is not None else 0.0:+.1f}deg)'
        )
        self.diag_overlap_on = 0
        self.diag_gate_reject = 0
        self.diag_jump_reject = 0
        self.diag_pub_ok = 0
        self.diag_lost = 0

    # ---- core tick ----
    def _tick(self) -> None:
        self.diag_tick += 1
        now = time.time()

        d435_r, d435_b, d435_c = self._d435_obs(now)
        d455_r, d455_b, d455_c = self._d455_obs(now)

        overlap_on = self._overlap_available(now)
        if overlap_on:
            self.diag_overlap_on += 1

        candidate = None  # (range, bearing, conf)

        if d455_r is not None and d455_b is not None:
            # base candidate from D455
            c_r = d455_r
            c_b = d455_b
            c_c = d455_c

            # if D435 bearing available, run geometric gate and optionally fuse bearing
            if d435_b is not None:
                gate_b = self.gate_bearing_deg_overlap if overlap_on else self.gate_bearing_deg
                db_deg = abs(math.degrees(ang_diff(d455_b, d435_b)))
                if db_deg <= gate_b:
                    a = self.bearing_mix_alpha_d455
                    c_b = wrap_pi(a * d455_b + (1.0 - a) * d435_b)
                    c_c = clamp(max(c_c, d435_c), 0.0, 1.0)
                else:
                    # gate reject only affects cross-camera fusion, still keep D455 as fallback
                    self.diag_gate_reject += 1

            # if D435 range exists, optional range gate
            if d435_r is not None:
                gate_r = self.gate_range_m_overlap if overlap_on else self.gate_range_m
                if abs(d455_r - d435_r) > gate_r:
                    self.diag_gate_reject += 1
                    # distance still trust D455 master by design

            candidate = (c_r, c_b, c_c)

        elif d435_r is not None and d435_b is not None:
            # pure D435 fallback (only available if D435 depth target enabled)
            candidate = (d435_r, d435_b, d435_c)

        elif d435_b is not None and self.last_fused_distance is not None and self._is_fresh(self.last_fused_time, now, self.coast_timeout_sec):
            # bearing-only fallback: keep last range shortly, update bearing from D435
            candidate = (self.last_fused_distance, d435_b, max(0.2, d435_c))

        # --- state machine ---
        if candidate is None:
            self.hit_count = 0
            self.miss_count += 1

            if self.state == 'LOCKED' and self.last_fused_distance is not None and self.last_fused_bearing is not None:
                if self.miss_count <= self.miss_tolerance and self._is_fresh(self.last_fused_time, now, self.coast_timeout_sec):
                    # short coast
                    self._publish_target(self.last_fused_distance, self.last_fused_bearing, self.last_fused_conf * 0.85)
                    self._publish_valid(True)
                    self.diag_pub_ok += 1
                    self._maybe_log()
                    return

            self.state = 'LOST'
            self.diag_lost += 1
            if self.stop_on_lost:
                self._publish_valid(False)
            self._maybe_log()
            return

        # candidate exists
        cand_r, cand_b, cand_c = candidate

        # confirm lock
        if self.state != 'LOCKED':
            self.hit_count += 1
            self.miss_count = 0
            if self.hit_count >= self.confirm_hits:
                self.state = 'LOCKED'
            else:
                self.state = 'UNLOCKED'
        else:
            self.miss_count = 0

        # output jump guard
        if self.last_fused_distance is not None and self.last_fused_bearing is not None:
            dr = abs(cand_r - self.last_fused_distance)
            db_deg = abs(math.degrees(ang_diff(cand_b, self.last_fused_bearing)))
            if dr > self.max_output_jump_m or db_deg > self.max_output_jump_deg:
                self.diag_jump_reject += 1
                # reject this candidate once, keep last output if still fresh
                if self._is_fresh(self.last_fused_time, now, self.coast_timeout_sec):
                    self._publish_target(self.last_fused_distance, self.last_fused_bearing, self.last_fused_conf * 0.9)
                    self._publish_valid(True)
                    self.diag_pub_ok += 1
                    self._maybe_log()
                    return

        # smooth output
        if self.last_fused_distance is None or self.last_fused_bearing is None:
            out_r = cand_r
            out_b = cand_b
        else:
            a = self.output_alpha
            out_r = self.last_fused_distance + a * (cand_r - self.last_fused_distance)
            out_b = wrap_pi(self.last_fused_bearing + a * ang_diff(cand_b, self.last_fused_bearing))

        out_c = clamp(cand_c, 0.0, 1.0)

        self.last_fused_distance = out_r
        self.last_fused_bearing = out_b
        self.last_fused_conf = out_c
        self.last_fused_time = now

        self._publish_target(out_r, out_b, out_c)
        self._publish_valid(True)
        self.diag_pub_ok += 1
        self._maybe_log()


def main(args=None):
    rclpy.init(args=args)
    node = PersonAssociationNode()
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
