#!/usr/bin/env python3
"""scene_servo_node.py — single-node scene-template visual servo.

All-in-one: feature matching → head control → chassis cmd_vel.

Architecture (single node):
  scene_servo_node → Torso action (head control)
                  → /smooth_cmd_vel (chassis velocity)
                  → /person_follow/follow_enabled
                  → /scene_template/servo_state (debug)

Chassis logic:
  - person_yolo: keep desired distance to detected person
  - scene_template: drive pose error to zero, then auto-complete
  - Both ramped with acceleration limits
"""
from __future__ import annotations

import math
import signal
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rclpy
import rclpy.duration
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, TransformStamped, Twist, TwistStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool, Float32MultiArray, Int32, String

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    import tf2_ros  # type: ignore
    import tf2_geometry_msgs  # type: ignore  # noqa: F401 — registers PointStamped transform
    from tf2_msgs.msg import TFMessage
except ImportError:
    tf2_ros = None
    TFMessage = None

from .feature_matcher import FeatureMatcherCfg
from .scene_template_store import load_template, template_to_all_ref_data
from .servo_estimator import ServoEstimator, ServoEstimatorCfg, estimate_best_keyframe

try:
    from interfaces.action import Torso  # type: ignore
except ImportError:
    Torso = None


SERVO_MODE_CODE = {
    'search': 0.0,
    'head_track_only': 1.0,
    'base_servo': 2.0,
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def resolve_yolo_model_path(configured_path: str) -> str:
    candidate = configured_path.strip()
    if candidate and Path(candidate).is_file():
        return candidate

    fallback_name = Path(candidate).name if candidate else 'yolov8s_fp16_512.engine'
    try:
        package_dir = Path(get_package_share_directory('person_follow'))
        if candidate:
            package_candidate = package_dir / candidate
            if package_candidate.is_file():
                return str(package_candidate)
        fallback = package_dir / 'models' / fallback_name
        if fallback.is_file():
            return str(fallback)
    except PackageNotFoundError:
        pass

    repo_fallback = Path(__file__).resolve().parents[1] / 'models' / fallback_name
    if repo_fallback.is_file():
        return str(repo_fallback)

    return candidate


class SceneServoNode(Node):

    def __init__(self) -> None:
        super().__init__('scene_servo_node')

        # =========================================================================
        # Parameter declarations
        # =========================================================================

        # --- vision / matching ---
        self.declare_parameter('template_path', '')
        self.declare_parameter('input_source', 'ros')
        self.declare_parameter('image_topic', '/cam_chest/d455/color/image_raw')
        self.declare_parameter('depth_topic', '/cam_chest/d455/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/cam_chest/d455/color/camera_info')
        self.declare_parameter('rs_device_name', 'D455')
        self.declare_parameter('rs_width', 640)
        self.declare_parameter('rs_height', 480)
        self.declare_parameter('rs_fps', 30)
        self.declare_parameter('control_hz', 10.0)
        self.declare_parameter('match_hz', 4.0)
        self.declare_parameter('log_period_sec', 1.0)
        self.declare_parameter('frame_timeout_sec', 0.75)
        self.declare_parameter('scene_frame_sync_tolerance_sec', 0.08)
        self.declare_parameter('publish_lost_state', True)
        self.declare_parameter('head_fov_h_deg', 69.0)
        self.declare_parameter('head_fov_v_deg', 42.0)
        self.declare_parameter('servo_state_topic', '/scene_template/servo_state')

        # --- follow mode ---
        self.declare_parameter('follow_mode', 'scene_template')  # scene_template | person_yolo

        # --- yolo ---
        self.declare_parameter('yolo_model_path', '')
        self.declare_parameter('yolo_device', '0')
        self.declare_parameter('yolo_imgsz', 512)
        self.declare_parameter('yolo_conf_thres', 0.16)
        self.declare_parameter('yolo_detect_class_id', 0)  # COCO person=0
        self.declare_parameter('yolo_use_half', True)
        # Vertical aim point inside bbox (0.0=top, 0.5=center, 1.0=bottom).
        # Use upper body/head-biased target so robot looks like it is watching the person.
        self.declare_parameter('yolo_target_y_ratio', 0.32)

        self.declare_parameter('detector', 'orb')
        self.declare_parameter('max_keypoints', 500)
        self.declare_parameter('match_ratio_test', 0.75)
        self.declare_parameter('ransac_reproj_th', 5.0)
        self.declare_parameter('ransac_3d_inlier_th_m', 0.08)
        self.declare_parameter('ransac_confidence', 0.995)
        self.declare_parameter('ransac_max_iters', 2000)
        self.declare_parameter('depth_min_m', 0.1)
        self.declare_parameter('depth_max_m', 8.0)
        self.declare_parameter('depth_patch_half', 2)
        self.declare_parameter('min_3d_pairs', 8)
        self.declare_parameter('min_pnp_pairs', 8)
        self.declare_parameter('min_2d_pairs', 8)
        self.declare_parameter('max_translation_m', 5.0)

        self.declare_parameter('ema_alpha', 0.35)
        self.declare_parameter('min_confidence_head', 0.15)
        self.declare_parameter('min_confidence_base', 0.30)
        self.declare_parameter('min_inlier_count_base', 12)
        self.declare_parameter('multi_keyframe', True)
        self.declare_parameter('base_ready_window', 3)
        self.declare_parameter('base_ready_min_count', 2)

        # --- head control ---
        self.declare_parameter('joint_state_topic', '/Torso/joint_states')
        self.declare_parameter('torso_action_name', '/Torso/torso_action_service')
        self.declare_parameter('yaw_sign', 1.0)
        self.declare_parameter('pitch_sign', 1.0)
        self.declare_parameter('head_smoothing', 0.35)
        self.declare_parameter('head_deadband_deg', 1.0)
        self.declare_parameter('head_yaw_vel_max_deg_s', 30.0)
        self.declare_parameter('head_pitch_vel_max_deg_s', 20.0)
        self.declare_parameter('home_torso_height', 0.149)
        self.declare_parameter('home_torso_yaw', 0.0)
        self.declare_parameter('home_head_yaw', 0.0)
        self.declare_parameter('home_head_pitch', 0.0)

        # --- chassis control ---
        self.declare_parameter('cmd_topic', '/smooth_cmd_vel')
        self.declare_parameter('use_twist_stamped', False)
        self.declare_parameter('desired_distance_m', 1.0)
        self.declare_parameter('k_dist', 0.55)
        self.declare_parameter('k_yaw', 0.55)
        self.declare_parameter('deadband_distance_m', 0.05)
        self.declare_parameter('deadband_yaw_deg', 3.0)
        self.declare_parameter('vx_min', -0.10)
        self.declare_parameter('vx_max', 0.12)
        self.declare_parameter('wz_max', 0.25)
        self.declare_parameter('vx_yaw_damp_start_deg', 20.0)  # start attenuating vx above this yaw
        self.declare_parameter('vx_yaw_damp_end_deg', 45.0)    # vx=0 at this yaw
        self.declare_parameter('head_homing_ratio', 0.0)  # head homing bias (0=off, 0.1=10% toward center)
        self.declare_parameter('accl_vx', 0.20)
        self.declare_parameter('accl_wz', 0.80)
        self.declare_parameter('lost_decel_vx', 0.18)
        self.declare_parameter('lost_decel_wz', 0.60)
        self.declare_parameter('scene_arrive_forward_m', 0.08)
        self.declare_parameter('scene_arrive_lateral_m', 0.12)
        self.declare_parameter('scene_arrive_yaw_deg', 4.0)
        self.declare_parameter('scene_arrive_stable_frames', 5)
        self.declare_parameter('scene_lateral_yaw_gain', 0.35)
        self.declare_parameter('scene_vx_min', -0.05)
        self.declare_parameter('scene_vx_max', 0.12)
        self.declare_parameter('scene_wz_max', 0.18)
        self.declare_parameter('scene_accl_vx', 0.12)
        self.declare_parameter('scene_accl_wz', 0.45)
        self.declare_parameter('scene_cmd_alpha', 0.35)
        self.declare_parameter('scene_min_vx_cmd', 0.04)
        self.declare_parameter('scene_min_wz_cmd', 0.045)
        self.declare_parameter('scene_pose_min_confidence', 0.45)
        self.declare_parameter('scene_pose_min_inliers', 8)
        self.declare_parameter('scene_pose_hold_sec', 0.8)
        self.declare_parameter('shutdown_home_wait_sec', 2.5)

        # --- follow enabled ---
        self.declare_parameter('follow_enabled_topic', '/person_follow/follow_enabled')
        self.declare_parameter('auto_enable_follow', True)
        self.declare_parameter('follow_enabled_republish_sec', 0.5)

        # --- action server bridge topics ---
        self.declare_parameter('action_master_enabled_topic', '/person_follow/action_master_enabled')
        self.declare_parameter('action_follow_enabled_topic', '/person_follow/action_follow_enabled')
        self.declare_parameter('fsm_force_stop_topic', '/person_follow/fsm_force_stop')
        self.declare_parameter('fsm_state_topic', '/person_follow/fsm_state')
        self.declare_parameter('gesture_stop_event_topic', '/person_follow/gesture_stop_event')

        # --- gesture control (built-in mediapipe) ---
        self.declare_parameter('gesture_start', 1)
        self.declare_parameter('gesture_stop', 2)

        # --- gesture detection (D435) ---
        self.declare_parameter('d435_image_topic', '/cam_head/d435/color/image_raw')
        self.declare_parameter('gesture_process_hz', 8.0)
        self.declare_parameter('gesture_hold_frames', 2)
        self.declare_parameter('gesture_cooldown_sec', 1.2)
        self.declare_parameter('gesture_min_det_conf', 0.6)
        self.declare_parameter('gesture_min_trk_conf', 0.5)

        # --- coast ---
        self.declare_parameter('base_coast_sec', 1.5)

        # --- TF2 ---
        self.declare_parameter('use_tf2', True)
        self.declare_parameter('camera_frame', 'camera_torso_optical')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tf_timeout_sec', 0.1)

        # =========================================================================
        # Read params
        # =========================================================================
        self.input_source = str(self.get_parameter('input_source').value).strip().lower()
        self.template_path = str(self.get_parameter('template_path').value).strip()
        self.servo_state_topic = str(self.get_parameter('servo_state_topic').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.rs_device_name = str(self.get_parameter('rs_device_name').value)
        self.rs_width = int(self.get_parameter('rs_width').value)
        self.rs_height = int(self.get_parameter('rs_height').value)
        self.rs_fps = int(self.get_parameter('rs_fps').value)
        self.control_hz = max(0.5, float(self.get_parameter('control_hz').value))
        self.match_hz = max(0.5, float(self.get_parameter('match_hz').value))
        self.log_period_sec = max(0.5, float(self.get_parameter('log_period_sec').value))
        self.frame_timeout_sec = max(0.1, float(self.get_parameter('frame_timeout_sec').value))
        self.scene_frame_sync_tolerance_sec = max(
            0.0, float(self.get_parameter('scene_frame_sync_tolerance_sec').value))
        self.publish_lost_state = bool(self.get_parameter('publish_lost_state').value)
        self.head_fov_h_deg = max(1.0, float(self.get_parameter('head_fov_h_deg').value))
        self.head_fov_v_deg = max(1.0, float(self.get_parameter('head_fov_v_deg').value))

        # follow mode
        self.follow_mode = str(self.get_parameter('follow_mode').value).strip().lower()

        # yolo params
        configured_yolo_model_path = str(self.get_parameter('yolo_model_path').value).strip()
        self.yolo_model_path = resolve_yolo_model_path(configured_yolo_model_path)
        self.yolo_device = str(self.get_parameter('yolo_device').value)
        self.yolo_imgsz = int(self.get_parameter('yolo_imgsz').value)
        self.yolo_conf_thres = float(self.get_parameter('yolo_conf_thres').value)
        self.yolo_detect_class_id = int(self.get_parameter('yolo_detect_class_id').value)
        self.yolo_use_half = bool(self.get_parameter('yolo_use_half').value)
        self.yolo_target_y_ratio = clamp(
            float(self.get_parameter('yolo_target_y_ratio').value), 0.0, 1.0)
        self.yolo_model = None
        if self.follow_mode == 'person_yolo':
            try:
                from ultralytics import YOLO
                mp = self.yolo_model_path or None
                self.yolo_model = YOLO(mp)
                if configured_yolo_model_path and configured_yolo_model_path != self.yolo_model_path:
                    self.get_logger().info(
                        f'YOLO model path fallback: {configured_yolo_model_path} -> {self.yolo_model_path}')
                self.get_logger().info(f'YOLO loaded: {mp} device={self.yolo_device} imgsz={self.yolo_imgsz}')
            except Exception as exc:
                self.get_logger().error(f'YOLO init failed: {exc}')
                self.follow_mode = 'scene_template'

        self.matcher_cfg = FeatureMatcherCfg(
            detector=str(self.get_parameter('detector').value),
            max_keypoints=int(self.get_parameter('max_keypoints').value),
            match_ratio_test=float(self.get_parameter('match_ratio_test').value),
            ransac_reproj_th=float(self.get_parameter('ransac_reproj_th').value),
            ransac_3d_inlier_th_m=float(self.get_parameter('ransac_3d_inlier_th_m').value),
            ransac_confidence=float(self.get_parameter('ransac_confidence').value),
            ransac_max_iters=int(self.get_parameter('ransac_max_iters').value),
            depth_min_m=float(self.get_parameter('depth_min_m').value),
            depth_max_m=float(self.get_parameter('depth_max_m').value),
            depth_patch_half=int(self.get_parameter('depth_patch_half').value),
            min_3d_pairs=int(self.get_parameter('min_3d_pairs').value),
            min_pnp_pairs=int(self.get_parameter('min_pnp_pairs').value),
            min_2d_pairs=int(self.get_parameter('min_2d_pairs').value),
            max_translation_m=float(self.get_parameter('max_translation_m').value),
        )
        self.estimator = ServoEstimator(
            ServoEstimatorCfg(
                ema_alpha=float(self.get_parameter('ema_alpha').value),
                min_confidence_head=float(self.get_parameter('min_confidence_head').value),
                min_confidence_base=float(self.get_parameter('min_confidence_base').value),
                min_inlier_count_base=int(self.get_parameter('min_inlier_count_base').value),
                multi_keyframe=bool(self.get_parameter('multi_keyframe').value),
                base_ready_window=int(self.get_parameter('base_ready_window').value),
                base_ready_min_count=int(self.get_parameter('base_ready_min_count').value),
            )
        )

        # head params
        self.joint_state_topic = str(self.get_parameter('joint_state_topic').value)
        self.torso_action_name = str(self.get_parameter('torso_action_name').value)
        self.yaw_sign = float(self.get_parameter('yaw_sign').value)
        self.pitch_sign = float(self.get_parameter('pitch_sign').value)
        self.head_smoothing = clamp(float(self.get_parameter('head_smoothing').value), 0.01, 1.0)
        self.head_deadband_deg = float(self.get_parameter('head_deadband_deg').value)
        self.head_yaw_vel_max_deg_s = float(self.get_parameter('head_yaw_vel_max_deg_s').value)
        self.head_pitch_vel_max_deg_s = float(self.get_parameter('head_pitch_vel_max_deg_s').value)
        self.home_torso_height = float(self.get_parameter('home_torso_height').value)
        self.home_torso_yaw = float(self.get_parameter('home_torso_yaw').value)
        self.home_head_yaw = float(self.get_parameter('home_head_yaw').value)
        self.home_head_pitch = float(self.get_parameter('home_head_pitch').value)

        # chassis params
        self.cmd_topic = str(self.get_parameter('cmd_topic').value)
        self.use_twist_stamped = bool(self.get_parameter('use_twist_stamped').value)
        self.desired_distance_m = float(self.get_parameter('desired_distance_m').value)
        self.k_dist = float(self.get_parameter('k_dist').value)
        self.k_yaw = float(self.get_parameter('k_yaw').value)
        self.deadband_distance_m = float(self.get_parameter('deadband_distance_m').value)
        self.deadband_yaw_deg = float(self.get_parameter('deadband_yaw_deg').value)
        self.vx_min = float(self.get_parameter('vx_min').value)
        self.vx_max = float(self.get_parameter('vx_max').value)
        self.wz_max = float(self.get_parameter('wz_max').value)
        self.vx_yaw_damp_start_deg = float(self.get_parameter('vx_yaw_damp_start_deg').value)
        self.vx_yaw_damp_end_deg = float(self.get_parameter('vx_yaw_damp_end_deg').value)
        self.head_homing_ratio = float(self.get_parameter('head_homing_ratio').value)
        self.accl_vx = float(self.get_parameter('accl_vx').value)
        self.accl_wz = float(self.get_parameter('accl_wz').value)
        self.lost_decel_vx = float(self.get_parameter('lost_decel_vx').value)
        self.lost_decel_wz = float(self.get_parameter('lost_decel_wz').value)
        self.scene_arrive_forward_m = abs(float(self.get_parameter('scene_arrive_forward_m').value))
        self.scene_arrive_lateral_m = abs(float(self.get_parameter('scene_arrive_lateral_m').value))
        self.scene_arrive_yaw_deg = abs(float(self.get_parameter('scene_arrive_yaw_deg').value))
        self.scene_arrive_stable_frames = max(1, int(self.get_parameter('scene_arrive_stable_frames').value))
        self.scene_lateral_yaw_gain = float(self.get_parameter('scene_lateral_yaw_gain').value)
        self.scene_vx_min = float(self.get_parameter('scene_vx_min').value)
        self.scene_vx_max = float(self.get_parameter('scene_vx_max').value)
        self.scene_wz_max = abs(float(self.get_parameter('scene_wz_max').value))
        self.scene_accl_vx = abs(float(self.get_parameter('scene_accl_vx').value))
        self.scene_accl_wz = abs(float(self.get_parameter('scene_accl_wz').value))
        self.scene_cmd_alpha = clamp(float(self.get_parameter('scene_cmd_alpha').value), 0.01, 1.0)
        self.scene_min_vx_cmd = abs(float(self.get_parameter('scene_min_vx_cmd').value))
        self.scene_min_wz_cmd = abs(float(self.get_parameter('scene_min_wz_cmd').value))
        self.scene_pose_min_confidence = float(self.get_parameter('scene_pose_min_confidence').value)
        self.scene_pose_min_inliers = max(1, int(self.get_parameter('scene_pose_min_inliers').value))
        self.scene_pose_hold_sec = max(0.0, float(self.get_parameter('scene_pose_hold_sec').value))
        self.shutdown_home_wait_sec = max(0.0, float(self.get_parameter('shutdown_home_wait_sec').value))

        # follow enabled
        self.follow_enabled_topic = str(self.get_parameter('follow_enabled_topic').value)
        self.auto_enable_follow = bool(self.get_parameter('auto_enable_follow').value)
        self.follow_enabled_republish_sec = max(0.1, float(self.get_parameter('follow_enabled_republish_sec').value))

        # action server bridge topics
        self.action_master_enabled_topic = str(self.get_parameter('action_master_enabled_topic').value)
        self.action_follow_enabled_topic = str(self.get_parameter('action_follow_enabled_topic').value)
        self.fsm_force_stop_topic = str(self.get_parameter('fsm_force_stop_topic').value)
        self.fsm_state_topic = str(self.get_parameter('fsm_state_topic').value)
        self.gesture_stop_event_topic = str(self.get_parameter('gesture_stop_event_topic').value)
        self.gesture_start = int(self.get_parameter('gesture_start').value)
        self.gesture_stop = int(self.get_parameter('gesture_stop').value)

        # gesture detection
        self.d435_image_topic = str(self.get_parameter('d435_image_topic').value)
        self.gesture_hold_frames = max(1, int(self.get_parameter('gesture_hold_frames').value))
        self.gesture_cooldown_sec = float(self.get_parameter('gesture_cooldown_sec').value)
        self.gesture_process_hz = max(1.0, float(self.get_parameter('gesture_process_hz').value))
        self.latest_d435_bgr: Optional[np.ndarray] = None
        self.gesture_candidate = 0
        self.gesture_hold_count = 0
        self.gesture_emit_latched = False
        self.gesture_last_emit_time: float = 0.0
        self.gesture_last_proc_time: float = 0.0
        self.mp_hands = None
        if self.follow_mode == 'person_yolo' and cv2 is not None:
            try:
                import mediapipe as mp  # type: ignore
                self.mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False, max_num_hands=1,
                    min_detection_confidence=float(self.get_parameter('gesture_min_det_conf').value),
                    min_tracking_confidence=float(self.get_parameter('gesture_min_trk_conf').value),
                )
                self.get_logger().info('mediapipe Hands initialized for gesture detection')
            except ImportError:
                self.get_logger().warn('mediapipe not installed; gesture detection disabled')
            except Exception as exc:
                self.get_logger().warn(f'mediapipe init failed: {exc}')
        self.base_coast_sec = float(self.get_parameter('base_coast_sec').value)

        # tf2
        self.use_tf2 = bool(self.get_parameter('use_tf2').value) and tf2_ros is not None
        self.camera_frame = str(self.get_parameter('camera_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        if self.use_tf2:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            from rclpy.qos import DurabilityPolicy
            static_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.static_tf_pub = self.create_publisher(
                TFMessage, '/tf_static', static_qos)
            static_pairs = [('camera_torso_link', 'd455_link')]
            if self.follow_mode == 'person_yolo':
                static_pairs.append(('camera_head_link', 'd435_link'))
            static_transforms = []
            for parent, child in static_pairs:
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = parent
                t.child_frame_id = child
                t.transform.rotation.w = 1.0
                static_transforms.append(t)
                self.get_logger().info(f'Static TF: {parent} → {child} (identity)')
            msg = TFMessage()
            msg.transforms = static_transforms
            self.static_tf_pub.publish(msg)
            self.get_logger().info(f'TF2 enabled: {self.camera_frame} → {self.base_frame}')
        else:
            self.tf_buffer = None
            self.tf_listener = None

        # derived
        self.dt = 1.0 / self.control_hz
        self.max_dvx = self.accl_vx * self.dt
        self.max_dwz = self.accl_wz * self.dt
        self.max_dvx_lost = max(1e-4, self.lost_decel_vx * self.dt)
        self.max_dwz_lost = max(1e-4, self.lost_decel_wz * self.dt)

        # =========================================================================
        # State
        # =========================================================================
        self.cv_bridge = CvBridge()
        self.latest_bgr: Optional[np.ndarray] = None
        self.latest_depth_m: Optional[np.ndarray] = None
        self.latest_image_stamp_ns: Optional[int] = None
        self.latest_depth_stamp_ns: Optional[int] = None
        self.cam_fx: Optional[float] = None
        self.cam_fy: Optional[float] = None
        self.cam_cx: Optional[float] = None
        self.cam_cy: Optional[float] = None
        self.last_log_time = 0.0
        self.last_head_debug_time = 0.0
        self.last_template_log_time = 0.0
        self.ref_data_list: list[dict[str, object]] = []
        self.template_scene_name = ''
        self.rs: Optional[Any] = None
        self.rs_pipeline: Optional[Any] = None
        self.rs_align: Optional[Any] = None

        # head state
        self.head_yaw_deg: Optional[float] = None
        self.head_pitch_deg: Optional[float] = None
        self.smooth_head_yaw: Optional[float] = None
        self.smooth_head_pitch: Optional[float] = None

        # chassis state
        self.cmd_vx: float = 0.0
        self.cmd_wz: float = 0.0
        self.tracking: bool = False
        self.last_base_ok_time: float = 0.0
        self.last_forward_m: Optional[float] = None
        self.scene_arrive_count: int = 0
        self.scene_completed: bool = False
        self.scene_base_forward_m: Optional[float] = None
        self.scene_base_lateral_m: Optional[float] = None
        self.scene_base_yaw_deg: Optional[float] = None
        self.scene_pose_ok: bool = False
        self.scene_state: str = 'WAIT_FRAME'  # WAIT_FRAME | TRACKING | HOLD | LOST_STOP | ARRIVED

        # follow enabled state
        self.last_follow_enabled_pub: Optional[bool] = None
        self.last_follow_enabled_pub_time: float = 0.0

        # action server bridge state
        self.action_master_enabled: bool = True
        self.fsm_force_stop_flag: bool = False
        self.action_follow_active: bool = self.auto_enable_follow if self.follow_mode == 'scene_template' else False

        # gesture state (YOLO mode only; scene mode uses action_follow_active)
        # auto_enable_follow only applies to scene_template; YOLO always starts
        # inactive and requires a gesture or action start to begin following.
        self.gesture_follow_active: bool = False

        # =========================================================================
        # Publishers
        # =========================================================================
        self.servo_pub = self.create_publisher(Float32MultiArray, self.servo_state_topic, 10)
        self.follow_enabled_pub = self.create_publisher(Bool, self.follow_enabled_topic, 10)
        self.fsm_state_pub = self.create_publisher(String, self.fsm_state_topic, 10)
        self.gesture_stop_event_pub = self.create_publisher(Bool, self.gesture_stop_event_topic, 10)
        if self.use_twist_stamped:
            self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_topic, 10)
        else:
            self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        # =========================================================================
        # Torso action client
        # =========================================================================
        if Torso is not None:
            self.torso_client = ActionClient(self, Torso, self.torso_action_name)
        else:
            self.torso_client = None
            self.get_logger().warn('interfaces.action.Torso not available — head control disabled')

        # =========================================================================
        # Subscriptions
        # =========================================================================
        if self.input_source == 'ros':
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=2,
            )
            self.create_subscription(Image, self.image_topic, self._image_cb, qos)
            self.create_subscription(Image, self.depth_topic, self._depth_cb, qos)
            self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, qos)
        elif self.input_source == 'realsense':
            self._init_realsense()
        else:
            raise ValueError(f'unsupported input_source={self.input_source}')
        self.create_subscription(JointState, self.joint_state_topic, self._joint_cb, 10)
        if self.follow_mode == 'person_yolo':
            d435_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=2)
            self.create_subscription(Image, self.d435_image_topic, self._d435_image_cb, d435_qos)
        # action server bridge subscriptions
        self.create_subscription(Bool, self.action_master_enabled_topic, self._on_action_master_enabled, 10)
        self.create_subscription(Bool, self.action_follow_enabled_topic, self._on_follow_enabled_command, 10)
        self.create_subscription(Bool, self.fsm_force_stop_topic, self._on_fsm_force_stop, 10)

        # match state (updated at match_hz)
        self.last_match_time: float = 0.0
        self.last_state: Optional[dict[str, object]] = None

        # YOLO async inference state
        self._yolo_lock = threading.Lock()
        self._yolo_latest_result: Optional[dict[str, object]] = None
        self._yolo_thread: Optional[threading.Thread] = None
        self._yolo_depth_ema: Optional[float] = None  # depth EMA for YOLO mode
        self._yolo_shutdown = False

        self._load_template(initial=True)

        self.timer = self.create_timer(1.0 / self.control_hz, self._tick)

        if self.follow_mode == 'scene_template':
            motion_cfg = (
                f'scene_vx=[{self.scene_vx_min},{self.scene_vx_max}] '
                f'scene_wz_max={self.scene_wz_max} '
                f'arrive=({self.scene_arrive_forward_m:.2f}m,'
                f'{self.scene_arrive_lateral_m:.2f}m,{self.scene_arrive_yaw_deg:.1f}deg) '
                f'pose_conf>={self.scene_pose_min_confidence:.2f}'
            )
        else:
            motion_cfg = (
                f'vx=[{self.vx_min},{self.vx_max}] wz_max={self.wz_max} '
                f'd_des={self.desired_distance_m}m coast={self.base_coast_sec}s '
                f'yolo_target_y_ratio={self.yolo_target_y_ratio:.2f}'
            )
        self.get_logger().info(
            f'scene_servo_node started (all-in-one). hz={self.control_hz} '
            f'mode={self.follow_mode} input={self.input_source} cmd={self.cmd_topic} '
            f'k_dist={self.k_dist} k_yaw={self.k_yaw} {motion_cfg}'
        )

    # =========================================================================
    # RealSense
    # =========================================================================
    def _init_realsense(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError('pyrealsense2 not installed') from exc
        self.rs = rs
        ctx = rs.context()
        serial = None
        for dev in ctx.query_devices():
            name = dev.get_info(rs.camera_info.name)
            if self.rs_device_name.lower() in name.lower():
                serial = dev.get_info(rs.camera_info.serial_number)
                break
        if serial is None:
            raise RuntimeError(f'no RealSense matched {self.rs_device_name!r}')
        self.rs_pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, self.rs_width, self.rs_height, rs.format.bgr8, self.rs_fps)
        cfg.enable_stream(rs.stream.depth, self.rs_width, self.rs_height, rs.format.z16, self.rs_fps)
        profile = self.rs_pipeline.start(cfg)
        self.rs_align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.cam_fx = float(intr.fx)
        self.cam_fy = float(intr.fy)
        self.cam_cx = float(intr.ppx)
        self.cam_cy = float(intr.ppy)
        self.get_logger().info(
            f'realsense: serial={serial} {self.rs_width}x{self.rs_height}@{self.rs_fps} '
            f'intr=({self.cam_fx:.2f},{self.cam_fy:.2f},{self.cam_cx:.2f},{self.cam_cy:.2f})'
        )

    def _poll_realsense(self) -> None:
        if self.rs_pipeline is None or self.rs_align is None:
            return
        try:
            frames = self.rs_pipeline.wait_for_frames(timeout_ms=200)
            aligned = self.rs_align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                return
            self.latest_bgr = np.asanyarray(color_frame.get_data())
            self.latest_depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * 0.001
            ns = self.get_clock().now().nanoseconds
            self.latest_image_stamp_ns = ns
            self.latest_depth_stamp_ns = ns
        except Exception as exc:
            now = time.time()
            if (now - self.last_log_time) >= self.log_period_sec:
                self.last_log_time = now
                self.get_logger().warn(f'realsense poll failed: {exc}')

    # =========================================================================
    # Template
    # =========================================================================
    def _load_template(self, initial: bool = False) -> None:
        if not self.template_path:
            if initial:
                self.get_logger().warn('template_path is empty; node will stay idle.')
            self.ref_data_list = []
            self.template_scene_name = ''
            return
        tf = Path(self.template_path)
        if not tf.exists():
            now = time.time()
            if initial or (now - self.last_template_log_time) >= self.log_period_sec:
                self.get_logger().warn(f'template_path not found: {tf}')
                self.last_template_log_time = now
            self.ref_data_list = []
            self.template_scene_name = ''
            return
        template = load_template(tf)
        self.ref_data_list = template_to_all_ref_data(template)
        self.template_scene_name = str(template.get('scene_name', tf.stem))
        intrinsics = template.get('camera_intrinsics', {})
        if self.cam_fx is None and intrinsics:
            self.cam_fx = float(intrinsics.get('fx', 0.0)) or None
            self.cam_fy = float(intrinsics.get('fy', 0.0)) or None
            self.cam_cx = float(intrinsics.get('cx', 0.0)) or None
            self.cam_cy = float(intrinsics.get('cy', 0.0)) or None
        self.get_logger().info(f'loaded template "{self.template_scene_name}" ({len(self.ref_data_list)} keyframe(s))')

    # =========================================================================
    # Callbacks
    # =========================================================================
    def _image_cb(self, msg: Image) -> None:
        try:
            self.latest_bgr = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image_stamp_ns = self._stamp_ns(msg.header.stamp)
        except Exception as exc:
            self.get_logger().warn(f'image convert failed: {exc}')

    def _depth_cb(self, msg: Image) -> None:
        try:
            d = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            d = np.asarray(d)
            self.latest_depth_m = d.astype(np.float32) * 0.001 if d.dtype == np.uint16 else d.astype(np.float32)
            self.latest_depth_stamp_ns = self._stamp_ns(msg.header.stamp)
        except Exception as exc:
            self.get_logger().warn(f'depth convert failed: {exc}')

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if len(msg.k) < 9:
            return
        self.cam_fx, self.cam_fy = float(msg.k[0]), float(msg.k[4])
        self.cam_cx, self.cam_cy = float(msg.k[2]), float(msg.k[5])

    def _joint_cb(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return
        n2p = {n: msg.position[i] for i, n in enumerate(msg.name) if i < len(msg.position)}
        for k in ('head_yaw', 'head_yaw_joint'):
            if k in n2p:
                self.head_yaw_deg = math.degrees(n2p[k])
                break
        for k in ('head_pitch', 'head_pitch_joint'):
            if k in n2p:
                self.head_pitch_deg = math.degrees(n2p[k])
                break

    # --- action server bridge callbacks ---
    def _on_action_master_enabled(self, msg: Bool) -> None:
        self.action_master_enabled = msg.data
        if not self.action_master_enabled:
            self.action_follow_active = False
            self.gesture_follow_active = False
        self.get_logger().info(f'action_master_enabled={self.action_master_enabled}')

    def _on_follow_enabled_command(self, msg: Bool) -> None:
        """Handle follow_enabled topic (from action server).

        scene_template: start/stop scene follow via action_follow_active.
        person_yolo: no-op here — YOLO follow is only triggered by gesture-1
        or gesture-2, not by the follow_enabled topic.
        """
        enabled = bool(msg.data)
        if not enabled:
            if self.action_follow_active:
                self.get_logger().info('follow_enabled=False -> stop follow')
            self.action_follow_active = False
            self.scene_completed = False
            self.scene_arrive_count = 0
            self.scene_state = 'WAIT_FRAME'
            return
        if self.follow_mode == 'scene_template':
            if not self.action_follow_active:
                self.get_logger().info('follow_enabled=True -> scene follow active')
            self.action_follow_active = True
            self.scene_completed = False
            self.scene_arrive_count = 0
            self.scene_state = 'WAIT_FRAME'

    def _on_fsm_force_stop(self, msg: Bool) -> None:
        if msg.data:
            self.fsm_force_stop_flag = True
            self.action_follow_active = False
            self.gesture_follow_active = False
            self.scene_completed = False
            self.scene_arrive_count = 0
            self.scene_state = 'WAIT_FRAME'
            self.get_logger().info('fsm_force_stop received, stopping follow')

    def _pub_fsm_state(self) -> None:
        state = 'FOLLOWING' if self._is_follow_active() else 'IDLE'
        msg = String()
        msg.data = state
        self.fsm_state_pub.publish(msg)

    def _is_follow_active(self) -> bool:
        if self.follow_mode == 'scene_template':
            return self.action_follow_active and not self.scene_completed
        return self.gesture_follow_active

    def _d435_image_cb(self, msg: Image) -> None:
        try:
            self.latest_d435_bgr = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            pass

    def _detect_gesture_frame(self) -> int:
        """Run mediapipe gesture detection on latest D435 frame. Returns 0/1/2."""
        if self.mp_hands is None or self.latest_d435_bgr is None:
            return 0
        try:
            rgb = cv2.cvtColor(self.latest_d435_bgr, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb)
            if not results.multi_hand_landmarks:
                return 0
            for hand_lm in results.multi_hand_landmarks:
                lm = hand_lm.landmark
                index_up = lm[8].y < lm[6].y - 0.006
                middle_up = lm[12].y < lm[10].y - 0.006
                ring_up = lm[16].y < lm[14].y - 0.006
                pinky_up = lm[20].y < lm[18].y - 0.006
                # gesture 1: only index up
                if index_up and not middle_up and not ring_up and not pinky_up:
                    return 1
                # gesture 2: index + middle up
                if index_up and middle_up and not ring_up and not pinky_up:
                    return 2
            return 0
        except Exception:
            return 0

    def _process_gesture(self, now: float) -> None:
        """Run gesture detection at gesture_process_hz and update state."""
        if (now - self.gesture_last_proc_time) < (1.0 / self.gesture_process_hz):
            return
        self.gesture_last_proc_time = now

        candidate = self._detect_gesture_frame()

        # hold gate
        if candidate == self.gesture_candidate and candidate in (1, 2):
            self.gesture_hold_count += 1
        else:
            self.gesture_candidate = candidate
            self.gesture_hold_count = 1 if candidate in (1, 2) else 0

        gid = candidate if (candidate in (1, 2) and self.gesture_hold_count >= self.gesture_hold_frames) else 0

        # cooldown + latch
        if gid == 0:
            self.gesture_emit_latched = False
        if gid != 0:
            if self.gesture_emit_latched or (now - self.gesture_last_emit_time) < self.gesture_cooldown_sec:
                gid = 0
            else:
                self.gesture_emit_latched = True
                self.gesture_last_emit_time = now

        # apply
        if gid == self.gesture_start and not self.gesture_follow_active:
            self.gesture_follow_active = True
            self.get_logger().info('Gesture START → follow active')
        elif gid == self.gesture_stop and self._is_follow_active():
            self.action_follow_active = False
            self.gesture_follow_active = False
            self.tracking = False
            self.last_forward_m = None
            self.get_logger().info('Gesture STOP → follow disabled, heading home')
            self._send_home_head()
            # publish stop event for action server (always, so behavior tree gets a result)
            self.gesture_stop_event_pub.publish(Bool(data=True))

    @staticmethod
    def _stamp_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _has_fresh_frame(self) -> bool:
        if self.latest_bgr is None or self.latest_depth_m is None:
            return False
        if self.latest_image_stamp_ns is None or self.latest_depth_stamp_ns is None:
            return False
        now_ns = self.get_clock().now().nanoseconds
        image_age_sec = (now_ns - self.latest_image_stamp_ns) / 1e9
        depth_age_sec = (now_ns - self.latest_depth_stamp_ns) / 1e9
        if image_age_sec > self.frame_timeout_sec or depth_age_sec > self.frame_timeout_sec:
            return False
        if self.follow_mode == 'scene_template' and self.scene_frame_sync_tolerance_sec > 0.0:
            stamp_diff_sec = abs(self.latest_image_stamp_ns - self.latest_depth_stamp_ns) / 1e9
            return stamp_diff_sec <= self.scene_frame_sync_tolerance_sec
        return True

    # =========================================================================
    # Head control
    # =========================================================================
    def _send_head_velocity(self, yaw_vel_deg_s: float, pitch_vel_deg_s: float) -> None:
        yaw_cmd = clamp(yaw_vel_deg_s, -self.head_yaw_vel_max_deg_s, self.head_yaw_vel_max_deg_s)
        pitch_cmd = clamp(pitch_vel_deg_s, -self.head_pitch_vel_max_deg_s, self.head_pitch_vel_max_deg_s)
        if self.torso_client is None:
            return
        if not self.torso_client.server_is_ready():
            self.get_logger().warn('Torso action server not ready, cannot move head', throttle_duration_sec=5.0)
            return
        try:
            goal = Torso.Goal()
            goal.torso_height = 0.0
            goal.torso_yaw = 0.0
            goal.head_yaw = yaw_cmd
            goal.head_pitch = pitch_cmd
            # Keep torso at zero-height while tracking head.
            goal.torso_mask = [False, True, False]
            goal.head_mask = [True, True]
            goal.max_velocity = 0.5
            goal.work_mode = 4  # VELOCITY_MODE (3588 torso_control supports mode 4)
            self.torso_client.send_goal_async(goal)
        except Exception:
            pass

    def _stop_head(self) -> None:
        self._send_head_velocity(0.0, 0.0)

    def _send_home_head(self) -> None:
        """Smoothly return head to home position."""
        if self.head_yaw_deg is None or self.head_pitch_deg is None:
            self._stop_head()
            return
        yaw_err = self.home_head_yaw - self.head_yaw_deg
        pitch_err = self.home_head_pitch - self.head_pitch_deg
        if abs(yaw_err) > 1.0 or abs(pitch_err) > 1.0:
            self._send_head_velocity(
                clamp(1.5 * yaw_err, -30.0, 30.0),
                clamp(1.5 * pitch_err, -20.0, 20.0))
        else:
            self._stop_head()

    def _update_head(self, track_error_x: float, track_error_y: float, head_ok: bool) -> None:
        if head_ok and self.head_yaw_deg is not None and self.head_pitch_deg is not None:
            raw_yaw = clamp(
                self.head_yaw_deg + (-self.yaw_sign * track_error_x * self.head_fov_h_deg), -90.0, 90.0)
            raw_pitch = clamp(
                self.head_pitch_deg + (-self.pitch_sign * track_error_y * self.head_fov_v_deg), -34.0, 19.0)

            if self.smooth_head_yaw is None:
                self.smooth_head_yaw = raw_yaw
            if self.smooth_head_pitch is None:
                self.smooth_head_pitch = raw_pitch

            # Adaptive smoothing: fast when error large, smooth when small
            dy = raw_yaw - self.smooth_head_yaw
            dp = raw_pitch - self.smooth_head_pitch
            max_d = max(abs(dy), abs(dp))
            alpha = clamp(0.35 + 0.25 * (max_d - 3.0) / 10.0, 0.35, 0.55)
            self.smooth_head_yaw += alpha * dy
            self.smooth_head_pitch += alpha * dp

            ye = self.smooth_head_yaw - self.head_yaw_deg
            pe = self.smooth_head_pitch - self.head_pitch_deg
            yaw_cmd = 0.0
            pitch_cmd = 0.0
            if abs(ye) > self.head_deadband_deg or abs(pe) > self.head_deadband_deg:
                # P control: slightly damped from 17:10 baseline to reduce overshoot.
                vel_gain = 1.7
                yaw_cmd = clamp(vel_gain * ye, -60.0, 60.0)
                pitch_cmd = clamp(vel_gain * pe, -40.0, 40.0)
                self._send_head_velocity(
                    yaw_cmd,
                    pitch_cmd)
            else:
                self._stop_head()
            now = time.time()
            if self.follow_mode == 'person_yolo' and (now - self.last_head_debug_time) >= 0.25:
                self.last_head_debug_time = now
                self.get_logger().info(
                    'HEAD_CTRL: '
                    f'track=({track_error_x:+.3f},{track_error_y:+.3f}) '
                    f'head=({self.head_yaw_deg:+.1f},{self.head_pitch_deg:+.1f}) '
                    f'raw=({raw_yaw:+.1f},{raw_pitch:+.1f}) '
                    f'smooth=({self.smooth_head_yaw:+.1f},{self.smooth_head_pitch:+.1f}) '
                    f'err=({ye:+.1f},{pe:+.1f}) '
                    f'cmd=({yaw_cmd:+.1f},{pitch_cmd:+.1f})'
                )
        else:
            self._stop_head()
            self.smooth_head_yaw = None
            self.smooth_head_pitch = None

    def _send_home_position(self) -> None:
        """Send one-shot position home goal for torso/head."""
        if self.torso_client is None:
            return
        if not self.torso_client.server_is_ready():
            self.get_logger().warn('Torso action server not ready during shutdown, skip home position')
            return
        try:
            goal = Torso.Goal()
            goal.work_mode = 0  # POSITION_MODE
            goal.input_mode = 0
            goal.torso_roll = 0.0
            goal.torso_height = 0.0
            goal.torso_yaw = 0.0
            goal.head_pitch = float(self.home_head_pitch)
            goal.head_yaw = float(self.home_head_yaw)
            # Send torso zero signal on shutdown home.
            goal.torso_mask = [False, True, False]
            goal.head_mask = [True, True]
            goal.max_velocity = 0.2
            future = self.torso_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            goal_handle = future.result() if future.done() else None
            if goal_handle is not None and goal_handle.accepted:
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future, timeout_sec=self.shutdown_home_wait_sec)
        except Exception as exc:
            self.get_logger().warn(f'failed to send home position on shutdown: {exc}')

    def graceful_shutdown(self) -> None:
        """Stop chassis/head and send one-shot home before rclpy shutdown."""
        self.get_logger().info('scene_servo graceful shutdown: stop cmd + home head...')
        try:
            self._publish_cmd(0.0, 0.0)
        except Exception:
            pass
        try:
            self._stop_head()
        except Exception:
            pass
        self._send_home_position()
        # Keep the process alive long enough for the torso controller to execute home.
        time.sleep(self.shutdown_home_wait_sec)

    # =========================================================================
    # Chassis control
    # =========================================================================
    def _publish_cmd(self, vx: float, wz: float) -> None:
        if self.use_twist_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.twist.linear.x = vx
            msg.twist.angular.z = wz
        else:
            msg = Twist()
            msg.linear.x = vx
            msg.angular.z = wz
        self.cmd_pub.publish(msg)

    def _transform_forward_error(self, forward_m: float, lateral_m: float) -> float:
        """Transform camera-frame (forward, lateral) to base_link forward distance."""
        if not self.use_tf2 or self.tf_buffer is None:
            return forward_m
        try:
            pt = PointStamped()
            pt.header.frame_id = self.camera_frame
            # Use Time(0) to get latest available transform (avoids clock mismatch)
            pt.header.stamp = rclpy.time.Time(seconds=0, nanoseconds=0).to_msg()
            pt.point.x = forward_m
            pt.point.y = lateral_m
            pt.point.z = 0.0
            transformed = self.tf_buffer.transform(pt, self.base_frame)
            return transformed.point.x
        except Exception as exc:
            if self.get_clock().now().nanoseconds // 5_000_000_000 != getattr(self, '_last_tf_warn_ns', 0) // 5_000_000_000:
                self._last_tf_warn_ns = self.get_clock().now().nanoseconds
                self.get_logger().warn(f'TF transform failed, using raw forward: {exc}')
            return forward_m

    @staticmethod
    def _transform_to_matrix(transform: TransformStamped) -> np.ndarray:
        q = transform.transform.rotation
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
        n = x * x + y * y + z * z + w * w
        if n <= 1e-12:
            rot = np.eye(3)
        else:
            s = 2.0 / n
            xx, yy, zz = x * x * s, y * y * s, z * z * s
            xy, xz, yz = x * y * s, x * z * s, y * z * s
            wx, wy, wz = w * x * s, w * y * s, w * z * s
            rot = np.array([
                [1.0 - yy - zz, xy - wz, xz + wy],
                [xy + wz, 1.0 - xx - zz, yz - wx],
                [xz - wy, yz + wx, 1.0 - xx - yy],
            ], dtype=np.float64)
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = rot
        t = transform.transform.translation
        mat[:3, 3] = [float(t.x), float(t.y), float(t.z)]
        return mat

    @staticmethod
    def _yaw_from_matrix_deg(mat: np.ndarray) -> float:
        return math.degrees(math.atan2(float(mat[1, 0]), float(mat[0, 0])))

    def _scene_error_in_base(
        self,
        state: dict[str, object],
        fallback_forward_m: float,
        fallback_lateral_m: float,
    ) -> tuple[float, float, float]:
        rot = state.get('rotation_ref_to_cur')
        trans = state.get('translation_ref_to_cur')
        if rot is None or trans is None or not self.use_tf2 or self.tf_buffer is None:
            return fallback_forward_m, fallback_lateral_m, float(state.get('yaw_error_deg', 0.0))
        try:
            t_base_camera = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time(seconds=0, nanoseconds=0),
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout_sec),
            )
            base_from_camera = self._transform_to_matrix(t_base_camera)
            camera_from_base = np.linalg.inv(base_from_camera)

            camera_cur_from_ref = np.eye(4, dtype=np.float64)
            camera_cur_from_ref[:3, :3] = np.array(rot, dtype=np.float64).reshape(3, 3)
            camera_cur_from_ref[:3, 3] = np.array(trans, dtype=np.float64).reshape(3)

            base_cur_from_ref_base = base_from_camera @ camera_cur_from_ref @ camera_from_base
            return (
                float(base_cur_from_ref_base[0, 3]),
                float(base_cur_from_ref_base[1, 3]),
                self._yaw_from_matrix_deg(base_cur_from_ref_base),
            )
        except Exception as exc:
            self.get_logger().warn(f'scene TF base error failed, using camera error: {exc}', throttle_duration_sec=2.0)
            return fallback_forward_m, fallback_lateral_m, float(state.get('yaw_error_deg', 0.0))

    def _update_chassis(self, forward_error_m: float, lateral_error_m: float, base_ok: bool, now: float) -> None:
        if self.follow_mode == 'scene_template':
            self._update_scene_chassis(forward_error_m, lateral_error_m, base_ok, now)
            return

        # --- wz from head yaw ---
        target_wz = 0.0
        if self.head_yaw_deg is not None:
            head_yaw_rad = math.radians(self.head_yaw_deg)
            if abs(math.degrees(head_yaw_rad)) > self.deadband_yaw_deg:
                target_wz = clamp(self.k_yaw * head_yaw_rad, -self.wz_max, self.wz_max)

        # --- vx from distance (TF2 corrected) ---
        target_vx = 0.0
        if base_ok:
            self.tracking = True
            self.last_base_ok_time = now
            base_forward = self._transform_forward_error(forward_error_m, lateral_error_m)
            # TF validation: compare with manual head_yaw projection
            if self.head_yaw_deg is not None:
                theta = math.radians(self.head_yaw_deg)
                manual_fwd = forward_error_m * math.cos(theta) - lateral_error_m * math.sin(theta)
                manual_lat = forward_error_m * math.sin(theta) + lateral_error_m * math.cos(theta)
                fwd_diff = abs(base_forward - manual_fwd)
                if fwd_diff > 0.05 or (now - self.last_log_time) < 0.1:
                    self.get_logger().info(
                        f'TF_CHECK: cam_fwd={forward_error_m:.3f} lat={lateral_error_m:.3f} '
                        f'head_yaw={self.head_yaw_deg:+.1f}° '
                        f'tf_base_fwd={base_forward:.3f} manual_fwd={manual_fwd:.3f} '
                        f'diff={fwd_diff:.3f} tf_ok={fwd_diff < 0.1}'
                    )
            self.last_forward_m = base_forward
            dist_err = base_forward - self.desired_distance_m
            if abs(dist_err) > self.deadband_distance_m:
                target_vx = clamp(self.k_dist * dist_err, self.vx_min, self.vx_max)

            # YOLO mode: attenuate vx when head yaw is large so robot
            # pivots in-place instead of driving a wide arc.
            if self.head_yaw_deg is not None:
                abs_yaw = abs(self.head_yaw_deg)
                if abs_yaw > self.vx_yaw_damp_start_deg:
                    ratio = max(0.0, 1.0 - (abs_yaw - self.vx_yaw_damp_start_deg)
                                / (self.vx_yaw_damp_end_deg - self.vx_yaw_damp_start_deg))
                    target_vx *= ratio
        elif self.tracking:
            # coast: keep going for base_coast_sec
            if (now - self.last_base_ok_time) < self.base_coast_sec and self.last_forward_m is not None:
                dist_err = self.last_forward_m - self.desired_distance_m
                if abs(dist_err) > self.deadband_distance_m:
                    target_vx = clamp(self.k_dist * dist_err, self.vx_min, self.vx_max)
            else:
                self.tracking = False
                self.last_forward_m = None

        # --- ramp ---
        if self.tracking or target_vx != 0.0 or target_wz != 0.0:
            max_dvx = self.max_dvx
            max_dwz = self.max_dwz
        else:
            max_dvx = self.max_dvx_lost
            max_dwz = self.max_dwz_lost

        self.cmd_vx = clamp(self.cmd_vx + clamp(target_vx - self.cmd_vx, -max_dvx, max_dvx), self.vx_min, self.vx_max)
        self.cmd_wz = clamp(self.cmd_wz + clamp(target_wz - self.cmd_wz, -max_dwz, max_dwz), -self.wz_max, self.wz_max)

        # decel to zero when lost
        if not self.tracking and target_vx == 0.0 and target_wz == 0.0:
            if abs(self.cmd_vx) < 0.005:
                self.cmd_vx = 0.0
            if abs(self.cmd_wz) < 0.005:
                self.cmd_wz = 0.0

        self._publish_cmd(self.cmd_vx, self.cmd_wz)

    def _update_scene_chassis(self, forward_error_m: float, lateral_error_m: float, base_ok: bool, now: float) -> None:
        state = self.last_state or {}
        level = str(state.get('level', 'LOST'))
        rot = state.get('rotation_ref_to_cur')
        trans = state.get('translation_ref_to_cur')
        has_pose = rot is not None and trans is not None
        scene_pose_ok = (
            has_pose
            and level == 'L0'
            and float(state.get('confidence', 0.0)) >= self.scene_pose_min_confidence
            and int(state.get('inlier_count', 0)) >= self.scene_pose_min_inliers
        )
        self.scene_pose_ok = scene_pose_ok

        target_vx = 0.0
        target_wz = 0.0

        if scene_pose_ok:
            # Strict scene return: use the D455 L0 3D-3D pose directly.
            # This intentionally does not use robot TF; D455 is fixed on the robot.
            rot_mat = np.array(rot, dtype=np.float64).reshape(3, 3)
            trans_vec = np.array(trans, dtype=np.float64).reshape(3)
            base_forward_m = float(trans_vec[2])
            base_lateral_m = float(trans_vec[0])
            base_yaw_deg = self._yaw_from_matrix_deg(rot_mat)
            self.scene_base_forward_m = base_forward_m
            self.scene_base_lateral_m = base_lateral_m
            self.scene_base_yaw_deg = base_yaw_deg
            self.tracking = True
            self.last_base_ok_time = now
            self.last_forward_m = base_forward_m

            arrived = (
                abs(base_forward_m) <= self.scene_arrive_forward_m
                and abs(base_lateral_m) <= self.scene_arrive_lateral_m
                and abs(base_yaw_deg) <= self.scene_arrive_yaw_deg
            )
            self.scene_arrive_count = self.scene_arrive_count + 1 if arrived else 0

            if self.scene_arrive_count >= self.scene_arrive_stable_frames:
                if not self.scene_completed:
                    self.scene_state = 'ARRIVED'
                    self.scene_completed = True
                    self.action_follow_active = False
                    self.gesture_follow_active = False
                    self.tracking = False
                    self.cmd_vx = 0.0
                    self.cmd_wz = 0.0
                    self._publish_cmd(0.0, 0.0)
                    self._pub_follow_enabled(False)
                    self.gesture_stop_event_pub.publish(Bool(data=True))
                    self.get_logger().info(
                        'scene target reached: stop follow '
                        f'base_fwd={base_forward_m:+.3f} base_lat={base_lateral_m:+.3f} base_yaw={base_yaw_deg:+.1f}'
                    )
                return

            self.scene_state = 'TRACKING'

            if abs(base_forward_m) > self.scene_arrive_forward_m:
                target_vx = clamp(self.k_dist * base_forward_m, self.scene_vx_min, self.scene_vx_max)
                if 0.0 < abs(target_vx) < self.scene_min_vx_cmd:
                    target_vx = math.copysign(min(self.scene_min_vx_cmd, self.scene_vx_max), target_vx)

            yaw_cmd = math.radians(base_yaw_deg)
            if abs(base_lateral_m) > self.scene_arrive_lateral_m:
                yaw_cmd += self.scene_lateral_yaw_gain * base_lateral_m
            if abs(math.degrees(yaw_cmd)) > self.deadband_yaw_deg:
                target_wz = clamp(self.k_yaw * yaw_cmd, -self.scene_wz_max, self.scene_wz_max)
                if 0.0 < abs(target_wz) < self.scene_min_wz_cmd:
                    target_wz = math.copysign(min(self.scene_min_wz_cmd, self.scene_wz_max), target_wz)
        elif self.tracking:
            # HOLD: was tracking, lost pose — smooth decel, wait for recovery
            self.scene_state = 'HOLD'
            self.scene_arrive_count = 0
            if (now - self.last_base_ok_time) > self.scene_pose_hold_sec:
                self.tracking = False
                self.last_forward_m = None
                self.scene_state = 'LOST_STOP'
        else:
            self.scene_state = 'LOST_STOP'
            self.scene_arrive_count = 0
            self.scene_base_forward_m = None
            self.scene_base_lateral_m = None
            self.scene_base_yaw_deg = None

        if self.tracking or target_vx != 0.0 or target_wz != 0.0:
            max_dvx = self.scene_accl_vx * self.dt
            max_dwz = self.scene_accl_wz * self.dt
        else:
            max_dvx = self.max_dvx_lost
            max_dwz = self.max_dwz_lost

        ramp_vx = self.cmd_vx + clamp(target_vx - self.cmd_vx, -max_dvx, max_dvx)
        ramp_wz = self.cmd_wz + clamp(target_wz - self.cmd_wz, -max_dwz, max_dwz)
        self.cmd_vx = clamp(
            self.cmd_vx + self.scene_cmd_alpha * (ramp_vx - self.cmd_vx),
            self.scene_vx_min,
            self.scene_vx_max,
        )
        self.cmd_wz = clamp(
            self.cmd_wz + self.scene_cmd_alpha * (ramp_wz - self.cmd_wz),
            -self.scene_wz_max,
            self.scene_wz_max,
        )

        if not self.tracking and target_vx == 0.0 and target_wz == 0.0:
            if abs(self.cmd_vx) < 0.005:
                self.cmd_vx = 0.0
            if abs(self.cmd_wz) < 0.005:
                self.cmd_wz = 0.0

        self._publish_cmd(self.cmd_vx, self.cmd_wz)

    # =========================================================================
    # Servo state (debug)
    # =========================================================================
    def _publish_servo_state(self, state: dict[str, object]) -> None:
        yaw_error_deg = float(state.get('yaw_error_deg', 0.0))
        msg = Float32MultiArray()
        msg.data = [
            clamp(-yaw_error_deg / self.head_fov_h_deg, -1.0, 1.0),
            clamp(-float(state.get('pitch_error_deg', 0.0)) / self.head_fov_v_deg, -1.0, 1.0),
            yaw_error_deg,
            float(state.get('forward_error_m', 0.0)),
            float(state.get('lateral_error_m', 0.0)),
            float(state.get('confidence', 0.0)),
            1.0 if bool(state.get('head_tracking_ok', False)) else 0.0,
            1.0 if bool(state.get('base_servo_ready', False)) else 0.0,
            float(SERVO_MODE_CODE.get(str(state.get('servo_mode', 'search')), 0.0)),
        ]
        self.servo_pub.publish(msg)

    # =========================================================================
    # Follow enabled
    # =========================================================================
    def _pub_follow_enabled(self, enabled: bool) -> None:
        now = time.time()
        if (self.last_follow_enabled_pub is not None
                and self.last_follow_enabled_pub == enabled
                and (now - self.last_follow_enabled_pub_time) < self.follow_enabled_republish_sec):
            return
        msg = Bool()
        msg.data = enabled
        self.follow_enabled_pub.publish(msg)
        self.last_follow_enabled_pub = enabled
        self.last_follow_enabled_pub_time = now

    # =========================================================================
    # Main tick
    # =========================================================================
    # =========================================================================
    # Scene template tick
    # =========================================================================
    def _tick_scene(self, now: float) -> None:
        # Separate input readiness from match timing:
        #   input_ready = template + fresh RGBD + intrinsics available
        #   match_due   = enough time since last match for a new one
        # When input is not ready, we must NOT use stale last_state for control.
        # When input is ready but match is not due, keep last_state for smooth control.
        input_ready = (
            bool(self.ref_data_list)
            and self._has_fresh_frame()
            and None not in (self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy)
        )
        if not input_ready:
            self.last_state = None
            self.scene_state = 'WAIT_FRAME'
            return

        match_due = (now - self.last_match_time) >= (1.0 / self.match_hz)
        if not match_due:
            return  # keep last_state, allow smooth control between matches

        self.last_match_time = now
        # Snapshot frame data + intrinsics + stamps so matching and control are
        # bound to the same input.  Matches what _tick_yolo() does.
        bgr_snap = self.latest_bgr.copy()
        depth_snap = self.latest_depth_m.copy()
        fx_snap = float(self.cam_fx)
        fy_snap = float(self.cam_fy)
        cx_snap = float(self.cam_cx)
        cy_snap = float(self.cam_cy)
        snap_img_stamp_ns = self.latest_image_stamp_ns
        snap_depth_stamp_ns = self.latest_depth_stamp_ns
        result = estimate_best_keyframe(
            self.ref_data_list, bgr_snap, depth_snap,
            fx=fx_snap, fy=fy_snap,
            cx=cx_snap, cy=cy_snap,
            cfg=self.matcher_cfg,
        )
        self.last_state = self.estimator.update(result)
        # Attach snapshot stamps for worldpilot data alignment
        if self.last_state is not None:
            self.last_state['_snap_img_stamp_ns'] = snap_img_stamp_ns
            self.last_state['_snap_depth_stamp_ns'] = snap_depth_stamp_ns
        self._publish_servo_state(self.last_state)

    # =========================================================================
    # YOLO person tick
    # =========================================================================
    # =========================================================================
    # YOLO async inference
    # =========================================================================
    def _detect_yolo_person(self, bgr: np.ndarray) -> Optional[tuple[float, float, float, float, float, int]]:
        """Return largest person bbox as (x1, y1, x2, y2, conf, count)."""
        if self.yolo_model is None:
            return None
        results = self.yolo_model.predict(
            source=bgr,
            classes=[self.yolo_detect_class_id],
            conf=self.yolo_conf_thres,
            imgsz=self.yolo_imgsz,
            device=self.yolo_device if self.yolo_device else None,
            half=self.yolo_use_half,
            verbose=False,
        )
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
        best = int(areas.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy()
        conf = float(boxes.conf[best].cpu())
        return float(x1), float(y1), float(x2), float(y2), conf, len(boxes)

    def _yolo_inference_worker(
        self,
        bgr: np.ndarray,
        depth_m: np.ndarray,
        d435_bgr: Optional[np.ndarray],
    ) -> None:
        """Run YOLO inference in background thread, store result.

        YOLO mode uses D455 for base distance and D435 for head tracking.  D455 is
        chest-fixed, so using D455 image error to command the head creates a
        positive feedback loop where the head can run to its yaw limit.
        """
        try:
            d455_det = self._detect_yolo_person(bgr)
            d435_det = self._detect_yolo_person(d435_bgr) if d435_bgr is not None else None
            if d455_det is None and d435_det is None:
                with self._yolo_lock:
                    self._yolo_latest_result = None
                return

            yaw_error_deg = 0.0
            pitch_error_deg = 0.0
            head_ok = False
            if d435_det is not None:
                hx1, hy1, hx2, hy2, hconf, _hcount = d435_det
                hh, hw = d435_bgr.shape[:2]
                hcx_px = (hx1 + hx2) / 2.0
                hbbox_h = max(1.0, (hy2 - hy1))
                hcy_px = hy1 + self.yolo_target_y_ratio * hbbox_h
                yaw_error_deg = -(hcx_px - hw / 2.0) / hw * self.head_fov_h_deg
                pitch_error_deg = -(hcy_px - hh / 2.0) / hh * self.head_fov_v_deg
                head_ok = hconf > self.yolo_conf_thres

            if d455_det is None:
                state = {
                    'servo_mode': 'head_track_only' if head_ok else 'search',
                    'level': 3,
                    'yaw_error_deg': yaw_error_deg,
                    'pitch_error_deg': pitch_error_deg,
                    'forward_error_m': 0.0,
                    'lateral_error_m': 0.0,
                    'confidence': float(d435_det[4]) if d435_det is not None else 0.0,
                    'inlier_count': 0,
                    'matched_count': 0,
                    'head_tracking_ok': head_ok,
                    'base_servo_ready': False,
                }
                with self._yolo_lock:
                    self._yolo_latest_result = state
                return

            x1, y1, x2, y2, conf, box_count = d455_det
            h, w = bgr.shape[:2]
            cx_px = (x1 + x2) / 2.0
            bbox_h = max(1.0, (y2 - y1))
            cy_px = y1 + self.yolo_target_y_ratio * bbox_h

            # depth at bbox center
            ix, iy = int(cx_px), int(cy_px)
            if 0 <= iy < depth_m.shape[0] and 0 <= ix < depth_m.shape[1]:
                patch = depth_m[max(0, iy-2):iy+3, max(0, ix-2):ix+3]
                valid = patch[patch > 0.1]
                depth_val = float(np.median(valid)) if len(valid) > 0 else None
            else:
                depth_val = None

            if depth_val is None or depth_val > self.matcher_cfg.depth_max_m:
                with self._yolo_lock:
                    self._yolo_latest_result = None
                return

            # D455 computes base distance/lateral error only.  Head yaw/pitch
            # errors come from D435 above because the head camera moves with the
            # head and closes the tracking loop.
            # EMA depth smoothing to suppress single-frame jumps
            if self._yolo_depth_ema is not None and depth_val is not None:
                delta = abs(depth_val - self._yolo_depth_ema)
                # reject outliers: >1.0m jump in one frame → use previous
                if delta > 1.0:
                    depth_val = self._yolo_depth_ema
                else:
                    alpha = 0.4  # 40% new, 60% old
                    depth_val = alpha * depth_val + (1 - alpha) * self._yolo_depth_ema
            self._yolo_depth_ema = depth_val

            forward_error_m = depth_val
            lateral_error_m = (cx_px - self.cam_cx) / self.cam_fx * depth_val if self.cam_fx else 0.0

            state = {
                'servo_mode': 'base_servo' if conf > 0.3 else 'head_track_only',
                'level': 3,
                'yaw_error_deg': yaw_error_deg,
                'pitch_error_deg': pitch_error_deg,
                'forward_error_m': forward_error_m,
                'lateral_error_m': lateral_error_m,
                'confidence': max(conf, float(d435_det[4]) if d435_det is not None else 0.0),
                'inlier_count': box_count,
                'matched_count': box_count,
                'head_tracking_ok': head_ok,
                'base_servo_ready': conf > 0.3,
            }
            with self._yolo_lock:
                self._yolo_latest_result = state
        except Exception as exc:
            self.get_logger().warn(f'YOLO inference error: {exc}', throttle_duration_sec=2.0)
            with self._yolo_lock:
                self._yolo_latest_result = None

    def _tick_yolo(self, now: float) -> None:
        if self.yolo_model is None:
            return

        # Collect latest inference result (non-blocking)
        with self._yolo_lock:
            yolo_result = self._yolo_latest_result
            self._yolo_latest_result = None  # consume

        if yolo_result is not None:
            self.last_state = yolo_result
            self._publish_servo_state(self.last_state)

        # Start new inference if previous thread is done
        if self._yolo_thread is not None and self._yolo_thread.is_alive():
            return  # still running, skip
        if not self._has_fresh_frame():
            return
        if None in (self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy):
            return
        if (now - self.last_match_time) < (1.0 / self.match_hz):
            return
        self.last_match_time = now

        # Snapshot frame data for thread safety
        bgr_snap = self.latest_bgr.copy()
        depth_snap = self.latest_depth_m.copy()
        d435_snap = self.latest_d435_bgr.copy() if self.latest_d435_bgr is not None else None
        self._yolo_thread = threading.Thread(
            target=self._yolo_inference_worker,
            args=(bgr_snap, depth_snap, d435_snap),
            daemon=True,
        )
        self._yolo_thread.start()

    def _tick(self) -> None:
        now = time.time()

        # publish FSM state for action server bridge
        self._pub_fsm_state()

        # action_master_enabled gate: if disabled by behavior tree, force stop
        if not self.action_master_enabled:
            self._send_home_head()
            if abs(self.cmd_vx) > 0.005:
                self.cmd_vx *= 0.8
            else:
                self.cmd_vx = 0.0
            self.cmd_wz = 0.0
            self._publish_cmd(self.cmd_vx, self.cmd_wz)
            self._pub_follow_enabled(False)
            self.fsm_force_stop_flag = False
            return

        # clear force stop flag after processing
        self.fsm_force_stop_flag = False

        if self.follow_mode == 'person_yolo':
            self._process_gesture(now)

        follow_active = self._is_follow_active()
        if follow_active:
            self._pub_follow_enabled(True)

        # follow inactive: home the head, decelerate chassis.
        if not follow_active:
            self._send_home_head()
            # decelerate chassis
            if abs(self.cmd_vx) > 0.005:
                self.cmd_vx *= 0.8
            else:
                self.cmd_vx = 0.0
            self.cmd_wz = 0.0
            self._publish_cmd(self.cmd_vx, self.cmd_wz)
            self._pub_follow_enabled(False)
            return

        if self.input_source == 'realsense':
            self._poll_realsense()

        # hot-reload template
        tp = str(self.get_parameter('template_path').value).strip()
        if tp != self.template_path:
            self.template_path = tp
            self.estimator.reset()
            self._load_template()

        # --- run detection/matching at match_hz ---
        if self.follow_mode == 'person_yolo':
            self._tick_yolo(now)
        else:
            self._tick_scene(now)

        # YOLO mode: suppress state when follow not active (no gesture-1 / action start)
        if self.follow_mode == 'person_yolo' and not self._is_follow_active():
            state = None
        else:
            state = self.last_state
        if state is None:
            reason = ('no detection yet' if self.follow_mode == 'person_yolo'
                      else 'template unavailable' if not self.ref_data_list
                      else 'waiting for frame' if not self._has_fresh_frame()
                      else 'waiting for intrinsics' if None in (self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy)
                      else 'no match yet')
            self._handle_lost(now, reason)
            return

        # extract
        track_error_x = clamp(-float(state.get('yaw_error_deg', 0.0)) / self.head_fov_h_deg, -1.0, 1.0)
        track_error_y = clamp(-float(state.get('pitch_error_deg', 0.0)) / self.head_fov_v_deg, -1.0, 1.0)
        forward_error_m = float(state.get('forward_error_m', 0.0))
        lateral_error_m = float(state.get('lateral_error_m', 0.0))
        confidence = float(state.get('confidence', 0.0))
        head_ok = bool(state.get('head_tracking_ok', False))
        base_ok = bool(state.get('base_servo_ready', False))

        # Head tracking: YOLO uses D435-derived error; scene_template keeps the
        # head unchanged in the current first-stage scene-return flow.
        if self.follow_mode == 'person_yolo':
            self._update_head(track_error_x, track_error_y, head_ok)

        # chassis
        self._update_chassis(forward_error_m, lateral_error_m, base_ok, now)

        # log
        if (now - self.last_log_time) >= self.log_period_sec:
            self.last_log_time = now
            scene_base = ''
            if self.follow_mode == 'scene_template' and self.scene_base_forward_m is not None:
                scene_base = (
                    f' base=({self.scene_base_forward_m:+.2f},'
                    f'{self.scene_base_lateral_m if self.scene_base_lateral_m is not None else 0.0:+.2f},'
                    f'{self.scene_base_yaw_deg if self.scene_base_yaw_deg is not None else 0.0:+.1f})'
                )
            scene_pose = f' pose_ok={self.scene_pose_ok}' if self.follow_mode == 'scene_template' else ''
            scene_state_str = f' st={self.scene_state}' if self.follow_mode == 'scene_template' else ''
            # worldpilot data alignment: log snapshot stamps (scene mode only)
            stamp_str = ''
            if self.follow_mode == 'scene_template':
                img_ts = state.get('_snap_img_stamp_ns')
                dp_ts = state.get('_snap_depth_stamp_ns')
                if img_ts is not None:
                    stamp_str = f' img_ts={img_ts}'
                    if dp_ts is not None:
                        stamp_str = f' ts=({img_ts},{dp_ts})'
            self.get_logger().info(
                f'scene_servo: {self.template_scene_name or "?"} '
                f'mode={state.get("servo_mode")} lv={state.get("level")} '
                f'yaw={float(state.get("yaw_error_deg", 0)):+.1f} '
                f'fwd={forward_error_m:+.2f} conf={confidence:.2f} '
                f'in={int(state.get("inlier_count", 0))}/{int(state.get("matched_count", 0))} '
                f'cmd=({self.cmd_vx:+.3f},{self.cmd_wz:+.3f}){scene_base}{scene_pose}{scene_state_str}{stamp_str}'
            )

    def _handle_lost(self, now: float, reason: str) -> None:
        self._stop_head()
        # Don't reset estimator — coast mechanism handles deceleration,
        # and preserving EMA lets us recover instantly on re-match.
        self.tracking = False
        # Keep last_forward_m so coast can use it
        self.scene_base_forward_m = None
        self.scene_base_lateral_m = None
        self.scene_base_yaw_deg = None
        if self.follow_mode == 'scene_template':
            self.cmd_vx = 0.0
            self.cmd_wz = 0.0
            self.scene_state = 'LOST_STOP'
            self._publish_cmd(0.0, 0.0)
            if self.publish_lost_state:
                self._publish_servo_state({
                    'servo_mode': 'search', 'yaw_error_deg': 0.0, 'pitch_error_deg': 0.0,
                    'forward_error_m': 0.0, 'lateral_error_m': 0.0, 'confidence': 0.0,
                    'head_tracking_ok': False, 'base_servo_ready': False,
                })
            if (now - self.last_log_time) >= self.log_period_sec:
                self.last_log_time = now
                self.get_logger().info(f'scene_servo idle: {reason} cmd=({self.cmd_vx:+.3f},{self.cmd_wz:+.3f})')
            return
        # decelerate
        self.cmd_vx = clamp(self.cmd_vx, -abs(self.cmd_vx) + self.max_dvx_lost * (-1 if self.cmd_vx < 0 else 1),
                            abs(self.cmd_vx) - self.max_dvx_lost if self.cmd_vx > 0 else 0.0)
        self.cmd_wz = 0.0
        if abs(self.cmd_vx) < 0.005:
            self.cmd_vx = 0.0
        self._publish_cmd(self.cmd_vx, self.cmd_wz)
        if self.publish_lost_state:
            self._publish_servo_state({
                'servo_mode': 'search', 'yaw_error_deg': 0.0, 'pitch_error_deg': 0.0,
                'forward_error_m': 0.0, 'lateral_error_m': 0.0, 'confidence': 0.0,
                'head_tracking_ok': False, 'base_servo_ready': False,
            })
        if (now - self.last_log_time) >= self.log_period_sec:
            self.last_log_time = now
            self.get_logger().info(f'scene_servo idle: {reason} cmd=({self.cmd_vx:+.3f},{self.cmd_wz:+.3f})')


def main(args=None) -> None:
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init(args=args)

    node = SceneServoNode()

    shutdown_event = threading.Event()

    def _sig_handler(signum, frame):
        del signum, frame
        if not shutdown_event.is_set():
            shutdown_event.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    spin_thread = threading.Thread(target=_spin_thread, args=(node, shutdown_event), daemon=True)
    spin_thread.start()

    try:
        shutdown_event.wait()
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        spin_thread.join(timeout=0.5)
        try:
            node.graceful_shutdown()
        except Exception:
            pass
        try:
            if node.rs_pipeline is not None:
                node.rs_pipeline.stop()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def _spin_thread(node: SceneServoNode, shutdown_event: threading.Event) -> None:
    try:
        while not shutdown_event.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except Exception:
        import traceback
        try:
            node.get_logger().error('spin thread exception:\n' + traceback.format_exc())
        except Exception:
            pass
    finally:
        shutdown_event.set()


if __name__ == '__main__':
    main()
