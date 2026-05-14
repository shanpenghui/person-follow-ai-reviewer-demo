#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸跟随一体化节点 v5 — 检测+控制合一，插值丝滑

方案：
- 订阅相机图像，直接用 YOLO 检测人
- 检测出 bbox 后立刻计算目标角度
- 控制定时器以高频（30Hz+）运行，对目标角做线性插值
- 每个控制周期发送插值后的角度 → 电机连续平滑运动
- 不依赖 action server 反馈，直接用 Torso goal 控制

关键：
- 检测频率 ~10-15Hz（GPU 推理耗时）
- 控制频率 30Hz（插值填充，保证丝滑）
- 目标角 = 当前实测角 + bbox偏差角 × gain
- 但 gain 已经不需要了：直接算 "人脸到中心需要转多少度"
  检测结果是实时的，每次都用最新 bbox 算一个新目标角
  然后控制器对目标角做插值平滑到达
"""

import math
import signal
import time
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, Float32MultiArray, Int32
from cv_bridge import CvBridge
from ultralytics import YOLO
import tf2_ros
from tf2_geometry_msgs import do_transform_point

from interfaces.action import Torso


def clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * clamp(t, 0.0, 1.0)


class PersonFollowNode(Node):

    def __init__(self) -> None:
        super().__init__('person_follow_node')

        # ---- parameters ----
        self.declare_parameter('torso_action_name', '/Torso/torso_action_service')
        self.declare_parameter('joint_state_topic', '/Torso/joint_states')
        self.declare_parameter('image_topic', '/cam_head/d435/color/image_raw')
        self.declare_parameter('depth_topic', '/cam_head/d435/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/cam_head/d435/color/camera_info')

        # YOLO
        default_model_path = str(Path(get_package_share_directory('person_follow')) / 'models' / 'yolov8s.pt')
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('device', '0')
        self.declare_parameter('imgsz', 512)
        self.declare_parameter('conf_thres', 0.16)
        self.declare_parameter('detect_max_det', 8)
        self.declare_parameter('use_half', True)
        # YOLO target class (COCO id): person=0, refrigerator=72
        self.declare_parameter('detect_class_id', 72)

        self.declare_parameter('image_width', 640.0)
        self.declare_parameter('image_height', 480.0)

        # D435 color FOV
        self.declare_parameter('fov_h_deg', 69.0)
        self.declare_parameter('fov_v_deg', 42.0)

        # 控制
        self.declare_parameter('control_hz', 12.0)    # 控制频率（插值）
        self.declare_parameter('deadband_deg', 1.0)    # 角度死区（度）
        self.declare_parameter('smoothing', 0.35)       # 插值平滑系数 (0~1, 越小越平滑)

        # bbox跟随点
        self.declare_parameter('bbox_target_x_ratio', 0.5)
        self.declare_parameter('bbox_target_y_ratio', 0.30)

        # 速度自适应
        self.declare_parameter('vel_min', 0.12)
        self.declare_parameter('vel_max', 0.50)
        self.declare_parameter('vel_ramp_deg', 20.0)

        # 方向
        self.declare_parameter('yaw_sign', 1.0)
        self.declare_parameter('pitch_sign', 1.0)

        # 丢目标
        self.declare_parameter('lost_timeout_sec', 1.0)
        self.declare_parameter('lost_behavior', 'hold')

        # 多目标下“跟住原目标”策略（轻量级，不依赖ReID）
        self.declare_parameter('enable_target_lock', True)
        self.declare_parameter('target_lock_timeout_sec', 1.2)
        self.declare_parameter('target_lock_iou_weight', 0.55)
        self.declare_parameter('target_lock_center_weight', 0.30)
        self.declare_parameter('target_lock_area_weight', 0.15)
        self.declare_parameter('target_lock_min_score', 0.30)
        self.declare_parameter('target_lock_hold_on_low_score', True)
        self.declare_parameter('target_lock_center_gate', 0.35)
        self.declare_parameter('target_lock_area_ratio_min', 0.45)
        self.declare_parameter('target_lock_area_ratio_max', 2.20)
        self.declare_parameter('target_lock_switch_iou_min', 0.20)
        self.declare_parameter('target_lock_switch_margin', 0.08)
        self.declare_parameter('target_lock_switch_margin_crowded', 0.16)
        self.declare_parameter('target_lock_dynamic_gate_enable', True)
        self.declare_parameter('target_lock_max_det_relax', 6)
        self.declare_parameter('target_lock_min_score_relaxed', 0.12)
        self.declare_parameter('target_lock_center_gate_relaxed', 0.55)
        self.declare_parameter('target_lock_switch_confirm_enable', True)
        self.declare_parameter('target_lock_switch_confirm_frames_d435', 3)
        self.declare_parameter('target_lock_switch_confirm_frames_d455', 2)
        self.declare_parameter('target_lock_switch_confirm_iou', 0.50)
        self.declare_parameter('target_lock_anchor_keep_iou_min', 0.08)

        # target interface for chassis follow (D435 main)
        self.declare_parameter('publish_target_interface', True)
        self.declare_parameter('target_topic', '/person_follow/target')
        self.declare_parameter('target_valid_topic', '/person_follow/target_valid')
        self.declare_parameter('d435_bbox_topic', '/person_follow/d435_bbox')
        self.declare_parameter('enable_bbox_meta_publish', True)
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('camera_optical_frame', 'camera_head_optical')
        self.declare_parameter('depth_roi_half', 7)
        self.declare_parameter('depth_min_m', 0.2)
        self.declare_parameter('depth_max_m', 5.0)
        self.declare_parameter('depth_min_valid_count', 3)
        self.declare_parameter('depth_hold_sec', 0.4)

        # D455 target channel (only publish target, no head control)
        self.declare_parameter('enable_d455_target_interface', True)
        self.declare_parameter('d455_image_topic', '/cam_chest/d455/color/image_raw')
        self.declare_parameter('d455_depth_topic', '/cam_chest/d455/aligned_depth_to_color/image_raw')
        self.declare_parameter('d455_camera_info_topic', '/cam_chest/d455/color/camera_info')
        self.declare_parameter('d455_target_topic', '/person_follow/target_d455')
        self.declare_parameter('d455_target_valid_topic', '/person_follow/target_valid_d455')
        self.declare_parameter('d455_bbox_topic', '/person_follow/d455_bbox')
        self.declare_parameter('d455_camera_optical_frame', 'camera_torso_link')
        self.declare_parameter('d455_detect_hz', 4.0)
        self.declare_parameter('d455_depth_roi_half', 11)
        self.declare_parameter('d455_depth_hold_sec', 0.4)

        # D455 adaptive depth strategy (Near/Mid/Far)
        self.declare_parameter('d455_depth_strategy_mode', 'adaptive')  # adaptive | legacy
        self.declare_parameter('d455_depth_roi_shrink_x', 0.20)
        self.declare_parameter('d455_depth_roi_shrink_top', 0.25)
        self.declare_parameter('d455_depth_roi_shrink_bottom', 0.25)
        self.declare_parameter('d455_depth_valid_ratio_min', 0.30)
        self.declare_parameter('d455_depth_valid_count_min', 80)
        self.declare_parameter('d455_depth_ref_quantile', 0.50)
        self.declare_parameter('d455_depth_near_quantile', 0.45)
        self.declare_parameter('d455_depth_mid_quantile', 0.35)
        self.declare_parameter('d455_depth_far_quantile', 0.30)
        self.declare_parameter('d455_depth_near_enter_m', 1.00)
        self.declare_parameter('d455_depth_near_exit_m', 1.15)
        self.declare_parameter('d455_depth_far_enter_m', 2.80)
        self.declare_parameter('d455_depth_far_exit_m', 2.50)
        self.declare_parameter('d455_depth_enter_frames', 3)
        self.declare_parameter('d455_depth_exit_frames', 3)
        self.declare_parameter('d455_depth_zone_hold_sec', 0.5)
        self.declare_parameter('d455_depth_dist_ema_alpha', 0.25)
        self.declare_parameter('d455_depth_output_ema_alpha', 0.35)
        self.declare_parameter('d455_depth_max_jump_m', 0.35)

        self.declare_parameter('fallback_without_tf', True)

        # 限位
        self.declare_parameter('yaw_min_deg', -90.0)
        self.declare_parameter('yaw_max_deg', 90.0)
        self.declare_parameter('pitch_min_deg', -34.0)
        self.declare_parameter('pitch_max_deg', 19.0)

        # home
        self.declare_parameter('home_before_start', True)
        self.declare_parameter('home_torso_height', 0.149)
        self.declare_parameter('home_torso_yaw', 0.0)
        self.declare_parameter('home_head_pitch', 0.0)
        self.declare_parameter('home_head_yaw', 0.0)
        self.declare_parameter('home_max_velocity', 0.15)
        self.declare_parameter('home_settle_sec', 1.5)

        # shutdown safety
        self.declare_parameter('shutdown_home_on_exit', True)
        self.declare_parameter('shutdown_home_settle_sec', 1.0)
        self.declare_parameter('shutdown_stop_repeat', 2)

        self.declare_parameter('log_period_sec', 1.0)
        self.declare_parameter('camera_alive_timeout_sec', 1.0)

        # external FSM control
        self.declare_parameter('follow_enabled_topic', '/person_follow/follow_enabled')
        self.declare_parameter('start_follow_enabled', False)
        self.declare_parameter('target_class_id_topic', '/person_follow/target_class_id')

        # ---- get params ----
        self.torso_action_name = self.get_parameter('torso_action_name').value
        self.joint_state_topic = self.get_parameter('joint_state_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value

        self.model_path = self.get_parameter('model_path').value
        self.device = self.get_parameter('device').value
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf_thres = float(self.get_parameter('conf_thres').value)
        self.use_half = bool(self.get_parameter('use_half').value)
        self.detect_class_id = int(self.get_parameter('detect_class_id').value)

        self.img_w = float(self.get_parameter('image_width').value)
        self.img_h = float(self.get_parameter('image_height').value)

        self.fov_h = float(self.get_parameter('fov_h_deg').value)
        self.fov_v = float(self.get_parameter('fov_v_deg').value)

        self.control_hz = float(self.get_parameter('control_hz').value)
        self.deadband_deg = float(self.get_parameter('deadband_deg').value)
        self.smoothing = clamp(float(self.get_parameter('smoothing').value), 0.01, 1.0)

        self.bbox_target_x_ratio = float(self.get_parameter('bbox_target_x_ratio').value)
        self.bbox_target_y_ratio = float(self.get_parameter('bbox_target_y_ratio').value)

        self.vel_min = float(self.get_parameter('vel_min').value)
        self.vel_max = float(self.get_parameter('vel_max').value)
        self.vel_ramp_deg = float(self.get_parameter('vel_ramp_deg').value)

        self.yaw_sign = float(self.get_parameter('yaw_sign').value)
        self.pitch_sign = float(self.get_parameter('pitch_sign').value)

        self.lost_timeout = float(self.get_parameter('lost_timeout_sec').value)
        self.lost_behavior = str(self.get_parameter('lost_behavior').value)

        self.enable_target_lock = bool(self.get_parameter('enable_target_lock').value)
        self.target_lock_timeout_sec = float(self.get_parameter('target_lock_timeout_sec').value)
        self.target_lock_iou_weight = float(self.get_parameter('target_lock_iou_weight').value)
        self.target_lock_center_weight = float(self.get_parameter('target_lock_center_weight').value)
        self.target_lock_area_weight = float(self.get_parameter('target_lock_area_weight').value)
        self.target_lock_min_score = float(self.get_parameter('target_lock_min_score').value)
        self.target_lock_hold_on_low_score = bool(self.get_parameter('target_lock_hold_on_low_score').value)
        self.target_lock_center_gate = float(self.get_parameter('target_lock_center_gate').value)
        self.target_lock_area_ratio_min = float(self.get_parameter('target_lock_area_ratio_min').value)
        self.target_lock_area_ratio_max = float(self.get_parameter('target_lock_area_ratio_max').value)
        self.target_lock_switch_iou_min = float(self.get_parameter('target_lock_switch_iou_min').value)
        self.target_lock_switch_margin = float(self.get_parameter('target_lock_switch_margin').value)
        self.target_lock_switch_margin_crowded = float(self.get_parameter('target_lock_switch_margin_crowded').value)
        self.target_lock_dynamic_gate_enable = bool(self.get_parameter('target_lock_dynamic_gate_enable').value)
        self.target_lock_max_det_relax = int(self.get_parameter('target_lock_max_det_relax').value)
        self.target_lock_min_score_relaxed = float(self.get_parameter('target_lock_min_score_relaxed').value)
        self.target_lock_center_gate_relaxed = float(self.get_parameter('target_lock_center_gate_relaxed').value)
        self.target_lock_switch_confirm_enable = bool(self.get_parameter('target_lock_switch_confirm_enable').value)
        self.target_lock_switch_confirm_frames_d435 = int(self.get_parameter('target_lock_switch_confirm_frames_d435').value)
        self.target_lock_switch_confirm_frames_d455 = int(self.get_parameter('target_lock_switch_confirm_frames_d455').value)
        self.target_lock_switch_confirm_iou = float(self.get_parameter('target_lock_switch_confirm_iou').value)
        self.target_lock_anchor_keep_iou_min = float(self.get_parameter('target_lock_anchor_keep_iou_min').value)

        self.detect_max_det = max(1, int(self.get_parameter('detect_max_det').value))

        self.publish_target_interface = bool(self.get_parameter('publish_target_interface').value)
        self.target_topic = str(self.get_parameter('target_topic').value)
        self.target_valid_topic = str(self.get_parameter('target_valid_topic').value)
        self.d435_bbox_topic = str(self.get_parameter('d435_bbox_topic').value)
        self.enable_bbox_meta_publish = bool(self.get_parameter('enable_bbox_meta_publish').value)
        self.target_frame = str(self.get_parameter('target_frame').value)
        self.camera_optical_frame = str(self.get_parameter('camera_optical_frame').value)
        self.depth_roi_half = int(self.get_parameter('depth_roi_half').value)
        self.depth_min_m = float(self.get_parameter('depth_min_m').value)
        self.depth_max_m = float(self.get_parameter('depth_max_m').value)
        self.depth_min_valid_count = int(self.get_parameter('depth_min_valid_count').value)
        self.depth_hold_sec = float(self.get_parameter('depth_hold_sec').value)

        self.enable_d455_target_interface = bool(self.get_parameter('enable_d455_target_interface').value)
        self.d455_image_topic = str(self.get_parameter('d455_image_topic').value)
        self.d455_depth_topic = str(self.get_parameter('d455_depth_topic').value)
        self.d455_camera_info_topic = str(self.get_parameter('d455_camera_info_topic').value)
        self.d455_target_topic = str(self.get_parameter('d455_target_topic').value)
        self.d455_target_valid_topic = str(self.get_parameter('d455_target_valid_topic').value)
        self.d455_bbox_topic = str(self.get_parameter('d455_bbox_topic').value)
        self.d455_camera_optical_frame = str(self.get_parameter('d455_camera_optical_frame').value)
        self.d455_detect_hz = float(self.get_parameter('d455_detect_hz').value)
        self.d455_depth_roi_half = int(self.get_parameter('d455_depth_roi_half').value)
        self.d455_depth_hold_sec = float(self.get_parameter('d455_depth_hold_sec').value)

        self.d455_depth_strategy_mode = str(self.get_parameter('d455_depth_strategy_mode').value).strip().lower()
        self.d455_depth_roi_shrink_x = float(self.get_parameter('d455_depth_roi_shrink_x').value)
        self.d455_depth_roi_shrink_top = float(self.get_parameter('d455_depth_roi_shrink_top').value)
        self.d455_depth_roi_shrink_bottom = float(self.get_parameter('d455_depth_roi_shrink_bottom').value)
        self.d455_depth_valid_ratio_min = float(self.get_parameter('d455_depth_valid_ratio_min').value)
        self.d455_depth_valid_count_min = int(self.get_parameter('d455_depth_valid_count_min').value)
        self.d455_depth_ref_quantile = float(self.get_parameter('d455_depth_ref_quantile').value)
        self.d455_depth_near_quantile = float(self.get_parameter('d455_depth_near_quantile').value)
        self.d455_depth_mid_quantile = float(self.get_parameter('d455_depth_mid_quantile').value)
        self.d455_depth_far_quantile = float(self.get_parameter('d455_depth_far_quantile').value)
        self.d455_depth_near_enter_m = float(self.get_parameter('d455_depth_near_enter_m').value)
        self.d455_depth_near_exit_m = float(self.get_parameter('d455_depth_near_exit_m').value)
        self.d455_depth_far_enter_m = float(self.get_parameter('d455_depth_far_enter_m').value)
        self.d455_depth_far_exit_m = float(self.get_parameter('d455_depth_far_exit_m').value)
        self.d455_depth_enter_frames = int(self.get_parameter('d455_depth_enter_frames').value)
        self.d455_depth_exit_frames = int(self.get_parameter('d455_depth_exit_frames').value)
        self.d455_depth_zone_hold_sec = float(self.get_parameter('d455_depth_zone_hold_sec').value)
        self.d455_depth_dist_ema_alpha = float(self.get_parameter('d455_depth_dist_ema_alpha').value)
        self.d455_depth_output_ema_alpha = float(self.get_parameter('d455_depth_output_ema_alpha').value)
        self.d455_depth_max_jump_m = float(self.get_parameter('d455_depth_max_jump_m').value)

        self.fallback_without_tf = bool(self.get_parameter('fallback_without_tf').value)

        self.yaw_min = float(self.get_parameter('yaw_min_deg').value)
        self.yaw_max = float(self.get_parameter('yaw_max_deg').value)
        self.pitch_min = float(self.get_parameter('pitch_min_deg').value)
        self.pitch_max = float(self.get_parameter('pitch_max_deg').value)

        self.home_before_start = bool(self.get_parameter('home_before_start').value)
        self.home_torso_height = float(self.get_parameter('home_torso_height').value)
        self.home_torso_yaw = float(self.get_parameter('home_torso_yaw').value)
        self.home_head_pitch = float(self.get_parameter('home_head_pitch').value)
        self.home_head_yaw = float(self.get_parameter('home_head_yaw').value)
        self.home_max_velocity = float(self.get_parameter('home_max_velocity').value)
        self.home_settle_sec = float(self.get_parameter('home_settle_sec').value)

        self.shutdown_home_on_exit = bool(self.get_parameter('shutdown_home_on_exit').value)
        self.shutdown_home_settle_sec = float(self.get_parameter('shutdown_home_settle_sec').value)
        self.shutdown_stop_repeat = int(self.get_parameter('shutdown_stop_repeat').value)

        self.log_period_sec = float(self.get_parameter('log_period_sec').value)
        self.camera_alive_timeout_sec = float(self.get_parameter('camera_alive_timeout_sec').value)

        self.follow_enabled_topic = str(self.get_parameter('follow_enabled_topic').value)
        self.follow_enabled = bool(self.get_parameter('start_follow_enabled').value)
        self.target_class_id_topic = str(self.get_parameter('target_class_id_topic').value)

        # ---- state ----
        self.lock = threading.Lock()

        # 实测关节角
        self.head_yaw_deg: Optional[float] = None
        self.head_pitch_deg: Optional[float] = None

        # 检测线程输出的"原始目标角"
        self.raw_target_yaw: Optional[float] = None
        self.raw_target_pitch: Optional[float] = None
        self.last_det_time = 0.0

        # 控制器当前发送的平滑目标角
        self.smooth_yaw: Optional[float] = None
        self.smooth_pitch: Optional[float] = None

        self.home_sent = False
        self.home_deadline = 0.0
        self.started = False

        self.last_log_time = 0.0
        self.det_count = 0
        self.infer_busy = False
        self.last_d455_infer_time = 0.0

        # 多目标目标锁（按 bbox 时序连续性，尽量跟住“原来那个人”）
        self.locked_bbox_d435: Optional[np.ndarray] = None
        self.locked_bbox_d435_time: float = 0.0
        self.locked_bbox_d455: Optional[np.ndarray] = None
        self.locked_bbox_d455_time: float = 0.0
        self.pending_switch_bbox_d435: Optional[np.ndarray] = None
        self.pending_switch_count_d435: int = 0
        self.pending_switch_bbox_d455: Optional[np.ndarray] = None
        self.pending_switch_count_d455: int = 0

        # depth / intrinsics cache (for target interface)
        self.depth_img: Optional[np.ndarray] = None
        self.depth_encoding: str = ''
        self.depth_frame_id: str = ''
        self.cam_fx: Optional[float] = None
        self.cam_fy: Optional[float] = None
        self.cam_cx: Optional[float] = None
        self.cam_cy: Optional[float] = None
        self.last_target_valid: bool = False
        self.last_target_valid_pub_time: float = 0.0
        self.last_tf_warn_time: float = 0.0
        self.last_good_depth_m: Optional[float] = None
        self.last_good_depth_time: float = 0.0

        # D455 channel cache
        self.d455_depth_img: Optional[np.ndarray] = None
        self.d455_depth_encoding: str = ''
        self.d455_depth_frame_id: str = ''
        self.d455_cam_fx: Optional[float] = None
        self.d455_cam_fy: Optional[float] = None
        self.d455_cam_cx: Optional[float] = None
        self.d455_cam_cy: Optional[float] = None
        self.d455_last_target_valid: bool = False
        self.d455_last_target_valid_pub_time: float = 0.0
        self.d455_last_good_depth_m: Optional[float] = None
        self.d455_last_good_depth_time: float = 0.0
        self.last_d455_det_time: float = 0.0

        # D455 adaptive depth state
        self.d455_depth_zone: str = 'MID'
        self.d455_depth_zone_last_switch_time: float = 0.0
        self.d455_depth_near_enter_count: int = 0
        self.d455_depth_near_exit_count: int = 0
        self.d455_depth_far_enter_count: int = 0
        self.d455_depth_far_exit_count: int = 0
        self.d455_depth_ref_ema: Optional[float] = None
        self.d455_depth_output_ema: Optional[float] = None
        self.d455_last_valid_ratio: float = 0.0
        self.d455_last_valid_count: int = 0

        # diagnostics counters (per log window)
        self.diag_det_ok: int = 0
        self.diag_depth_fail: int = 0
        self.diag_tf_fail: int = 0
        self.diag_publish_ok: int = 0

        self.diag_d455_det_ok: int = 0
        self.diag_d455_depth_fail: int = 0
        self.diag_d455_tf_fail: int = 0
        self.diag_d455_publish_ok: int = 0
        self.diag_lock_d435_hold: int = 0
        self.diag_lock_d435_switch: int = 0
        self.diag_lock_d455_hold: int = 0
        self.diag_lock_d455_switch: int = 0
        self.last_diag_log_time: float = time.time()

        self.last_follow_stop_cmd_time: float = 0.0

        # ---- YOLO ----
        self.get_logger().info(f'Loading YOLO: {self.model_path} device={self.device}')
        self.model = YOLO(self.model_path)
        # warmup
        try:
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model.predict(source=dummy, verbose=False, imgsz=320, conf=0.1,
                              classes=[self.detect_class_id], device=self.device if self.device else None)
        except Exception:
            pass
        self.get_logger().info('YOLO loaded.')

        self.bridge = CvBridge()

        # ---- ROS ----
        self.torso_client = ActionClient(self, Torso, self.torso_action_name)

        # TF for camera->base transform (target interface)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, self.image_topic, self._image_cb, qos)
        self.create_subscription(Image, self.depth_topic, self._depth_cb, qos)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, qos)

        if self.enable_d455_target_interface:
            self.create_subscription(Image, self.d455_image_topic, self._d455_image_cb, qos)
            self.create_subscription(Image, self.d455_depth_topic, self._d455_depth_cb, qos)
            self.create_subscription(CameraInfo, self.d455_camera_info_topic, self._d455_camera_info_cb, qos)

        self.create_subscription(JointState, self.joint_state_topic, self._joint_cb, 10)
        self.create_subscription(Bool, self.follow_enabled_topic, self._follow_enabled_cb, 10)
        self.create_subscription(Int32, self.target_class_id_topic, self._target_class_cb, 10)

        self.target_pub = None
        self.target_valid_pub = None
        self.d435_bbox_pub = None
        self.d455_target_pub = None
        self.d455_target_valid_pub = None
        self.d455_bbox_pub = None

        if self.publish_target_interface:
            self.target_pub = self.create_publisher(PointStamped, self.target_topic, 10)
            self.target_valid_pub = self.create_publisher(Bool, self.target_valid_topic, 10)

        if self.enable_bbox_meta_publish:
            self.d435_bbox_pub = self.create_publisher(Float32MultiArray, self.d435_bbox_topic, 10)

        if self.enable_d455_target_interface:
            self.d455_target_pub = self.create_publisher(PointStamped, self.d455_target_topic, 10)
            self.d455_target_valid_pub = self.create_publisher(Bool, self.d455_target_valid_topic, 10)
            if self.enable_bbox_meta_publish:
                self.d455_bbox_pub = self.create_publisher(Float32MultiArray, self.d455_bbox_topic, 10)

        # 控制定时器
        self.ctrl_timer = self.create_timer(1.0 / max(1.0, self.control_hz), self._ctrl_tick)
        # 启动定时器
        self.start_timer = self.create_timer(0.5, self._startup_tick)

        self.get_logger().info(
            f'person_follow_node v5 started. ctrl={self.control_hz}Hz detect_class_id={self.detect_class_id} '
            f'follow_enabled={self.follow_enabled} target_topic={self.target_class_id_topic}')

    # ---- startup ----
    def _startup_tick(self) -> None:
        if self.started:
            return
        if not self.torso_client.server_is_ready():
            self.get_logger().warn('waiting torso action server...')
            return
        if self.home_before_start and not self.home_sent:
            g = Torso.Goal()
            g.torso_height = float(self.home_torso_height)
            g.torso_yaw = float(self.home_torso_yaw)
            g.head_yaw = float(self.home_head_yaw)
            g.head_pitch = float(self.home_head_pitch)
            g.max_velocity = float(self.home_max_velocity)
            g.work_mode = 0
            self.torso_client.send_goal_async(g)
            self.home_sent = True
            self.home_deadline = time.time() + self.home_settle_sec
            self.get_logger().info(f'home: yaw={g.head_yaw} pitch={g.head_pitch} vel={g.max_velocity}')
            return
        if time.time() < self.home_deadline:
            return
        self.started = True
        self.get_logger().info('startup complete, tracking active.')

    def _send_velocity_stop(self, wait_sec: float = 0.0) -> None:
        """Send velocity=0 stop command. Best-effort, swallows all exceptions."""
        try:
            g = Torso.Goal()
            g.torso_height = float(self.home_torso_height)
            g.torso_yaw = float(self.home_torso_yaw)
            g.head_yaw = 0.0
            g.head_pitch = 0.0
            g.max_velocity = 0.5
            g.work_mode = 1  # VELOCITY_MODE, vel=0 -> stop
            self.torso_client.send_goal_async(g)
        except Exception:
            pass

    def _send_home_position(self, wait_sec: float = 0.0) -> None:
        """Send position-mode home command. Best-effort, swallows all exceptions."""
        try:
            g = Torso.Goal()
            g.torso_height = float(self.home_torso_height)
            g.torso_yaw = float(self.home_torso_yaw)
            g.head_yaw = float(self.home_head_yaw)
            g.head_pitch = float(self.home_head_pitch)
            g.max_velocity = float(self.home_max_velocity)
            g.work_mode = 0  # POSITION_MODE
            self.torso_client.send_goal_async(g)
        except Exception:
            pass

    def graceful_shutdown(self) -> None:
        """Stop motor velocity first, then position-home before process exits.

        Must be called while rclpy context is still alive (before rclpy.shutdown).
        """
        if not self.shutdown_home_on_exit:
            self.get_logger().info('shutdown_home_on_exit=false, skip shutdown homing.')
            return

        self.get_logger().info('shutdown safeguard: sending velocity stop + position home...')

        # Step 1: 多次发送速度清零，确保底层收到
        repeat = max(1, int(self.shutdown_stop_repeat))
        for i in range(repeat):
            self._send_velocity_stop()
            time.sleep(0.05)

        # Step 2: 发送位置模式回中
        self._send_home_position()

        # Step 3: 等待底层执行（这里不需要 spin，只是给底层时间）
        settle = max(0.3, float(self.shutdown_home_settle_sec))
        self.get_logger().info(f'shutdown: waiting {settle:.1f}s for home to complete...')
        time.sleep(settle)

        # Step 4: 最后再发一次速度清零保险
        self._send_velocity_stop()
        time.sleep(0.05)

        self.get_logger().info('shutdown safeguard complete.')

    # ---- sensor callbacks ----
    def _depth_cb(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_img = depth
            self.depth_encoding = msg.encoding
            self.depth_frame_id = msg.header.frame_id
        except Exception:
            pass

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self.cam_fx is not None:
            return
        if len(msg.k) >= 9:
            self.cam_fx = float(msg.k[0])
            self.cam_fy = float(msg.k[4])
            self.cam_cx = float(msg.k[2])
            self.cam_cy = float(msg.k[5])
            self.get_logger().info(
                f'camera intrinsics cached: fx={self.cam_fx:.2f} fy={self.cam_fy:.2f} '
                f'cx={self.cam_cx:.2f} cy={self.cam_cy:.2f}'
            )

    def _d455_depth_cb(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.d455_depth_img = depth
            self.d455_depth_encoding = msg.encoding
            self.d455_depth_frame_id = msg.header.frame_id
        except Exception:
            pass

    def _d455_camera_info_cb(self, msg: CameraInfo) -> None:
        if self.d455_cam_fx is not None:
            return
        if len(msg.k) >= 9:
            self.d455_cam_fx = float(msg.k[0])
            self.d455_cam_fy = float(msg.k[4])
            self.d455_cam_cx = float(msg.k[2])
            self.d455_cam_cy = float(msg.k[5])
            self.get_logger().info(
                f'd455 intrinsics cached: fx={self.d455_cam_fx:.2f} fy={self.d455_cam_fy:.2f} '
                f'cx={self.d455_cam_cx:.2f} cy={self.d455_cam_cy:.2f}'
            )

    def _publish_target_valid(self, valid: bool) -> None:
        if not self.publish_target_interface or self.target_valid_pub is None:
            return
        # 节流：状态变化时立刻发；不变化时低频发
        now = time.time()
        if valid != self.last_target_valid or (now - self.last_target_valid_pub_time) > self.log_period_sec:
            b = Bool()
            b.data = bool(valid)
            self.target_valid_pub.publish(b)
            self.last_target_valid = valid
            self.last_target_valid_pub_time = now

    def _publish_target(self, stamp, distance_m: float, bearing_rad: float, confidence: float = 1.0) -> None:
        if not self.publish_target_interface or self.target_pub is None:
            return
        m = PointStamped()
        m.header.stamp = stamp
        m.header.frame_id = self.target_frame
        m.point.x = float(distance_m)
        m.point.y = float(bearing_rad)
        m.point.z = float(clamp(confidence, 0.0, 1.0))
        self.target_pub.publish(m)

    def _publish_d455_target_valid(self, valid: bool) -> None:
        if not self.enable_d455_target_interface or self.d455_target_valid_pub is None:
            return
        now = time.time()
        if valid != self.d455_last_target_valid or (now - self.d455_last_target_valid_pub_time) > self.log_period_sec:
            b = Bool()
            b.data = bool(valid)
            self.d455_target_valid_pub.publish(b)
            self.d455_last_target_valid = valid
            self.d455_last_target_valid_pub_time = now

    def _publish_d455_target(self, stamp, distance_m: float, bearing_rad: float, confidence: float = 1.0) -> None:
        if not self.enable_d455_target_interface or self.d455_target_pub is None:
            return
        m = PointStamped()
        m.header.stamp = stamp
        m.header.frame_id = self.target_frame
        m.point.x = float(distance_m)
        m.point.y = float(bearing_rad)
        m.point.z = float(clamp(confidence, 0.0, 1.0))
        self.d455_target_pub.publish(m)

    def _publish_d435_bbox(self, stamp, x1: float, y1: float, x2: float, y2: float,
                           conf: float, img_w: float, img_h: float) -> None:
        if (not self.enable_bbox_meta_publish) or (self.d435_bbox_pub is None):
            return
        m = Float32MultiArray()
        # [x1,y1,x2,y2,conf,img_w,img_h,stamp_sec]
        sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9 if hasattr(stamp, 'sec') else time.time()
        m.data = [
            float(x1), float(y1), float(x2), float(y2),
            float(clamp(conf, 0.0, 1.0)), float(img_w), float(img_h), float(sec)
        ]
        self.d435_bbox_pub.publish(m)

    def _publish_d455_bbox(self, stamp, x1: float, y1: float, x2: float, y2: float,
                           conf: float, img_w: float, img_h: float) -> None:
        if (not self.enable_bbox_meta_publish) or (self.d455_bbox_pub is None):
            return
        m = Float32MultiArray()
        sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9 if hasattr(stamp, 'sec') else time.time()
        m.data = [
            float(x1), float(y1), float(x2), float(y2),
            float(clamp(conf, 0.0, 1.0)), float(img_w), float(img_h), float(sec)
        ]
        self.d455_bbox_pub.publish(m)

    # ---- multi-target lock helpers ----
    @staticmethod
    def _bbox_area(b: np.ndarray) -> float:
        return max(1.0, float((b[2] - b[0]) * (b[3] - b[1])))

    def _bbox_iou(self, a: np.ndarray, b: np.ndarray) -> float:
        xx1 = max(float(a[0]), float(b[0]))
        yy1 = max(float(a[1]), float(b[1]))
        xx2 = min(float(a[2]), float(b[2]))
        yy2 = min(float(a[3]), float(b[3]))
        iw = max(0.0, xx2 - xx1)
        ih = max(0.0, yy2 - yy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        ua = self._bbox_area(a) + self._bbox_area(b) - inter
        return float(inter / max(1e-6, ua))

    @staticmethod
    def _bbox_center_norm_xy(b: np.ndarray, w: float, h: float) -> Tuple[float, float]:
        cx = 0.5 * float(b[0] + b[2])
        cy = 0.5 * float(b[1] + b[3])
        return cx / max(1.0, w), cy / max(1.0, h)

    def _select_best_detection(
        self,
        xyxy: np.ndarray,
        w_img: float,
        h_img: float,
        locked_bbox: Optional[np.ndarray],
        locked_time: float,
        now: float,
        pending_bbox: Optional[np.ndarray],
        pending_count: int,
        confirm_frames: int,
    ) -> Tuple[Optional[int], Optional[np.ndarray], int]:
        # fallback: 没有锁定信息时，选最大框（通常更近更稳定）
        if (not self.enable_target_lock) or locked_bbox is None or (now - locked_time) > self.target_lock_timeout_sec:
            best_i = 0
            best_area = 0.0
            for i in range(xyxy.shape[0]):
                area = self._bbox_area(xyxy[i])
                if area > best_area:
                    best_area = area
                    best_i = i
            return best_i, None, 0

        lock = locked_bbox.astype(np.float32)
        lock_cx, lock_cy = self._bbox_center_norm_xy(lock, w_img, h_img)
        lock_area = self._bbox_area(lock)

        crowded = self.target_lock_dynamic_gate_enable and (xyxy.shape[0] >= self.target_lock_max_det_relax)
        gate = self.target_lock_center_gate_relaxed if crowded else self.target_lock_center_gate
        min_score = self.target_lock_min_score_relaxed if crowded else self.target_lock_min_score
        switch_margin = self.target_lock_switch_margin_crowded if crowded else self.target_lock_switch_margin

        gate = max(1e-3, gate)

        scores = np.zeros((xyxy.shape[0],), dtype=np.float32)
        ious = np.zeros((xyxy.shape[0],), dtype=np.float32)

        for i in range(xyxy.shape[0]):
            b = xyxy[i].astype(np.float32)
            iou = self._bbox_iou(lock, b)
            ious[i] = float(iou)

            cx, cy = self._bbox_center_norm_xy(b, w_img, h_img)
            center_dist = math.hypot(cx - lock_cx, cy - lock_cy)
            center_score = clamp(1.0 - center_dist / gate, 0.0, 1.0)

            area = self._bbox_area(b)
            area_ratio = area / max(1.0, lock_area)
            if self.target_lock_area_ratio_min <= area_ratio <= self.target_lock_area_ratio_max:
                area_score = clamp(1.0 - abs(math.log(max(1e-6, area_ratio))) / math.log(2.5), 0.0, 1.0)
            else:
                area_score = 0.0

            score = (
                self.target_lock_iou_weight * iou
                + self.target_lock_center_weight * center_score
                + self.target_lock_area_weight * area_score
            )

            # 软惩罚：中心偏差过大时大幅降分，防止瞬间切到另一人
            if center_dist > gate:
                score *= 0.35

            scores[i] = float(score)

        best_i = int(np.argmax(scores))
        best_score = float(scores[best_i])

        # 锚点候选：与锁定框 IoU 最大（最可能是“原目标”）
        anchor_i = int(np.argmax(ious))
        anchor_score = float(scores[anchor_i])

        # 拥挤/交叉场景防跳人：
        # 若 best 与 anchor 不一致，并且 best 与锁定框重叠很小，
        # 则必须显著优于 anchor 才允许切换。
        need_confirm = False
        if best_i != anchor_i:
            best_iou = float(ious[best_i])
            if best_iou < self.target_lock_switch_iou_min:
                if best_score < (anchor_score + switch_margin):
                    best_i = anchor_i
                    best_score = anchor_score
                else:
                    need_confirm = True

        # 切换确认（自动，无需交互）：候选切换需连续 N 帧稳定出现
        if self.target_lock_switch_confirm_enable and need_confirm and confirm_frames > 1:
            cand_bbox = xyxy[best_i].astype(np.float32)
            if pending_bbox is not None and self._bbox_iou(pending_bbox, cand_bbox) >= self.target_lock_switch_confirm_iou:
                pending_count += 1
            else:
                pending_bbox = cand_bbox
                pending_count = 1

            if pending_count < confirm_frames:
                # 锚点仍有一定重叠时，坚持原目标；否则输出 invalid，避免乱切
                if float(ious[anchor_i]) >= self.target_lock_anchor_keep_iou_min:
                    best_i = anchor_i
                    best_score = anchor_score
                else:
                    return None, pending_bbox, pending_count
            else:
                # 确认成功，允许切换
                pending_bbox = None
                pending_count = 0
        else:
            pending_bbox = None
            pending_count = 0

        # 若候选都不可信：
        # - hold_on_low_score=True 时：短时保持当前锁（返回 None，交由上层发布 invalid）
        # - 否则回退最大框
        if best_score < min_score:
            if self.target_lock_hold_on_low_score and (now - locked_time) <= self.target_lock_timeout_sec:
                return None, pending_bbox, pending_count

            fallback_i = 0
            fallback_area = 0.0
            for i in range(xyxy.shape[0]):
                area = self._bbox_area(xyxy[i])
                if area > fallback_area:
                    fallback_area = area
                    fallback_i = i
            return fallback_i, pending_bbox, pending_count

        return best_i, pending_bbox, pending_count

    def _maybe_log_diag(self) -> None:
        now = time.time()
        if (now - self.last_diag_log_time) < max(0.5, self.log_period_sec):
            return
        self.get_logger().info(
            f'diag_d435: det_ok={self.diag_det_ok} depth_fail={self.diag_depth_fail} '
            f'tf_fail={self.diag_tf_fail} publish_ok={self.diag_publish_ok} '
            f'lock_hold={self.diag_lock_d435_hold} lock_switch={self.diag_lock_d435_switch}'
        )
        if self.enable_d455_target_interface:
            self.get_logger().info(
                f'diag_d455: det_ok={self.diag_d455_det_ok} depth_fail={self.diag_d455_depth_fail} '
                f'tf_fail={self.diag_d455_tf_fail} publish_ok={self.diag_d455_publish_ok} '
                f'lock_hold={self.diag_lock_d455_hold} lock_switch={self.diag_lock_d455_switch} '
                f'zone={self.d455_depth_zone} vr={self.d455_last_valid_ratio:.2f} vc={self.d455_last_valid_count}'
            )
        self.diag_det_ok = 0
        self.diag_depth_fail = 0
        self.diag_tf_fail = 0
        self.diag_publish_ok = 0
        self.diag_d455_det_ok = 0
        self.diag_d455_depth_fail = 0
        self.diag_d455_tf_fail = 0
        self.diag_d455_publish_ok = 0
        self.diag_lock_d435_hold = 0
        self.diag_lock_d435_switch = 0
        self.diag_lock_d455_hold = 0
        self.diag_lock_d455_switch = 0
        self.last_diag_log_time = now

    def _depth_value_to_m(self, arr: np.ndarray) -> np.ndarray:
        # RealSense depth 常见是 16UC1(mm)，也可能是 32FC1(m)
        if arr.dtype == np.uint16:
            return arr.astype(np.float32) * 0.001
        return arr.astype(np.float32)

    def _estimate_depth_m(self, px: float, py: float) -> Optional[float]:
        if self.depth_img is None:
            return None

        d = self.depth_img
        h, w = d.shape[:2]
        cx = int(round(px))
        cy = int(round(py))
        half = max(1, int(self.depth_roi_half))

        x1 = max(0, cx - half)
        x2 = min(w, cx + half + 1)
        y1 = max(0, cy - half)
        y2 = min(h, cy + half + 1)
        if x1 >= x2 or y1 >= y2:
            return None

        roi = d[y1:y2, x1:x2]
        roi_m = self._depth_value_to_m(roi)
        valid = roi_m[np.isfinite(roi_m)]
        valid = valid[(valid > self.depth_min_m) & (valid < self.depth_max_m)]
        if valid.size >= max(1, int(self.depth_min_valid_count)):
            depth_m = float(np.median(valid))
            self.last_good_depth_m = depth_m
            self.last_good_depth_time = time.time()
            return depth_m

        # 短时保持：深度偶发空洞时使用最近一次有效深度，降低抖动
        if self.last_good_depth_m is not None and (time.time() - self.last_good_depth_time) <= self.depth_hold_sec:
            return float(self.last_good_depth_m)

        return None

    def _estimate_d455_depth_m(self, x1: float, y1: float, x2: float, y2: float,
                               px: Optional[float] = None, py: Optional[float] = None) -> Optional[float]:
        if self.d455_depth_img is None:
            return None

        d = self.d455_depth_img
        h, w = d.shape[:2]

        # Legacy path (center small ROI + median)
        if self.d455_depth_strategy_mode != 'adaptive':
            cx = int(round((x1 + x2) * 0.5 if px is None else px))
            cy = int(round((y1 + y2) * 0.5 if py is None else py))
            half = max(1, int(self.d455_depth_roi_half))
            lx1 = max(0, cx - half)
            lx2 = min(w, cx + half + 1)
            ly1 = max(0, cy - half)
            ly2 = min(h, cy + half + 1)
            if lx1 >= lx2 or ly1 >= ly2:
                return None
            roi = d[ly1:ly2, lx1:lx2]
            roi_m = self._depth_value_to_m(roi)
            valid = roi_m[np.isfinite(roi_m)]
            valid = valid[(valid > self.depth_min_m) & (valid < self.depth_max_m)]
            self.d455_last_valid_count = int(valid.size)
            self.d455_last_valid_ratio = float(valid.size / max(1, roi_m.size))
            if valid.size >= max(1, int(self.depth_min_valid_count)):
                depth_m = float(np.median(valid))
                self.d455_last_good_depth_m = depth_m
                self.d455_last_good_depth_time = time.time()
                return depth_m
            if self.d455_last_good_depth_m is not None and (time.time() - self.d455_last_good_depth_time) <= self.d455_depth_hold_sec:
                return float(self.d455_last_good_depth_m)
            return None

        # Adaptive path: core ROI + zone quantile (Near/Mid/Far)
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        sx = clamp(self.d455_depth_roi_shrink_x, 0.0, 0.45)
        st = clamp(self.d455_depth_roi_shrink_top, 0.0, 0.45)
        sb = clamp(self.d455_depth_roi_shrink_bottom, 0.0, 0.45)

        rx1 = int(round(x1 + bw * sx))
        rx2 = int(round(x2 - bw * sx))
        ry1 = int(round(y1 + bh * st))
        ry2 = int(round(y2 - bh * sb))

        rx1 = max(0, min(w - 1, rx1))
        rx2 = max(1, min(w, rx2))
        ry1 = max(0, min(h - 1, ry1))
        ry2 = max(1, min(h, ry2))

        if rx1 >= rx2 or ry1 >= ry2:
            # fallback to legacy center ROI if adaptive core invalid
            cx = int(round((x1 + x2) * 0.5 if px is None else px))
            cy = int(round((y1 + y2) * 0.5 if py is None else py))
            half = max(1, int(self.d455_depth_roi_half))
            rx1 = max(0, cx - half)
            rx2 = min(w, cx + half + 1)
            ry1 = max(0, cy - half)
            ry2 = min(h, cy + half + 1)
            if rx1 >= rx2 or ry1 >= ry2:
                return None

        roi = d[ry1:ry2, rx1:rx2]
        roi_m = self._depth_value_to_m(roi)
        valid = roi_m[np.isfinite(roi_m)]
        valid = valid[(valid > self.depth_min_m) & (valid < self.depth_max_m)]

        valid_count = int(valid.size)
        total_count = int(max(1, roi_m.size))
        valid_ratio = float(valid_count / total_count)
        self.d455_last_valid_count = valid_count
        self.d455_last_valid_ratio = valid_ratio

        if valid_count < max(1, int(self.d455_depth_valid_count_min)) or valid_ratio < self.d455_depth_valid_ratio_min:
            if self.d455_last_good_depth_m is not None and (time.time() - self.d455_last_good_depth_time) <= self.d455_depth_hold_sec:
                return float(self.d455_last_good_depth_m)
            return None

        # Ref distance for zone switching
        ref_q = clamp(self.d455_depth_ref_quantile, 0.05, 0.95)
        d_ref = float(np.quantile(valid, ref_q))
        a_ref = clamp(self.d455_depth_dist_ema_alpha, 0.01, 1.0)
        if self.d455_depth_ref_ema is None:
            self.d455_depth_ref_ema = d_ref
        else:
            self.d455_depth_ref_ema = (1.0 - a_ref) * self.d455_depth_ref_ema + a_ref * d_ref
        d_smooth = float(self.d455_depth_ref_ema)

        # Zone FSM with hysteresis + consecutive frames + hold
        now = time.time()
        hold_ok = (now - self.d455_depth_zone_last_switch_time) >= max(0.0, self.d455_depth_zone_hold_sec)
        zone = self.d455_depth_zone
        enter_n = max(1, int(self.d455_depth_enter_frames))
        exit_n = max(1, int(self.d455_depth_exit_frames))

        if hold_ok:
            if zone == 'NEAR':
                if d_smooth >= self.d455_depth_near_exit_m:
                    self.d455_depth_near_exit_count += 1
                else:
                    self.d455_depth_near_exit_count = 0
                self.d455_depth_far_enter_count = 0
                self.d455_depth_near_enter_count = 0
                self.d455_depth_far_exit_count = 0
                if self.d455_depth_near_exit_count >= exit_n:
                    zone = 'MID'
            elif zone == 'FAR':
                if d_smooth <= self.d455_depth_far_exit_m:
                    self.d455_depth_far_exit_count += 1
                else:
                    self.d455_depth_far_exit_count = 0
                self.d455_depth_far_enter_count = 0
                self.d455_depth_near_enter_count = 0
                self.d455_depth_near_exit_count = 0
                if self.d455_depth_far_exit_count >= exit_n:
                    zone = 'MID'
            else:  # MID
                if d_smooth <= self.d455_depth_near_enter_m:
                    self.d455_depth_near_enter_count += 1
                else:
                    self.d455_depth_near_enter_count = 0

                if d_smooth >= self.d455_depth_far_enter_m:
                    self.d455_depth_far_enter_count += 1
                else:
                    self.d455_depth_far_enter_count = 0

                self.d455_depth_near_exit_count = 0
                self.d455_depth_far_exit_count = 0

                if self.d455_depth_near_enter_count >= enter_n:
                    zone = 'NEAR'
                elif self.d455_depth_far_enter_count >= enter_n:
                    zone = 'FAR'

            if zone != self.d455_depth_zone:
                self.d455_depth_zone = zone
                self.d455_depth_zone_last_switch_time = now
                self.d455_depth_near_enter_count = 0
                self.d455_depth_near_exit_count = 0
                self.d455_depth_far_enter_count = 0
                self.d455_depth_far_exit_count = 0

        zone = self.d455_depth_zone
        if zone == 'NEAR':
            q = clamp(self.d455_depth_near_quantile, 0.05, 0.95)
        elif zone == 'FAR':
            q = clamp(self.d455_depth_far_quantile, 0.05, 0.95)
        else:
            q = clamp(self.d455_depth_mid_quantile, 0.05, 0.95)

        depth_m = float(np.quantile(valid, q))

        # jump clamp w.r.t last good
        max_jump = max(0.05, float(self.d455_depth_max_jump_m))
        if self.d455_last_good_depth_m is not None:
            delta = depth_m - float(self.d455_last_good_depth_m)
            if abs(delta) > max_jump:
                depth_m = float(self.d455_last_good_depth_m) + math.copysign(max_jump, delta)

        # output ema
        a_out = clamp(self.d455_depth_output_ema_alpha, 0.01, 1.0)
        if self.d455_depth_output_ema is None:
            self.d455_depth_output_ema = depth_m
        else:
            self.d455_depth_output_ema = (1.0 - a_out) * self.d455_depth_output_ema + a_out * depth_m

        out = float(self.d455_depth_output_ema)
        self.d455_last_good_depth_m = out
        self.d455_last_good_depth_time = now
        return out

    def _pixel_to_camera_point(self, u: float, v: float, z_m: float) -> Optional[np.ndarray]:
        if self.cam_fx is None or self.cam_fy is None or self.cam_cx is None or self.cam_cy is None:
            return None
        x = (u - self.cam_cx) * z_m / self.cam_fx
        y = (v - self.cam_cy) * z_m / self.cam_fy
        z = z_m
        return np.array([x, y, z], dtype=np.float32)

    def _pixel_to_d455_camera_point(self, u: float, v: float, z_m: float) -> Optional[np.ndarray]:
        if self.d455_cam_fx is None or self.d455_cam_fy is None or self.d455_cam_cx is None or self.d455_cam_cy is None:
            return None
        x = (u - self.d455_cam_cx) * z_m / self.d455_cam_fx
        y = (v - self.d455_cam_cy) * z_m / self.d455_cam_fy
        z = z_m
        return np.array([x, y, z], dtype=np.float32)

    def _camera_to_base(self, p_cam: np.ndarray, stamp):
        pt = PointStamped()
        pt.header.stamp = stamp
        src_frame = self.depth_frame_id if self.depth_frame_id else self.camera_optical_frame
        pt.header.frame_id = src_frame
        pt.point.x = float(p_cam[0])
        pt.point.y = float(p_cam[1])
        pt.point.z = float(p_cam[2])
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.target_frame,
                src_frame,
                rclpy.time.Time())
            return do_transform_point(pt, tf_stamped)
        except Exception as e:
            if self.fallback_without_tf:
                # No-TF fallback: publish ONLY forward depth as distance.
                # Do not use camera x/y for range, otherwise large yaw can look like "too close".
                fb = PointStamped()
                fb.header.stamp = stamp
                fb.header.frame_id = src_frame
                fb.point.x = float(p_cam[2])  # use Z depth as range proxy
                fb.point.y = 0.0
                fb.point.z = 0.0
                return fb
            now = time.time()
            if (now - self.last_tf_warn_time) > max(0.5, self.log_period_sec):
                self.last_tf_warn_time = now
                self.get_logger().warn(
                    f'tf lookup failed: {self.target_frame} <- {src_frame}: {e}')
            return None

    def _d455_camera_to_base(self, p_cam: np.ndarray, stamp):
        pt = PointStamped()
        pt.header.stamp = stamp
        src_frame = self.d455_depth_frame_id if self.d455_depth_frame_id else self.d455_camera_optical_frame
        pt.header.frame_id = src_frame
        pt.point.x = float(p_cam[0])
        pt.point.y = float(p_cam[1])
        pt.point.z = float(p_cam[2])
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.target_frame,
                src_frame,
                rclpy.time.Time())
            return do_transform_point(pt, tf_stamped)
        except Exception as e:
            if self.fallback_without_tf:
                fb = PointStamped()
                fb.header.stamp = stamp
                fb.header.frame_id = src_frame
                fb.point.x = float(p_cam[2])
                fb.point.y = 0.0
                fb.point.z = 0.0
                return fb
            now = time.time()
            if (now - self.last_tf_warn_time) > max(0.5, self.log_period_sec):
                self.last_tf_warn_time = now
                self.get_logger().warn(
                    f'd455 tf lookup failed: {self.target_frame} <- {src_frame}: {e}')
            return None

    # ---- joint callback ----
    def _joint_cb(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return
        name_to_pos = {n: msg.position[i] for i, n in enumerate(msg.name) if i < len(msg.position)}
        if 'head_yaw' in name_to_pos:
            self.head_yaw_deg = math.degrees(name_to_pos['head_yaw'])
        elif 'head_yaw_joint' in name_to_pos:
            self.head_yaw_deg = math.degrees(name_to_pos['head_yaw_joint'])
        if 'head_pitch' in name_to_pos:
            self.head_pitch_deg = math.degrees(name_to_pos['head_pitch'])
        elif 'head_pitch_joint' in name_to_pos:
            self.head_pitch_deg = math.degrees(name_to_pos['head_pitch_joint'])



    def _target_class_cb(self, msg: Int32) -> None:
        new_id = int(msg.data)
        if new_id == self.detect_class_id:
            return
        self.detect_class_id = new_id
        self.get_logger().info(f'target class switched by FSM: detect_class_id={self.detect_class_id}')

    def _follow_enabled_cb(self, msg: Bool) -> None:
        new_value = bool(msg.data)
        if new_value == self.follow_enabled:
            return
        self.follow_enabled = new_value

        if self.follow_enabled:
            self.get_logger().info('FSM -> FOLLOWING: enable realtime detect/control.')
            return

        # enter IDLE: clear tracking state and stop motion/control outputs
        self.get_logger().info('FSM -> IDLE: disable realtime detect/control, stop + home.')

        with self.lock:
            self.raw_target_yaw = None
            self.raw_target_pitch = None
            self.last_det_time = 0.0
            self.locked_bbox_d435 = None
            self.locked_bbox_d455 = None
            self.pending_switch_bbox_d435 = None
            self.pending_switch_count_d435 = 0
            self.pending_switch_bbox_d455 = None
            self.pending_switch_count_d455 = 0

        self.smooth_yaw = None
        self.smooth_pitch = None

        if self.publish_target_interface:
            self._publish_target_valid(False)
        if self.enable_d455_target_interface:
            self._publish_d455_target_valid(False)

        self._send_velocity_stop()
        self._send_home_position()

    # ---- image callback: 检测 + 直接算目标角 ----
    def _image_cb(self, msg: Image) -> None:
        if not self.started or (not self.follow_enabled):
            return
        # 跳过：上一帧还没推理完
        if self.infer_busy:
            return
        self.infer_busy = True
        try:
            self._do_detect(msg)
        finally:
            self.infer_busy = False

    def _do_detect(self, msg: Image) -> None:
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h_img, w_img = cv_img.shape[:2]

        results = self.model.predict(
            source=cv_img, classes=[self.detect_class_id], conf=self.conf_thres,
            imgsz=self.imgsz, max_det=self.detect_max_det, verbose=False,
            device=self.device if self.device else None,
            half=self.use_half)

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            self.pending_switch_bbox_d435 = None
            self.pending_switch_count_d435 = 0
            return

        # 多目标选择：优先跟住已锁定目标，否则回退最大框
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
        cx_img, cy_img = w_img * 0.5, h_img * 0.5
        prev_locked_bbox = self.locked_bbox_d435.copy() if self.locked_bbox_d435 is not None else None

        now = time.time()
        best_i, self.pending_switch_bbox_d435, self.pending_switch_count_d435 = self._select_best_detection(
            xyxy=xyxy,
            w_img=float(w_img),
            h_img=float(h_img),
            locked_bbox=self.locked_bbox_d435,
            locked_time=self.locked_bbox_d435_time,
            now=now,
            pending_bbox=self.pending_switch_bbox_d435,
            pending_count=self.pending_switch_count_d435,
            confirm_frames=max(1, self.target_lock_switch_confirm_frames_d435),
        )

        if best_i is None:
            self.diag_lock_d435_hold += 1
            if self.publish_target_interface:
                self._publish_target_valid(False)
                self._maybe_log_diag()
            return

        x1, y1, x2, y2 = xyxy[best_i]
        det_conf = float(confs[best_i]) if confs is not None and best_i < len(confs) else 1.0

        # 更新 D435 锁定框，并统计是否发生“切换”
        new_locked_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        if self.enable_target_lock and prev_locked_bbox is not None:
            if self._bbox_iou(prev_locked_bbox, new_locked_bbox) < 0.25:
                self.diag_lock_d435_switch += 1
            else:
                self.diag_lock_d435_hold += 1
        self.locked_bbox_d435 = new_locked_bbox
        self.locked_bbox_d435_time = now
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return

        # bbox 跟随点
        px = x1 + self.bbox_target_x_ratio * bw
        py = y1 + self.bbox_target_y_ratio * bh

        # 发布 D435 bbox 元信息（用于跨相机关联）
        self._publish_d435_bbox(msg.header.stamp, x1, y1, x2, y2, det_conf, w_img, h_img)

        # 归一化偏差 [-1, +1]
        ex = (px - cx_img) / cx_img
        ey = (py - cy_img) / cy_img

        # 偏差角度
        delta_yaw = -self.yaw_sign * ex * (self.fov_h / 2.0)
        delta_pitch = -self.pitch_sign * ey * (self.fov_v / 2.0)

        # 目标角 = 当前实测角 + 偏差角（这是此刻应该去到的角度）
        cur_yaw = self.head_yaw_deg
        cur_pitch = self.head_pitch_deg
        if cur_yaw is None or cur_pitch is None:
            return

        target_yaw = clamp(cur_yaw + delta_yaw, self.yaw_min, self.yaw_max)
        target_pitch = clamp(cur_pitch + delta_pitch, self.pitch_min, self.pitch_max)

        # 每次检测都打一行详细日志（排查方向问题）
        self.get_logger().debug(
            f'DET px={px:.0f}/{w_img} ex={ex:+.2f} dy={delta_yaw:+.1f} '
            f'cur={cur_yaw:+.1f} tgt={target_yaw:+.1f}'
        )

        # 发布给底盘跟随控制器的目标接口（distance + bearing）
        if self.publish_target_interface:
            self.diag_det_ok += 1
            depth_m = self._estimate_depth_m(px, py)
            if depth_m is not None:
                p_cam = self._pixel_to_camera_point(px, py, depth_m)
                if p_cam is not None:
                    p_base = self._camera_to_base(p_cam, msg.header.stamp)
                    if p_base is not None:
                        bx = float(p_base.point.x)
                        by = float(p_base.point.y)
                        distance_m = max(0.0, math.hypot(bx, by))
                        bearing_rad = math.atan2(by, bx)
                        self._publish_target(msg.header.stamp, distance_m, bearing_rad, 1.0)
                        self._publish_target_valid(True)
                        self.diag_publish_ok += 1
                    else:
                        self._publish_target_valid(False)
                        self.diag_tf_fail += 1
                else:
                    self._publish_target_valid(False)
                    self.diag_depth_fail += 1
            else:
                self._publish_target_valid(False)
                self.diag_depth_fail += 1

            self._maybe_log_diag()

        with self.lock:
            self.raw_target_yaw = target_yaw
            self.raw_target_pitch = target_pitch
            self.last_det_time = time.time()
            self.det_count += 1

    # ---- D455 image callback: detect + publish target_d455 only ----
    def _d455_image_cb(self, msg: Image) -> None:
        if (not self.started) or (not self.enable_d455_target_interface) or (not self.follow_enabled):
            return
        now = time.time()
        interval = 1.0 / max(0.5, self.d455_detect_hz)
        if (now - self.last_d455_infer_time) < interval:
            return
        if self.infer_busy:
            return

        self.infer_busy = True
        self.last_d455_infer_time = now
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h_img, w_img = cv_img.shape[:2]

            results = self.model.predict(
                source=cv_img, classes=[self.detect_class_id], conf=self.conf_thres,
                imgsz=self.imgsz, max_det=self.detect_max_det, verbose=False,
                device=self.device if self.device else None,
                half=self.use_half)

            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                self.pending_switch_bbox_d455 = None
                self.pending_switch_count_d455 = 0
                self._publish_d455_target_valid(False)
                return

            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
            prev_locked_bbox = self.locked_bbox_d455.copy() if self.locked_bbox_d455 is not None else None

            now = time.time()
            best_i, self.pending_switch_bbox_d455, self.pending_switch_count_d455 = self._select_best_detection(
                xyxy=xyxy,
                w_img=float(w_img),
                h_img=float(h_img),
                locked_bbox=self.locked_bbox_d455,
                locked_time=self.locked_bbox_d455_time,
                now=now,
                pending_bbox=self.pending_switch_bbox_d455,
                pending_count=self.pending_switch_count_d455,
                confirm_frames=max(1, self.target_lock_switch_confirm_frames_d455),
            )

            if best_i is None:
                self.diag_lock_d455_hold += 1
                self._publish_d455_target_valid(False)
                self._maybe_log_diag()
                return

            x1, y1, x2, y2 = xyxy[best_i]
            det_conf = float(confs[best_i]) if confs is not None and best_i < len(confs) else 1.0

            # 更新 D455 锁定框，并统计是否发生“切换”
            new_locked_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
            if self.enable_target_lock and prev_locked_bbox is not None:
                if self._bbox_iou(prev_locked_bbox, new_locked_bbox) < 0.25:
                    self.diag_lock_d455_switch += 1
                else:
                    self.diag_lock_d455_hold += 1
            self.locked_bbox_d455 = new_locked_bbox
            self.locked_bbox_d455_time = now
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                self._publish_d455_target_valid(False)
                return

            px = x1 + self.bbox_target_x_ratio * bw
            py = y1 + self.bbox_target_y_ratio * bh

            # 发布 D455 bbox 元信息（用于跨相机关联）
            self._publish_d455_bbox(msg.header.stamp, x1, y1, x2, y2, det_conf, w_img, h_img)

            self.diag_d455_det_ok += 1
            depth_m = self._estimate_d455_depth_m(x1, y1, x2, y2, px, py)
            if depth_m is None:
                self.diag_d455_depth_fail += 1
                self._publish_d455_target_valid(False)
                self._maybe_log_diag()
                return

            p_cam = self._pixel_to_d455_camera_point(px, py, depth_m)
            if p_cam is None:
                self.diag_d455_depth_fail += 1
                self._publish_d455_target_valid(False)
                self._maybe_log_diag()
                return

            p_base = self._d455_camera_to_base(p_cam, msg.header.stamp)
            if p_base is None:
                self.diag_d455_tf_fail += 1
                self._publish_d455_target_valid(False)
                self._maybe_log_diag()
                return

            bx = float(p_base.point.x)
            by = float(p_base.point.y)
            distance_m = max(0.0, math.hypot(bx, by))
            bearing_rad = math.atan2(by, bx)
            self._publish_d455_target(msg.header.stamp, distance_m, bearing_rad, 1.0)
            self._publish_d455_target_valid(True)
            self.diag_d455_publish_ok += 1
            self.last_d455_det_time = now
            self._maybe_log_diag()

        except Exception:
            self._publish_d455_target_valid(False)
        finally:
            self.infer_busy = False
    def _ctrl_tick(self) -> None:
        if not self.started:
            return

        now = time.time()

        if not self.follow_enabled:
            if self.publish_target_interface:
                self._publish_target_valid(False)
            if self.enable_d455_target_interface:
                self._publish_d455_target_valid(False)

            # throttle stop command when staying in IDLE
            if (now - self.last_follow_stop_cmd_time) > 0.5:
                self.last_follow_stop_cmd_time = now
                self._send_velocity_stop()
            return

        with self.lock:
            raw_yaw = self.raw_target_yaw
            raw_pitch = self.raw_target_pitch
            det_time = self.last_det_time
            det_cnt = self.det_count

        # 没有检测目标 or 超时
        if raw_yaw is None or raw_pitch is None or (now - det_time) > self.lost_timeout:
            # 释放目标锁，允许重新选择
            if (now - det_time) > self.target_lock_timeout_sec:
                self.locked_bbox_d435 = None
                self.locked_bbox_d455 = None
                self.pending_switch_bbox_d435 = None
                self.pending_switch_count_d435 = 0
                self.pending_switch_bbox_d455 = None
                self.pending_switch_count_d455 = 0

            if self.publish_target_interface:
                self._publish_target_valid(False)
                self._maybe_log_diag()

            sent_home_vel = False
            if self.lost_behavior == 'home':
                # 丢目标后回中位，以便重新扫到人
                home_yaw = self.home_head_yaw
                home_pitch = self.home_head_pitch
                if self.smooth_yaw is not None:
                    self.smooth_yaw += 0.08 * (home_yaw - self.smooth_yaw)
                    self.smooth_pitch += 0.08 * (home_pitch - self.smooth_pitch)
                    cur_yaw = self.head_yaw_deg if self.head_yaw_deg is not None else 0.0
                    cur_pitch = self.head_pitch_deg if self.head_pitch_deg is not None else 0.0
                    # 用 velocity mode 平滑回中
                    yaw_err = self.smooth_yaw - cur_yaw
                    pitch_err = self.smooth_pitch - cur_pitch
                    if abs(yaw_err) > 1.0 or abs(pitch_err) > 1.0:
                        if self.torso_client.server_is_ready():
                            g = Torso.Goal()
                            g.torso_height = float(self.home_torso_height)
                            g.torso_yaw = float(self.home_torso_yaw)
                            g.head_yaw = float(clamp(1.5 * yaw_err, -30.0, 30.0))
                            g.head_pitch = float(clamp(1.5 * pitch_err, -20.0, 20.0))
                            g.max_velocity = 0.5
                            g.work_mode = 1
                            self.torso_client.send_goal_async(g)
                            sent_home_vel = True

            # 关键修复：不要在同一个 tick 里又发回中速度又发 stop(0速度)
            # 否则底层会出现“启停打架”，表现为丢目标时电机咔咔响。
            if not sent_home_vel:
                self._send_velocity_stop()

            if (now - self.last_log_time) > self.log_period_sec * 3:
                self.last_log_time = now
                sy = self.smooth_yaw if self.smooth_yaw is not None else 0.0
                sp = self.smooth_pitch if self.smooth_pitch is not None else 0.0
                self.get_logger().info(f'no target -> {self.lost_behavior} smooth=({sy:+.1f},{sp:+.1f})')
            return

        # 初始化平滑值
        if self.smooth_yaw is None:
            self.smooth_yaw = raw_yaw
        if self.smooth_pitch is None:
            self.smooth_pitch = raw_pitch

        # 自适应插值：偏差大时快速跟进，偏差小时丝滑收尾
        dy = raw_yaw - self.smooth_yaw
        dp = raw_pitch - self.smooth_pitch
        max_d = max(abs(dy), abs(dp))
        # 偏差 >10° 时 alpha 接近 0.6（快追），<3° 时 alpha 接近 smoothing（丝滑）
        alpha = lerp(self.smoothing, 0.6, clamp((max_d - 3.0) / 10.0, 0.0, 1.0))
        self.smooth_yaw += alpha * dy
        self.smooth_pitch += alpha * dp

        target_yaw = clamp(self.smooth_yaw, self.yaw_min, self.yaw_max)
        target_pitch = clamp(self.smooth_pitch, self.pitch_min, self.pitch_max)

        # 死区：如果目标与当前位置差距很小就不发
        cur_yaw = self.head_yaw_deg if self.head_yaw_deg is not None else 0.0
        cur_pitch = self.head_pitch_deg if self.head_pitch_deg is not None else 0.0
        diff_yaw = abs(target_yaw - cur_yaw)
        diff_pitch = abs(target_pitch - cur_pitch)

        if diff_yaw < self.deadband_deg and diff_pitch < self.deadband_deg:
            return

        # 用 velocity mode (work_mode=1)：直接发角速度，底层电机自己平滑
        # Torso action 在 velocity mode 下：
        #   head_yaw = yaw 速度 (deg/s)
        #   head_pitch = pitch 速度 (deg/s)
        # 速度 > 0 → 朝正方向，< 0 → 朝负方向，≈ 0 → 停

        # P控制：速度 = gain × 偏差角度
        vel_gain = 2.0  # deg/s per deg error
        yaw_vel = vel_gain * (target_yaw - cur_yaw)
        pitch_vel = vel_gain * (target_pitch - cur_pitch)

        # 限速
        max_vel_deg = 60.0  # deg/s
        yaw_vel = clamp(yaw_vel, -max_vel_deg, max_vel_deg)
        pitch_vel = clamp(pitch_vel, -max_vel_deg, max_vel_deg)

        if self.torso_client.server_is_ready():
            g = Torso.Goal()
            g.torso_height = float(self.home_torso_height)
            g.torso_yaw = float(self.home_torso_yaw)
            g.head_yaw = float(yaw_vel)      # velocity mode: 这是速度不是角度
            g.head_pitch = float(pitch_vel)   # velocity mode: 这是速度不是角度
            g.max_velocity = 0.5
            g.work_mode = 1  # VELOCITY_MODE
            self.torso_client.send_goal_async(g)

        vel = max(abs(yaw_vel), abs(pitch_vel))

        # 日志
        if (now - self.last_log_time) > self.log_period_sec:
            self.last_log_time = now
            self.get_logger().info(
                f'target=({target_yaw:+.1f},{target_pitch:+.1f}) '
                f'head=({cur_yaw:+.1f},{cur_pitch:+.1f}) '
                f'raw=({raw_yaw:+.1f},{raw_pitch:+.1f}) '
                f'yaw_v={yaw_vel:+.1f} pitch_v={pitch_vel:+.1f} '
                f'det={det_cnt}'
            )


def main(args=None):
    # ---- 关键：禁用 rclpy 默认的 SIGINT handler ----
    # rclpy.init() 默认会安装自己的 SIGINT handler，收到 Ctrl+C 后
    # 立刻调 rclpy.shutdown()，导致后续 send_goal_async 全部失败。
    # 我们需要在 rclpy context 活着的时候先发归位命令。
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        # Humble 可能没有 SignalHandlerOptions，退回默认
        rclpy.init(args=args)

    node = PersonFollowNode()

    # 用 threading.Event 控制退出
    shutdown_event = threading.Event()

    def _sigint_handler(signum, frame):
        if not shutdown_event.is_set():
            shutdown_event.set()

    # 注册自己的 SIGINT/SIGTERM handler
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    # 在独立线程中 spin（这样主线程可以等待信号）
    spin_thread = threading.Thread(target=_spin_thread, args=(node, shutdown_event), daemon=True)
    spin_thread.start()

    # 主线程等待退出信号
    try:
        shutdown_event.wait()
    except KeyboardInterrupt:
        shutdown_event.set()

    node.get_logger().info('Ctrl+C / SIGTERM received, starting graceful shutdown...')

    # ---- 在 rclpy context 还活着的时候执行归位 ----
    try:
        node.graceful_shutdown()
    except Exception as e:
        try:
            node.get_logger().warn(f'graceful_shutdown error: {e}')
        except Exception:
            pass

    # ---- 现在才关闭 rclpy ----
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


def _spin_thread(node: PersonFollowNode, shutdown_event: threading.Event) -> None:
    """Spin node until shutdown_event is set."""
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
