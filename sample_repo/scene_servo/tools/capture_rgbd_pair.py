#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Capture one aligned RGB-D pair from D455 via ROS topics or pyrealsense2")
    ap.add_argument("--out-dir", default="", help="output directory, default: ./data/pbvs_eval/YYYYMMDD/HHMMSS")
    ap.add_argument("--name", required=True, help="basename, e.g. ref or back_0p5")
    ap.add_argument("--input-source", choices=["ros", "realsense"], default="ros")
    ap.add_argument("--image-topic", default="/cam_chest/d455/color/image_raw")
    ap.add_argument("--depth-topic", default="/cam_chest/d455/aligned_depth_to_color/image_raw")
    ap.add_argument("--camera-info-topic", default="/cam_chest/d455/color/camera_info")
    ap.add_argument("--timeout-sec", type=float, default=8.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--warmup-frames", type=int, default=30)
    ap.add_argument("--device-name", default="D455", help="substring matched against camera name")
    ap.add_argument("--allow-fallback-intrinsics", action="store_true",
                     help="allow default D455 intrinsics when camera_info is unavailable (NOT recommended for data collection)")
    return ap.parse_args()


def resolve_out_dir(out_dir_arg: str) -> Path:
    if out_dir_arg:
        out_dir = Path(out_dir_arg)
    else:
        repo_root = Path(__file__).resolve().parents[2]
        session_dir_env = os.environ.get("PBVS_EVAL_SESSION_DIR", "").strip()
        if session_dir_env:
            out_dir = Path(session_dir_env)
        else:
            date_dir = datetime.now().strftime("%Y%m%d")
            time_dir = datetime.now().strftime("%H%M%S")
            out_dir = repo_root / "data" / "pbvs_eval" / date_dir / time_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_capture(out_dir: Path, name: str, color: np.ndarray, depth_m: np.ndarray, camera_info: dict) -> None:
    img_path = out_dir / f"{name}.jpg"
    depth_path = out_dir / f"{name}_depth.npy"
    info_path = out_dir / "camera_info.json"

    cv2.imwrite(str(img_path), color)
    np.save(str(depth_path), depth_m.astype(np.float32))
    info_path.write_text(json.dumps(camera_info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "session_dir": str(out_dir),
        "image_path": str(img_path),
        "depth_path": str(depth_path),
        "camera_info": str(info_path),
        "fx": camera_info["fx"],
        "fy": camera_info["fy"],
        "cx": camera_info["cx"],
        "cy": camera_info["cy"],
    }, ensure_ascii=False, indent=2))


def capture_from_ros(args: argparse.Namespace, out_dir: Path) -> None:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image

    class CaptureNode(Node):
        def __init__(self) -> None:
            super().__init__("capture_rgbd_pair_once")
            self.bridge = CvBridge()
            self.color: np.ndarray | None = None
            self.depth_m: np.ndarray | None = None
            self.camera_info: dict | None = None
            self.ready = threading.Event()
            qos_sensor = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=2,
            )
            self.create_subscription(Image, args.image_topic, self._image_cb, qos_sensor)
            self.create_subscription(Image, args.depth_topic, self._depth_cb, qos_sensor)
            self.create_subscription(CameraInfo, args.camera_info_topic, self._camera_info_cb, qos_sensor)

        def _maybe_ready(self) -> None:
            if self.color is not None and self.depth_m is not None and self.camera_info is not None:
                self.ready.set()

        def _image_cb(self, msg: Image) -> None:
            try:
                self.color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                self._maybe_ready()
            except Exception as exc:
                self.get_logger().warn(f"image convert failed: {exc}")

        def _depth_cb(self, msg: Image) -> None:
            try:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                depth_np = np.asarray(depth)
                if depth_np.dtype == np.uint16:
                    self.depth_m = depth_np.astype(np.float32) * 0.001
                else:
                    self.depth_m = depth_np.astype(np.float32)
                self._maybe_ready()
            except Exception as exc:
                self.get_logger().warn(f"depth convert failed: {exc}")

        def _camera_info_cb(self, msg: CameraInfo) -> None:
            self.camera_info = {
                "fx": float(msg.k[0]),
                "fy": float(msg.k[4]),
                "cx": float(msg.k[2]),
                "cy": float(msg.k[5]),
                "width": int(msg.width),
                "height": int(msg.height),
                "device_name": "D455",
                "source": "ros",
                "image_topic": args.image_topic,
                "depth_topic": args.depth_topic,
                "camera_info_topic": args.camera_info_topic,
            }
            self._maybe_ready()

    rclpy.init()
    node = CaptureNode()
    deadline = time.time() + max(1.0, args.timeout_sec)
    try:
        while time.time() < deadline and not node.ready.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.color is None or node.depth_m is None:
            missing = []
            if node.color is None:
                missing.append(f"image({args.image_topic})")
            if node.depth_m is None:
                missing.append(f"depth({args.depth_topic})")
            raise RuntimeError("failed to receive required ROS data: missing " + ", ".join(missing))

        if node.camera_info is None:
            if not args.allow_fallback_intrinsics:
                raise RuntimeError(
                    "camera_info not received before timeout — "
                    "cannot capture with unknown intrinsics. "
                    "Use --allow-fallback-intrinsics to override (NOT recommended for data collection)."
                )
            h, w = node.color.shape[:2]
            node.get_logger().warn(
                "camera_info not received, using default D455 intrinsics (--allow-fallback-intrinsics)"
            )
            node.camera_info = {
                "fx": 382.6,
                "fy": 382.6,
                "cx": float(w) / 2.0,
                "cy": float(h) / 2.0,
                "width": int(w),
                "height": int(h),
                "device_name": "D455",
                "source": "ros_fallback",
                "image_topic": args.image_topic,
                "depth_topic": args.depth_topic,
                "camera_info_topic": args.camera_info_topic,
            }
        save_capture(out_dir, args.name, node.color, node.depth_m, node.camera_info)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def capture_from_realsense(args: argparse.Namespace, out_dir: Path) -> None:
    import pyrealsense2 as rs

    def find_device_serial(device_name_substr: str) -> str | None:
        ctx = rs.context()
        for dev in ctx.query_devices():
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            if device_name_substr.lower() in name.lower():
                return serial
        return None

    serial = find_device_serial(args.device_name)
    if serial is None:
        raise RuntimeError(f"no RealSense device matched device-name={args.device_name!r}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        for _ in range(max(1, args.warmup_frames)):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("failed to get aligned color/depth frames")

        color = np.asanyarray(color_frame.get_data())
        depth_u16 = np.asanyarray(depth_frame.get_data())
        depth_m = depth_u16.astype(np.float32) * 0.001

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        camera_info = {
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
            "width": intr.width,
            "height": intr.height,
            "device_serial": serial,
            "device_name": args.device_name,
            "source": "realsense",
        }
        save_capture(out_dir, args.name, color, depth_m, camera_info)
    finally:
        pipeline.stop()


def main() -> None:
    args = parse_args()
    out_dir = resolve_out_dir(args.out_dir)
    if args.input_source == "ros":
        capture_from_ros(args, out_dir)
    else:
        capture_from_realsense(args, out_dir)


if __name__ == "__main__":
    main()
