#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D455 深度准确性小实验（支持 YOLO 人体 ROI）。

用途：
- 订阅 D455 aligned depth 图像
- 可选：订阅 D455 RGB，用 YOLO 检测人体 bbox
- 在人体 bbox 对应深度区域做鲁棒统计（默认 P30）
- 与人工真值(gt_distance_m)对比并输出统计结果，可追加 CSV

推荐：use_yolo_person_roi=True（更贴近跟随场景）
"""

import csv
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def _to_meters(depth: np.ndarray) -> np.ndarray:
    # RealSense 常见 16UC1(mm)；也可能 32FC1(m)
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) * 0.001
    return depth.astype(np.float32)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class D455DepthAccuracyTest(Node):

    def __init__(self) -> None:
        super().__init__('d455_depth_accuracy_test')

        # topics
        self.declare_parameter('depth_topic', '/cam_chest/d455/aligned_depth_to_color/image_raw')
        self.declare_parameter('rgb_topic', '/cam_chest/d455/color/image_raw')

        # sampling mode
        self.declare_parameter('use_yolo_person_roi', True)
        self.declare_parameter('use_center_pixel', True)  # 仅在 use_yolo_person_roi=False 时生效
        self.declare_parameter('pixel_u', -1)
        self.declare_parameter('pixel_v', -1)
        self.declare_parameter('roi_half', 15)

        # YOLO params
        default_model_path = str(Path(get_package_share_directory('person_follow')) / 'models' / 'yolov8s.pt')
        self.declare_parameter('yolo_model_path', default_model_path)
        self.declare_parameter('yolo_device', '')
        self.declare_parameter('conf_thres', 0.20)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('yolo_detect_hz', 8.0)
        self.declare_parameter('min_bbox_area', 2500.0)
        self.declare_parameter('bbox_inner_shrink_ratio', 0.25)  # 从 bbox 四边内缩比例，减少背景污染
        self.declare_parameter('bbox_timeout_sec', 0.7)

        # depth filter + statistic
        self.declare_parameter('depth_min_m', 0.2)
        self.declare_parameter('depth_max_m', 6.0)
        self.declare_parameter('min_valid_count', 5)
        self.declare_parameter('depth_stat_method', 'p30')  # p30|p40|median

        # runtime
        self.declare_parameter('settle_sec', 1.5)
        self.declare_parameter('sample_duration_sec', 6.0)
        self.declare_parameter('min_samples', 20)

        # eval
        self.declare_parameter('gt_distance_m', 1.0)
        self.declare_parameter('output_csv', '')

        # ---- get params ----
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.rgb_topic = str(self.get_parameter('rgb_topic').value)

        self.use_yolo_person_roi = bool(self.get_parameter('use_yolo_person_roi').value)
        self.use_center_pixel = bool(self.get_parameter('use_center_pixel').value)
        self.pixel_u = int(self.get_parameter('pixel_u').value)
        self.pixel_v = int(self.get_parameter('pixel_v').value)
        self.roi_half = max(1, int(self.get_parameter('roi_half').value))

        self.yolo_model_path = str(self.get_parameter('yolo_model_path').value)
        self.yolo_device = str(self.get_parameter('yolo_device').value)
        self.conf_thres = float(self.get_parameter('conf_thres').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.yolo_detect_hz = max(0.5, float(self.get_parameter('yolo_detect_hz').value))
        self.min_bbox_area = float(self.get_parameter('min_bbox_area').value)
        self.bbox_inner_shrink_ratio = _clamp(float(self.get_parameter('bbox_inner_shrink_ratio').value), 0.0, 0.45)
        self.bbox_timeout_sec = max(0.1, float(self.get_parameter('bbox_timeout_sec').value))

        self.depth_min_m = float(self.get_parameter('depth_min_m').value)
        self.depth_max_m = float(self.get_parameter('depth_max_m').value)
        self.min_valid_count = max(1, int(self.get_parameter('min_valid_count').value))
        self.depth_stat_method = str(self.get_parameter('depth_stat_method').value).strip().lower()

        self.settle_sec = max(0.0, float(self.get_parameter('settle_sec').value))
        self.sample_duration_sec = max(0.5, float(self.get_parameter('sample_duration_sec').value))
        self.min_samples = max(1, int(self.get_parameter('min_samples').value))

        self.gt_distance_m = float(self.get_parameter('gt_distance_m').value)
        self.output_csv = str(self.get_parameter('output_csv').value)

        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # timing
        self.start_time = time.time()
        self.collect_begin = self.start_time + self.settle_sec
        self.collect_end = self.collect_begin + self.sample_duration_sec

        # stats
        self.samples = []
        self.frames = 0
        self.frames_valid = 0
        self.last_log_time = 0.0

        # yolo states
        self.model = None
        self.infer_busy = False
        self.last_infer_time = 0.0
        self.last_rgb_wh: Optional[Tuple[int, int]] = None
        self.last_person_roi_rgb: Optional[Tuple[int, int, int, int]] = None  # x1,y1,x2,y2 in RGB
        self.last_person_roi_time = 0.0
        self.diag_det_ok = 0
        self.diag_det_miss = 0

        if self.use_yolo_person_roi:
            self._init_yolo()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, self.depth_topic, self._depth_cb, qos)
        if self.use_yolo_person_roi:
            self.create_subscription(Image, self.rgb_topic, self._rgb_cb, qos)

        self.timer = self.create_timer(0.1, self._tick)

        self.get_logger().info(
            f'D455 depth accuracy test started. mode={"yolo_person_roi" if self.use_yolo_person_roi else "center_roi"} '
            f'depth_topic={self.depth_topic} rgb_topic={self.rgb_topic if self.use_yolo_person_roi else "<unused>"} '
            f'gt={self.gt_distance_m:.3f}m settle={self.settle_sec:.1f}s sample={self.sample_duration_sec:.1f}s '
            f'stat={self.depth_stat_method} out_csv={self.output_csv if self.output_csv else "<none>"}'
        )

    def _init_yolo(self) -> None:
        if YOLO is None:
            self.get_logger().error('ultralytics not available. Please install it or disable use_yolo_person_roi.')
            raise SystemExit(3)
        if not os.path.isfile(self.yolo_model_path):
            self.get_logger().error(f'YOLO model not found: {self.yolo_model_path}')
            raise SystemExit(3)

        self.get_logger().info(
            f'Loading YOLO model={self.yolo_model_path} device={self.yolo_device if self.yolo_device else "auto"}'
        )
        self.model = YOLO(self.yolo_model_path)
        # warmup
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        _ = self.model.predict(
            source=dummy,
            classes=[0],
            conf=0.1,
            imgsz=320,
            max_det=1,
            verbose=False,
            device=self.yolo_device if self.yolo_device else None,
        )
        self.get_logger().info('YOLO ready.')

    def _rgb_cb(self, msg: Image) -> None:
        if self.model is None:
            return

        now = time.time()
        if (now - self.last_infer_time) < (1.0 / self.yolo_detect_hz):
            return
        if self.infer_busy:
            return

        self.infer_busy = True
        self.last_infer_time = now
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = img.shape[:2]
            self.last_rgb_wh = (w, h)

            results = self.model.predict(
                source=img,
                classes=[0],
                conf=self.conf_thres,
                imgsz=self.imgsz,
                max_det=5,
                verbose=False,
                device=self.yolo_device if self.yolo_device else None,
            )

            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                self.diag_det_miss += 1
                return

            xyxy = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else None

            best = None
            best_area = 0.0
            best_conf = 0.0
            for i in range(xyxy.shape[0]):
                x1, y1, x2, y2 = xyxy[i]
                area = (x2 - x1) * (y2 - y1)
                if area < self.min_bbox_area:
                    continue
                if area > best_area:
                    best_area = area
                    best = (x1, y1, x2, y2)
                    best_conf = float(confs[i]) if confs is not None and i < len(confs) else 0.0

            if best is None:
                self.diag_det_miss += 1
                return

            x1, y1, x2, y2 = best
            # clamp and int
            x1 = int(max(0, min(w - 1, x1)))
            y1 = int(max(0, min(h - 1, y1)))
            x2 = int(max(1, min(w, x2)))
            y2 = int(max(1, min(h, y2)))
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                self.diag_det_miss += 1
                return

            # inner shrink to reduce background contamination
            bw = x2 - x1
            bh = y2 - y1
            sx = int(bw * self.bbox_inner_shrink_ratio)
            sy = int(bh * self.bbox_inner_shrink_ratio)
            rx1 = x1 + sx
            ry1 = y1 + sy
            rx2 = x2 - sx
            ry2 = y2 - sy
            if rx2 <= rx1 + 2 or ry2 <= ry1 + 2:
                rx1, ry1, rx2, ry2 = x1, y1, x2, y2

            with self.lock:
                self.last_person_roi_rgb = (rx1, ry1, rx2, ry2)
                self.last_person_roi_time = now

            self.diag_det_ok += 1
            if (now - self.last_log_time) > 1.0:
                self.last_log_time = now
                self.get_logger().info(
                    f'yolo: roi=({rx1},{ry1})-({rx2},{ry2}) conf={best_conf:.2f} '
                    f'det_ok={self.diag_det_ok} miss={self.diag_det_miss}'
                )
                self.diag_det_ok = 0
                self.diag_det_miss = 0

        except Exception as e:
            self.get_logger().warn(f'YOLO detect failed: {e}')
        finally:
            self.infer_busy = False

    def _resolve_roi_on_depth(self, depth_w: int, depth_h: int) -> Optional[Tuple[int, int, int, int]]:
        if not self.use_yolo_person_roi:
            if self.use_center_pixel:
                u = depth_w // 2
                v = depth_h // 2
            else:
                u = self.pixel_u if self.pixel_u >= 0 else (depth_w // 2)
                v = self.pixel_v if self.pixel_v >= 0 else (depth_h // 2)
            u = int(max(0, min(depth_w - 1, u)))
            v = int(max(0, min(depth_h - 1, v)))
            x1 = max(0, u - self.roi_half)
            x2 = min(depth_w, u + self.roi_half + 1)
            y1 = max(0, v - self.roi_half)
            y2 = min(depth_h, v + self.roi_half + 1)
            return (x1, y1, x2, y2) if (x2 > x1 and y2 > y1) else None

        with self.lock:
            roi_rgb = self.last_person_roi_rgb
            roi_t = self.last_person_roi_time
            rgb_wh = self.last_rgb_wh

        now = time.time()
        if roi_rgb is None or (now - roi_t) > self.bbox_timeout_sec or rgb_wh is None:
            return None

        rgb_w, rgb_h = rgb_wh
        if rgb_w <= 1 or rgb_h <= 1:
            return None

        x1, y1, x2, y2 = roi_rgb
        # depth 与 color 理论已对齐同分辨率；这里仍做尺度兼容
        sx = depth_w / float(rgb_w)
        sy = depth_h / float(rgb_h)
        dx1 = int(max(0, min(depth_w - 1, round(x1 * sx))))
        dy1 = int(max(0, min(depth_h - 1, round(y1 * sy))))
        dx2 = int(max(1, min(depth_w, round(x2 * sx))))
        dy2 = int(max(1, min(depth_h, round(y2 * sy))))

        if dx2 <= dx1 + 1 or dy2 <= dy1 + 1:
            return None
        return dx1, dy1, dx2, dy2

    def _depth_cb(self, msg: Image) -> None:
        self.frames += 1
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            now = time.time()
            if (now - self.last_log_time) > 1.0:
                self.last_log_time = now
                self.get_logger().warn(f'cv_bridge convert failed: {e}')
            return

        if depth_raw is None or depth_raw.ndim < 2:
            return

        h, w = depth_raw.shape[:2]
        if h < 2 or w < 2:
            return

        roi = self._resolve_roi_on_depth(w, h)
        if roi is None:
            return

        x1, y1, x2, y2 = roi
        if x1 >= x2 or y1 >= y2:
            return

        crop = depth_raw[y1:y2, x1:x2]
        crop_m = _to_meters(crop)
        valid = crop_m[np.isfinite(crop_m)]
        valid = valid[(valid > self.depth_min_m) & (valid < self.depth_max_m)]

        if valid.size < self.min_valid_count:
            return

        self.frames_valid += 1

        method = self.depth_stat_method
        if method == 'median':
            m = float(np.median(valid))
        elif method == 'p40':
            m = float(np.percentile(valid, 40.0))
        else:  # default p30
            m = float(np.percentile(valid, 30.0))

        now = time.time()
        if self.collect_begin <= now <= self.collect_end:
            self.samples.append(m)

    def _tick(self) -> None:
        now = time.time()
        if now < self.collect_begin:
            remain = self.collect_begin - now
            if int(remain * 10) % 10 == 0:
                self.get_logger().info(f'settling... {remain:.1f}s')
            return

        if now <= self.collect_end:
            remain = self.collect_end - now
            if int(remain * 10) % 10 == 0:
                self.get_logger().info(
                    f'collecting... {remain:.1f}s left, samples={len(self.samples)} '
                    f'frames={self.frames} valid_frames={self.frames_valid}'
                )
            return

        self._finish_and_exit()

    def _finish_and_exit(self) -> None:
        n = len(self.samples)
        if n == 0:
            self.get_logger().error('no valid samples collected. check topic / bbox / ROI / scene.')
            self._shutdown(2)
            return

        arr = np.array(self.samples, dtype=np.float64)
        mean_m = float(np.mean(arr))
        median_m = float(np.median(arr))
        std_m = float(np.std(arr))
        min_m = float(np.min(arr))
        max_m = float(np.max(arr))

        err_m = mean_m - self.gt_distance_m
        abs_err_m = abs(err_m)
        rel_err_pct = abs_err_m / max(1e-6, self.gt_distance_m) * 100.0

        quality = 'PASS'
        # 小实验默认门限：绝对误差 <= 5cm 且 相对误差 <= 5%
        if (abs_err_m > 0.05) and (rel_err_pct > 5.0):
            quality = 'WARN'

        self.get_logger().info('========== D455 Depth Accuracy Result ==========' )
        self.get_logger().info(
            f'mode={"yolo_person_roi" if self.use_yolo_person_roi else "center_roi"} stat={self.depth_stat_method}'
        )
        self.get_logger().info(
            f'gt={self.gt_distance_m:.3f} m | n={n} (min_required={self.min_samples}) '
            f'frames={self.frames} valid_frames={self.frames_valid}'
        )
        self.get_logger().info(
            f'mean={mean_m:.4f} m median={median_m:.4f} m std={std_m:.4f} m range=[{min_m:.4f}, {max_m:.4f}] m'
        )
        self.get_logger().info(
            f'error={err_m:+.4f} m abs_error={abs_err_m:.4f} m rel_error={rel_err_pct:.2f}% => {quality}'
        )

        if n < self.min_samples:
            self.get_logger().warn(
                f'sample count too low ({n} < {self.min_samples}), recommend increase sample_duration_sec.'
            )

        if self.output_csv:
            try:
                self._append_csv(
                    gt=self.gt_distance_m,
                    n=n,
                    mean_m=mean_m,
                    median_m=median_m,
                    std_m=std_m,
                    min_m=min_m,
                    max_m=max_m,
                    err_m=err_m,
                    abs_err_m=abs_err_m,
                    rel_err_pct=rel_err_pct,
                    quality=quality,
                )
                self.get_logger().info(f'csv appended: {self.output_csv}')
            except Exception as e:
                self.get_logger().warn(f'write csv failed: {e}')

        self._shutdown(0)

    def _append_csv(self, **row) -> None:
        path = self.output_csv
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

        headers = [
            'timestamp', 'gt_m', 'n', 'mean_m', 'median_m', 'std_m', 'min_m', 'max_m',
            'err_m', 'abs_err_m', 'rel_err_pct', 'quality'
        ]
        write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0

        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                w.writeheader()
            w.writerow({
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'gt_m': f'{row["gt"]:.4f}',
                'n': int(row['n']),
                'mean_m': f'{row["mean_m"]:.6f}',
                'median_m': f'{row["median_m"]:.6f}',
                'std_m': f'{row["std_m"]:.6f}',
                'min_m': f'{row["min_m"]:.6f}',
                'max_m': f'{row["max_m"]:.6f}',
                'err_m': f'{row["err_m"]:.6f}',
                'abs_err_m': f'{row["abs_err_m"]:.6f}',
                'rel_err_pct': f'{row["rel_err_pct"]:.3f}',
                'quality': row['quality'],
            })

    def _shutdown(self, code: int) -> None:
        self.get_logger().info(f'exit_code={code}')
        raise SystemExit(code)


def main(args=None):
    rclpy.init(args=args)
    node = D455DepthAccuracyTest()
    try:
        rclpy.spin(node)
    except SystemExit:
        raise
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
