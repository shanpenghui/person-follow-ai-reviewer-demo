#!/usr/bin/env python3
"""quick_test.py — 用现有 scene_memory 快速验证特征点匹配

自身匹配（参考帧 vs 参考帧）应该得到：
  - level=L0, yaw≈0, forward≈0, inlier 接近总特征点数
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_servo.feature_matcher import FeatureMatcherCfg, extract_keypoints_3d, match_and_estimate


def main() -> None:
    # 用 qwen-vl-scene-match 里的现有数据
    mem_dir = Path("/home/dev/midea_humanoid_robot/src/qwen-vl-scene-match/scene_memory_test_d455")
    if not mem_dir.exists():
        print(f"[ERROR] {mem_dir} not found")
        sys.exit(1)

    idx = json.loads((mem_dir / "memory_index.json").read_text())
    sc = idx["scenes"][0]
    img_path = mem_dir / sc["image"]
    dep_path = mem_dir / sc["depth"]

    bgr = cv2.imread(str(img_path))
    depth = np.load(str(dep_path)).astype(np.float32)
    if depth.max() > 100:  # uint16 毫米
        depth = depth * 0.001
    h, w = bgr.shape[:2]

    # D455 默认内参 (640x480)
    fx = fy = 382.6
    cx = w / 2.0
    cy = h / 2.0

    print(f"[INFO] image: {img_path} ({w}x{h})")
    print(f"[INFO] depth range: {depth[depth>0].min():.3f} ~ {depth[depth>0].max():.3f} m")

    for det in ["orb", "sift"]:
        cfg = FeatureMatcherCfg(detector=det, max_keypoints=500)
        if det == "sift":
            cfg.sift_n_features = 500

        print(f"\n{'='*50}")
        print(f"  Detector: {det}")
        print(f"{'='*50}")

        # 提取参考帧特征
        ref_data = extract_keypoints_3d(bgr, depth, fx, fy, cx, cy, cfg)
        print(f"  keypoints: {len(ref_data['keypoints'])}")
        print(f"  with 3D:   {int(ref_data['xyz_mask'].sum())}")

        # 自身匹配（sanity check）
        result = match_and_estimate(ref_data, bgr, depth, fx, fy, cx, cy, cfg)
        print(f"  --- self-match ---")
        print(f"  level:        {result.level}")
        print(f"  matched:      {result.matched_count}")
        print(f"  inlier:       {result.inlier_count}")
        print(f"  yaw_err:      {result.yaw_error_deg:.3f}°")
        print(f"  forward_err:  {result.forward_error_m:.4f} m")
        print(f"  lateral_err:  {result.lateral_error_m:.4f} m")
        print(f"  confidence:   {result.confidence:.4f}")

        # 模拟后退：把图像裁中心 80% 然后 resize 回原尺寸（模拟 zoom-out）
        crop_ratio = 0.80
        ch = int(h * crop_ratio)
        cw = int(w * crop_ratio)
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        cropped_bgr = cv2.resize(bgr[y0:y0+ch, x0:x0+cw], (w, h))
        cropped_depth = cv2.resize(depth[y0:y0+ch, x0:x0+cw], (w, h), interpolation=cv2.INTER_NEAREST)

        result2 = match_and_estimate(ref_data, cropped_bgr, cropped_depth, fx, fy, cx, cy, cfg)
        print(f"  --- simulated retreat (crop {crop_ratio}) ---")
        print(f"  level:        {result2.level}")
        print(f"  matched:      {result2.matched_count}")
        print(f"  inlier:       {result2.inlier_count}")
        print(f"  yaw_err:      {result2.yaw_error_deg:.3f}°")
        print(f"  forward_err:  {result2.forward_error_m:.4f} m")
        print(f"  lateral_err:  {result2.lateral_error_m:.4f} m")
        print(f"  confidence:   {result2.confidence:.4f}")
        print(f"  reason:       {result2.reason}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
