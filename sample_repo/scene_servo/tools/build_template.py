#!/usr/bin/env python3
"""build_template_from_rgbd.py — 从 RGB-D 录制生成特征点场景模板

用法:
  # 从已有的 memory_index.json 构建
  python3 -m scene_servo.tools.build_template \
      --memory-dir /path/to/scene_memory \
      --out template_v2.json

  # 从单张 RGB + depth.npy 构建
  python3 -m scene_servo.tools.build_template \
      --image ref.jpg --depth ref_depth.npy \
      --fx 382.6 --fy 382.6 --cx 319.5 --cy 238.5 \
      --scene-name "test_scene" \
      --out template_v2.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# 允许作为 -m scene_servo.tools.build_template 运行
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_servo.feature_matcher import FeatureMatcherCfg
from scene_servo.scene_template_store import build_template, save_template


def main() -> None:
    ap = argparse.ArgumentParser(description="从 RGB-D 构建特征点场景模板 v2")
    ap.add_argument("--memory-dir", default="", help="包含 memory_index.json 的目录")
    ap.add_argument("--image", default="", help="单张参考图")
    ap.add_argument("--depth", default="", help="单张 depth .npy")
    ap.add_argument("--fx", type=float, default=0.0)
    ap.add_argument("--fy", type=float, default=0.0)
    ap.add_argument("--cx", type=float, default=0.0)
    ap.add_argument("--cy", type=float, default=0.0)
    ap.add_argument("--scene-name", default="scene")
    ap.add_argument("--detector", default="orb", choices=["orb", "sift"])
    ap.add_argument("--max-keypoints", type=int, default=500)
    ap.add_argument("--out", required=True, help="输出模板 JSON 路径")
    args = ap.parse_args()

    cfg = FeatureMatcherCfg(detector=args.detector, max_keypoints=args.max_keypoints)
    if args.detector == "sift":
        cfg.sift_n_features = args.max_keypoints

    images: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    img_paths: list[str] = []
    dep_paths: list[str] = []
    fx = fy = cx = cy = 0.0
    scene_name = args.scene_name

    if args.memory_dir:
        mem = Path(args.memory_dir)
        idx_path = mem / "memory_index.json"
        if not idx_path.exists():
            print(f"[ERROR] memory_index.json not found in {mem}")
            sys.exit(1)
        index = json.loads(idx_path.read_text(encoding="utf-8"))
        scenes = index.get("scenes", [])
        if not scenes:
            print("[ERROR] no scenes in memory_index.json")
            sys.exit(1)
        for sc in scenes:
            img_path = mem / sc["image"]
            dep_path = mem / sc["depth"]
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"[WARN] skip {img_path}: failed to read")
                continue
            depth = np.load(str(dep_path))
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) * 0.001
            images.append(bgr)
            depths.append(depth.astype(np.float32))
            img_paths.append(str(sc["image"]))
            dep_paths.append(str(sc["depth"]))
        scene_name = scenes[0].get("name", args.scene_name)
        # 尝试从 camera_info 获取内参
        info_path = mem / "camera_info.json"
        if info_path.exists():
            ci = json.loads(info_path.read_text(encoding="utf-8"))
            fx = ci.get("fx", 0.0)
            fy = ci.get("fy", 0.0)
            cx = ci.get("cx", 0.0)
            cy = ci.get("cy", 0.0)

    elif args.image and args.depth:
        bgr = cv2.imread(args.image)
        if bgr is None:
            print(f"[ERROR] failed to read image: {args.image}")
            sys.exit(1)
        depth = np.load(args.depth)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * 0.001
        images.append(bgr)
        depths.append(depth.astype(np.float32))
        img_paths.append(args.image)
        dep_paths.append(args.depth)
        image_path = Path(args.image)
        info_path = image_path.parent / "camera_info.json"
        if info_path.exists():
            ci = json.loads(info_path.read_text(encoding="utf-8"))
            fx = ci.get("fx", 0.0)
            fy = ci.get("fy", 0.0)
            cx = ci.get("cx", 0.0)
            cy = ci.get("cy", 0.0)
    else:
        print("[ERROR] provide --memory-dir or --image + --depth")
        sys.exit(1)

    # 命令行内参覆盖
    if args.fx > 0:
        fx = args.fx
    if args.fy > 0:
        fy = args.fy
    if args.cx > 0:
        cx = args.cx
    if args.cy > 0:
        cy = args.cy

    # 默认 D455 内参
    if fx <= 0:
        h, w = images[0].shape[:2]
        fx = fy = 382.6
        cx = w / 2.0
        cy = h / 2.0
        print(f"[WARN] no camera intrinsics provided, using D455 defaults: fx={fx} fy={fy} cx={cx} cy={cy}")

    print(f"[INFO] building template: scene_name={scene_name} keyframes={len(images)} detector={cfg.detector} max_kp={cfg.max_keypoints}")
    template = build_template(
        scene_name=scene_name,
        images_bgr=images,
        depths_m=depths,
        fx=fx, fy=fy, cx=cx, cy=cy,
        cfg=cfg,
        image_paths=img_paths,
        depth_paths=dep_paths,
    )

    save_template(template, args.out)
    for kf in template["keyframes"]:
        print(f"  keyframe {kf['keyframe_id']}: {kf['keypoint_count']} kp, {kf['keypoint_3d_count']} with 3D")
    print(f"[OK] saved to {args.out}")


if __name__ == "__main__":
    main()
