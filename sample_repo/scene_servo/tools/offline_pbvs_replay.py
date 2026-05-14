#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from scene_servo.feature_matcher import FeatureMatcherCfg, match_and_estimate
from scene_servo.scene_template_store import load_template, template_to_all_ref_data
from scene_servo.servo_estimator import ServoEstimator, ServoEstimatorCfg, estimate_best_keyframe


def find_latest_session(repo_root: Path) -> Path:
    root = repo_root / "data" / "pbvs_eval"
    sessions = [p for p in root.glob("*/*") if p.is_dir()]
    if not sessions:
        raise FileNotFoundError(f"no session dirs found under {root}")
    return sorted(sessions, key=lambda p: p.stat().st_mtime)[-1]


def collect_test_pairs(session_dir: Path) -> Iterable[tuple[str, Path, Path]]:
    for img_path in sorted(session_dir.glob("*.jpg")):
        stem = img_path.stem
        if stem == "ref":
            continue
        depth_path = session_dir / f"{stem}_depth.npy"
        if depth_path.exists():
            yield stem, img_path, depth_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Offline PBVS replay from recorded RGB-D pairs")
    ap.add_argument("--session-dir", default="", help="session dir, default: latest under ./data/pbvs_eval/YYYYMMDD/HHMMSS")
    ap.add_argument("--template", default="", help="reference_frame json path, default: <session-dir>/reference_frame.json or pbvs_test_template.json")
    ap.add_argument("--detector", default="orb", choices=["orb", "sift"])
    ap.add_argument("--max-keypoints", type=int, default=500)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    session_dir = Path(args.session_dir) if args.session_dir else find_latest_session(repo_root)
    if args.template:
        template_path = Path(args.template)
    else:
        preferred = session_dir / "reference_frame.json"
        fallback = session_dir / "pbvs_test_template.json"
        template_path = preferred if preferred.exists() else fallback
    if not template_path.exists():
        raise FileNotFoundError(f"reference_frame not found: {template_path}")

    info_path = session_dir / "camera_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"camera_info.json not found in {session_dir}")
    camera_info = json.loads(info_path.read_text(encoding="utf-8"))

    matcher_cfg = FeatureMatcherCfg(detector=args.detector, max_keypoints=args.max_keypoints)
    if args.detector == "sift":
        matcher_cfg.sift_n_features = args.max_keypoints
    estimator = ServoEstimator(ServoEstimatorCfg())
    template = load_template(template_path)
    ref_data_list = template_to_all_ref_data(template)

    outputs = []
    for name, img_path, depth_path in collect_test_pairs(session_dir):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        depth = np.load(str(depth_path))
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * 0.001
        result = estimate_best_keyframe(
            ref_data_list,
            bgr,
            depth.astype(np.float32),
            fx=float(camera_info["fx"]),
            fy=float(camera_info["fy"]),
            cx=float(camera_info["cx"]),
            cy=float(camera_info["cy"]),
            cfg=matcher_cfg,
        )
        state = estimator.update(result)
        outputs.append({
            "sample": name,
            "level": result.level,
            "matched_count": result.matched_count,
            "inlier_count": result.inlier_count,
            "yaw_error_deg": state["yaw_error_deg"],
            "forward_error_m": state["forward_error_m"],
            "lateral_error_m": state["lateral_error_m"],
            "confidence": state["confidence"],
            "servo_mode": state["servo_mode"],
        })

    print(json.dumps({
        "session_dir": str(session_dir),
        "template": str(template_path),
        "results": outputs,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
