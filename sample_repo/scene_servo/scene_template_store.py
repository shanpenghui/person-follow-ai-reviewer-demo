#!/usr/bin/env python3
"""scene_template_store.py — 场景模板的序列化/反序列化

模板格式 (JSON):
{
    "version": 2,
    "scene_name": "...",
    "camera_intrinsics": {"fx": ..., "fy": ..., "cx": ..., "cy": ...},
    "image_resolution": [W, H],
    "detector": "orb",
    "keyframes": [
        {
            "keyframe_id": 0,
            "image_path": "...",
            "depth_path": "...",
            "keypoints_uv": [[u,v], ...],
            "descriptors_b64": "base64...",
            "descriptors_dtype": "uint8",
            "descriptors_shape": [N, 32],
            "xyz_cam": [[x,y,z], ...],
            "xyz_mask": [true, false, ...],
            "depth_vals": [1.23, 0.0, ...],
        },
        ...
    ]
}
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .feature_matcher import FeatureMatcherCfg, extract_keypoints_3d


def _encode_ndarray(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _decode_ndarray(b64: str, dtype: str, shape: list[int]) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()


def build_template(
    scene_name: str,
    images_bgr: list[np.ndarray],
    depths_m: list[np.ndarray],
    fx: float, fy: float, cx: float, cy: float,
    cfg: FeatureMatcherCfg | None = None,
    image_paths: list[str] | None = None,
    depth_paths: list[str] | None = None,
) -> dict[str, Any]:
    """从一组 RGB-D 关键帧构建场景模板 (dict, 可直接 json.dumps)"""
    cfg = cfg or FeatureMatcherCfg()
    h, w = images_bgr[0].shape[:2]
    keyframes: list[dict[str, Any]] = []

    for i, (bgr, depth) in enumerate(zip(images_bgr, depths_m)):
        data = extract_keypoints_3d(bgr, depth, fx, fy, cx, cy, cfg)
        kf: dict[str, Any] = {
            "keyframe_id": i,
            "image_path": (image_paths[i] if image_paths else f"keyframe_{i:03d}.jpg"),
            "depth_path": (depth_paths[i] if depth_paths else f"keyframe_{i:03d}_depth.npy"),
        }
        if data["descriptors"] is not None:
            desc = data["descriptors"]
            kf["keypoints_uv"] = data["uv"].tolist()
            kf["descriptors_b64"] = _encode_ndarray(desc)
            kf["descriptors_dtype"] = str(desc.dtype)
            kf["descriptors_shape"] = list(desc.shape)
            kf["xyz_cam"] = data["xyz_cam"].tolist()
            kf["xyz_mask"] = data["xyz_mask"].tolist()
            kf["depth_vals"] = data["depth_vals"].tolist()
            kf["keypoint_count"] = len(data["keypoints"])
            kf["keypoint_3d_count"] = int(data["xyz_mask"].sum())
        else:
            kf["keypoint_count"] = 0
            kf["keypoint_3d_count"] = 0

        keyframes.append(kf)

    return {
        "version": 2,
        "scene_name": scene_name,
        "camera_intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "image_resolution": [w, h],
        "detector": cfg.detector,
        "max_keypoints": cfg.max_keypoints,
        "keyframes": keyframes,
    }


def save_template(template: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")


def load_template(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def template_to_ref_data(template: dict[str, Any], keyframe_id: int = 0) -> dict[str, Any]:
    """把模板里的某个关键帧还原成 feature_matcher.match_and_estimate() 需要的 ref_data 格式"""
    kf = None
    for k in template["keyframes"]:
        if k["keyframe_id"] == keyframe_id:
            kf = k
            break
    if kf is None:
        raise ValueError(f"keyframe_id={keyframe_id} not found in template")

    if kf.get("keypoint_count", 0) == 0:
        return {
            "keypoints": [],
            "descriptors": None,
            "uv": np.zeros((0, 2), dtype=np.float32),
            "xyz_cam": np.zeros((0, 3), dtype=np.float64),
            "xyz_mask": np.zeros(0, dtype=bool),
            "depth_vals": np.zeros(0, dtype=np.float32),
        }

    desc = _decode_ndarray(kf["descriptors_b64"], kf["descriptors_dtype"], kf["descriptors_shape"])
    uv = np.array(kf["keypoints_uv"], dtype=np.float32)
    xyz_cam = np.array(kf["xyz_cam"], dtype=np.float64)
    xyz_mask = np.array(kf["xyz_mask"], dtype=bool)
    depth_vals = np.array(kf["depth_vals"], dtype=np.float32)

    # 重建 cv2.KeyPoint（只需要 pt 和 size）
    kps = [cv2.KeyPoint(x=float(uv[i, 0]), y=float(uv[i, 1]), size=7.0) for i in range(len(uv))]

    return {
        "keypoints": kps,
        "descriptors": desc,
        "uv": uv,
        "xyz_cam": xyz_cam,
        "xyz_mask": xyz_mask,
        "depth_vals": depth_vals,
    }


def template_to_all_ref_data(template: dict[str, Any]) -> list[dict[str, Any]]:
    """把模板中所有关键帧都还原"""
    return [template_to_ref_data(template, kf["keyframe_id"]) for kf in template["keyframes"]]
