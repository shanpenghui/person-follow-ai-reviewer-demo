#!/usr/bin/env python3
"""feature_matcher.py — 全图特征点 + 深度融合 3D 配准

核心函数:
  extract_keypoints_3d()  — 从 RGB-D 提取特征点并反投影到 3D
  match_and_estimate()    — 当前帧 vs 参考模板，输出 6-DOF 位姿误差

降级链:
  L0  3D-3D ≥ min_3d_pairs  → SVD (Umeyama) 刚体配准
  L1  3D+2D 混合 ≥ 8        → solvePnP (参考3D + 当前2D)
  L2  2D-2D ≥ 8             → Homography 分解 → yaw only
  L3  匹配点 < 8            → LOST
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class FeatureMatcherCfg:
    """配置参数，可序列化到 yaml"""
    # 特征提取
    detector: str = "orb"             # "orb" | "sift"
    max_keypoints: int = 500
    # ORB specific
    orb_scale_factor: float = 1.2
    orb_n_levels: int = 8
    orb_fast_threshold: int = 20
    # SIFT specific
    sift_n_features: int = 500
    sift_contrast_threshold: float = 0.04
    sift_edge_threshold: float = 10.0

    # 匹配
    match_ratio_test: float = 0.75    # Lowe's ratio test
    ransac_reproj_th: float = 5.0     # RANSAC 重投影阈值 (px)
    ransac_3d_inlier_th_m: float = 0.08  # 3D-3D RANSAC 内点阈值 (m)
    ransac_confidence: float = 0.995
    ransac_max_iters: int = 2000

    # 深度
    depth_min_m: float = 0.1
    depth_max_m: float = 8.0
    depth_patch_half: int = 4         # 取特征点周围 (2*n+1)^2 区域的中值深度

    # 配准
    min_3d_pairs: int = 6             # L0 SVD 最少 3D-3D 点对
    min_pnp_pairs: int = 6            # L1 PnP 最少点对
    min_2d_pairs: int = 6             # L2 Homography 最少点对
    max_translation_m: float = 5.0    # 过滤明显错误的配准结果


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------
def _create_detector(cfg: FeatureMatcherCfg):
    """创建 OpenCV 特征检测器"""
    if cfg.detector == "sift":
        return cv2.SIFT_create(
            nfeatures=cfg.sift_n_features,
            contrastThreshold=cfg.sift_contrast_threshold,
            edgeThreshold=cfg.sift_edge_threshold,
        )
    # default ORB
    return cv2.ORB_create(
        nfeatures=cfg.max_keypoints,
        scaleFactor=cfg.orb_scale_factor,
        nlevels=cfg.orb_n_levels,
        fastThreshold=cfg.orb_fast_threshold,
    )


def _sample_depth(depth_m: np.ndarray, u: float, v: float,
                   half: int, d_min: float, d_max: float) -> float | None:
    """在 (u,v) 周围取 patch 中值深度，返回 None 表示无效"""
    h, w = depth_m.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    y0 = max(0, vi - half)
    y1 = min(h, vi + half + 1)
    x0 = max(0, ui - half)
    x1 = min(w, ui + half + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[(patch > d_min) & (patch < d_max) & np.isfinite(patch)]
    if valid.size < 1:
        return None
    return float(np.median(valid))


def _backproject(u: float, v: float, depth_m: float,
                  fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """像素 + 深度 → 相机系 3D 坐标"""
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float64)


def extract_keypoints_3d(
    bgr: np.ndarray,
    depth_m: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    cfg: FeatureMatcherCfg | None = None,
) -> dict[str, Any]:
    """从 RGB-D 提取全图特征点，返回 2D kp + descriptors + 3D 坐标

    Returns:
        {
            "keypoints": list[cv2.KeyPoint],
            "descriptors": np.ndarray | None,       # shape (N, D)
            "uv": np.ndarray,                        # shape (N, 2) float32
            "xyz_cam": np.ndarray | None,            # shape (M, 3) float64, 有深度的子集
            "xyz_mask": np.ndarray,                  # shape (N,) bool, 哪些有 3D
            "depth_vals": np.ndarray,                # shape (N,) float32, 深度值 (无效为 0)
        }
    """
    cfg = cfg or FeatureMatcherCfg()
    detector = _create_detector(cfg)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kps, descs = detector.detectAndCompute(gray, None)

    if kps is None or len(kps) == 0 or descs is None:
        n = 0
        return {
            "keypoints": [],
            "descriptors": None,
            "uv": np.zeros((0, 2), dtype=np.float32),
            "xyz_cam": np.zeros((0, 3), dtype=np.float64),
            "xyz_mask": np.zeros(0, dtype=bool),
            "depth_vals": np.zeros(0, dtype=np.float32),
        }

    n = len(kps)
    uv = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
    depth_vals = np.zeros(n, dtype=np.float32)
    xyz_mask = np.zeros(n, dtype=bool)
    xyz_list: list[np.ndarray] = []

    for i in range(n):
        d = _sample_depth(depth_m, uv[i, 0], uv[i, 1],
                          cfg.depth_patch_half, cfg.depth_min_m, cfg.depth_max_m)
        if d is not None:
            depth_vals[i] = d
            xyz_mask[i] = True
            xyz_list.append(_backproject(uv[i, 0], uv[i, 1], d, fx, fy, cx, cy))

    xyz_cam = np.array(xyz_list, dtype=np.float64) if xyz_list else np.zeros((0, 3), dtype=np.float64)

    return {
        "keypoints": kps,
        "descriptors": descs,
        "uv": uv,
        "xyz_cam": xyz_cam,
        "xyz_mask": xyz_mask,
        "depth_vals": depth_vals,
    }


# ---------------------------------------------------------------------------
# 描述子匹配
# ---------------------------------------------------------------------------
def _match_descriptors(
    desc_ref: np.ndarray,
    desc_cur: np.ndarray,
    detector_type: str,
    ratio_test: float,
) -> list[tuple[int, int]]:
    """返回 (ref_idx, cur_idx) 匹配对列表"""
    if desc_ref is None or desc_cur is None:
        return []
    if len(desc_ref) < 2 or len(desc_cur) < 2:
        return []

    if detector_type == "sift":
        # SIFT: float descriptors → FLANN
        index_params = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        # ORB: binary descriptors → BF Hamming
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    try:
        raw_matches = matcher.knnMatch(desc_ref, desc_cur, k=2)
    except cv2.error:
        return []

    good: list[tuple[int, int]] = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            good.append((m.queryIdx, m.trainIdx))
    return good


# ---------------------------------------------------------------------------
# 3D-3D 刚体配准 (Umeyama / SVD)
# ---------------------------------------------------------------------------
def _estimate_rigid_svd(
    pts_ref: np.ndarray,    # (N, 3)
    pts_cur: np.ndarray,    # (N, 3)
) -> tuple[np.ndarray, np.ndarray]:
    """Umeyama alignment (无尺度): R, t  使得 pts_cur ≈ R @ pts_ref + t

    Returns: R (3x3), t (3,)
    """
    assert pts_ref.shape == pts_cur.shape and pts_ref.shape[0] >= 3
    centroid_ref = pts_ref.mean(axis=0)
    centroid_cur = pts_cur.mean(axis=0)
    ref_c = pts_ref - centroid_ref
    cur_c = pts_cur - centroid_cur
    H = ref_c.T @ cur_c  # (3, 3)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = centroid_cur - R @ centroid_ref
    return R, t


def _ransac_rigid_3d(
    pts_ref: np.ndarray,
    pts_cur: np.ndarray,
    max_iters: int = 1000,
    inlier_th_m: float = 0.05,
    confidence: float = 0.995,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC 3D-3D 刚体配准

    Returns: R (3x3), t (3,), inlier_mask (N,) bool
    """
    n = len(pts_ref)
    best_inliers = np.zeros(n, dtype=bool)
    best_R = np.eye(3)
    best_t = np.zeros(3)
    best_count = 0

    for _ in range(max_iters):
        # 最少 3 个点
        idx = np.random.choice(n, 3, replace=False)
        try:
            R, t = _estimate_rigid_svd(pts_ref[idx], pts_cur[idx])
        except Exception:
            continue
        # 计算所有点的残差
        transformed = (R @ pts_ref.T).T + t
        residuals = np.linalg.norm(transformed - pts_cur, axis=1)
        inliers = residuals < inlier_th_m
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_R = R
            best_t = t
            # 自适应迭代次数
            inlier_ratio = count / n
            if inlier_ratio > 0.1:
                needed = math.log(1.0 - confidence) / math.log(1.0 - inlier_ratio ** 3 + 1e-12)
                if _ > needed:
                    break

    # 用所有 inlier 重新估计
    if best_count >= 3:
        try:
            best_R, best_t = _estimate_rigid_svd(pts_ref[best_inliers], pts_cur[best_inliers])
        except Exception:
            pass

    return best_R, best_t, best_inliers


# ---------------------------------------------------------------------------
# 从 R 分解欧拉角
# ---------------------------------------------------------------------------
def rotation_to_euler_deg(R: np.ndarray) -> dict[str, float]:
    """从旋转矩阵提取 yaw/pitch/roll (度)，ZYX 顺序"""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return {
        "yaw_deg": math.degrees(yaw),
        "pitch_deg": math.degrees(pitch),
        "roll_deg": math.degrees(roll),
    }


# ---------------------------------------------------------------------------
# 主接口: match_and_estimate
# ---------------------------------------------------------------------------
@dataclass
class MatchResult:
    """匹配与配准结果"""
    level: str = "LOST"               # "L0" | "L1" | "L2" | "LOST"
    yaw_error_deg: float = 0.0        # 水平角偏差 (正=目标在左，需左转)
    pitch_error_deg: float = 0.0
    roll_error_deg: float = 0.0
    forward_error_m: float = 0.0      # 前后偏差 (正=目标更远，需前进)
    lateral_error_m: float = 0.0      # 左右偏差 (正=目标在左)
    vertical_error_m: float = 0.0
    confidence: float = 0.0           # 0~1
    matched_count: int = 0            # 匹配点对数
    inlier_count: int = 0             # RANSAC 内点数
    total_ref_kp: int = 0
    total_cur_kp: int = 0
    ref_3d_count: int = 0
    cur_3d_count: int = 0
    rotation_ref_to_cur: Any = None
    translation_ref_to_cur: Any = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def match_and_estimate(
    ref_data: dict[str, Any],
    cur_bgr: np.ndarray,
    cur_depth_m: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    cfg: FeatureMatcherCfg | None = None,
) -> MatchResult:
    """核心接口: 参考模板 vs 当前帧 → 6-DOF 误差

    Args:
        ref_data: extract_keypoints_3d() 的输出 (参考帧，离线构建)
        cur_bgr: 当前帧 BGR
        cur_depth_m: 当前帧对齐深度 (float32, 米)
        fx, fy, cx, cy: 相机内参
        cfg: 配置

    Returns:
        MatchResult
    """
    cfg = cfg or FeatureMatcherCfg()
    result = MatchResult()

    # 1. 提取当前帧特征
    cur_data = extract_keypoints_3d(cur_bgr, cur_depth_m, fx, fy, cx, cy, cfg)
    result.total_ref_kp = len(ref_data.get("keypoints", []))
    result.total_cur_kp = len(cur_data["keypoints"])
    result.ref_3d_count = int(ref_data.get("xyz_mask", np.zeros(0)).sum())
    result.cur_3d_count = int(cur_data["xyz_mask"].sum())

    if result.total_ref_kp < 4 or result.total_cur_kp < 4:
        result.reason = f"too few keypoints: ref={result.total_ref_kp} cur={result.total_cur_kp}"
        return result

    # 2. 描述子匹配
    matches = _match_descriptors(
        ref_data["descriptors"], cur_data["descriptors"],
        cfg.detector, cfg.match_ratio_test,
    )
    result.matched_count = len(matches)

    if len(matches) < cfg.min_2d_pairs:
        result.reason = f"too few matches: {len(matches)}"
        return result

    ref_idx = np.array([m[0] for m in matches])
    cur_idx = np.array([m[1] for m in matches])

    # 3. 分类: 哪些匹配对有双 3D
    ref_mask = ref_data["xyz_mask"]
    cur_mask = cur_data["xyz_mask"]
    both_3d = ref_mask[ref_idx] & cur_mask[cur_idx]
    n_3d_pairs = int(both_3d.sum())

    # 构建 3D 点对 (用在有 3D 的子集上)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # --- L0: 3D-3D SVD ---
    if n_3d_pairs >= cfg.min_3d_pairs:
        # 获取 ref 和 cur 的 3D 坐标（按匹配顺序）
        ref_3d_all = ref_data["xyz_cam"]  # shape (M_ref, 3)
        cur_3d_all = cur_data["xyz_cam"]  # shape (M_cur, 3)

        # 建立从原始 kp index → xyz_cam 索引的映射
        ref_3d_idx = np.cumsum(ref_mask) - 1  # ref_mask[i]=True 的累积计数 - 1
        cur_3d_idx = np.cumsum(cur_mask) - 1

        pairs_3d_idx = np.where(both_3d)[0]
        pts_ref = np.array([ref_3d_all[ref_3d_idx[ref_idx[i]]] for i in pairs_3d_idx])
        pts_cur = np.array([cur_3d_all[cur_3d_idx[cur_idx[i]]] for i in pairs_3d_idx])

        R, t, inlier_mask = _ransac_rigid_3d(
            pts_ref, pts_cur,
            max_iters=cfg.ransac_max_iters,
            inlier_th_m=cfg.ransac_3d_inlier_th_m,
            confidence=cfg.ransac_confidence,
        )
        inlier_count = int(inlier_mask.sum())

        if inlier_count >= cfg.min_3d_pairs and np.linalg.norm(t) < cfg.max_translation_m:
            euler = rotation_to_euler_deg(R)
            result.level = "L0"
            result.yaw_error_deg = euler["yaw_deg"]
            result.pitch_error_deg = euler["pitch_deg"]
            result.roll_error_deg = euler["roll_deg"]
            result.forward_error_m = float(t[2])     # Z = forward
            result.lateral_error_m = float(t[0])      # X = right (相机系)
            result.vertical_error_m = float(t[1])     # Y = down (相机系)
            result.rotation_ref_to_cur = R.tolist()
            result.translation_ref_to_cur = t.tolist()
            result.inlier_count = inlier_count
            result.confidence = min(1.0, inlier_count / max(20.0, n_3d_pairs * 0.5))
            return result

    # --- L1: PnP (参考3D + 当前2D) ---
    ref_has_3d = ref_mask[ref_idx]
    n_pnp = int(ref_has_3d.sum())
    if n_pnp >= cfg.min_pnp_pairs:
        ref_3d_all = ref_data["xyz_cam"]
        ref_3d_idx = np.cumsum(ref_mask) - 1

        pnp_idx = np.where(ref_has_3d)[0]
        obj_pts = np.array([ref_3d_all[ref_3d_idx[ref_idx[i]]] for i in pnp_idx], dtype=np.float64)
        img_pts = cur_data["uv"][cur_idx[pnp_idx]].reshape(-1, 1, 2).astype(np.float64)

        success, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
            obj_pts, img_pts, K, None,
            reprojectionError=cfg.ransac_reproj_th,
            confidence=cfg.ransac_confidence,
            iterationsCount=cfg.ransac_max_iters,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success and inliers_pnp is not None:
            inlier_count = len(inliers_pnp)
            if inlier_count >= cfg.min_pnp_pairs:
                R_pnp, _ = cv2.Rodrigues(rvec)
                t_pnp = tvec.flatten()
                if np.linalg.norm(t_pnp) < cfg.max_translation_m:
                    euler = rotation_to_euler_deg(R_pnp)
                    result.level = "L1"
                    result.yaw_error_deg = euler["yaw_deg"]
                    result.pitch_error_deg = euler["pitch_deg"]
                    result.roll_error_deg = euler["roll_deg"]
                    result.forward_error_m = float(t_pnp[2])
                    result.lateral_error_m = float(t_pnp[0])
                    result.vertical_error_m = float(t_pnp[1])
                    result.rotation_ref_to_cur = R_pnp.tolist()
                    result.translation_ref_to_cur = t_pnp.tolist()
                    result.inlier_count = inlier_count
                    result.confidence = min(1.0, inlier_count / max(15.0, n_pnp * 0.5))
                    return result

    # --- L2: Homography (2D-2D) → yaw only ---
    ref_pts_2d = ref_data["uv"][ref_idx].reshape(-1, 1, 2)
    cur_pts_2d = cur_data["uv"][cur_idx].reshape(-1, 1, 2)

    if len(matches) >= cfg.min_2d_pairs:
        H, mask_h = cv2.findHomography(
            ref_pts_2d, cur_pts_2d,
            cv2.RANSAC,
            ransacReprojThreshold=cfg.ransac_reproj_th,
            maxIters=cfg.ransac_max_iters,
            confidence=cfg.ransac_confidence,
        )
        if H is not None and mask_h is not None:
            inlier_count = int(mask_h.sum())
            if inlier_count >= cfg.min_2d_pairs:
                # 用 decomposeHomographyMat 提取 yaw
                n_solutions, Rs, Ts, normals = cv2.decomposeHomographyMat(H, K)
                # 选最合理的解 (translation Z > 0, normal 接近 [0,0,1])
                best_yaw = 0.0
                best_score = -1.0
                for i in range(n_solutions):
                    normal_i = normals[i].flatten()
                    # 法向量应接近 (0, 0, 1)
                    score = abs(normal_i[2])
                    if Ts[i][2, 0] < 0:
                        score *= 0.1  # 惩罚 Z 为负
                    if score > best_score:
                        best_score = score
                        euler_i = rotation_to_euler_deg(Rs[i])
                        best_yaw = euler_i["yaw_deg"]

                result.level = "L2"
                result.yaw_error_deg = best_yaw
                result.inlier_count = inlier_count
                result.confidence = min(1.0, inlier_count / max(20.0, len(matches) * 0.5))
                result.reason = "homography only: no depth-based translation"
                return result

    result.reason = f"all levels failed: matches={len(matches)} 3d_pairs={n_3d_pairs}"
    return result
