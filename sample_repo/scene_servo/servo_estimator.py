#!/usr/bin/env python3
"""servo_estimator.py — 从 feature_matcher 结果计算伺服控制误差

把相机系的 [R|t] 转换成机器人底盘坐标系下的控制量:
  - yaw_error_deg:    需要转的角度 (正=左转)
  - forward_error_m:  需要前进的距离 (正=前进)
  - lateral_error_m:  横向偏差 (正=左移)

并产出与 chassis_follow_node 兼容的 Float32MultiArray / PointStamped 格式。
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .feature_matcher import (
    FeatureMatcherCfg,
    MatchResult,
    match_and_estimate,
)


@dataclass
class ServoEstimatorCfg:
    """伺服估计器配置"""
    # 平滑
    ema_alpha: float = 0.35                # EMA 系数 (越小越平滑)
    # 状态门控
    min_confidence_head: float = 0.15      # head_tracking_ok 阈值
    min_confidence_base: float = 0.30      # base_servo_ready 阈值
    min_inlier_count_base: int = 12        # base 至少内点
    # 多关键帧：取最佳匹配
    multi_keyframe: bool = True
    # base servo ready 连续帧确认
    base_ready_window: int = 3
    base_ready_min_count: int = 2


class ServoEstimator:
    """有状态的伺服估计器，维护平滑和门控逻辑"""

    def __init__(self, cfg: ServoEstimatorCfg | None = None) -> None:
        self.cfg = cfg or ServoEstimatorCfg()
        self._smooth: dict[str, float] = {}
        self._ready_history: deque[bool] = deque(maxlen=self.cfg.base_ready_window)
        self._last_result: MatchResult | None = None

    def _ema(self, key: str, value: float) -> float:
        prev = self._smooth.get(key)
        if prev is None:
            self._smooth[key] = value
        else:
            self._smooth[key] = prev + self.cfg.ema_alpha * (value - prev)
        return self._smooth[key]

    def reset(self) -> None:
        self._smooth.clear()
        self._ready_history.clear()
        self._last_result = None

    def update(self, result: MatchResult) -> dict[str, Any]:
        """输入原始 MatchResult，输出平滑后的伺服状态 dict"""
        self._last_result = result

        if result.level == "LOST":
            self._ready_history.append(False)
            # Don't clear EMA — preserve history so recovery is instant
            return {
                "tracking_ok": False,
                "head_tracking_ok": False,
                "base_servo_ready": False,
                "servo_mode": "search",
                "yaw_error_deg": 0.0,
                "forward_error_m": 0.0,
                "lateral_error_m": 0.0,
                "confidence": 0.0,
                "level": result.level,
                "matched_count": result.matched_count,
                "inlier_count": result.inlier_count,
                "reason": result.reason,
            }

        # L2 (pure 2D homography) yaw is unreliable — skip EMA update for yaw,
        # use last smoothed yaw if available; always update fwd/lat/conf.
        if result.level == "L2" and "yaw" in self._smooth:
            yaw = self._smooth["yaw"]  # freeze yaw on L2
        else:
            yaw = self._ema("yaw", result.yaw_error_deg)
        fwd = self._ema("fwd", result.forward_error_m)
        lat = self._ema("lat", result.lateral_error_m)
        conf = self._ema("conf", result.confidence)

        head_ok = conf >= self.cfg.min_confidence_head
        base_raw = (
            conf >= self.cfg.min_confidence_base
            and result.inlier_count >= self.cfg.min_inlier_count_base
            and result.level in ("L0", "L1")
        )
        self._ready_history.append(base_raw)
        base_ok = sum(1 for x in self._ready_history if x) >= self.cfg.base_ready_min_count

        if base_ok:
            mode = "base_servo"
        elif head_ok:
            mode = "head_track_only"
        else:
            mode = "search"

        return {
            "tracking_ok": True,
            "head_tracking_ok": head_ok,
            "base_servo_ready": base_ok,
            "servo_mode": mode,
            "yaw_error_deg": round(yaw, 3),
            "forward_error_m": round(fwd, 4),
            "lateral_error_m": round(lat, 4),
            "confidence": round(conf, 4),
            "level": result.level,
            "matched_count": result.matched_count,
            "inlier_count": result.inlier_count,
            "reason": result.reason,
            "rotation_ref_to_cur": result.rotation_ref_to_cur,
            "translation_ref_to_cur": result.translation_ref_to_cur,
        }


def estimate_best_keyframe(
    ref_data_list: list[dict[str, Any]],
    cur_bgr: np.ndarray,
    cur_depth_m: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    cfg: FeatureMatcherCfg | None = None,
) -> MatchResult:
    """对多个关键帧分别匹配，返回最佳结果"""
    cfg = cfg or FeatureMatcherCfg()
    best: MatchResult | None = None
    for ref_data in ref_data_list:
        r = match_and_estimate(ref_data, cur_bgr, cur_depth_m, fx, fy, cx, cy, cfg)
        if best is None:
            best = r
            continue
        # 优先级: level 更高 > inlier 更多 > confidence 更高
        level_order = {"L0": 3, "L1": 2, "L2": 1, "LOST": 0}
        if level_order.get(r.level, 0) > level_order.get(best.level, 0):
            best = r
        elif r.level == best.level and r.inlier_count > best.inlier_count:
            best = r
    return best if best is not None else MatchResult()
