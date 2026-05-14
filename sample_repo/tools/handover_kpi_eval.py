#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于离线 split 指标 + 当前控制参数，估算 handover 时延（丢失->缓停->重跟随）。

输出：
- d435 / d455 的 T_handover avg/p50/p95/min/max
- 复用最新离线 split 指标（MOT17-06/14）

说明：
- 这是工程估算器，不是实车闭环时间戳回放器。
- 若需要实测精确值，应在 ROS topic 中加事件时间戳并做bag回放统计。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def weighted_quantile(values: List[float], weights: List[float], q: float) -> float:
    idx = np.argsort(values)
    v = np.array(values, dtype=np.float64)[idx]
    w = np.array(weights, dtype=np.float64)[idx]
    c = np.cumsum(w)
    t = q * c[-1]
    return float(v[np.searchsorted(c, t, side='left')])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Estimate handover KPI from current params + offline split metrics')
    ap.add_argument('--split06', default='/tmp/phaseb_videos/mot17_06_eval_split_latest.json', type=str)
    ap.add_argument('--split14', default='/tmp/phaseb_videos/mot17_14_eval_split_latest.json', type=str)

    ap.add_argument('--control-hz', default=20.0, type=float)
    ap.add_argument('--lost-decel-vx', default=0.35, type=float)
    ap.add_argument('--lost-decel-wz', default=1.2, type=float)
    ap.add_argument('--confirm-frames-d435', default=3, type=int)
    ap.add_argument('--confirm-frames-d455', default=2, type=int)
    ap.add_argument('--invalid-immediate-lost', action='store_true', default=True)
    ap.add_argument('--target-valid-hold-sec', default=0.4, type=float)

    ap.add_argument('--output', default='/tmp/phaseb_videos/handover_kpi_eval_latest.json', type=str)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    j06 = json.loads(Path(args.split06).read_text(encoding='utf-8'))['target_lock']
    j14 = json.loads(Path(args.split14).read_text(encoding='utf-8'))['target_lock']

    # 代表性丢失瞬间速度样本（可按实车日志替换）
    speed_samples: List[Tuple[float, float, float]] = [
        # (weight, vx0[m/s], wz0[rad/s])
        (0.20, 0.08, 0.08),
        (0.30, 0.12, 0.12),
        (0.25, 0.15, 0.20),
        (0.15, 0.20, 0.30),
        (0.10, 0.25, 0.60),
    ]

    def t_stop95(v0: float, w0: float) -> float:
        base = max(
            abs(v0) / max(1e-6, args.lost_decel_vx),
            abs(w0) / max(1e-6, args.lost_decel_wz),
        )
        # 传输/控制余量
        return base + 0.10

    def t_reacq(confirm_frames: int) -> float:
        hold = 0.0 if args.invalid_immediate_lost else args.target_valid_hold_sec
        return float(confirm_frames) / max(1e-6, args.control_hz) + hold

    vals_d435: List[float] = []
    vals_d455: List[float] = []
    wts: List[float] = []

    for w, v0, wz0 in speed_samples:
        ts = t_stop95(v0, wz0)
        vals_d435.append(ts + t_reacq(args.confirm_frames_d435))
        vals_d455.append(ts + t_reacq(args.confirm_frames_d455))
        wts.append(w)

    result = {
        'definition': {
            'kpi': 'T_handover_sec',
            'formula': 'T_loss_to_stop95 + T_reacquire_confirm',
            'T_loss_to_stop95': 'max(|vx0|/lost_decel_vx, |wz0|/lost_decel_wz) + 0.10s',
            'T_reacquire_confirm': 'switch_confirm_frames / control_hz (+ hold_sec when invalid_immediate_lost=false)',
        },
        'config': {
            'control_hz': args.control_hz,
            'lost_decel_vx': args.lost_decel_vx,
            'lost_decel_wz': args.lost_decel_wz,
            'confirm_frames_d435': args.confirm_frames_d435,
            'confirm_frames_d455': args.confirm_frames_d455,
            'invalid_immediate_lost': args.invalid_immediate_lost,
            'target_valid_hold_sec': args.target_valid_hold_sec,
            'speed_samples': speed_samples,
        },
        'handover_estimate': {
            'd435': {
                'avg': round(float(np.average(vals_d435, weights=wts)), 3),
                'p50': round(weighted_quantile(vals_d435, wts, 0.50), 3),
                'p95': round(weighted_quantile(vals_d435, wts, 0.95), 3),
                'min': round(float(min(vals_d435)), 3),
                'max': round(float(max(vals_d435)), 3),
            },
            'd455': {
                'avg': round(float(np.average(vals_d455, weights=wts)), 3),
                'p50': round(weighted_quantile(vals_d455, wts, 0.50), 3),
                'p95': round(weighted_quantile(vals_d455, wts, 0.95), 3),
                'min': round(float(min(vals_d455)), 3),
                'max': round(float(max(vals_d455)), 3),
            },
        },
        'offline_split_metrics': {
            'mot17_06': {k: j06[k] for k in [
                'keep_ratio', 'wrong_ratio', 'invalid_ratio', 'switch_per_100f',
                'switch_present_per_100f', 'switch_absent_per_100f', 'switch_edge_per_100f',
            ]},
            'mot17_14': {k: j14[k] for k in [
                'keep_ratio', 'wrong_ratio', 'invalid_ratio', 'switch_per_100f',
                'switch_present_per_100f', 'switch_absent_per_100f', 'switch_edge_per_100f',
            ]},
        },
    }

    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n[OK] saved => {args.output}")


if __name__ == '__main__':
    main()
