#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-B 离线评测：多目标下“原目标保持”能力

特点：
- 纯离线，不依赖 ROS/相机/机器人硬件
- 可跑合成场景（cross / short_occlusion / long_occlusion）
- 可读取外部 JSON 序列评测
- 输出关键指标：keep_ratio / wrong_ratio / invalid_ratio / switch_count / reacquire_delay

JSON 输入格式（--mode json）：
[
  {
    "t": 0.0,
    "target_id": "A",
    "detections": [
      {"id": "A", "bbox": [x1, y1, x2, y2]},
      {"id": "B", "bbox": [x1, y1, x2, y2]}
    ]
  },
  ...
]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


BBox = np.ndarray


@dataclass
class SelectorParams:
    enable_target_lock: bool = True
    target_lock_timeout_sec: float = 1.2
    target_lock_iou_weight: float = 0.55
    target_lock_center_weight: float = 0.30
    target_lock_area_weight: float = 0.15
    target_lock_min_score: float = 0.30
    target_lock_hold_on_low_score: bool = True
    target_lock_center_gate: float = 0.35
    target_lock_area_ratio_min: float = 0.45
    target_lock_area_ratio_max: float = 2.20


class TargetLockSelector:
    def __init__(self, params: SelectorParams):
        self.p = params
        self.locked_bbox: Optional[BBox] = None
        self.locked_time: float = 0.0

    @staticmethod
    def _clamp(v: float, a: float, b: float) -> float:
        return max(a, min(b, v))

    @staticmethod
    def _bbox_area(b: BBox) -> float:
        return max(1.0, float((b[2] - b[0]) * (b[3] - b[1])))

    @staticmethod
    def _bbox_iou(a: BBox, b: BBox) -> float:
        xx1 = max(float(a[0]), float(b[0]))
        yy1 = max(float(a[1]), float(b[1]))
        xx2 = min(float(a[2]), float(b[2]))
        yy2 = min(float(a[3]), float(b[3]))
        iw = max(0.0, xx2 - xx1)
        ih = max(0.0, yy2 - yy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        ua = TargetLockSelector._bbox_area(a) + TargetLockSelector._bbox_area(b) - inter
        return float(inter / max(1e-6, ua))

    @staticmethod
    def _bbox_center_norm_xy(b: BBox, w: float, h: float) -> Tuple[float, float]:
        cx = 0.5 * float(b[0] + b[2])
        cy = 0.5 * float(b[1] + b[3])
        return cx / max(1.0, w), cy / max(1.0, h)

    def select(self, boxes: np.ndarray, w_img: float, h_img: float, now: float) -> Optional[int]:
        if boxes.shape[0] <= 0:
            return None

        p = self.p
        if (not p.enable_target_lock) or self.locked_bbox is None or (now - self.locked_time) > p.target_lock_timeout_sec:
            best_i = int(np.argmax([(self._bbox_area(b)) for b in boxes]))
            self.locked_bbox = boxes[best_i].astype(np.float32)
            self.locked_time = now
            return best_i

        lock = self.locked_bbox.astype(np.float32)
        lock_cx, lock_cy = self._bbox_center_norm_xy(lock, w_img, h_img)
        lock_area = self._bbox_area(lock)

        gate = max(1e-3, p.target_lock_center_gate)
        best_i = -1
        best_score = -1e9

        for i in range(boxes.shape[0]):
            b = boxes[i].astype(np.float32)
            iou = self._bbox_iou(lock, b)

            cx, cy = self._bbox_center_norm_xy(b, w_img, h_img)
            center_dist = math.hypot(cx - lock_cx, cy - lock_cy)
            center_score = self._clamp(1.0 - center_dist / gate, 0.0, 1.0)

            area = self._bbox_area(b)
            area_ratio = area / max(1.0, lock_area)
            if p.target_lock_area_ratio_min <= area_ratio <= p.target_lock_area_ratio_max:
                area_score = self._clamp(1.0 - abs(math.log(max(1e-6, area_ratio))) / math.log(2.5), 0.0, 1.0)
            else:
                area_score = 0.0

            score = (
                p.target_lock_iou_weight * iou
                + p.target_lock_center_weight * center_score
                + p.target_lock_area_weight * area_score
            )
            if center_dist > gate:
                score *= 0.35

            if score > best_score:
                best_score = score
                best_i = i

        if best_i < 0:
            best_i = int(np.argmax([(self._bbox_area(b)) for b in boxes]))

        if best_score < p.target_lock_min_score:
            if p.target_lock_hold_on_low_score and (now - self.locked_time) <= p.target_lock_timeout_sec:
                return None
            best_i = int(np.argmax([(self._bbox_area(b)) for b in boxes]))

        self.locked_bbox = boxes[best_i].astype(np.float32)
        self.locked_time = now
        return best_i


@dataclass
class Metrics:
    n_frames: int
    target_present_frames: int
    keep_frames: int
    wrong_frames: int
    invalid_frames: int
    switch_count: int
    switch_count_target_present: int
    switch_count_target_absent: int
    switch_count_target_edge: int
    reacquire_count: int
    reacquire_delay_mean: float

    def as_dict(self) -> Dict[str, Any]:
        keep_ratio = self.keep_frames / max(1, self.target_present_frames)
        wrong_ratio = self.wrong_frames / max(1, self.target_present_frames)
        invalid_ratio = self.invalid_frames / max(1, self.n_frames)
        switch_per_100f = self.switch_count * 100.0 / max(1, self.n_frames)
        switch_present_per_100f = self.switch_count_target_present * 100.0 / max(1, self.n_frames)
        switch_absent_per_100f = self.switch_count_target_absent * 100.0 / max(1, self.n_frames)
        switch_edge_per_100f = self.switch_count_target_edge * 100.0 / max(1, self.n_frames)
        return {
            'frames': self.n_frames,
            'target_present_frames': self.target_present_frames,
            'keep_ratio': round(keep_ratio, 4),
            'wrong_ratio': round(wrong_ratio, 4),
            'invalid_ratio': round(invalid_ratio, 4),
            'switch_count': self.switch_count,
            'switch_per_100f': round(switch_per_100f, 3),
            'switch_present_count': self.switch_count_target_present,
            'switch_absent_count': self.switch_count_target_absent,
            'switch_edge_count': self.switch_count_target_edge,
            'switch_present_per_100f': round(switch_present_per_100f, 3),
            'switch_absent_per_100f': round(switch_absent_per_100f, 3),
            'switch_edge_per_100f': round(switch_edge_per_100f, 3),
            'reacquire_count': self.reacquire_count,
            'reacquire_delay_mean_sec': round(self.reacquire_delay_mean, 4),
        }


def _mkbox(cx: float, cy: float, w: float, h: float) -> List[float]:
    return [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]


def gen_synthetic(scene: str, frames: int, dt: float, seed: int, noise: float = 1.2) -> List[Dict[str, Any]]:
    random.seed(seed)
    out: List[Dict[str, Any]] = []

    for t in range(frames):
        ts = t * dt
        # target A
        x_a = 170.0 + 2.0 * t + random.uniform(-noise, noise)
        y_a = 252.0 + 3.0 * math.sin(t / 8.0)
        a = {'id': 'A', 'bbox': _mkbox(x_a, y_a, 88.0, 188.0)}

        if scene == 'cross':
            if t < 20:
                det = [a]
            else:
                boost = 1.2 + 0.45 * math.exp(-((t - 60.0) ** 2) / (2.0 * 10.0 ** 2))
                x_b = 620.0 - 3.5 * (t - 20.0) + random.uniform(-noise, noise)
                y_b = 248.0 + 3.0 * math.cos(t / 7.0)
                b = {'id': 'B', 'bbox': _mkbox(x_b, y_b, 95.0 * boost, 195.0 * boost)}
                det = [a, b]

        elif scene == 'short_occ':
            x_b = 500.0 - 1.4 * t + random.uniform(-noise, noise)
            b = {'id': 'B', 'bbox': _mkbox(x_b, 250.0, 104.0, 206.0)}
            if t < 18:
                det = [a]
            elif 36 <= t <= 44:  # ~0.9s
                det = [b]
            else:
                det = [a, b]

        elif scene == 'long_occ':
            x_b = 500.0 - 1.4 * t + random.uniform(-noise, noise)
            b = {'id': 'B', 'bbox': _mkbox(x_b, 250.0, 104.0, 206.0)}
            if t < 18:
                det = [a]
            elif 33 <= t <= 49:  # ~1.7s > default timeout
                det = [b]
            else:
                det = [a, b]
        else:
            raise ValueError(f'unsupported scene: {scene}')

        out.append({'t': ts, 'target_id': 'A', 'detections': det})

    return out


def eval_sequence(sequence: List[Dict[str, Any]], policy: str, image_w: float, image_h: float,
                  params: SelectorParams) -> Metrics:
    selector = TargetLockSelector(params)

    n = len(sequence)
    present_frames = 0
    keep_frames = 0
    wrong_frames = 0
    invalid_frames = 0
    switch_count = 0
    switch_count_target_present = 0
    switch_count_target_absent = 0
    switch_count_target_edge = 0

    last_pred_id: Optional[str] = None
    last_target_present: Optional[bool] = None

    in_gap = False
    reacquire_delays: List[float] = []
    gap_reappear_t: Optional[float] = None

    for f in sequence:
        t = float(f['t'])
        target_id = str(f['target_id'])
        dets = f['detections']

        ids: List[str] = [str(d['id']) for d in dets]
        boxes = np.array([d['bbox'] for d in dets], dtype=np.float32) if dets else np.zeros((0, 4), dtype=np.float32)

        target_present = target_id in ids
        if target_present:
            present_frames += 1

        pred_id: Optional[str] = None
        if boxes.shape[0] > 0:
            if policy == 'baseline_max_area':
                i = int(np.argmax([(boxes[k][2] - boxes[k][0]) * (boxes[k][3] - boxes[k][1]) for k in range(boxes.shape[0])]))
            elif policy == 'target_lock':
                i = selector.select(boxes, image_w, image_h, t)
                if i is None:
                    i = None
            else:
                raise ValueError(f'unsupported policy: {policy}')

            if i is not None:
                pred_id = ids[i]

        if pred_id is None:
            invalid_frames += 1

        # switch count（仅统计非空预测）
        if pred_id is not None and last_pred_id is not None and pred_id != last_pred_id:
            switch_count += 1
            if last_target_present is True and target_present is True:
                switch_count_target_present += 1
            elif last_target_present is False and target_present is False:
                switch_count_target_absent += 1
            else:
                switch_count_target_edge += 1
        if pred_id is not None:
            last_pred_id = pred_id
        last_target_present = target_present

        # keep/wrong（仅目标存在时）
        if target_present:
            if pred_id == target_id:
                keep_frames += 1
            elif pred_id is not None:
                wrong_frames += 1

        # reacquire
        # 目标缺失期间，记 gap；目标重现后直到再次预测到 target_id 计时
        if not target_present:
            in_gap = True
            gap_reappear_t = None
        else:
            if in_gap and gap_reappear_t is None:
                gap_reappear_t = t
            if in_gap and gap_reappear_t is not None and pred_id == target_id:
                reacquire_delays.append(max(0.0, t - gap_reappear_t))
                in_gap = False
                gap_reappear_t = None

    mean_reacquire = float(np.mean(reacquire_delays)) if reacquire_delays else 0.0
    return Metrics(
        n_frames=n,
        target_present_frames=present_frames,
        keep_frames=keep_frames,
        wrong_frames=wrong_frames,
        invalid_frames=invalid_frames,
        switch_count=switch_count,
        switch_count_target_present=switch_count_target_present,
        switch_count_target_absent=switch_count_target_absent,
        switch_count_target_edge=switch_count_target_edge,
        reacquire_count=len(reacquire_delays),
        reacquire_delay_mean=mean_reacquire,
    )


def summarize_metrics(ms: List[Metrics]) -> Dict[str, float]:
    if not ms:
        return {}
    ks = [m.as_dict() for m in ms]
    keys = [
        'keep_ratio', 'wrong_ratio', 'invalid_ratio',
        'switch_count', 'switch_per_100f',
        'switch_present_count', 'switch_absent_count', 'switch_edge_count',
        'switch_present_per_100f', 'switch_absent_per_100f', 'switch_edge_per_100f',
        'reacquire_count', 'reacquire_delay_mean_sec',
    ]
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = float(np.mean([float(d[k]) for d in ks]))
    out['trials'] = float(len(ms))
    return out


def run_synthetic(args: argparse.Namespace) -> Dict[str, Any]:
    params = SelectorParams(
        target_lock_timeout_sec=args.lock_timeout,
        target_lock_min_score=args.lock_min_score,
        target_lock_hold_on_low_score=args.lock_hold_on_low_score,
        target_lock_center_gate=args.lock_center_gate,
    )

    scene_names = [s.strip() for s in args.scenes.split(',') if s.strip()]
    all_results: Dict[str, Any] = {}

    for scene in scene_names:
        ms_base: List[Metrics] = []
        ms_lock: List[Metrics] = []

        for i in range(args.trials):
            seq = gen_synthetic(scene=scene, frames=args.frames, dt=args.dt, seed=args.seed + i)
            ms_base.append(eval_sequence(seq, 'baseline_max_area', args.image_w, args.image_h, params))
            ms_lock.append(eval_sequence(seq, 'target_lock', args.image_w, args.image_h, params))

        all_results[scene] = {
            'baseline_max_area': summarize_metrics(ms_base),
            'target_lock': summarize_metrics(ms_lock),
            'params': {
                'lock_timeout': args.lock_timeout,
                'lock_min_score': args.lock_min_score,
                'lock_hold_on_low_score': args.lock_hold_on_low_score,
                'lock_center_gate': args.lock_center_gate,
            }
        }
    return all_results


def run_json(args: argparse.Namespace) -> Dict[str, Any]:
    p = Path(args.input)
    raw = json.loads(p.read_text(encoding='utf-8'))

    # 兼容两种输入：
    # 1) 直接 list[frame]
    # 2) {"meta":..., "sequence":[frame,...]}（video_to_eval_json.py 输出）
    if isinstance(raw, dict) and 'sequence' in raw:
        seq = raw['sequence']
        meta = raw.get('meta', {})
    elif isinstance(raw, list):
        seq = raw
        meta = {}
    else:
        raise ValueError('unsupported json format: expect list or dict with key "sequence"')

    params = SelectorParams(
        target_lock_timeout_sec=args.lock_timeout,
        target_lock_min_score=args.lock_min_score,
        target_lock_hold_on_low_score=args.lock_hold_on_low_score,
        target_lock_center_gate=args.lock_center_gate,
    )

    m_base = eval_sequence(seq, 'baseline_max_area', args.image_w, args.image_h, params)
    m_lock = eval_sequence(seq, 'target_lock', args.image_w, args.image_h, params)

    return {
        'input': str(p),
        'meta': meta,
        'baseline_max_area': m_base.as_dict(),
        'target_lock': m_lock.as_dict(),
        'params': {
            'lock_timeout': args.lock_timeout,
            'lock_min_score': args.lock_min_score,
            'lock_hold_on_low_score': args.lock_hold_on_low_score,
            'lock_center_gate': args.lock_center_gate,
        }
    }


def print_pretty(result: Dict[str, Any]) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Phase-B offline evaluator for target-lock')
    ap.add_argument('--mode', choices=['synthetic', 'json'], default='synthetic')
    ap.add_argument('--input', type=str, default='')

    ap.add_argument('--image-w', type=float, default=640.0)
    ap.add_argument('--image-h', type=float, default=480.0)

    # synthetic
    ap.add_argument('--scenes', type=str, default='cross,short_occ,long_occ')
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--frames', type=int, default=100)
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=20260328)

    # lock params
    ap.add_argument('--lock-timeout', type=float, default=1.2)
    ap.add_argument('--lock-min-score', type=float, default=0.30)
    ap.add_argument('--lock-hold-on-low-score', action='store_true', default=False)
    ap.add_argument('--lock-center-gate', type=float, default=0.35)

    ap.add_argument('--output', type=str, default='')
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.mode == 'json' and not args.input:
        raise SystemExit('--mode json 时必须提供 --input')

    if args.mode == 'synthetic':
        result = run_synthetic(args)
    else:
        result = run_json(args)

    print_pretty(result)

    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\n[OK] saved => {args.output}')


if __name__ == '__main__':
    main()
