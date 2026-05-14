#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将真实视频转换为 Phase-B 离线评测 JSON 序列。

输出 JSON 每帧格式：
{
  "t": 0.0,
  "target_id": "trk_7",
  "detections": [
    {"id": "trk_7", "bbox": [x1,y1,x2,y2]},
    ...
  ]
}

说明：
- 优先使用 YOLO track（ByteTrack）生成 track id
- 若 track id 不可用，自动回退到内置 IoU 简易跟踪器
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class TrackState:
    bbox: np.ndarray
    last_frame: int


class SimpleIoUTracker:
    def __init__(self, iou_thres: float = 0.35, max_age: int = 20):
        self.iou_thres = iou_thres
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, TrackState] = {}

    @staticmethod
    def _area(b: np.ndarray) -> float:
        return max(1.0, float((b[2] - b[0]) * (b[3] - b[1])))

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        xx1 = max(float(a[0]), float(b[0]))
        yy1 = max(float(a[1]), float(b[1]))
        xx2 = min(float(a[2]), float(b[2]))
        yy2 = min(float(a[3]), float(b[3]))
        iw = max(0.0, xx2 - xx1)
        ih = max(0.0, yy2 - yy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        ua = SimpleIoUTracker._area(a) + SimpleIoUTracker._area(b) - inter
        return float(inter / max(1e-6, ua))

    def update(self, dets: np.ndarray, frame_idx: int) -> List[int]:
        # 清理超时轨迹
        stale = [tid for tid, st in self.tracks.items() if (frame_idx - st.last_frame) > self.max_age]
        for tid in stale:
            self.tracks.pop(tid, None)

        if dets.shape[0] == 0:
            return []

        assigned_ids = [-1] * dets.shape[0]
        used_tracks = set()

        pairs: List[Tuple[float, int, int]] = []  # (iou, det_i, track_id)
        for i in range(dets.shape[0]):
            for tid, st in self.tracks.items():
                iou = self._iou(dets[i], st.bbox)
                if iou >= self.iou_thres:
                    pairs.append((iou, i, tid))

        pairs.sort(key=lambda x: x[0], reverse=True)

        for _, di, tid in pairs:
            if assigned_ids[di] != -1:
                continue
            if tid in used_tracks:
                continue
            assigned_ids[di] = tid
            used_tracks.add(tid)

        for i in range(dets.shape[0]):
            if assigned_ids[i] == -1:
                assigned_ids[i] = self.next_id
                self.next_id += 1

        for i, tid in enumerate(assigned_ids):
            self.tracks[tid] = TrackState(bbox=dets[i].copy(), last_frame=frame_idx)

        return assigned_ids


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Convert real video to target-lock eval JSON')
    ap.add_argument('--video', required=True, type=str)
    ap.add_argument('--output', required=True, type=str)
    ap.add_argument('--model-path', default='/home/dev/yolov8s.pt', type=str)
    ap.add_argument('--class-id', default=0, type=int)
    ap.add_argument('--imgsz', default=640, type=int)
    ap.add_argument('--conf', default=0.20, type=float)
    ap.add_argument('--tracker', default='bytetrack.yaml', type=str)
    ap.add_argument('--device', default='0', type=str)
    ap.add_argument('--frame-stride', default=2, type=int)
    ap.add_argument('--start-sec', default=0.0, type=float)
    ap.add_argument('--end-sec', default=-1.0, type=float)
    ap.add_argument('--target-track-id', default='', type=str,
                    help='指定原目标 track id（例如 trk_7）；不填则按 --target-mode 自动选择')
    ap.add_argument('--target-mode', choices=['first', 'longest'], default='longest',
                    help='自动选择 target_id 的策略：first=首帧最大框，longest=出现帧数最多')
    ap.add_argument('--id-source', choices=['auto', 'bytetrack', 'iou'], default='iou',
                    help='检测ID来源：auto=优先bytetrack缺失回退iou，bytetrack=强制用bytetrack，iou=强制用IoU跟踪')
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f'cannot open video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(round(args.start_sec * fps)))
    end_frame = total_frames - 1
    if args.end_sec > args.start_sec >= 0.0:
        end_frame = min(total_frames - 1, int(round(args.end_sec * fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f'[INFO] video={video_path} fps={fps:.3f} frames={total_frames} use_range=[{start_frame},{end_frame}]')
    print(f'[INFO] model={args.model_path} class_id={args.class_id} conf={args.conf} stride={args.frame_stride}')

    model = YOLO(args.model_path)
    iou_tracker = SimpleIoUTracker(iou_thres=0.35, max_age=max(10, int(round(1.0 * fps / max(1, args.frame_stride)))))

    sequence: List[Dict] = []
    chosen_target_id: Optional[str] = args.target_track_id.strip() or None
    seen_area: Dict[str, float] = {}
    seen_count: Dict[str, int] = {}

    frame_idx = start_frame
    kept = 0
    used_fallback_iou = 0

    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if ((frame_idx - start_frame) % max(1, args.frame_stride)) != 0:
            frame_idx += 1
            continue

        t = frame_idx / fps

        # YOLO track（优先）
        results = model.track(
            source=frame,
            persist=True,
            classes=[args.class_id],
            conf=args.conf,
            imgsz=args.imgsz,
            tracker=args.tracker,
            verbose=False,
            device=(args.device if args.device else None),
        )

        boxes = results[0].boxes if results and len(results) > 0 else None
        dets_np = np.zeros((0, 4), dtype=np.float32)
        ids_out: List[str] = []

        if boxes is not None and len(boxes) > 0:
            dets_np = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            track_ids = boxes.id.detach().cpu().numpy().astype(np.int32) if boxes.id is not None else None

            use_iou = (args.id_source == 'iou')
            use_bt = (args.id_source == 'bytetrack')
            use_auto = (args.id_source == 'auto')

            if use_iou:
                assigned = iou_tracker.update(dets_np, kept)
                ids_out = [f'iou_{int(tid)}' for tid in assigned]
            elif use_bt:
                if track_ids is not None and len(track_ids) == dets_np.shape[0]:
                    ids_out = [f'trk_{int(tid)}' for tid in track_ids.tolist()]
                else:
                    # 强制 bytetrack 时若无 id，则退化为逐帧临时 id（会导致强烈抖动）
                    ids_out = [f'frame_{kept}_{i}' for i in range(dets_np.shape[0])]
                    used_fallback_iou += 1
            else:
                # auto: 优先 bytetrack，缺失时回退 iou
                if track_ids is not None and len(track_ids) == dets_np.shape[0]:
                    ids_out = [f'trk_{int(tid)}' for tid in track_ids.tolist()]
                else:
                    assigned = iou_tracker.update(dets_np, kept)
                    ids_out = [f'iou_{int(tid)}' for tid in assigned]
                    used_fallback_iou += 1

        detections = []
        max_area = -1.0
        max_id = None

        for i in range(dets_np.shape[0]):
            b = dets_np[i]
            det_id = ids_out[i]
            area = float((b[2] - b[0]) * (b[3] - b[1]))
            seen_area[det_id] = seen_area.get(det_id, 0.0) + max(0.0, area)
            seen_count[det_id] = seen_count.get(det_id, 0) + 1
            detections.append({
                'id': det_id,
                'bbox': [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
            })
            if area > max_area:
                max_area = area
                max_id = det_id

        # 自动选 target
        if chosen_target_id is None and max_id is not None and args.target_mode == 'first':
            chosen_target_id = max_id

        sequence.append({
            't': float(t),
            'target_id': '__PENDING__',
            'detections': detections,
        })

        kept += 1
        if kept % 50 == 0:
            print(f'[INFO] processed={kept} frames, current_t={t:.2f}s, dets={len(detections)}')

        frame_idx += 1

    cap.release()

    if chosen_target_id is None:
        # 自动策略：默认 longest（出现帧数最多）
        if args.target_mode == 'longest' and seen_count:
            chosen_target_id = max(seen_count.items(), key=lambda kv: kv[1])[0]
        elif seen_area:
            chosen_target_id = max(seen_area.items(), key=lambda kv: kv[1])[0]
        else:
            raise SystemExit('no detections found in selected range, cannot choose target_id')

    for f in sequence:
        f['target_id'] = chosen_target_id

    payload = {
        'meta': {
            'video': str(video_path),
            'fps': float(fps),
            'start_frame': int(start_frame),
            'end_frame': int(min(end_frame, frame_idx - 1)),
            'frame_stride': int(args.frame_stride),
            'class_id': int(args.class_id),
            'model_path': str(args.model_path),
            'tracker': str(args.tracker),
            'device': str(args.device),
            'chosen_target_id': str(chosen_target_id),
            'fallback_iou_frames': int(used_fallback_iou),
        },
        'sequence': sequence,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[OK] saved: {out_path}')
    print(f'[OK] frames_out={len(sequence)} target_id={chosen_target_id} fallback_iou_frames={used_fallback_iou}')


if __name__ == '__main__':
    main()
