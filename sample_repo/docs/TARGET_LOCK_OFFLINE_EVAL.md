# Phase-B 离线评测：Target Lock 稳定性验证

> 更新时间：2026-03-28

## 0) 先回答：这次离线评测用的是哪个视频？
本轮我跑的结果**不是来自真实视频**，而是脚本内置的**合成场景序列**（synthetic）：
- `cross`：多人交叉穿行
- `short_occ`：短时遮挡（< lock timeout）
- `long_occ`：长时遮挡（> lock timeout）

这样做的目的：先验证算法行为是否符合预期，且不依赖机器人/相机硬件。

---

## 1) 目标
在**不依赖机器人硬件**的情况下，评估并比较：
- 旧策略：`baseline_max_area`（总选最大框）
- 新策略：`target_lock`（IoU + 中心距离 + 面积比）

输出指标：
- `keep_ratio`：目标存在时，持续选中原目标比例（越高越好）
- `wrong_ratio`：目标存在时，选错目标比例（越低越好）
- `invalid_ratio`：输出无效（hold）比例
- `switch_count`：目标 ID 切换次数（越低越好）
- `reacquire_delay_mean_sec`：遮挡后重获目标平均延时

---

## 2) 脚本位置
`tools/target_lock_offline_eval.py`

支持两种输入：
1. `--mode synthetic`：内置合成场景（不需要视频）
2. `--mode json`：读取外部标注序列 JSON（可来自真实视频）

---

## 3) 原理与流程

### 3.1 算法核心（与主代码一致）
每一帧对候选框打分：

`score = w_iou * IoU + w_center * center_score + w_area * area_score`

- `IoU`：与上一帧锁定框的重叠度
- `center_score`：与上一帧中心距离的归一化得分
- `area_score`：与上一帧面积比相似度得分
- 默认权重：`0.55 / 0.30 / 0.15`

并且：
- 若 `best_score < target_lock_min_score`：
  - `target_lock_hold_on_low_score=true` 时：短时返回 invalid（不切人）
  - 否则：回退最大框
- 超过 `target_lock_timeout_sec` 后：解锁，允许重选

### 3.2 评测流程
1. 输入一段“逐帧检测序列”（synthetic 或 JSON）
2. 同时跑两套策略：
   - baseline_max_area
   - target_lock
3. 每帧记录预测 ID / invalid
4. 聚合统计 keep/wrong/switch/reacquire 指标
5. 输出 JSON 结果，便于回归对比

---

## 4) 使用方法

### A. 合成场景回归（推荐先跑）

#### 1) 启用 hold-on-low-score（建议与当前主代码一致）
```bash
python3 tools/target_lock_offline_eval.py \
  --mode synthetic \
  --trials 20 \
  --frames 100 \
  --dt 0.1 \
  --lock-hold-on-low-score \
  --output /tmp/target_lock_eval_hold.json
```

#### 2) 关闭 hold-on-low-score（做A/B对照）
```bash
python3 tools/target_lock_offline_eval.py \
  --mode synthetic \
  --trials 20 \
  --frames 100 \
  --dt 0.1 \
  --output /tmp/target_lock_eval_nohold.json
```

### C. 从真实视频一键转 JSON（新增）

新增脚本：`tools/video_to_eval_json.py`

作用：
- 输入真实视频
- 用 YOLO 人体检测（class=0）逐帧生成 bbox
- 生成评测 JSON（可直接喂给 `target_lock_offline_eval.py`）

示例（MOT17-14 片段）：
```bash
# 1) 视频 -> JSON
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
python3 tools/video_to_eval_json.py \
  --video /tmp/phaseb_videos/mot17_14_clip_05_25.mp4 \
  --output /tmp/phaseb_videos/mot17_14_clip_05_25_eval_iou.json \
  --model-path /home/dev/yolov8s.pt \
  --class-id 0 \
  --imgsz 640 \
  --conf 0.25 \
  --frame-stride 2 \
  --id-source iou \
  --target-mode longest

# 2) JSON -> 评测结果
python3 tools/target_lock_offline_eval.py \
  --mode json \
  --input /tmp/phaseb_videos/mot17_14_clip_05_25_eval_iou.json \
  --image-w 960 --image-h 540 \
  --output /tmp/phaseb_videos/mot17_14_eval_report_iou_nohold.json
```

参数建议：
- `--id-source iou`：在离线评测里更稳定（避免某些视频上 ByteTrack id 抖动）
- `--target-mode longest`：自动选“出现帧数最多”的目标作为 `target_id`
- `--target-track-id trk_xxx`：若你已知要跟的人，可手工指定


---

## 5) 本轮实测摘要（10 trials）

参数：`lock_timeout=1.2, lock_min_score=0.3, lock_center_gate=0.35`

1) `cross`（多人交叉）
- baseline: `keep_ratio=0.20`
- target_lock: `keep_ratio=1.00`, `switch_count=0`

2) `short_occ`（短遮挡）
- no-hold: `keep_ratio≈0.396`
- hold: `keep_ratio=1.00`, `invalid_ratio≈0.09`

3) `long_occ`（长遮挡）
- hold/no-hold 均会在超时后可能重选，`keep_ratio≈0.398`

结论：
- hold 策略显著改善短时遮挡下乱切目标；
- 代价是少量 invalid（保守但稳定）；
- 长遮挡超过超时阈值后重选属于预期行为。

## 5) 指标解释升级（建议看这组）

除了总 `switch_per_100f`，新增了分解指标：
- `switch_present_per_100f`：目标在前后两帧都存在时发生的切换（真正的“跳人”）
- `switch_absent_per_100f`：目标在前后两帧都缺失时发生的切换（通常是缺失期误追）
- `switch_edge_per_100f`：目标进/出场边界时发生的切换

工程上建议优先优化顺序：
1. 先压 `switch_present_per_100f`（真跳人）
2. 再压 `switch_edge_per_100f`（进出场边界抖动）
3. `switch_absent_per_100f` 可通过 invalid+底盘缓停策略兜底

---
- 本脚本是**旁路工具**，不依赖 ROS，不影响主运行链路。
- 可用于参数回归：每次调整 `target_lock_*` 参数后，建议先跑 synthetic，再跑真实 JSON 集。
