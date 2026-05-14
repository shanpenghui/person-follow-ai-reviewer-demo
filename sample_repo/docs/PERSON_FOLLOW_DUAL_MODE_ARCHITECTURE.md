# Person Follow 双模式架构设计

> 版本: v0.2 (2026-04-15)
> 代码: `scene_servo/` 目录
> 参考 tag: `v0.1.0` (scene_template), `v0.1.1` (person_yolo)

## 1. 概述

`scene_servo_node` 是一个 ROS2 节点，通过 `follow_mode` 参数在两种模式下切换：

| 维度 | `scene_template` | `person_yolo` |
|------|------------------|---------------|
| 用途 | 回到预录场景位置 | 实时跟随人体 |
| 视觉输入 | ORB 特征匹配 + 3D-3D PnP | YOLOv8 目标检测 + 深度 |
| 相机 | D455 (胸部固定) | D455 (距离) + D435 (头部跟踪) |
| 匹配方式 | 同步，主线程 | 异步，后台线程 |
| Head 控制 | 不控制（保持朝前） | D435 图像误差驱动 |
| 底盘控制 | `scene_template_servo` | `person_yolo_servo` |
| 激活方式 | `action_master_enabled=True` 自动 | 手势 1 或 action start |
| 到达判定 | fwd/lat/yaw 阈值 + 连续帧确认 | 无（持续跟随） |

## 2. 系统架构

```
                          ┌─────────────────────┐
                          │  scene_servo_node    │
                          │  (10 Hz control loop)│
                          └─────────┬───────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
            follow_mode?    follow_mode?     (person_yolo only)
          scene_template   person_yolo              │
                    │               │               │
            ┌───────┴──┐    ┌──────┴───┐    ┌──────┴──────┐
            │ scene_    │    │ person_  │    │ _process_   │
            │ template_ │    │ yolo_    │    │ gesture()   │
            │ servo.py  │    │ servo.py │    │ D435 hand   │
            │           │    │          │    │ mediapipe   │
            │ ORB match │    │ YOLO det │    └─────────────┘
            │ (同步)     │    │ (异步线程)│
            └──────┬────┘    └────┬─────┘
                   │              │
              last_state     last_state
                   │              │
            ┌──────┴──────────────┴──────┐
            │                            │
    scene底盘控制()              yolo底盘控制()
    (D455 3D pose 直接用)         (TF2 + head_yaw)
            │                            │
    ┌───────┴────────┐          ┌────────┴────────┐
    │ vx: PID on fwd │          │ vx: PID on dist │
    │ wz: PID on yaw │          │ wz: head_yaw    │
    │ + lateral→yaw  │          │ + vx yaw damp   │
    └───────┬────────┘          └────────┬────────┘
            │                            │
            └──────────┬─────────────────┘
                       │
              /smooth_cmd_vel (Twist)
```

## 3. 硬件拓扑 (MIRA3)

```
base_link
  └── ... → torso_link
              ├── camera_torso_link ─── D455 (胸部, 固定)
              │                        话题: /cam_chest/d455/*
              │                        用途: scene匹配 / YOLO距离
              │
              └── head_link ─── D435 (头部, 可转)
                              话题: /cam_head/d435/*
                              用途: 手势检测 / YOLO head跟踪
```

### 关键约束
- **D455 胸部固定**: head 转动不影响 D455 画面 → 用于精确测距和场景匹配
- **D435 随 head 转**: head 转动改变 D435 画面 → 用于 head 闭环跟踪
- **TF2**: `camera_torso_optical → base_link` 变换包含 head_yaw 信息

## 4. 主循环流程 (`_tick`, 10Hz)

```
_tick()
  │
  ├─ action_master_enabled? ── No ──→ home head, 停底盘, return
  │
  ├─ person_yolo? ── Yes ──→ _process_gesture()  [D435 手势检测]
  │
  ├─ follow_active? ── No ──→ home head, 减速底盘, return
  │
  ├─ 获取帧数据 (ROS topic / RealSense)
  │
  ├─ 热加载 template
  │
  ├─ follow_mode?
  │   ├─ person_yolo ──→ _tick_yolo()   [异步 YOLO 推理]
  │   └─ scene_template ──→ _tick_scene() [同步 ORB 匹配]
  │
  ├─ 更新 last_state
  │
  ├─ person_yolo? ── Yes ──→ _update_head()  [D435 误差驱动]
  │
  ├─ _update_chassis() or _update_scene_chassis()
  │
  └─ 日志输出
```

### 4.1 follow_active 判定

```
_is_follow_active():
  action_follow_active OR gesture_follow_active
```

- `action_follow_active`: 由 behavior tree 通过 `/person_follow/action_master_enabled` 设置
- `gesture_follow_active`: D435 检测到手势 1 时设为 True，手势 2 时设为 False

**scene_template 模式**: `action_master_enabled=True` 即激活，不依赖手势
**person_yolo 模式**: 需要手势 1 或 action start 才激活

## 5. Scene Template 模式

### 5.1 数据流

```
D455 RGBD 帧
    │
    ▼
_tick_scene() [同步, 主线程]
    │
    ├─ input_ready? (template + 帧新鲜 + 内参)
    │   └─ No → last_state=None, WAIT_FRAME
    │
    ├─ match_due? (1/match_hz 秒间隔)
    │   └─ No → 保持 last_state, 继续控制
    │
    └─ estimate_best_keyframe()  ←── feature_matcher.py
       │                              servo_estimator.py
       │   ORB 检测 → BF匹配 → RANSAC
       │   → Homography → 2D-2D + depth → 3D-3D PnP
       │
       ▼
    last_state = estimator.update(result)
       │
       ├─ yaw_error_deg (D455 图像误差)
       ├─ forward_error_m (D455 深度)
       ├─ confidence (inlier_ratio)
       ├─ level (L0/L1/LOST)
       ├─ rotation_ref_to_cur (3x3, L0 only)
       └─ translation_ref_to_cur (3x1, L0 only)
```

### 5.2 底盘控制 (`_update_scene_chassis`)

**使用 D455 3D-3D pose 直接计算**，不经过 TF 变换（因为 D455 固定在胸部，pose 已经是相对于机器人的）。

```python
# 从 pose 提取
base_forward_m = trans_vec[2]   # Z 轴 = 前进
base_lateral_m = trans_vec[0]   # X 轴 = 横向
base_yaw_deg = yaw_from_matrix  # 旋转矩阵→偏航角

# 速度计算
target_vx = clamp(k_dist * base_forward_m, scene_vx_min, scene_vx_max)
target_wz = clamp(k_yaw * yaw_cmd, -scene_wz_max, scene_wz_max)

# 到达判定
arrived = |fwd| < 0.08m AND |lat| < 0.12m AND |yaw| < 4°
需要连续 scene_arrive_stable_frames 帧满足 → ARRIVED → 停止
```

### 5.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `scene_vx_max` | 0.10 m/s | 场景模式最大前进速度 |
| `scene_vx_min` | -0.05 m/s | 最大后退速度 |
| `scene_wz_max` | 0.18 rad/s | 最大旋转速度 |
| `scene_accl_vx` | 0.12 | 前进加速度限制 |
| `scene_cmd_alpha` | 0.35 | 输出 EMA 平滑系数 |
| `scene_arrive_forward_m` | 0.08 m | 到达判定前进阈值 |
| `scene_arrive_lateral_m` | 0.12 m | 到达判定横向阈值 |
| `scene_arrive_yaw_deg` | 4.0° | 到达判定偏航阈值 |
| `scene_arrive_stable_frames` | 5 | 到达需要连续帧数 |
| `scene_pose_hold_sec` | 0.8 | pose 丢失后保持时间 |
| `match_hz` | 4.0 | 匹配频率 |

### 5.4 状态机

```
WAIT_FRAME → TRACKING → ARRIVED
                │
                ├─ pose_ok=False 持续 > hold_sec → LOST_STOP
                │
                └─ (手动中断)
```

### 5.5 为什么用同步匹配

v0.1.0 使用同步匹配（`estimate_best_keyframe` 在主线程调用）。曾尝试改为异步线程匹配，但 ORB 匹配耗时 0.8-2.4s，异步期间反复使用旧 state 导致底盘一卡一卡。同步匹配保证了每一帧 state 都是新鲜的。

代价是匹配期间（~1s）控制循环被阻塞，但因为 `match_hz=4`，实际上每 0.25s 才做一次匹配，其余 tick 用 `last_state` 继续控制。

## 6. Person YOLO 模式

### 6.1 双相机架构

```
┌─────────────────────────────────────────────┐
│            _yolo_inference_worker            │
│            (后台线程, ~4-10 Hz)               │
│                                             │
│  D455 BGR ──→ YOLO 检测 ──→ 最大bbox        │
│  D455 depth ──→ bbox中心5x5中值 ──→ 距离     │
│  D435 BGR ──→ YOLO 检测 ──→ head 误差       │
│                                             │
│  输出 state:                                │
│    forward_error_m (D455 距离)               │
│    lateral_error_m (D455 bbox偏移×depth)     │
│    yaw_error_deg (D435 bbox中心)             │
│    pitch_error_deg (D435 bbox中心)           │
│    head_tracking_ok (D435 检测到人)          │
│    base_servo_ready (D455 conf>0.3)          │
└─────────────────────────────────────────────┘
```

**为什么需要双相机**: D455 装在胸部固定，head 转了 D455 画面不变。如果用 D455 误差驱动 head → 正反馈死循环（head 追到极限 89°）。D435 随 head 转，用 D435 的误差驱动 head 形成稳定闭环。

### 6.2 Depth 处理

```python
# D455 bbox 中心 5x5 patch 中值
depth_val = median(depth_m[iy-2:iy+3, ix-2:ix+3])

# EMA 平滑
if delta > 1.0m:  depth_val = prev      # 跳变过大 → 丢弃
else:             depth_val = 0.4*new + 0.6*old  # 平滑

# 横向误差
lateral_error_m = (cx - cam_cx) / cam_fx * depth_val
```

### 6.3 Head 控制 (`_update_head`)

```python
# 目标位置 = 当前head角 + 图像误差
raw_yaw = head_yaw_deg + (-track_error_x * fov_h)

# EMA 平滑 (自适应 alpha)
alpha = clamp(0.35 + 0.45*(max_delta - 2)/8, 0.35, 0.80)
smooth_yaw += alpha * (raw_yaw - smooth_yaw)

# P 控制输出速度
vel = vel_gain * (smooth_yaw - head_yaw_deg)   # vel_gain=1.7
clamp(vel, -60, +60) deg/s
```

### 6.4 底盘控制 (`_update_chassis`)

```python
# wz: 直接用 head_yaw (不是图像误差)
target_wz = k_yaw * head_yaw_rad    # head 偏多少就转多少

# vx: TF2 变换后的距离误差
base_forward = TF2_transform(forward_error_m, lateral_error_m)
target_vx = k_dist * (base_forward - desired_distance)

# vx yaw 衰减: head 偏角大时压低 vx → 原地旋转
if |head_yaw| > 20°: target_vx *= ratio  # 20°~45° 线性衰减到 0
```

### 6.5 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `vx_max` | 0.40 m/s | 最大前进速度 |
| `wz_max` | 0.25 rad/s | 最大旋转速度 |
| `k_dist` | 0.75 | 距离增益 |
| `k_yaw` | 0.50 | 偏航增益 |
| `desired_distance_m` | 1.0 m | 目标跟随距离 |
| `vx_yaw_damp_start_deg` | 20° | vx 衰减起始角 |
| `vx_yaw_damp_end_deg` | 45° | vx=0 的角度 |
| `head_homing_ratio` | 0.0 | head 回中力 (当前关闭) |
| `yolo_conf_thres` | 0.16 | YOLO 置信度阈值 |

## 7. 共用模块

### 7.1 帧输入

两种模式共用同一套帧获取逻辑：

```python
# D455 (胸部, 两种模式都用)
latest_bgr ← /cam_chest/d455/color/image_raw (CompressedImage)
latest_depth_m ← /cam_chest/d455/aligned_depth_to_color (depth, mm→m)
cam_intrinsics ← /cam_chest/d455/color/camera_info

# D435 (头部, person_yolo only)
latest_d435_bgr ← /cam_head/d435/color/image_raw (CompressedImage)
```

### 7.2 TF2 变换 (`_transform_forward_error`)

仅 person_yolo 模式使用。将 D455 相机坐标系的 forward/lateral 转换到 base_link 坐标系：

```
base_forward = TF_lookup(camera_torso_optical → base_link) × [forward, lateral, 0]
```

当 TF 失败时 fallback 到手动 head_yaw 投影：
```
base_forward = forward * cos(head_yaw) - lateral * sin(head_yaw)
```

### 7.3 Head 控制器

`head_controller.py` 提供两类接口：
- **跟踪模式** (`_update_head`): P 控制器，输入 D435 图像误差，输出 head 速度指令
- **回中模式** (`_send_home_head`): head 缓慢回到 (0, 0) 位置

scene_template 模式不调用跟踪，仅用回中。

### 7.4 手势检测 (`_process_gesture`)

**仅 person_yolo 模式调用**。D435 图像 + MediaPipe Hands → 手指计数：
- 手势 1 (竖1指) → `gesture_follow_active = True`
- 手势 2 (竖2指) → `gesture_follow_active = False` + `gesture_stop_event`

scene_template 不依赖手势，但 gesture_stop_event 仍可触发 scene 停止。

### 7.5 Follow Action Server

独立节点 `follow_action_server_node.py`：
- 接收 behavior tree 的 action 调用
- 通过 `/person_follow/action_master_enabled` 控制 servo 启停
- 通过 `/person_follow/fsm_state` 上报状态

## 8. 启动方式

### 8.1 Scene Template 模式

```bash
REFERENCE_FRAME_PATH=./data/reference_frame.json \
  ROBOT_TYPE=MIRA3 INPUT_SOURCE=ros \
  ./scripts/run_scene_template_chain.sh
```

自动识别: `REFERENCE_FRAME_PATH != "none"` → `follow_mode=scene_template`

### 8.2 Person YOLO 模式

```bash
REFERENCE_FRAME_PATH=none \
  ROBOT_TYPE=MIRA3 INPUT_SOURCE=ros \
  ./scripts/run_scene_template_chain.sh
```

自动识别: `REFERENCE_FRAME_PATH == "none"` → `follow_mode=person_yolo`

两个模式使用同一个启动脚本，通过环境变量区分。

## 9. 代码结构 (v0.2 拆分后)

```
scene_servo/
├── scene_servo_node.py           # 共用基座: 参数、ROS接口、主循环 _tick()
│                                  #   帧获取、head控制、手势检测
├── scene_template_servo.py        # scene 模式专用:
│   ├── tick_scene()               #   同步 ORB 匹配
│   └── update_scene_chassis()     #   3D pose → PID → vx/wz
│
├── person_yolo_servo.py           # yolo 模式专用:
│   ├── tick_yolo()                #   异步推理调度
│   ├── yolo_inference_worker()    #   双相机 YOLO (后台线程)
│   ├── detect_yolo_person()       #   单相机检测
│   └── update_chassis()           #   TF2 + head_yaw → vx/wz
│
├── head_controller.py             # head P 控制器 (共用):
│   ├── update_head()              #   跟踪: 图像误差 → head 速度
│   ├── send_home_head()           #   回中
│   └── stop_head()                #   停止
│
├── feature_matcher.py             # ORB + BF + RANSAC (scene only)
├── servo_estimator.py             # EMA 状态估计、level 判定 (scene only)
└── config/
    └── person_follow_all.yaml     # 全部参数 (两种模式共用一个 yaml)

person_follow/
└── follow_action_server_node.py   # Action server (独立节点)

scripts/
├── run_scene_template_chain.sh        # 统一启动脚本
└── run_scene_template_servo_logged.sh # servo 启动 + 日志
```

### 9.1 职责划分

| 文件 | 行数(估) | 依赖 | 说明 |
|------|----------|------|------|
| `scene_servo_node.py` | ~800 | 所有 servo 模块 | 主循环、帧输入、ROS 接口 |
| `scene_template_servo.py` | ~300 | feature_matcher, servo_estimator | scene 匹配+底盘 |
| `person_yolo_servo.py` | ~400 | ultralytics, torch | YOLO 检测+底盘 |
| `head_controller.py` | ~100 | torso action client | head 速度控制 |
| `feature_matcher.py` | ~300 | opencv | ORB 匹配 |
| `servo_estimator.py` | ~150 | numpy | EMA + level 判定 |

### 9.2 模式切换

```python
# scene_servo_node.py _tick() 中:
if self.follow_mode == 'person_yolo':
    self.yolo_servo.tick(now)
else:
    self.scene_servo.tick(now)

# chassis 也分离:
if self.follow_mode == 'scene_template':
    self.scene_servo.update_chassis(forward, lateral, base_ok, now)
    return
else:
    self.yolo_servo.update_chassis(forward, lateral, base_ok, now)
```

## 10. 已知问题与待优化

### 10.1 Person YOLO 模式

| 问题 | 优先级 | 状态 |
|------|--------|------|
| YOLO bbox 抖动导致 depth 不稳 | P1 | depth EMA 已缓解，需目标锁定 |
| 无目标锁定（每帧选最大框） | P1 | 待实现 IoU + 中心距离关联 |
| Head 增益需精细调 | P2 | vel_gain=1.7, max=60°/s, 需实测确认 |
| TF_CHECK 日志刷屏 | P3 | 改为 debug level |
| Homing bias 关闭后 head 可能卡在极限 | P2 | 需要更好的回中策略 |

### 10.2 Scene Template 模式

| 问题 | 优先级 | 状态 |
|------|--------|------|
| ORB 匹配耗时 0.8-2.4s 阻塞主循环 | P1 | 当前用同步匹配，可以接受 |
| 仅 1 keyframe | P2 | multi_keyframe 已声明但未验证 |
| match_hz=4 但实际 ~1Hz | P2 | 匹配耗时长导致实际频率低 |

### 10.3 共用问题

| 问题 | 优先级 | 状态 |
|------|--------|------|
| 残留进程杀不干净 | P2 | 需要更好的 pid 管理 |
| D435 帧 rate 不稳定 | P2 | gesture_process_hz=8 但实际更低 |
