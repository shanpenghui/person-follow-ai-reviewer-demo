# person_follow scene_template 第一阶段问题清单

> **版本**：v0.3 — 2026-04-15  
> **状态**：问题确认稿，先讨论并冻结边界，再改代码  
> **范围**：仅限 `/home/nano/Documents/mira-deploy/person_follow` 当前仓库  
> **第一阶段目标**：在不影响 YOLO 跟随的前提下，让 `scene_template` 模式稳定、平滑、可重复地控制机器人底盘回到记忆场景拍照位置，并产出可用于 `~/worldpilot` 的数据。

---

## 1. 第一阶段目标

第一阶段只解决 `scene_template` 场景回位链路，不做大规模重构。

### 1.1 要解决的问题

1. D455 RGB-D 输入稳定，匹配使用同一时刻的图像、深度和相机内参快照。  
2. L0 3D-3D 场景位姿估计可解释，R/t 符号经过离线验证。  
3. 底盘控制连续、不过冲、不抖动，能克服静摩擦但接近目标时能减速。  
4. 丢匹配时状态清晰：不继续用旧 pose 硬追，也不产生残留滑行。  
5. 到位判定稳定，到位后自动停车并反馈完成。  
6. 日志/数据字段足够支撑 `worldpilot` 后续数据集分析。

### 1.2 不在第一阶段处理

1. 不改变 YOLO 跟随现有手势/头部控制行为。  
2. 不把 YOLO 的 action start 语义改成直接启动。  
3. 不启用 `scene_template` 头部伺服。  
4. 不依赖 TF tree 做严格 base frame SE(3) 控制。  
5. 不清理旧 launch、旧 `person_follow_node`、association、TTS 等历史问题。  
6. 不拆分 `scene_servo_node`。

---

## 2. 当前启动契约

第一阶段只维护脚本启动路径：

```bash
./scripts/run_scene_template_chain.sh
```

该脚本当前启动：

| 节点 | 说明 |
|------|------|
| `follow_action_server_node.py` | 提供 `/person_follow/skill_behavior_tree` action |
| `scene_servo_node` | all-in-one 视觉匹配、底盘控制、模式切换 |

第一阶段不维护 `person_follow_all.launch.py` 的 scene 启动路径。

### 2.1 场景模式启动

当前测试命令：

```bash
REFERENCE_FRAME_PATH=data/reference_frame.json INPUT_SOURCE=ros ROBOT_TYPE=MIRA3 ./scripts/run_scene_template_chain.sh
```

或显式写模式：

```bash
FOLLOW_MODE=scene_template REFERENCE_FRAME_PATH=data/reference_frame.json INPUT_SOURCE=ros ROBOT_TYPE=MIRA3 ./scripts/run_scene_template_chain.sh
```

约束：

- `REFERENCE_FRAME_PATH` 必须存在。
- `FOLLOW_MODE=scene_template` 时使用 D455 RGB-D 模板。
- 第一阶段头部不做场景跟踪。

### 2.2 YOLO 模式启动

当前脚本仍要求 `REFERENCE_FRAME_PATH` 非空，因此 YOLO 模式需要显式传 `none`：

```bash
FOLLOW_MODE=person_yolo REFERENCE_FRAME_PATH=none INPUT_SOURCE=ros ROBOT_TYPE=MIRA3 ./scripts/run_scene_template_chain.sh
```

第一阶段只保证该模式不被场景修复破坏。  
后续可以把脚本改成：`FOLLOW_MODE=person_yolo` 时不再要求 `REFERENCE_FRAME_PATH`。

### 2.3 Action 当前能力

当前 `follow_action_server_node.py` 只支持：

| action_name | 当前含义 |
|-------------|----------|
| `start` | 发布 follow enabled，长运行等待 stop/cancel/完成事件 |
| `stop` | 强制停止 |
| `status` | 查询状态 |

第一阶段不新增：

- `start_yolo`
- `start_scene`
- `capture_scene`

第一阶段语义：

- 由 `FOLLOW_MODE` 决定 `start` 实际进入 YOLO 还是 scene。
- `scene_template` 可通过脚本 auto enable 或 action start 进入跟随，具体安全语义需在修复前确认。
- `person_yolo` 保持当前行为，不改 action 与手势关系。

---

## 3. 第一阶段控制状态机

建议把 `scene_template` 底盘控制显式整理成以下状态。当前代码未完全按该状态机实现，这是第一阶段待确认和修复目标。

| 状态 | 进入条件 | 底盘命令 | 退出条件 | 数据记录 |
|------|----------|----------|----------|----------|
| `WAIT_FRAME` | 未收到 RGB-D/内参/模板 | `cmd=(0,0)` | 帧和模板可用 | 记录等待原因 |
| `TRACKING` | `pose_ok=True` 且未到位 | 根据 pose error 连续伺服 | 到位、短暂丢 pose、stop/cancel | 记录 pose 与 cmd |
| `HOLD` | 从 `TRACKING` 短暂掉到 `pose_ok=False` | 快速平滑减速，不使用旧 pose 继续追 | pose 恢复或超时 | 记录 hold 时长 |
| `LOST_STOP` | HOLD 超过 `scene_pose_hold_sec` 或匹配长期不可用 | 持续 `cmd=(0,0)` | pose 恢复或 stop/cancel | 标记 lost |
| `ARRIVED` | pose error 连续满足到位阈值 | `cmd=(0,0)` | 新 start | 记录最终误差并反馈完成 |

关键原则：

- 无帧时不能残留速度。
- 丢 pose 后不能继续使用旧 pose 前进。
- 到位后不能 auto restart。
- 状态变化必须可在日志中回放。

---

## 4. 第一阶段 P0 问题

### 4.1 [P0] `_tick_scene()` 缺少 RGB-D 快照

当前问题：

- `_tick_scene()` 直接把 `self.latest_bgr` 和 `self.latest_depth_m` 传给 `estimate_best_keyframe()`。
- `_image_cb()` 和 `_depth_cb()` 可能在匹配过程中更新最新帧。
- 即使 numpy 数组本身未被原地改写，scene 匹配也缺少“本次控制对应哪一组 RGB/depth/intrinsics”的快照语义。

对 `worldpilot` 数据的影响：

- 控制命令可能对应不稳定的 RGB-D 输入。
- 后续做数据回放时，难以保证图像、深度和动作是一组样本。

第一阶段目标：

- scene 匹配前复制 RGB、depth、camera intrinsics、stamp。
- 单次匹配和单次控制命令绑定同一份输入快照。
- 不改变 YOLO 分支；YOLO 当前已有 snapshot。

### 4.2 [P0] R/t 符号必须离线验证

`feature_matcher.py` 中 `_estimate_rigid_svd()` 的语义是：

```text
pts_cur ~= R @ pts_ref + t
```

也就是 `T_cur_ref`。当前控制直接使用：

```text
base_forward_m = t[2]
base_lateral_m = t[0]
base_yaw_deg = yaw(R)
```

需要验证：

| 运动 | 期望验证 |
|------|----------|
| 相机/机器人向场景前进 0.5m | 静态场景点在当前相机坐标中的 Z 变小，`t[2]` 应接近负值 |
| 相机/机器人向右平移 | 验证 `t[0]` 与底盘转向/横向修正符号 |
| 相机/机器人原地旋转 | 验证 `yaw(R)` 还是 `yaw(R.T)` 才是回位控制方向 |

第一阶段目标：

- 先用离线数据验证符号，不在实机上盲调。
- 若符号确认有误，只改 `scene_template` 控制分支。

### 4.3 [P0] LOST/HOLD/STOP 状态机不够清晰

当前问题：

- `scene_pose_ok=False` 且 `tracking=True` 时会进入 hold 分支并 ramp 到 0。
- 超过 `scene_pose_hold_sec` 后切 `tracking=False`，可能还有速度未完全衰减。
- `_handle_lost()` 对 scene 直接置零，避免残留但存在硬切。

第一阶段目标：

- 明确 `HOLD` 与 `LOST_STOP` 两个状态。
- HOLD 内快速平滑减速。
- HOLD 超时后持续零速。
- 到位或 stop/cancel 时立即进入安全停车状态。
- 不改变 YOLO 的 lost/decel 行为。

---

## 5. 第一阶段 P1 问题

### 5.1 [P1] 底盘控制需要更丝滑

当前 scene 控制链路存在多层滤波/限幅：

1. `servo_estimator.py` 中 EMA 平滑 pose。  
2. `_update_scene_chassis()` 中加速度 ramp。  
3. `scene_cmd_alpha` 对命令再做一次 EMA。

当前默认值：

- servo_estimator EMA alpha = 0.35
- scene_accl_vx = 0.12 m/s², scene_accl_wz = 0.45 rad/s²
- scene_cmd_alpha = 0.35

风险：

- 三层滤波叠加相位延迟大，接近目标时可能过冲。  
- 调参困难：改一个 alpha 不一定能看出效果，因为还有另外两层在起作用。

第一阶段目标：

- 明确每层滤波职责。
- pose 平滑只处理测量噪声。
- cmd ramp 只限制加速度。
- 命令 EMA 要么弱化，要么明确其存在原因。
- 用实机日志验证速度曲线是否连续。

### 5.2 [P1] 最小速度与静摩擦策略

已观察到过极小速度命令导致底盘不动，例如 `cmd_vx` 只有几 mm/s。

需要平衡：

- 速度太小：克服不了底盘静摩擦，机器人看起来不动。
- 速度太大：接近拍照点时容易冲过。

第一阶段需要确认：

| 参数 | 作用 |
|------|------|
| `scene_min_vx_cmd` | 最小有效线速度 |
| `scene_vx_max` | 场景模式最大线速度 |
| `scene_accl_vx` | 加速度限制 |
| `scene_arrive_forward_m` | 到位前后误差阈值 |

### 5.3 [P1] 到位判定需要可重复

当前到位逻辑基于：

```text
abs(base_forward_m) <= scene_arrive_forward_m
abs(base_lateral_m) <= scene_arrive_lateral_m
abs(base_yaw_deg) <= scene_arrive_yaw_deg
scene_arrive_count >= scene_arrive_stable_frames
```

第一阶段目标：

- 多次测试最终误差稳定。
- 到位前不能过早成功。
- 到位后必须持续发 0。
- 日志必须记录最终 `base_fwd/base_lat/base_yaw/conf/inliers`。

### 5.4 [P1] `follow_enabled=True` 的状态语义容易误解

当前 `_tick()` 中会较早发布 `follow_enabled=True`，但此时可能还没加载模板、没完成匹配或 `pose_ok=False`。

问题：

- 该 topic 更像“命令允许跟随”，不是“视觉伺服有效跟随中”。
- 如果外部或数据采集把它当作真实 tracking 状态，会误判。

第一阶段目标：

- 日志/数据中单独记录 scene state：`WAIT_FRAME/TRACKING/HOLD/LOST_STOP/ARRIVED`。
- 不仅依赖 `follow_enabled` 判断是否真的在视觉伺服。

### 5.5 [P1] `scene_completed` 重启语义需要写死

第一阶段定义：

- `ARRIVED` 后进入完成态。
- 不允许 auto enable 自动重启。
- 必须收到下一次明确 start 才能重新开始 scene 跟随。

这样避免机器人到位后再次动起来，保证数据采集安全。

### 5.6 [P1] 模板加载错误需要可解释

`scene_template_store.py` 的 `build_template()` 写入 `version: 2`，但 `load_template()` 当前缺少版本/字段校验。

对第一阶段的影响：

- 如果模板文件损坏或格式旧，节点可能在后续转换时 KeyError。
- 对现场测试来说，错误信息不够直接。

第一阶段最低要求：

- 缺模板或模板格式错误时，不发底盘速度。
- 日志明确说明模板不可用原因。

---

## 6. `worldpilot` 数据要求

第一阶段日志/数据至少应能表达一条样本对应的视觉输入、状态和控制命令。

建议字段：

| 字段 | 说明 |
|------|------|
| `timestamp` | 控制周期时间 |
| `image_stamp` | RGB 图像时间戳 |
| `depth_stamp` | depth 图像时间戳 |
| `state` | `WAIT_FRAME/TRACKING/HOLD/LOST_STOP/ARRIVED` |
| `pose_ok` | 本帧是否通过 L0 pose 门控 |
| `level` | L0/L1/L2/LOST |
| `confidence` | 匹配置信度 |
| `inlier_count` | 内点数量 |
| `matched_count` | 匹配数量 |
| `base_fwd` | 当前估计前后误差 |
| `base_lat` | 当前估计横向误差 |
| `base_yaw` | 当前估计角度误差 |
| `cmd_vx` | 输出线速度 |
| `cmd_wz` | 输出角速度 |
| `arrived` | 是否到位 |
| `template_path` | 使用的模板路径 |

第一阶段可以先用结构化日志或 rosbag 辅助，不强制立即定义最终数据格式。  
但必须保证日志足够回放分析“为什么机器人动/停”。

---

## 7. 第一阶段验收标准

### 7.1 场景回位功能

1. 无 RGB-D 帧或模板不可用时，`cmd=(0,0)`。  
2. `pose_ok=True` 且未到位时，底盘连续输出合理速度。  
3. 速度无明显跳变，接近目标时能减速。  
4. 短暂丢匹配时进入 HOLD，不继续用旧 pose 前进。  
5. 长时间丢匹配时进入 LOST_STOP，持续零速。  
6. 到位后出现 `scene target reached`，并持续发 0。  
7. action server 能收到完成事件并返回成功。

### 7.2 数据可用性

1. 每条控制命令能关联到对应 RGB/depth stamp。  
2. 日志能区分 `TRACKING/HOLD/LOST_STOP/ARRIVED`。  
3. 多次从相似起点回位，最终 `base_fwd/base_lat/base_yaw` 分布稳定。  
4. 失败样本能看出是模板不可用、无帧、pose 不通过、还是 LOST。

### 7.3 YOLO 不受影响

1. `FOLLOW_MODE=person_yolo REFERENCE_FRAME_PATH=none` 能启动。  
2. YOLO 原有手势/头部/底盘跟随行为不因 scene 修复改变。  
3. 第一阶段不修改 YOLO action start 语义。

---

## 8. 后续阶段问题池

以下问题存在，但不进入第一阶段。

### 8.1 模式与协议

1. 是否增加 `start_yolo/start_scene/capture_scene`。  
2. YOLO action start 是否等价于手势 start。  
3. 行为树协议与 `PersonFollow.action` 扩展。

### 8.2 TF/base frame 严格控制

1. `_scene_error_in_base()` 当前保留但第一阶段禁用。  
2. 若要严格回到 base pose，需要先解决 `base_link` TF 在 AGX 上可用性。  
3. 严格 SE(3) 控制放后续阶段。

### 8.3 旧节点和 launch 问题

1. `person_follow_all.launch.py` 中的 `scene_template_servo_bridge_node` 遗留引用。  
2. `person_follow_all.launch.py` 中的 `chassis_follow_node` entry point/源文件缺失问题。  
3. 旧 `person_follow_node` 的 D435/D455 推理调度。  
4. `person_association_node` 日志格式化问题。  
5. `speak_pyttsx3_node` service callback 阻塞问题。  
6. **YOLO `_handle_lost()` 负速度减速 bug**：L1493-1494 的 clamp 范围在 `cmd_vx < 0` 时完全包含当前值，导致负速度永远不减速。需回归测试 YOLO 丢目标缓停。  
7. **YOLO level 字段类型不一致**：`_yolo_inference_worker()` 写 `'level': 3`（int），场景模式用 `'L0'`/`'L1'`（string）。当前 YOLO 分支不检查 level 字符串所以没触发，但后续改代码时可能踩坑。

### 8.4 性能与维护性

1. matcher 每帧重建。  
2. 3D 点索引用 Python list comprehension。  
3. 3D RANSAC 纯 Python 循环。  
4. RANSAC 迭代变量命名。  
5. `quick_test.py` 硬编码路径。  
6. `scene_servo_node.py` 长期拆分。

---

## 9. 第一阶段建议修复顺序

1. 增加 scene RGB-D snapshot，保证单次匹配输入一致。  
2. 整理 scene 状态机：`WAIT_FRAME/TRACKING/HOLD/LOST_STOP/ARRIVED`。  
3. 调整 scene LOST/HOLD 速度策略，避免硬切和残留滑行。  
4. 补充结构化日志字段，支撑 `worldpilot` 数据分析。  
5. 做 R/t 符号离线验证，确认是否需要修正 yaw/lateral 符号。  
6. 根据实机日志微调 `scene_vx_max/scene_min_vx_cmd/scene_cmd_alpha/scene_arrive_*`。  
7. 验证 `person_yolo` 启动与原行为未被影响。

---

## 10. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-04-15 | 初版问题清单 |
| v0.2 | 2026-04-15 | 补齐 `_scene_error_in_base` 跳过历史原因、R/t 离线验证方案、scene 模式 head 行为表 |
| v0.3 | 2026-04-15 | 收敛第一阶段目标：聚焦 scene_template 稳定丝滑回位和 worldpilot 数据采集；将旧节点、launch、性能债移入后续问题池 |
| v0.4 | 2026-04-15 | 补充后续池：YOLO 负速度 bug、level 类型不一致；补 5.1 三重滤波默认参数值 |
