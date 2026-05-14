# MIRA2 Person Follow 实机验收操作单（2026-03-28版）

> 目的：周一回公司后，按固定流程快速验证当前“多人场景稳定性 + 丢目标缓停”版本。
> 版本基线：请使用本仓库当前主分支（含 2026-03-28 当日提交）。

---

## 0. 验收通过标准（建议）

### 核心指标（建议按 D435 主链路）
1. `switch_present_per_100f <= 3`
   - 目标可见时，基本不跳人。
2. `T_handover_p95 <= 1.0s`
   - 丢目标到缓停再到跟新目标，95分位不超过 1 秒。
3. 主观体验
   - 目标丢失时底盘“缓停”自然，无突兀硬刹。
   - 新目标接管时无明显来回抖动。

---

## 1. 验收前准备

### 1.1 环境
```bash
source /opt/ros/humble/setup.bash
cd /home/dev/midea_humanoid_robot
source install/setup.bash
export ROS_DOMAIN_ID=55
```

### 1.2 构建（如周一代码有变更）
```bash
colcon build --packages-select person_follow --symlink-install --merge-install
source install/setup.bash
```

### 1.3 设备检查
- D435 / D455 相机正常供电与识别
- 底盘急停可用
- 现场留出 3m x 3m 安全测试区域

---

## 2. 启动与基础连通性检查

### 2.1 启动一体化跟随
```bash
ros2 launch person_follow person_follow_all.launch.py
```

### 2.2 关键 topic 检查（另开终端）
```bash
source /opt/ros/humble/setup.bash
source /home/dev/midea_humanoid_robot/install/setup.bash
export ROS_DOMAIN_ID=55

ros2 topic hz /mira2/target
ros2 topic hz /mira2/target_valid
ros2 topic hz /cmd_vel
```

### 2.3 丢目标缓停行为快速验证
让目标暂时离开画面，观察：
- `/mira2/target_valid` 变为 `false`
- `/cmd_vel` 速度在短时间内平滑降至 0（不是瞬间硬清零）

可用命令观察：
```bash
ros2 topic echo /mira2/target_valid
ros2 topic echo /cmd_vel
```

---

## 3. 场景化验收（建议录 bag）

### 3.1 录制 rosbag（全程）
```bash
mkdir -p /tmp/person_follow_acceptance
ros2 bag record \
  /mira2/target \
  /mira2/target_valid \
  /mira2/d455/target \
  /mira2/d455/target_valid \
  /cmd_vel \
  -o /tmp/person_follow_acceptance/bag_$(date +%Y%m%d_%H%M%S)
```

### 3.2 场景A：中等拥挤（2~4人）
- 目标A先被锁定
- 其他人从左右穿行，偶尔近距离遮挡
- 时长：5分钟

观察点：
- 目标可见时是否持续跟 A（不跳到 B/C）
- 遮挡期间是否“缓停而不是乱追”
- 遮挡解除后能否快速恢复跟随

### 3.3 场景B：拥挤（5+人）
- 多人交叉、并排、近距离前后遮挡
- 时长：5分钟

观察点同上，重点关注“误切换次数”与“恢复速度”。

---

## 4. 离线评估命令（周一回放）

> 注：当前仓库的离线评估工具是“视频/JSON口径”；
> 实机 bag 可先导出关键时间段视频，或另做 bag->JSON 转换后复用。

### 4.1 若已有评测 JSON
```bash
python3 tools/target_lock_offline_eval.py \
  --mode json \
  --input /path/to/sequence.json \
  --image-w 640 --image-h 480 \
  --output /tmp/person_follow_acceptance/eval_report.json
```

### 4.2 handover 指标统计（avg/p50/p95）
```bash
python3 tools/handover_kpi_eval.py \
  --split06 /tmp/phaseb_videos/mot17_06_eval_split_latest.json \
  --split14 /tmp/phaseb_videos/mot17_14_eval_split_latest.json \
  --output /tmp/person_follow_acceptance/handover_kpi_eval.json
```

---

## 5. 参数微调顺序（仅当不达标）

### 5.1 先调“体验”
- 缓停太猛：
  - 降低 `lost_decel_vx`（如 0.35 -> 0.28）
  - 降低 `lost_decel_wz`（如 1.2 -> 0.9）

### 5.2 再调“抗跳人”
- 目标可见时仍跳人：
  - 增大 `target_lock_switch_confirm_frames_d435`（3 -> 4）
  - 增大 `target_lock_switch_margin`（0.08 -> 0.10）
  - 增大 `target_lock_switch_margin_crowded`（0.16 -> 0.20）

### 5.3 最后调“恢复速度”
- 恢复太慢：
  - 适当降低 `target_lock_switch_confirm_frames_d435`（4 -> 3）
  - 或轻降 `target_lock_min_score`（0.20 -> 0.18）

---

## 6. 验收记录模板（建议）

- 日期：
- 场景：A / B
- 时长：
- 主观体验：
- 关键参数快照：
- 结果：
  - switch_present_per_100f =
  - switch_absent_per_100f =
  - switch_edge_per_100f =
  - T_handover_p95 =
- 结论：通过 / 待调参

---

## 7. 安全注意事项
- 首轮实机建议将 `enable_output` 与 `max_vx` 保守配置（例如 0.20m/s）。
- 必须有人工旁站，急停手柄可达。
- 禁止在人群密集且无保护区域直接跑高速参数。
