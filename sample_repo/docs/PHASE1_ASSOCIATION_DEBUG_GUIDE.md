# Phase1+ 联调与验证文档（D435+D455 关联融合）

> 版本：2026-03-27
> 包：`person_follow`
> 范围：几何+时序 + overlap gating（无 ReID）

---

## 1. 本次功能概览

本次新增了一个关联节点：`person_association_node`

- 输入：
  - D435 target: `/person_follow/target`
  - D455 target: `/person_follow/target_d455`
  - bbox meta:
    - `/person_follow/d435_bbox`
    - `/person_follow/d455_bbox`
- 输出：
  - `/person_follow/target_fused`
  - `/person_follow/target_fused_valid`

底盘 `chassis_follow_node` 已改为订阅 fused 目标。

---

## 2. 代码位置

- 关联节点：
  - `/home/dev/midea_humanoid_robot/src/person_follow/person_follow/person_association_node.py`
- 检测节点（新增 bbox meta 发布）：
  - `/home/dev/midea_humanoid_robot/src/person_follow/person_follow/person_follow_node.py`
- 总启动：
  - `/home/dev/midea_humanoid_robot/src/person_follow/launch/person_follow_all.launch.py`

---

## 3. 一次性准备（编译）

```bash
cd /home/dev/midea_humanoid_robot
source /opt/ros/humble/setup.bash
colcon build --packages-select person_follow --symlink-install --merge-install
source install/setup.bash
```

---

## 4. 启动前检查

### 4.1 启动 D455（若未启动）

```bash
source /opt/ros/humble/setup.bash
source /home/dev/midea_humanoid_robot/install/setup.bash
export ROS_DOMAIN_ID=55
bash /home/dev/midea_humanoid_robot/src/agx_startup/scripts/realsense/launch_d455.sh
```

### 4.2 相机话题检查

```bash
export ROS_DOMAIN_ID=55
ros2 topic hz /cam_head/d435/color/image_raw
ros2 topic hz /cam_head/d435/aligned_depth_to_color/image_raw
ros2 topic hz /cam_chest/d455/color/image_raw
ros2 topic hz /cam_chest/d455/aligned_depth_to_color/image_raw
```

---

## 5. 正式联调启动

```bash
cd /home/dev/midea_humanoid_robot
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=55

ros2 launch person_follow person_follow_all.launch.py
```

---

## 6. 关键话题验证命令

## 6.1 关联输入是否存在

```bash
ros2 topic hz /person_follow/target
ros2 topic hz /person_follow/target_d455
ros2 topic echo /person_follow/d435_bbox --once
ros2 topic echo /person_follow/d455_bbox --once
```

## 6.2 融合输出是否正常

```bash
ros2 topic hz /person_follow/target_fused
ros2 topic echo /person_follow/target_fused_valid
ros2 topic echo /person_follow/target_fused --once
```

## 6.3 底盘控制链路

```bash
ros2 topic hz /person_follow/cmd_preview
ros2 topic echo /person_follow/cmd_preview --once
```

---

## 7. 你在现场怎么测（建议流程）

1. 单人站在视野内正前方（1.2~2.0m）
2. 缓慢左右移动
3. 观察：
   - `/person_follow/target_fused_valid` 大部分时间为 `true`
   - `target_fused` 的 bearing 连续变化，不剧烈跳变
   - `cmd_preview` 平滑变化，不突变

再做多人穿越场景：
- 当前跟随人前方有第二人短暂经过
- 观察是否减少“跳人”现象（相比旧版）

---

## 8. 常见问题排查

### Q1: fused 没有输出
- 检查 `person_association_node` 是否在运行：
```bash
ros2 node list | grep person_association_node
```
- 检查输入话题是否有数据：`target/target_d455/bbox`

### Q2: fused_valid 经常 false
- 人体可能不在重叠区域或两路检测不同步
- 先放宽参数（launch里 association node 参数）：
  - `gate_bearing_deg` 从 15 -> 18
  - `max_input_age_sec` 从 0.45 -> 0.6
  - `bbox_timeout_sec` 从 0.60 -> 0.8

### Q3: 仍有跳变
- 收紧：
  - `max_output_jump_deg` 15 -> 10
  - `max_output_jump_m` 0.50 -> 0.35
- 提高平滑：
  - `output_alpha` 0.55 -> 0.40

---

## 9. 调参建议（第一轮）

先保持默认参数跑一轮，再按下列顺序调：
1. `gate_bearing_deg / gate_bearing_deg_overlap`
2. `max_input_age_sec / bbox_timeout_sec`
3. `output_alpha`
4. `max_output_jump_m / max_output_jump_deg`

---

## 10. 回退方案（紧急）

如果现场异常，快速回退到“无 association”可用旧链路：
- 把 `chassis_follow_node` 的 `target_topic` 改回 `/person_follow/target`
- `target_valid_topic` 改回 `/person_follow/target_valid`

---

## 11. 说明

本阶段仅为 Phase1+：
- 已做：几何+时序 + overlap gating
- 未做：ReID / 图像特征相似度

后续如果需要再进入 Phase2（外观特征增强）。
