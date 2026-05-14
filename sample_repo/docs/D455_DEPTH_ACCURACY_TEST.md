# D455 深度准确性小实验（快速版）

本实验用于验证 D455 在你现场环境下的深度准确性。

- 方法A（推荐）：YOLO 先检测人体 bbox，再在 aligned depth 对应 ROI 做鲁棒统计（默认 P30）
- 方法B（备选）：中心点 ROI 采样（容易被背景污染）
- 输出：控制台统计 + 可选 CSV（多距离点汇总）

---

## 1. 代码放在哪里

已放置在包内：

- 脚本：`/home/dev/midea_humanoid_robot/src/person_follow/person_follow/d455_depth_accuracy_test.py`
- 入口：`setup.py` 已有 `d455_depth_accuracy_test` 可执行项

---

## 2. 编译

```bash
cd /home/dev/midea_humanoid_robot
source /opt/ros/humble/setup.bash
colcon build --packages-select person_follow --symlink-install --merge-install
source install/setup.bash
```

---

## 3. 启动前准备

### 3.1 确认 D455 在跑

```bash
export ROS_DOMAIN_ID=55
ros2 topic hz /cam_chest/d455/aligned_depth_to_color/image_raw
ros2 topic hz /cam_chest/d455/color/image_raw
```

### 3.2 场地建议

- 用卷尺从**相机中心**到人体（或目标板）测量 GT 距离
- 人体测试时：人尽量正对相机、少摆动
- 每个距离点测试时保持 2~3 秒静止后再采样

推荐距离点：`0.8m / 1.2m / 1.6m / 2.0m`

---

## 4. 直接可跑命令（YOLO人体ROI，推荐）

例如 GT=1.2m：

```bash
cd /home/dev/midea_humanoid_robot
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=55

ros2 run person_follow d455_depth_accuracy_test --ros-args \
  -p use_yolo_person_roi:=true \
  -p rgb_topic:=/cam_chest/d455/color/image_raw \
  -p depth_topic:=/cam_chest/d455/aligned_depth_to_color/image_raw \
  -p yolo_model_path:=/home/dev/.openclaw/workspace/yolov8s.pt \
  -p conf_thres:=0.20 \
  -p bbox_inner_shrink_ratio:=0.25 \
  -p depth_stat_method:=p30 \
  -p gt_distance_m:=1.2 \
  -p settle_sec:=1.5 \
  -p sample_duration_sec:=6.0 \
  -p output_csv:=/tmp/d455_depth_accuracy_human.csv
```

---

## 5. 多距离点一键循环

```bash
for d in 0.8 1.2 1.6 2.0; do
  echo "=== 请站到 ${d}m，准备好后按回车 ==="
  read
  ros2 run person_follow d455_depth_accuracy_test --ros-args \
    -p use_yolo_person_roi:=true \
    -p rgb_topic:=/cam_chest/d455/color/image_raw \
    -p depth_topic:=/cam_chest/d455/aligned_depth_to_color/image_raw \
    -p yolo_model_path:=/home/dev/.openclaw/workspace/yolov8s.pt \
    -p conf_thres:=0.20 \
    -p bbox_inner_shrink_ratio:=0.25 \
    -p depth_stat_method:=p30 \
    -p gt_distance_m:=${d} \
    -p settle_sec:=1.5 \
    -p sample_duration_sec:=6.0 \
    -p output_csv:=/tmp/d455_depth_accuracy_human.csv
  done
```

查看 CSV：

```bash
cat /tmp/d455_depth_accuracy_human.csv
```

---

## 6. 参数说明（核心）

- `use_yolo_person_roi`：是否启用 YOLO 人体 ROI（推荐 true）
- `bbox_inner_shrink_ratio`：bbox 内缩比例，防止采到背景
- `depth_stat_method`：`p30|p40|median`，人体建议 `p30`
- `conf_thres`：YOLO 置信度阈值
- `gt_distance_m`：人工真值（米）

---

## 7. 结果解释

输出重点：

- `mean / median / std`
- `error = mean - gt`
- `abs_error`
- `rel_error%`

经验门限：

- `abs_error <= 0.05m` 或 `rel_error <= 5%` 基本可接受
- 若偏差大且波动高（std 大）：通常是 bbox 漂移或 ROI 混入背景
