# person_follow

基于 YOLOv8 + 深度 / 场景模板的 MIRA 跟随模块。当前仓库主入口是单节点 all-in-one 伺服，支持 `person_yolo` 和 `scene_template` 两种模式。

## 依赖安装

```bash
cd /home/dev/midea_humanoid_robot/src/person_follow
python3 -m pip install -r requirements.txt
```

ROS2 依赖（`rclpy`、`sensor_msgs`、`cv_bridge`、`tf2_ros` 等）通过 ROS apt 环境提供。

## 快速编译

```bash
cd /home/dev/midea_humanoid_robot
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=56

colcon build --merge-install --packages-select interfaces
source install/setup.bash

colcon build --merge-install --packages-select person_follow
source install/setup.bash
```

## 快速启动

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=56
source /home/dev/midea_humanoid_robot/install/setup.bash

/home/dev/midea_humanoid_robot/src/task_planner/person_follow/scripts/run_person_yolo.sh
```

这条命令是 YOLO 跟人模式的推荐入口，可从任意目录执行。

通用入口仍然可用，也可以从任意目录执行：

```bash
REFERENCE_FRAME_PATH=none ROBOT_TYPE=MIRA3 INPUT_SOURCE=ros \
/home/dev/midea_humanoid_robot/src/task_planner/person_follow/scripts/run_scene_template_chain.sh
```

- `run_person_yolo.sh`：默认进入 `person_yolo` 跟人模式
- `REFERENCE_FRAME_PATH=none`：进入 `person_yolo` 跟人模式
- `REFERENCE_FRAME_PATH=<json路径>`：进入 `scene_template` 场景模板模式

补充：如果需要直接走 ROS launch，也可以使用：

```bash
cd /home/dev/midea_humanoid_robot
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=56
source install/setup.bash

ros2 launch person_follow person_follow_all.launch.py
```

注意：相机驱动、URDF/TF、底盘与 torso 控制需提前启动；无论脚本还是 launch，都不负责启动整机依赖。

## PersonFollow Action 接口

### Action 定义

```text
# Goal
string action_name
---
# Result
bool success
uint16 status
---
# Feedback
float32 action_percentage
uint16 status
```

接口文件：[`PersonFollow.action`](/home/nano/Documents/mira-deploy/interfaces/action/PersonFollow.action)

### Action Server 名称

`/person_follow/skill_behavior_tree`

### 环境准备

```bash
cd /home/dev/midea_humanoid_robot
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=56
source install/setup.bash
```

### 行为树调用指令

行为树对外就是这个 Action 接口，调用入口固定为：

`/person_follow/skill_behavior_tree`

常用指令如下。

### 开始

`start` 是长时间运行的 goal：

- `scene_template` 模式：发送后直接开始场景跟随
- `person_yolo` 跟人：发送 `start` / `start:person` 后 server 保持 goal 活跃，等待手势 1 进入跟随
- `person_yolo` 跟其他 COCO 类别：发送 `start:<COCO_CLASS_ID>` 后直接进入跟随，不需要手势启动
- 停止或到位后返回 result

```bash
ros2 action send_goal --feedback /person_follow/skill_behavior_tree \
  interfaces/action/PersonFollow "{action_name: 'start'}"
```

### 停止

```bash
ros2 action send_goal /person_follow/skill_behavior_tree \
  interfaces/action/PersonFollow "{action_name: 'stop'}"
```

### 查询状态

```bash
ros2 action send_goal /person_follow/skill_behavior_tree \
  interfaces/action/PersonFollow "{action_name: 'status'}"
```

如果你是从行为树或外部调度系统接入，本质上就是向 `/person_follow/skill_behavior_tree` 发送 `PersonFollow` goal，当前支持的 `action_name` 为：

- `start`
- `start:person`
- `start:<COCO_CLASS_ID>`
- `stop`
- `status`

`person_yolo` 模式下，`start` 默认跟人。`start:person` 也跟人，需要手势 1 启动、手势 2 结束。
如果通过 action 切换到其他 COCO 类别，则不需要手势，action 发送后直接跟随，并使用 `object_desired_distance_m` 作为停车距离。非人目标到达后会发 0 速度并让 action goal 返回成功：

```bash
# 跟人
ros2 action send_goal --feedback /person_follow/skill_behavior_tree \
  interfaces/action/PersonFollow "{action_name: 'start:person'}"

# 按 COCO class id 跟踪，例如 refrigerator=72，直接开始，默认停在 1.5m
ros2 action send_goal --feedback /person_follow/skill_behavior_tree \
  interfaces/action/PersonFollow "{action_name: 'start:72'}"
```

内置名称映射：

| action_name | COCO class id |
|-------------|---------------|
| `start:person` / `start:人` | `0` |
| `start:fridge` / `start:refrigerator` / `start:冰箱` | `72` |
| `start:sofa` / `start:couch` / `start:沙发` | `57` |
| `start:chair` / `start:椅子` | `56` |

### Status 码

| status | 含义 |
|--------|------|
| `0` | `IDLE` |
| `1` | `FOLLOWING` |
| `2` | `STOPPED_BY_GESTURE` / 正常停止返回 |
| `3` | `CANCELLED` |
| `4` | `ERROR` |

### 联动观测

```bash
ros2 topic echo /person_follow/action_master_enabled
ros2 topic echo /person_follow/follow_enabled
ros2 topic echo /person_follow/fsm_state
```

## 节点列表

| 节点 | 说明 |
|------|------|
| `scene_servo_node` | all-in-one 场景/跟人伺服节点 |
| `person_yolo_servo_node` | YOLO 目标跟随 all-in-one 伺服节点 |
| `scene_template_servo_node` | 场景模板伺服节点 |
| `follow_action_server_node` | `/person_follow/skill_behavior_tree` Action Server |
| `follow_fsm_langgraph_node` | 手势 FSM 状态机 |
| `gesture_detector_node` | 手势检测 |
| `person_follow_node` | 旧版头部视觉伺服节点 |
| `person_association_node` | 多源目标关联 |
| `speak_pyttsx3_node` | TTS 播报 |


## 跟随模式

通过 `follow_mode` 参数切换：

| 模式 | 说明 | 输入 | 启动方式 |
|------|------|------|----------|
| `scene_template` | 场景模板匹配 PBVS（回到记忆位置） | reference_frame.json | action start 自动开始 |
| `person_yolo` | YOLO 目标检测跟随 | 无需模板 | 跟人用手势；非人目标用 action |

> **注意**：`auto_enable_follow` 参数只对 `scene_template` 生效。`person_yolo` 跟人时需要手势触发跟随；跟非人 COCO 类别时由 action 直接触发跟随。

## 手势控制

内置 mediapipe 手势检测（使用 D435 头部相机），用于 `person_yolo` 的 person 目标：

- **手势1**（食指朝上）→ 开始跟人
- **手势2**（食指+中指朝上）→ 停止跟人，头部归位

## 场景跟随完整流程

### 步骤 1：拍照（采集记忆图片）

```bash
cd ~/midea_humanoid_robot/src/person_follow
source /opt/ros/humble/setup.bash
source ../../install/setup.bash
export ROS_DOMAIN_ID=56

# 拍一张 RGB-D 对 + camera_info，保存到 data/ 目录
python3 -m scene_servo.tools.capture_rgbd_pair \
    --name ref \
    --out-dir ./data/scene_memory \
    --input-source ros
```

输出文件：
- `data/scene_memory/ref.jpg` — RGB 图像
- `data/scene_memory/ref_depth.npy` — 深度图（float32, 单位 m）
- `data/scene_memory/ref_camera_info.json` — 相机内参

> 如果 `camera_info` 话题收不到，脚本会报错退出（防止错误内参污染数据）。
> 紧急情况下可以加 `--allow-fallback-intrinsics` 使用默认 D455 内参（不推荐用于正式采集）。

### 步骤 2：构建模板

```bash
# 从单张拍照结果构建特征模板
python3 -m scene_servo.tools.build_template \
    --image ./data/scene_memory/ref.jpg \
    --depth ./data/scene_memory/ref_depth.npy \
    --scene-name "kitchen_table" \
    --out ./data/reference_frame.json
```

模板会自动从同目录 `ref_camera_info.json` 读取内参。也可以用 `--fx --fy --cx --cy` 手动覆盖。

### 步骤 3：启动场景跟随

```bash
# 启动 scene_template 模式
REFERENCE_FRAME_PATH=./data/reference_frame.json \
ROBOT_TYPE=MIRA3 INPUT_SOURCE=ros \
./scripts/run_scene_template_chain.sh
```

启动后，scene 模式通过 `auto_enable_follow` 自动开始跟随。也可以通过行为树触发：

```bash
# 通过 action start 启动
ros2 action send_goal /person_follow/skill_behavior_tree interfaces/action/PersonFollow '{action_name: "start"}'
```

### 步骤 4：观察状态

```bash
# 查看实时日志
LATEST="$(find logs/scene_template_chain -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
tail -f "$LATEST/servo.log"
```

日志示例：
```
scene_servo: kitchen_table mode=servo lv=L0 yaw=+2.1 fwd=+0.35 conf=0.78 in=23/31 cmd=(+0.080,+0.012) base=(+0.35,-0.02,+2.1) pose_ok=True st=TRACKING ts=(1744700123456789012,1744700123451234567)
```

关键字段：
- `st=` — 状态机（WAIT_FRAME/TRACKING/HOLD/LOST_STOP/ARRIVED）
- `pose_ok` — 当前帧位姿是否通过门控
- `base=(fwd,lat,yaw)` — 当前估计误差
- `ts=(img_ns,depth_ns)` — 快照时间戳（用于 worldpilot 数据对齐）

### 停止

```bash
./scripts/stop_scene_template_chain.sh
```

到位后 scene 模式会自动停车并输出 `scene target reached`，不会 auto restart。

## YOLO 跟随模式

```bash
# 启动 YOLO 跟人模式
/home/dev/midea_humanoid_robot/src/task_planner/person_follow/scripts/run_person_yolo.sh
```

- 默认跟人不会自动跟随，需要对着 D435 比**手势1**开始、比**手势2**停止
- 通过 `start:<COCO_CLASS_ID>` 跟非人目标时，不需要手势，action 后直接开始

> `run_person_yolo.sh` 默认设置 `REFERENCE_FRAME_PATH=none`、`FOLLOW_MODE=person_yolo`、`ROBOT_TYPE=MIRA3`、`INPUT_SOURCE=ros`。

## 行为树调用

```bash
# 默认跟人。scene_template 模式下 start 会直接启动场景跟随；person_yolo 模式下等待手势1开始跟随。
ros2 action send_goal /person_follow/skill_behavior_tree interfaces/action/PersonFollow '{action_name: "start"}'

# 显式跟人，需要手势1开始、手势2结束
ros2 action send_goal /person_follow/skill_behavior_tree interfaces/action/PersonFollow '{action_name: "start:person"}'

# 切换 YOLO 跟踪类别，例如 refrigerator=72，直接开始，默认停在 1.5m
ros2 action send_goal /person_follow/skill_behavior_tree interfaces/action/PersonFollow '{action_name: "start:72"}'

# 停止
ros2 action send_goal /person_follow/skill_behavior_tree interfaces/action/PersonFollow '{action_name: "stop"}'

# 查询状态
ros2 action send_goal /person_follow/skill_behavior_tree interfaces/action/PersonFollow '{action_name: "status"}'
```

## 关键配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `follow_mode` | `person_yolo` | 跟随模式 |
| `auto_enable_follow` | `true` | scene 模式自动开始跟随（不影响 YOLO） |
| `desired_distance_m` | `1.0` | YOLO 跟人距离 |
| `object_desired_distance_m` | `1.5` | YOLO 非人目标跟随距离 |
| `object_arrive_stable_frames` | `5` | YOLO 非人目标到达确认帧数 |
| `k_dist` | `0.75` | 距离增益 |
| `k_yaw` | `0.5` | 转向增益 |
| `vx_max` | `0.45` | 线速度上限 |
| `scene_arrive_forward_m` | `0.08` | scene 到位前后阈值 |
| `scene_arrive_lateral_m` | `0.12` | scene 到位横向阈值 |
| `scene_arrive_yaw_deg` | `4.0` | scene 到位角度阈值 |
| `scene_vx_max` | `0.12` | scene 最大线速度 |
| `scene_wz_max` | `0.18` | scene 最大角速度 |
| `scene_cmd_alpha` | `0.35` | scene 命令 EMA 系数 |
| `scene_pose_hold_sec` | `0.8` | scene 丢 pose 保持时间 |
| `yolo_detect_class_id` | `0` | YOLO 检测类别 (COCO person=0) |
| `yolo_model_path` | `''` | YOLO 模型路径 (.pt/.engine) |
| `gesture_start` | `1` | 开始手势 ID |
| `gesture_stop` | `2` | 停止手势 ID |
