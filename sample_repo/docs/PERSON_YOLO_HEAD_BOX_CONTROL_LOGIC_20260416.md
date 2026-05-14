# Person YOLO 头部对框控制逻辑梳理

基线分支：`dev/yolo-stable-20260416`  
目标文件：`scene_servo/person_yolo_servo_node.py`

## 1. 目标

这条控制链的目标不是“让底盘对准人”，而是：

- 用 `D435` 头部相机画面中的检测框误差，驱动头部转动
- 让头部相机的目标点逐步对准 bbox 内指定位置
- 同时把头部当前偏航角提供给底盘控制，用于后续底盘转向与前进速度衰减

这里“头部对准 box”的闭环只依赖：

- `D435` 图像中的 bbox
- 当前头部实测角 `head_yaw_deg / head_pitch_deg`
- `Torso` action 的速度模式命令

## 2. 输入来源

### 2.1 检测输入

`_yolo_inference_worker()` 同时跑两路 YOLO：

- `D455`：负责距离和底盘横向误差
- `D435`：负责头部跟踪误差

其中头控只使用 `D435` 检测结果。

### 2.2 目标点定义

对于 `D435` bbox：

- 水平目标点：bbox 中心 `hcx_px = (hx1 + hx2) / 2`
- 垂直目标点：不是 bbox 几何中心，而是
  `hcy_px = hy1 + yolo_target_y_ratio * bbox_h`

默认参数：

- `yolo_target_y_ratio = 0.32`

这意味着头部不是盯着人框正中心，而是更偏上半身/头肩区域。

## 3. 图像误差到角度误差

`_yolo_inference_worker()` 内先把像素偏差换成角度偏差：

```python
yaw_error_deg = -(hcx_px - hw / 2.0) / hw * self.head_fov_h_deg
pitch_error_deg = -(hcy_px - hh / 2.0) / hh * self.head_fov_v_deg
```

含义：

- 目标点在图像中心右边时，`yaw_error_deg` 为负
- 目标点在图像中心左边时，`yaw_error_deg` 为正
- 目标点在图像中心下方时，`pitch_error_deg` 为负
- 目标点在图像中心上方时，`pitch_error_deg` 为正

所以这里输出的是“图像误差对应的视角误差”，不是直接的电机命令。

## 4. 控制链总览

头控真正执行在 `_update_head()`，完整链路是：

1. YOLO 产出 `yaw_error_deg / pitch_error_deg`
2. 主循环把它归一化成 `track_error_x / track_error_y`
3. `_update_head()` 对归一化误差做低通
4. 用当前头部实测角 + 图像误差，生成原始目标角 `raw_yaw/raw_pitch`
5. 对目标角做步长限制
6. 对目标角做自适应平滑，得到 `smooth_head_yaw/pitch`
7. 用当前实测头角和目标角的差，算速度命令
8. 通过 `Torso` action 以 `work_mode=4` 下发

## 5. 详细分解

### 5.1 主循环进入头控

主循环 `_tick()` 中：

- 先由 `_tick_yolo()` 更新 `last_state`
- 从 `last_state` 里取出：
  - `yaw_error_deg`
  - `pitch_error_deg`
  - `head_tracking_ok`
- 再算：

```python
track_error_x = clamp(-yaw_error_deg / self.head_fov_h_deg, -1.0, 1.0)
track_error_y = clamp(-pitch_error_deg / self.head_fov_v_deg, -1.0, 1.0)
```

也就是说 `_update_head()` 接收的是归一化误差，范围约为 `[-1, 1]`。

### 5.2 一阶低通滤波

`_update_head()` 入口先对误差做低通：

```python
filtered += head_track_filter_alpha * (raw - filtered)
```

参数：

- `head_track_filter_alpha = 0.22`

作用：

- 抑制 bbox 抖动
- 避免头部直接跟着单帧检测噪声抽动

代价：

- 响应会变慢
- 如果目标快速横移，头会出现明显滞后

### 5.3 当前头角 + 图像误差 -> 原始目标角

当前实测角来自 `/Torso/joint_states`：

- `head_yaw_deg`
- `head_pitch_deg`

然后计算：

```python
raw_yaw = head_yaw_deg + (-yaw_sign * filt_track_error_x * head_fov_h_deg)
raw_pitch = head_pitch_deg + (-pitch_sign * filt_track_error_y * head_fov_v_deg)
```

这里的思路不是“直接把图像误差映射成速度”，而是：

- 先根据当前头角，求一个“我下一步应该看向哪里”的目标角
- 再由后级控制器去追这个目标角

相关参数：

- `yaw_sign`
- `pitch_sign`
- `head_fov_h_deg`
- `head_fov_v_deg`

如果左右反了，优先看 `yaw_sign`。  
如果上下反了，优先看 `pitch_sign`。

### 5.4 目标角步长限制

在对 `raw_yaw/raw_pitch` 做平滑前，代码先做一步限幅：

```python
raw_yaw = clamp(raw_yaw, smooth_head_yaw - head_target_step_max_deg, smooth_head_yaw + head_target_step_max_deg)
raw_pitch = clamp(raw_pitch, smooth_head_pitch - head_target_step_max_deg, smooth_head_pitch + head_target_step_max_deg)
```

参数：

- `head_target_step_max_deg = 10.0`

作用：

- 即使 bbox 单帧跳得很远，目标角每次也只允许跳有限角度
- 抑制检测闪跳带来的大角度瞬时切换

这是一个“目标层限速”，不是电机速度限速。

### 5.5 自适应目标平滑

接着对目标角做自适应平滑：

```python
dy = raw_yaw - smooth_head_yaw
dp = raw_pitch - smooth_head_pitch
max_d = max(abs(dy), abs(dp))
alpha = lerp(head_smoothing, 0.6, clamp((max_d - 3.0) / 10.0, 0.0, 1.0))
smooth += alpha * delta
```

参数：

- `head_smoothing = 0.35`

行为：

- 误差小：`alpha` 接近 `0.35`，更稳更柔
- 误差大：`alpha` 接近 `0.6`，追得更快

所以它不是固定 EMA，而是“误差大时快追、误差小时软收尾”。

### 5.6 目标角到速度命令

平滑后的目标角：

- `target_yaw = smooth_head_yaw`
- `target_pitch = smooth_head_pitch`

再与当前实测头角做差：

```python
ye = target_yaw - head_yaw_deg
pe = target_pitch - head_pitch_deg
```

如果误差足够大，则进入速度命令分支。

### 5.7 死区与 soft-zone

先判断死区：

```python
if abs(ye) >= head_deadband_deg or abs(pe) >= head_deadband_deg:
```

参数：

- `head_deadband_deg = 1.0`

接着不是直接 `2.0 * error`，而是先乘一个 soft-zone 缩放系数：

```python
yaw_scale = _soft_zone_speed_scale(ye, head_deadband_deg, head_soft_zone_yaw_deg)
pitch_scale = _soft_zone_speed_scale(pe, head_deadband_deg, head_soft_zone_pitch_deg)
```

其中：

- 误差小于死区：scale = 0
- 误差大于 soft zone：scale = 1
- 中间区间：用二次曲线平滑拉升，最小不低于 `0.15`

参数：

- `head_soft_zone_yaw_deg = 8.0`
- `head_soft_zone_pitch_deg = 6.0`

作用：

- 靠近中心时减速更明显
- 防止快到目标时速度还太大，导致来回过冲

### 5.8 速度命令公式

最后的速度命令：

```python
vel_gain = 2.0
yaw_cmd = clamp(vel_gain * ye * yaw_scale, -60.0, 60.0)
pitch_cmd = clamp(vel_gain * pe * pitch_scale, -60.0, 60.0)
```

然后 `_send_head_velocity()` 会再按参数限幅：

```python
yaw_cmd = clamp(yaw_cmd, -head_yaw_vel_max_deg_s, head_yaw_vel_max_deg_s)
pitch_cmd = clamp(pitch_cmd, -head_pitch_vel_max_deg_s, head_pitch_vel_max_deg_s)
```

实际配置：

- `head_yaw_vel_max_deg_s = 40.0`
- `head_pitch_vel_max_deg_s = 30.0`

所以真正生效的上限最终是：

- yaw：`40 deg/s`
- pitch：`30 deg/s`

## 6. 下发到 Torso 的方式

`_send_head_velocity()` 里使用的是 `Torso` action 的速度模式：

```python
goal.head_yaw = yaw_cmd
goal.head_pitch = pitch_cmd
goal.work_mode = 4
```

同时它还会带上：

```python
goal.torso_height = 0.0
goal.torso_yaw = 0.0
goal.torso_mask = [False, True, False]
goal.head_mask = [True, True]
```

从代码意图看，作者想表达的是：

- 只控制 head
- torso 不参与跟踪

但这里的 `torso_mask` 语义要结合远端 `torso_control` 的实际轴顺序再确认，不能只看字面。

## 7. 丢目标时的行为

### 7.1 `head_ok=False`

如果当前帧 `D435` 没有有效头部检测：

- `_update_head()` 直接进 `else`
- 执行：
  - `_stop_head()`
  - `smooth_head_yaw = None`
  - `smooth_head_pitch = None`
  - `filtered_track_error_x = None`
  - `filtered_track_error_y = None`

这意味着：

- 头部不会继续回家
- 也不会继续保持上一目标
- 而是直接停住，并清空内部控制状态

### 7.2 `follow inactive`

当手势停用或 action 不激活时：

- `_tick()` 中会调用 `_send_home_head()`
- 头部尝试回 home

`_send_home_head()` 是比较简单的回中控制：

```python
yaw_err = home_head_yaw - head_yaw_deg
pitch_err = home_head_pitch - head_pitch_deg
cmd = 1.5 * err
```

并且限幅为：

- yaw：`30 deg/s`
- pitch：`20 deg/s`

## 8. 现有设计的核心特点

这版逻辑的头控可以概括成：

1. `D435 bbox` 提供图像误差
2. 图像误差先低通
3. 再转换成“目标头角”
4. 目标头角做步长限制
5. 再做自适应平滑
6. 最后用带 soft-zone 的 P 控制输出速度

它不是一个纯 PID，也不是直接 image-error -> velocity 的单层控制，而是一个两层结构：

- 外层：目标角生成与平滑
- 内层：速度型 P 控制

## 9. 优点

- 对检测框抖动比较宽容
- 目标接近中心时明显减速，视觉上更稳
- 步长限制 + 软区缩放一起工作时，不容易因为单帧误检突然猛甩头

## 10. 风险点

### 10.1 响应偏慢

慢的来源叠加了四层：

- `head_track_filter_alpha` 低通
- `head_target_step_max_deg` 限制目标跳变
- `head_smoothing` 平滑目标角
- `soft-zone` 在小误差区再减速

如果人走得快，头容易“追不上框中心”。

### 10.2 多个平滑环节叠加后，调参耦合很强

只改一个参数往往不够，因为：

- 改 `head_smoothing`
- 改 `head_target_step_max_deg`
- 改 `head_soft_zone_*`
- 改 `head_track_filter_alpha`

这些都会改变最终体感，而且相互耦合明显。

### 10.3 丢头框时直接清状态

`head_ok=False` 时直接 stop + clear 状态，这会让重新锁回目标时控制器从头开始建状态，恢复手感可能偏硬。

### 10.4 控制目标点固定在 bbox 上部

`yolo_target_y_ratio=0.32` 会使系统更偏向盯上半身。如果实际现场希望“框中心对框中心”，这个值会直接影响 pitch 行为。

## 11. 当前配置总结

基于 `config/person_follow_all.yaml` 的 `scene_servo_node.ros__parameters`，这条分支上的头控关键参数是：

- `yaw_sign = 1.0`
- `pitch_sign = 1.0`
- `yolo_target_y_ratio = 0.32`
- `head_smoothing = 0.22`
- `head_track_filter_alpha = 0.22`
- `head_target_step_max_deg = 8.0`
- `head_deadband_deg = 1.5`
- `head_soft_zone_yaw_deg = 8.0`
- `head_soft_zone_pitch_deg = 6.0`
- `head_yaw_vel_max_deg_s = 40.0`
- `head_pitch_vel_max_deg_s = 30.0`

注意：

- 代码默认值和 yaml 实际值不完全一样
- 运行时应以 yaml 为准

## 12. 我建议后续改文档时重点讨论的点

如果后面要改这条控制链，最应该先定清楚的是：

1. 是否还需要“低通 + 步长限制 + 平滑 + soft-zone”四层同时存在
2. `yaw_sign / pitch_sign` 是否要按机型固化
3. 丢目标时是“停住”还是“缓慢回中”
4. 对准点是否继续使用 `bbox 上 0.32 高度`
5. `torso_mask` 在当前 MIRA3 torso_control 上是否完全符合预期

