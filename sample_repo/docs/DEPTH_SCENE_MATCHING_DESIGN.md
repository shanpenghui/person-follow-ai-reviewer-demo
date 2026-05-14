# 深度几何场景匹配设计文档

> **版本**: v0.1 | **日期**: 2026-04-15 | **状态**: 草案，待讨论
> **基线**: person_follow v0.1.0 (tag `v0.1.0`)

## 1. 背景与问题

### 1.1 现状

当前 `scene_template` 模式使用 **ORB 2D 描述子匹配 + 深度反投影 3D-3D SVD** 的流程：

```
RGB → ORB 特征提取 → Hamming 匹配 → 取匹配点深度 → 反投影 3D → RANSAC SVD → R,t
```

### 1.2 失败模式

实机测试（2026-04-15）表明：

| 视角偏差 | 匹配表现 |
|---------|---------|
| 0°~10° | inlier 100+, conf ≥0.99 ✅ |
| 10°~20° | inlier 开始下降, conf 0.9 ⚠️ |
| 20°~30° | inlier 急降至 9, conf 0.52 ❌ |
| >30° | 完全匹配失败, base 估计飞天 💀 |

**根本原因**：瓶颈在 2D 描述子匹配阶段。ORB 是二进制描述子，对透视变化极度敏感。深度信息只在匹配成功后的后处理阶段参与，没有在前端匹配中发挥作用。

### 1.3 需求

WorldPilot P0 数据采集需要从**不同起始角度**（±30°~45°）驱动机器人返回目标位置。当前 ±15° 的有效范围意味着每次采集都要精确对准，严重限制采集效率和轨迹多样性。

---

## 2. 设计目标

| 指标 | 当前 (v0.1.0) | 目标 |
|------|--------------|------|
| 有效视角范围 | ±15° | **±45°** |
| 匹配成功率 (±30°) | ~0% | **>90%** |
| 匹配成功率 (±45°) | 0% | **>70%** |
| 单帧延迟 (AGX Orin) | ~10ms | **<50ms** |
| 依赖 | OpenCV | OpenCV + Open3D |

---

## 3. 方案设计

### 3.1 核心思路：深度几何先，RGB 辅助

```
当前: RGB匹配 → 深度后处理        (视角差大就崩)
改后: 深度→3D特征→几何匹配→精修   (视角无关)
```

深度图描述的是场景的**几何结构**，天然不受视角变化影响。一张桌子从正面和侧面看，RGB 可能完全不同，但 3D 点云的局部几何（法向量、曲率）是稳定的。

### 3.2 新匹配流程

```
┌─────────────┐    ┌─────────────┐
│  参考帧深度  │    │  当前帧深度  │
│  (离线)      │    │  (实时)      │
└──────┬──────┘    └──────┬──────┘
       │                   │
       ▼                   ▼
  ┌─────────────────────────────┐
  │  Step 1: 深度 → 有组织点云   │
  │  降采样 (voxel ~5mm)        │
  └─────────────┬───────────────┘
                │
       ▼                   ▼
  ┌──────────┐      ┌──────────┐
  │ FPFH 特征 │      │ FPFH 特征 │
  │ (参考)    │      │ (当前)    │
  └─────┬────┘      └─────┬────┘
        │                 │
        └────────┬────────┘
                 ▼
  ┌─────────────────────────────┐
  │  Step 2: FPFH 匹配 + RANSAC │
  │  3D 空间粗配准 → 初始 R,t    │
  └─────────────┬───────────────┘
                │
                ▼
  ┌─────────────────────────────┐
  │  Step 3: 彩色 ICP 精修       │
  │  几何 + 颜色联合优化          │
  │  → 精确 R,t (6-DOF)         │
  └─────────────┬───────────────┘
                │
                ▼
  ┌─────────────────────────────┐
  │  Step 4: 输出伺服误差         │
  │  (fwd, lat, yaw)            │
  └─────────────────────────────┘
```

### 3.3 各步骤详解

#### Step 1: 深度 → 有组织点云

```python
# 参考: Open3D create_from_depth_image
# 使用相机内参 (fx, fy, cx, cy) 反投影
# D455 内参已知，深度已对齐到 RGB
# 降采样: voxel_down_sample(voxel_size=0.005) → ~5mm 精度
# 预计算法向量: estimate_normals(radius=0.02)
```

- 参考帧：离线完成，存入 reference_frame.json（点云 + FPFH）
- 当前帧：每帧实时计算

#### Step 2: FPFH 特征 + RANSAC 粗配准

FPFH (Fast Point Feature Histograms) 是纯 3D 局部几何描述子：
- 基于法向量的局部几何关系
- 对刚体变换、旋转、平移完全不变
- 不依赖 RGB，不受光照和视角影响

```python
# Open3D 实现
fpfh_ref = o3d.pipelines.registration.compute_fpfh_feature(
    pcd_ref, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100))
fpfh_cur = o3d.pipelines.registration.compute_fpfh_feature(
    pcd_cur, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100))

result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    pcd_cur, pcd_ref, fpfh_cur, fpfh_ref,
    mutual_filter=True,
    max_correspondence_distance=0.05,  # 5cm
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    ransac_n=3,
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05),
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
```

#### Step 3: 彩色 ICP 精修

粗配准后用 Colored ICP 同时优化几何和颜色对齐：

```python
result = o3d.pipelines.registration.registration_colored_icp(
    pcd_cur, pcd_ref,
    max_correspondence_distance=0.03,  # 3cm
    init=result_ransac.transformation,
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30))
```

彩色 ICP 比纯几何 ICP 更精确，因为它同时利用了几何和外观信息。

#### Step 4: 从 RT 提取伺服误差

从最终 `transformation` (4x4) 提取与现有 servo_estimator 兼容的误差：

```python
R = transformation[:3, :3]
t = transformation[:3, 3]
euler = rotation_to_euler_deg(R)  # 已有函数
fwd = t[2]    # Z = forward
lat = t[0]    # X = lateral
yaw = euler["yaw_deg"]
```

### 3.4 降级策略

```
L0: FPFH + RANSAC + 彩色ICP   → 完整 6-DOF (fwd, lat, yaw, pitch, roll)
L1: FPFH + RANSAC (ICP失败)    → 粗 6-DOF
L2: ORB/SIFT 2D 匹配 (现有流程) → yaw + fwd
L3: 匹配失败                    → LOST
```

现有 ORB 流程作为 **fallback 保留**，在场景几何过于简单（白墙、空桌面）FPFH 无法匹配时降级到 RGB 匹配。

### 3.5 模板存储变更

`reference_frame.json` 需要扩展：

```json
{
  "name": "kitchen_table",
  "version": 2,
  "keyframes": [
    {
      "image_path": "kitchen_table_000.jpg",
      "depth_path": "kitchen_table_000_depth.npy",
      "camera_intrinsics": {"fx": 382.0, "fy": 382.0, "cx": 322.0, "cy": 240.0},
      "fpfh_path": "kitchen_table_000_fpfh.npz",
      "pcd_path": "kitchen_table_000_pcd.ply",
      "pose": [0, 0, 0, 0, 0, 0]
    }
  ]
}
```

- `depth_path`: 参考帧原始深度（float32, 米）
- `pcd_path`: 预计算降采样点云
- `fpfh_path`: 预计算 FPFH 特征
- 参考帧的点云和 FPFH **离线预计算**，运行时只加载

---

## 4. 性能预估 (AGX Orin)

| 步骤 | 预估耗时 | 备注 |
|------|---------|------|
| 深度→点云+降采样 | ~5ms | D455 640x480 → 降采样后 ~5000 点 |
| FPFH 计算 | ~10ms | KDTree 半径搜索 |
| FPFH RANSAC | ~15ms | 100k iterations, 实际提前退出 |
| 彩色 ICP (30 iter) | ~10ms | 降采样点云上做 |
| **总计** | **~40ms** | 满足 10Hz servo 频率 |

如果超时，可降低采样密度或限制 ICP 迭代次数。

---

## 5. 依赖与部署

### 5.1 新增依赖

```
open3d >= 0.17    # 点云处理、FPFH、ICP
```

- AGX Orin 上安装: `pip install open3d`（有预编译 aarch64 wheel）
- 或者从源码编译 Open3D（AGX Orin 支持 CUDA 加速）

### 5.2 不变的部分

- `servo_estimator.py`: 接口不变，仍然输出 (fwd, lat, yaw) 给 servo loop
- `scene_servo_node.py`: ROS 节点不变
- `follow_action_server_node.py`: 不变
- `run_scene_template_chain.sh`: 不变

改动集中在 `feature_matcher.py`，新增一个 3D 几何匹配路径。

---

## 6. 实现计划

### Phase 1: 最小可行 (预估 1-2 天)

1. `pip install open3d` 在 AGX 上
2. `feature_matcher.py` 新增 `match_3d_geometric()` 函数
3. 参考: 录制时保存深度图，离线预计算点云 + FPFH
4. 在线: 实时点云 + FPFH → RANSAC → ICP → RT
5. 降级: 失败时走现有 ORB 流程
6. **验收**: 偏 30° 成功返回

### Phase 2: 优化 (可选)

- 彩色 ICP 替换普通 ICP
- 体素降采样参数调优
- 参考 keyframe 加入多视角（±20° 各一张）
- CUDA 加速（Open3D CUDA backend）

### Phase 3: 长期

- SuperPoint + SuperGlue 替代 FPFH（学习型特征，更大视角范围）
- 场景图语义匹配（利用桌椅等物体级信息）

---

## 7. 风险与备选

| 风险 | 概率 | 应对 |
|------|------|------|
| Open3D aarch64 wheel 不兼容 AGX | 中 | 从源码编译，或用 pytorch3d |
| FPFH 在简单场景（白墙）匹配失败 | 高 | 降级到现有 ORB 流程 |
| 40ms 延迟超预算 | 低 | 降低采样密度，或隔帧计算 |
| D455 深度噪声影响 FPFH 质量 | 低 | 降采样时自带滤波，5mm voxel 足够平滑 |

---

## 8. 与现有系统的接口

```
                    ┌──────────────────────┐
                    │   scene_servo_node    │
                    │  (ROS2 node, 不变)     │
                    └──────────┬───────────┘
                               │ 每帧调用
                               ▼
                    ┌──────────────────────┐
                    │   servo_estimator     │
                    │  (调度层, 不变)        │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    │                      │
                    ▼                      ▼
          ┌─────────────────┐   ┌─────────────────┐
          │ feature_matcher │   │ feature_matcher  │
          │ (新增 3D 几何)   │   │ (现有 ORB 流程)   │
          │ L0: FPFH+ICP    │   │ L0: ORB+3D-3D    │
          │ L1: FPFH RANSAC │   │ L1: PnP          │
          └─────────────────┘   │ L2: Homography   │
                                └─────────────────┘
```

`servo_estimator` 选择匹配方式的优先级：
1. **先尝试 3D 几何匹配**（新流程）
2. 失败则降级到 **ORB RGB 匹配**（现有流程）
3. 都失败则 LOST

---

## 9. 开放问题

1. **参考帧深度图格式**：存原始 float32 numpy 还是存 PLY？前者简单，后者通用。
2. **FPFH 参数**：`radius=0.05` 和 `max_nn=100` 是否最优？需要实测调参。
3. **是否需要 head 先对齐**：即使 3D 匹配不怕视角差，先用 head 转向减少底盘转弯是否更优？
4. **多 keyframe 策略**：Phase 1 先单 keyframe，Phase 2 加多 keyframe 的选择逻辑。
