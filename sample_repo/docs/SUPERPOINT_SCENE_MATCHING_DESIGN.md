# SuperPoint + LightGlue 场景匹配方案

> **版本**: v0.1 | **日期**: 2026-04-15 | **状态**: 草案
> **基线**: person_follow v0.2 | **目标**: 将场景匹配视角容忍度从 ±15° 提升到 ±60°

## 1. 为什么不用 FPFH/3D 方案

见 [DEPTH_SCENE_MATCHING_DESIGN.md](./DEPTH_SCENE_MATCHING_DESIGN.md) 的分析。核心问题：
- 单视角深度图在大视角差下没有几何重叠，FPFH 无法工作
- Open3D 在 AGX Orin (aarch64) 上性能存疑，预估 300ms-1s
- 引入重依赖（Open3D ~200MB），部署复杂

**SuperPoint + LightGlue 是更好的选择**：
- 学习型 2D 特征，天然对视角变化鲁棒（±60°+）
- ONNX Runtime 部署，AGX Orin 有 TensorRT EP 加速
- 不需要新硬件，不需要深度图参与匹配（深度仅用于后处理测距）

## 2. 方案对比

| 方案 | 视角容忍度 | 延迟 (AGX 估) | 依赖 | 改动量 |
|------|-----------|--------------|------|--------|
| ORB (当前) | ±15° | ~10ms | OpenCV | - |
| SIFT | ±30° | ~30ms | OpenCV | 改一行配置 |
| **SuperPoint + LightGlue** | **±60°** | **~50-80ms** | **onnxruntime** | **新增模块** |
| FPFH + ICP | ±45° (理论上) | 300ms-1s | Open3D | 重写 |

**推荐路线**: 先试 SIFT（0 成本），再上 SuperPoint（中等成本），跳过 FPFH。

## 3. SuperPoint + LightGlue 架构

### 3.1 模型简介

- **SuperPoint**: 自监督关键点检测 + 描述子提取网络
  - 输入: 灰度图 (H×W)
  - 输出: 关键点 (N×2) + 描述子 (N×256)
  - 对光照、视角、模糊变化鲁棒

- **LightGlue**: 轻量级特征匹配网络（SuperGlue 的快速替代）
  - 输入: 两组 SuperPoint 描述子 + 关键点
  - 输出: 匹配对 + confidence
  - 自适应深度：简单场景早退，复杂场景深计算
  - 比 SuperGlue 快 2-3x，精度持平

### 3.2 匹配流程

```
┌──────────────┐          ┌──────────────┐
│  参考帧 RGB   │          │  当前帧 RGB   │
│  (离线预提取)  │          │  (实时)       │
└──────┬───────┘          └──────┬───────┘
       │                         │
       ▼                         ▼
┌──────────────────────────────────────┐
│         SuperPoint (ONNX)            │
│    关键点检测 + 256维描述子提取        │
└──────┬──────────────────────┬───────┘
       │                      │
  ref_keypoints (N_ref)  cur_keypoints (N_cur)
  ref_descriptors        cur_descriptors
       │                      │
       └──────────┬───────────┘
                  ▼
┌──────────────────────────────────────┐
│         LightGlue (ONNX)             │
│    Transformer 注意力匹配             │
│    自适应深度 (1-9 层)                │
└──────────────┬───────────────────────┘
               │
          matches (M pairs)
          + confidence scores
               │
               ├──────────────────────┐
               ▼                      ▼
┌──────────────────────┐  ┌──────────────────────┐
│  现有 3D-3D SVD      │  │  现有 PnP / Homog     │
│  (取匹配点深度)       │  │  (纯 2D fallback)     │
│  → L0: 6-DOF        │  │  → L1/L2             │
└──────────────────────┘  └──────────────────────┘
```

**关键点**: SuperPoint + LightGlue 只替代 `feature_matcher.py` 里的"特征提取 + 匹配"两步。后面的 3D-3D SVD / PnP / Homography 流程完全不变。

### 3.3 与现有代码的接口

```python
# feature_matcher.py 新增一个 extract 函数:
def extract_keypoints_3d_superpoint(
    bgr: np.ndarray,
    depth_m: np.ndarray,
    fx, fy, cx, cy,
    cfg, sp_engine  # SuperPoint ONNX 引擎
) -> dict:
    """返回和 extract_keypoints_3d() 完全相同的格式"""
    # SuperPoint 提取关键点 + 描述子
    keypoints, descriptors = sp_engine.infer(gray)
    # 深度采样 (复用现有 _sample_depth)
    # 反投影 (复用现有 _backproject)
    return {
        "keypoints": [...],
        "descriptors": np.ndarray,  # float32 (N, 256)
        "uv": np.ndarray,
        "xyz_cam": np.ndarray,
        "xyz_mask": np.ndarray,
        "depth_vals": np.ndarray,
    }

# 匹配部分:
def _match_descriptors_sp(desc_ref, desc_cur, lg_engine):
    """LightGlue 匹配，替代 BFMatcher / FlannBasedMatcher"""
    matches = lg_engine.infer(keypoints_ref, desc_ref, keypoints_cur, desc_cur)
    return [(m[0], m[1]) for m in matches]  # (ref_idx, cur_idx)
```

`match_and_estimate()` 只需要改两行：提取函数 + 匹配函数。后面的 RANSAC SVD / PnP / Homography 不动。

## 4. ONNX 部署方案

### 4.1 模型来源

```
fabio-sim/LightGlue-ONNX (GitHub)
├── superpoint.onnx      # ~5MB, FP16
├── lightglue.onnx       # ~15MB, FP16
└── superpoint_lightglue.onnx  # 融合版 ~18MB
```

### 4.2 推理引擎

```python
import onnxruntime as ort

# AGX Orin: 使用 TensorRT EP (GPU 加速)
providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_max_workspace_size': 1 << 30,  # 1GB
        'trt_fp16_enable': True,
    }),
    'CUDAExecutionProvider',  # fallback
]

session = ort.InferenceSession("superpoint.onnx", providers=providers)
```

### 4.3 参考帧预提取

参考帧的关键点和描述子**离线预计算**，运行时只加载：

```python
# 离线 (录制模板时一次性执行)
ref_data = extract_keypoints_3d_superpoint(ref_bgr, ref_depth, fx, fy, cx, cy, cfg, sp)
np.savez("kitchen_table_sp.npz",
         keypoints=ref_data["uv"],
         descriptors=ref_data["descriptors"],
         xyz_cam=ref_data["xyz_cam"],
         xyz_mask=ref_data["xyz_mask"])
```

模板文件扩展：
```json
{
  "name": "kitchen_table",
  "version": 3,
  "keyframes": [{
    "image_path": "kitchen_table_000.jpg",
    "depth_path": "kitchen_table_000_depth.npy",
    "superpoint_path": "kitchen_table_000_sp.npz",
    "camera_intrinsics": {"fx": 382.0, "fy": 382.0, "cx": 322.0, "cy": 240.0}
  }]
}
```

## 5. 性能预估 (AGX Orin)

| 步骤 | TensorRT FP16 | CPU FP32 | 备注 |
|------|--------------|----------|------|
| SuperPoint 推理 (640×480) | ~15ms | ~60ms | 单帧 |
| LightGlue 匹配 (500 对) | ~15ms | ~80ms | 自适应层数 |
| 深度采样 + 反投影 | ~2ms | ~2ms | 复用现有代码 |
| RANSAC SVD | ~5ms | ~5ms | 复用现有代码 |
| **总计 (TensorRT)** | **~37ms** | - | 满足 4Hz match_hz |
| **总计 (CPU)** | - | **~147ms** | 可接受，~7Hz |

TensorRT FP16 预估基于:
- SuperPoint: ~10M FLOPs，AGX Orin FP16 ~200 TOPS → ~10-15ms
- LightGlue: Transformer 注意力，500 点 × 9 层 → ~15ms
- 实际需要基准测试确认

## 6. 实现计划

### Phase 0: SIFT 基线 (0.5 天)

不改架构，验证视角改善：
```python
# person_follow_all.yaml
detector: "sift"  # 从 "orb" 改为 "sift"
```
- 验收: ±25° 成功匹配
- 这一步不需要任何代码改动

### Phase 1: SuperPoint ONNX 集成 (2-3 天)

1. **环境准备**
   ```bash
   pip install onnxruntime   # ARM64 wheel 可用
   # TensorRT EP: 随 JetPack 自带
   ```
2. **模型下载**
   ```bash
   # 从 fabio-sim/LightGlue-ONNX 下载 superpoint.onnx + lightglue.onnx
   wget https://huggingface.co/fabio-sim/LightGlue-ONNX/resolve/main/superpoint.onnx
   wget https://huggingface.co/fabio-sim/LightGlue-ONNX/resolve/main/lightglue.onnx
   ```
3. **feature_matcher.py 改动**
   - 新增 `SuperPointEngine` 类封装 ONNX 推理
   - 新增 `LightGlueEngine` 类封装匹配
   - 新增 `extract_keypoints_3d_superpoint()` 函数
   - `match_and_estimate()` 根据 `cfg.detector` 选择 ORB/SIFT/SuperPoint
4. **模板录制脚本更新**
   - `p0_record_trajectory.py` 或单独脚本：录制时同时生成 `.npz`
5. **验收**: ±45° 成功返回，延迟 <100ms

### Phase 2: 优化 (1-2 天)

- 融合 SuperPoint + LightGlue 为单一 ONNX（减少推理调用开销）
- TensorRT FP16 精度验证
- 描述子缓存（当前帧跨 tick 复用）
- 参考帧多视角 keyframe（±30° 各一张，匹配时选最佳）

### Phase 3: head 辅助对齐 (可选)

```
匹配失败 (视角差 >60°)
    ↓
head 向参考帧方向旋转搜索 (±90° range, 10° step)
    ↓
每步尝试匹配
    ↓
匹配成功 → 切回正常 servo
```

利用 head 可以转 ±90° 的特性，先让 D455 粗略对齐，再做精确匹配。结合 SuperPoint 的 ±60° 容忍度，理论上覆盖 ±150° 范围。

## 7. 风险与备选

| 风险 | 概率 | 应对 |
|------|------|------|
| ONNX TensorRT EP 在 AGX 上不稳定 | 中 | 回退 CUDAExecutionProvider 或 CPU EP |
| SuperPoint 关键点在弱纹理场景少于 ORB | 低 | 混合策略: SuperPoint 失败时 fallback ORB |
| LightGlue 匹配耗时 >100ms | 低 | 减少 max_keypoints (500→200) 或使用 Disk 替代 |
| 融合 ONNX 模型 shape 兼容问题 | 中 | 分别推理，不融合 |
| AGX Orin 内存不足 (模型 + YOLO) | 低 | SuperPoint ~5MB，LightGlue ~15MB，总共 <25MB |

## 8. 降级策略

```
SuperPoint + LightGlue (首选)
    │ 失败 (关键点太少 / 匹配太少)
    ▼
SIFT + FLANN (次选, 无需 ONNX)
    │ 失败
    ▼
ORB + BFMatcher (兜底, 现有流程)
    │ 失败
    ▼
LOST
```

每级降级自动切换，对 `servo_estimator` 和 `scene_servo_node` 完全透明。

## 9. 与拆分后代码的关系

```
scene_servo/
├── feature_matcher.py          # ← 改动主要在这个文件
│   ├── FeatureMatcherCfg       #   新增 detector="superpoint"
│   ├── SuperPointEngine        #   新增类
│   ├── LightGlueEngine         #   新增类
│   ├── extract_keypoints_3d_superpoint()  # 新增函数
│   ├── extract_keypoints_3d()            # 现有 (ORB/SIFT)
│   └── match_and_estimate()              # 现有, 加 sp 分支
│
├── scene_template_servo.py     # 不变
├── person_yolo_servo.py        # 不变
├── scene_servo_node.py         # 不变 (只改 yaml 参数)
└── models/                     # 新增目录
    ├── superpoint.onnx
    └── lightglue.onnx
```

改动集中在 `feature_matcher.py` 一个文件，其他模块零改动。

## 10. 开放问题

1. **SuperPoint 输入尺寸**: 640×480 直接推理还是 resize 到 320×240 再推理？后者更快但关键点定位精度下降。
2. **LightGlue 最大匹配点数**: 设 500 还是 1000？影响延迟和精度。
3. **是否需要 APAP 对齐辅助**: SuperPoint 匹配后是否需要局部单应变换验证？
4. **TensorRT engine 缓存**: 首次推理慢（构建 TRT engine ~30s），需要预编译缓存。
