# YOLO GPU 加速部署指南（Jetson AGX Orin / JetPack 6.0）

本文档说明如何将 person_follow 模块的 YOLOv8s 从 CPU 推理切换到 GPU 推理（CUDA / FP16 / TensorRT）。

## 环境信息

| 项目 | 版本 |
|------|------|
| 平台 | Jetson AGX Orin |
| JetPack | 6.0 (R36.3.0) |
| CUDA | 12.2 |
| cuDNN | 8.9.4 |
| TensorRT | 8.6.2 |
| Python | 3.10 |
| PyTorch | 2.4.0a0+nv24.07（Jetson 专用） |
| Ultralytics | 8.3.37 |

## 性能对比（实测 yolov8s, imgsz=512, 640x480 输入）

| 模式 | 推理延迟 | 相对加速 |
|------|---------|---------|
| CPU (numpy) | ~894 ms | 1x |
| GPU FP32 | ~27 ms | **33x** |
| GPU FP16 | ~22 ms | **41x** |
| TensorRT FP16 | ~10-15 ms（预期） | **60-90x** |

---

## 前置条件：CUDA 环境变量

**所有 GPU 方案的前提**——必须先让系统找到 CUDA 和 cusparseLt 库。

### 方法一：写入 ldconfig（推荐，一劳永逸）

```bash
echo "/home/dev/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib" | sudo tee /etc/ld.so.conf.d/cusparselt.conf
echo "/usr/local/cuda-12.2/lib64" | sudo tee /etc/ld.so.conf.d/cuda.conf
sudo ldconfig
```

### 方法二：写入 ~/.bashrc

```bash
cat >> ~/.bashrc << 'EOF'
# CUDA + cusparseLt for PyTorch Jetson
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/home/dev/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib:/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}
EOF
source ~/.bashrc
```

### 方法三：每次启动前手动 export

```bash
export LD_LIBRARY_PATH=/home/dev/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
```

### 验证

```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
# 预期输出：CUDA: True, Device: Orin
```

> 如果不设置这些环境变量，`import torch` 会报错 `ImportError: libcusparseLt.so.0`，
> YOLO 会静默降级到 CPU 推理（894ms vs 22ms）。

---

## 方案一：CUDA GPU 推理（最简单，推荐先用）

**无需额外安装**，设好环境变量后即可使用。

launch 参数确认（`person_follow_all.launch.py` 中 `person_follow_node`）：

```python
'device': '0',        # '0' = GPU:0, 'cpu' = CPU
'use_half': True,     # FP16 半精度（GPU 模式下自动生效，额外快 ~20%）
```

启动命令：

```bash
cd /home/dev/midea_humanoid_robot
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=55
export LD_LIBRARY_PATH=/home/dev/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
source install/setup.bash

ros2 launch person_follow person_follow_all.launch.py
```

验证 GPU 推理生效：

```bash
# 日志中应看到 "Loading YOLO: ... device=0"
# 用 jtop 观察 GPU 利用率 > 0%
jtop
```

如需强制 CPU（调试用），改 launch 参数：

```python
'device': 'cpu',
'use_half': False,
```

---

## 方案二：TensorRT 加速（最快，推荐生产环境）

TensorRT 是 Jetson 上最快的推理方式。

> 参考：同 workspace 下 `face_tracking` 已在用 TensorRT engine：
> `/home/dev/midea_humanoid_robot/install/share/face_tracking/weights/yolov8n-face.engine`

### 步骤

#### 1. 导出 TensorRT engine

```bash
# 确保 CUDA 环境变量已设置
export LD_LIBRARY_PATH=/home/dev/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

cd /home/dev/midea_humanoid_robot/src/person_follow/models

# 导出 FP16 TensorRT engine（首次导出需要 5-15 分钟）
yolo export model=yolov8s.pt format=engine imgsz=512 half=True device=0
```

导出完成后会生成 `yolov8s.engine`。

#### 2. 安装 engine 到包里

```bash
# 方法 A：直接放到 install 目录（快，不需重新编译）
cp yolov8s.engine /home/dev/midea_humanoid_robot/install/share/person_follow/models/

# 方法 B：放到源码目录，修改 setup.py 后重新编译（规范）
cp yolov8s.engine /home/dev/midea_humanoid_robot/src/person_follow/models/
```

如果用方法 B，需要修改 `setup.py` 的 data_files：

```python
(os.path.join('share', package_name, 'models'), glob('models/*.pt') + glob('models/*.engine')),
```

然后重新编译：

```bash
cd /home/dev/midea_humanoid_robot
colcon build --merge-install --packages-select person_follow
source install/setup.bash
```

#### 3. 修改 launch 参数指向 engine

在 `person_follow_all.launch.py` 的 `person_follow_node` 参数中添加：

```python
'model_path': '/home/dev/midea_humanoid_robot/install/share/person_follow/models/yolov8s.engine',
```

> 代码使用 `ultralytics.YOLO()`，会根据文件后缀自动选择 PyTorch (.pt) 或 TensorRT (.engine) 推理。

#### 4. 验证 TensorRT 推理

```bash
export LD_LIBRARY_PATH=/home/dev/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

python3 -c "
from ultralytics import YOLO
import numpy as np, time

model = YOLO('/home/dev/midea_humanoid_robot/src/person_follow/models/yolov8s.engine')
dummy = np.random.randint(0,255,(480,640,3), dtype=np.uint8)
model.predict(dummy, verbose=False, imgsz=512)  # warmup
t0 = time.time()
for _ in range(50):
    model.predict(dummy, verbose=False, imgsz=512)
ms = (time.time()-t0)/50*1000
print(f'TensorRT FP16: {ms:.1f} ms/frame')
"
```

### TensorRT 注意事项

- `.engine` 文件**不可跨平台**，必须在目标 Jetson 上导出
- 更换 `imgsz` 或模型版本后**必须重新导出**
- 导出过程 GPU 内存占用较高，建议先关闭其他 GPU 进程
- 同一台机器上导出的 engine 可以在同型号机器间复制使用

---

## requirements.txt（GPU 版）

当前机器已装好 GPU 依赖。以下是全新 Jetson 环境的安装步骤：

```txt
# Jetson AGX Orin (JetPack 6.0, CUDA 12.2) GPU 版依赖
# 注意：PyTorch 必须使用 NVIDIA Jetson 专用版本，不要 pip install torch！

# ========== 核心依赖 ==========
numpy>=1.23,<2.0
opencv-python>=4.8,<4.11
ultralytics>=8.2,<9.0
mediapipe>=0.10.14,<0.11
pyttsx3>=2.90,<3.0

# ========== GPU 专用（手动安装，见下方步骤） ==========
# PyTorch Jetson 版 wheel：
# wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07-cp310-cp310-linux_aarch64.whl
# pip3 install torch-2.4.0a0+3bcc3cddb5.nv24.07-cp310-cp310-linux_aarch64.whl
#
# torchvision Jetson 版（源码编译）：
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/

# Optional
langgraph>=0.2,<0.4
```

### 全新 Jetson 安装步骤

```bash
# 1. 安装 Jetson 版 PyTorch
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07-cp310-cp310-linux_aarch64.whl
pip3 install torch-2.4.0a0+3bcc3cddb5.nv24.07-cp310-cp310-linux_aarch64.whl

# 2. 安装 torchvision（从源码编译）
sudo apt install -y libjpeg-dev zlib1g-dev libpng-dev
git clone --branch v0.19.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.19.0
python3 setup.py install --user
cd ..

# 3. 安装其他依赖
pip3 install ultralytics>=8.2,<9.0 mediapipe>=0.10.14 pyttsx3>=2.90 langgraph>=0.2

# 4. 修复 CUDA 库路径
echo "/home/dev/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib" | sudo tee /etc/ld.so.conf.d/cusparselt.conf
echo "/usr/local/cuda-12.2/lib64" | sudo tee /etc/ld.so.conf.d/cuda.conf
sudo ldconfig

# 5. 验证
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# 预期：CUDA: True

# 6. （可选）导出 TensorRT engine
cd /home/dev/midea_humanoid_robot/src/person_follow/models
yolo export model=yolov8s.pt format=engine imgsz=512 half=True device=0
```

---

## 常见问题

### Q: `ImportError: libcusparseLt.so.0`
PyTorch nv24.07 依赖 cusparseLt，库文件在 `/home/dev/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib/` 但不在系统搜索路径中。按"前置条件"部分修复。

### Q: `torch.cuda.is_available()` 返回 False
- 检查 PyTorch 版本：`pip3 show torch`，应含 `nv24` 字样（Jetson 专用版）
- 如果是 x86 版本，需要卸载后安装 Jetson wheel
- 检查 CUDA：`ls /usr/local/cuda-12.2/`

### Q: TensorRT 导出报错 `out of memory`
关闭其他 GPU 进程后重试，或降低 `imgsz`。

### Q: 从 CPU 切到 GPU 后检测结果微不同
正常现象，FP16 有微小精度差异，不影响跟随效果。设 `use_half: False` 可获得完全一致的 FP32 结果。

### Q: person_follow_node 启动后仍用 CPU
1. 检查日志 `Loading YOLO: ... device=` 是否为 `0`
2. 确认 `LD_LIBRARY_PATH` 已设好（或 ldconfig 已配置）
3. 确认不是在 launch 前忘了 export 环境变量
