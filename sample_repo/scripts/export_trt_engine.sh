#!/usr/bin/env bash
# ============================================================
# export_trt_engine.sh
# 在 Jetson AGX Orin 上将 yolov8s.pt → ONNX → TensorRT FP16 engine
#
# 用法:
#   bash scripts/export_trt_engine.sh [model.pt] [imgsz]
#
# 示例:
#   bash scripts/export_trt_engine.sh models/yolov8s.pt 512
# ============================================================
set -euo pipefail

MODEL_PT="${1:-models/yolov8s.pt}"
IMGSZ="${2:-512}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$PROJECT_DIR/models"

cd "$PROJECT_DIR"

# 文件名（不带扩展名）
BASE="$(basename "$MODEL_PT" .pt)"
ONNX_FILE="$MODEL_DIR/${BASE}.onnx"
ENGINE_FILE="$MODEL_DIR/${BASE}_fp16_${IMGSZ}.engine"

echo "============================================================"
echo "YOLO → TensorRT Engine 导出"
echo "  输入:  $MODEL_PT"
echo "  ONNX:  $ONNX_FILE"
echo "  Engine: $ENGINE_FILE"
echo "  imgsz: $IMGSZ"
echo "============================================================"

# Step 1: PT → ONNX (不需要 CUDA PyTorch)
if [ -f "$ONNX_FILE" ]; then
    echo "[1/2] ONNX 已存在，跳过导出: $ONNX_FILE"
else
    echo "[1/2] 导出 ONNX..."
    python3 -c "
from ultralytics import YOLO
model = YOLO('$MODEL_PT')
model.export(format='onnx', imgsz=$IMGSZ, opset=17, simplify=True, dynamic=False)
print('ONNX export done.')
"
    # ultralytics 默认输出到同目录，移到 models/
    GENERATED_ONNX="$(dirname "$MODEL_PT")/${BASE}.onnx"
    if [ "$GENERATED_ONNX" != "$ONNX_FILE" ] && [ -f "$GENERATED_ONNX" ]; then
        mv "$GENERATED_ONNX" "$ONNX_FILE"
    fi
    echo "  ONNX: $(ls -lh "$ONNX_FILE" | awk '{print $5}')"
fi

# Step 2: ONNX → TensorRT Engine (用 trtexec)
if [ -f "$ENGINE_FILE" ]; then
    echo "[2/2] Engine 已存在，跳过: $ENGINE_FILE"
else
    echo "[2/2] 构建 TensorRT FP16 engine (可能需要 5-15 分钟)..."
    
    TRTEXEC=""
    for p in /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec "$(which trtexec 2>/dev/null)"; do
        if [ -x "$p" ] 2>/dev/null; then
            TRTEXEC="$p"
            break
        fi
    done
    
    if [ -z "$TRTEXEC" ]; then
        echo "错误: 找不到 trtexec"
        echo "尝试用 Python TensorRT API 导出..."
        python3 -c "
import tensorrt as trt
import os

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open('$ONNX_FILE', 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError('ONNX parse failed')

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
config.set_flag(trt.BuilderFlag.FP16)

engine_bytes = builder.build_serialized_network(network, config)
with open('$ENGINE_FILE', 'wb') as f:
    f.write(engine_bytes)
print(f'Engine saved: $ENGINE_FILE ({os.path.getsize(\"$ENGINE_FILE\") / 1024 / 1024:.1f} MB)')
"
    else
        "$TRTEXEC" \
            --onnx="$ONNX_FILE" \
            --saveEngine="$ENGINE_FILE" \
            --fp16 \
            --workspace=4096 \
            --verbose 2>&1 | tail -20
    fi
    
    echo "  Engine: $(ls -lh "$ENGINE_FILE" | awk '{print $5}')"
fi

echo ""
echo "============================================================"
echo "导出完成!"
echo "  Engine: $ENGINE_FILE"
echo ""
echo "在 launch 中使用:"
echo "  'model_path': '$ENGINE_FILE',"
echo "============================================================"
