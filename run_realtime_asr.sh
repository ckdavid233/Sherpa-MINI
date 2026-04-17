#!/bin/bash
# 实时 ASR 启动脚本 - GPU 版本

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sherpa

# 模型目录
MODEL_DIR="/home/tsingwin/apps/sherpa-onnx/models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"

# 检查模型是否存在
if [ ! -d "$MODEL_DIR" ]; then
    echo "模型目录不存在: $MODEL_DIR"
    exit 1
fi

echo "=========================================="
echo "  sherpa-onnx 实时 ASR (GPU 加速)"
echo "=========================================="
echo ""
echo "模型: 中英文双语流式 Zipformer (INT8)"
echo "设备: Jetson Orin AGX (CUDA 12.6)"
echo ""
echo "请对着麦克风说话，按 Ctrl+C 退出"
echo "=========================================="
echo ""

# 运行实时识别
cd /home/tsingwin/apps/sherpa-onnx/python-api-examples

python speech-recognition-from-microphone.py \
    --tokens="$MODEL_DIR/tokens.txt" \
    --encoder="$MODEL_DIR/encoder-epoch-99-avg-1.int8.onnx" \
    --decoder="$MODEL_DIR/decoder-epoch-99-avg-1.int8.onnx" \
    --joiner="$MODEL_DIR/joiner-epoch-99-avg-1.int8.onnx" \
    --provider=cuda \
    --decoding-method=greedy_search
