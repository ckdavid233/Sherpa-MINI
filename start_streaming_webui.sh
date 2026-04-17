#!/bin/bash
# sherpa-onnx HTTPS WebSocket 实时流式识别启动脚本

cd /home/tsingwin/apps/sherpa-onnx

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sherpa

# 安装依赖
echo "检查依赖..."
pip install flask flask-sock pyopenssl numpy -q

echo ""
echo "=========================================="
echo "  sherpa-onnx 实时流式 ASR WebUI"
echo "=========================================="
echo ""

# 运行 WebUI
python webui_streaming.py --port 6007
