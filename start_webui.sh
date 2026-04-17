#!/bin/bash
# sherpa-onnx HTTPS WebUI 启动脚本

cd /home/tsingwin/apps/sherpa-onnx

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sherpa

# 安装依赖
echo "检查依赖..."
pip install flask pyopenssl soundfile -q

echo ""
echo "=========================================="
echo "  sherpa-onnx HTTPS WebUI ASR 服务"
echo "=========================================="
echo ""

# 运行 WebUI
python webui_asr.py --port 6006
