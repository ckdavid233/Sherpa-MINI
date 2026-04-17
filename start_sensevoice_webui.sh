#!/bin/bash
# sherpa-onnx HTTPS SenseVoice 语音识别启动脚本
# 更高识别准确率，支持多语言

cd /home/tsingwin/apps/sherpa-onnx

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sherpa

# 安装依赖
echo "检查依赖..."
pip install flask flask-sock pyopenssl numpy -q

echo ""
echo "=========================================="
echo "  sherpa-onnx SenseVoice 语音识别服务"
echo "=========================================="
echo ""
echo "模型: SenseVoice (INT8 量化)"
echo "支持: 中文/英文/日语/韩语/粤语"
echo "GPU: CUDA 加速"
echo ""

# 运行 WebUI
python webui_streaming_sensevoice.py --port 6008
