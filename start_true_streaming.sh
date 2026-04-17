#!/bin/bash
# sherpa-onnx HTTPS 原生流式语音识别启动脚本
# 真正原生流式，每个音频块直接解码，响应超快

cd /home/tsingwin/apps/sherpa-onnx

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sherpa

# 安装依赖
echo "检查依赖..."
pip install flask flask-sock pyopenssl numpy -q

echo ""
echo "=========================================="
echo "  sherpa-onnx 原生流式 ASR 服务"
echo "=========================================="
echo ""
echo "模型: Zipformer 原生流式 (INT8 量化)"
echo "支持: 中文/英文 双语"
echo "GPU: CUDA 加速"
echo "特点: 真正流式，边说话边出字，超低延迟"
echo ""

# 运行 WebUI
python webui_true_streaming.py --port 6009
