#!/bin/bash
# sherpa-onnx HTTPS VAD + SenseVoice 智能分段识别启动脚本
# VAD 自动检测停顿切分句子 + SenseVoice 高准确率识别

cd /home/tsingwin/apps/sherpa-onnx

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sherpa

# 安装依赖
echo "检查依赖..."
pip install flask flask-sock pyopenssl numpy -q

echo ""
echo "=========================================="
echo "  sherpa-onnx VAD + SenseVoice 智能分段"
echo "=========================================="
echo ""
echo "架构: Silero VAD 自动断句 + SenseVoice 识别"
echo "支持: 中文/英文/日语/韩语/粤语"
echo "GPU: CUDA 加速"
echo "特点: 自动切分句子，准确率高，不会累积变慢"
echo ""

# 运行 WebUI（默认一组偏“放宽、少丢句”：更敏感拾音、更长段、更短段也识别）
python webui_vad_sensevoice.py --port 6010 \
  --vad-threshold 0.35 \
  --vad-min-silence 0.20 \
  --vad-min-speech 0.12 \
  --vad-max-speech 12.0 \
  --min-segment-seconds 0.03 \
  --segment-context-seconds 0.18 \
  --pre-speech-context-seconds 0.22 \
  --partial-interval 0.40 \
  --partial-max-seconds 5.0 \
  --preroll-overlap-search-seconds 0.25
