#!/bin/bash
# sherpa-onnx HTTPS VAD + SenseVoice 智能分段识别启动脚本
# VAD 自动检测停顿切分句子 + SenseVoice 高准确率识别

cd "$(dirname "$0")"

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

# 监听端口：默认 6010；若端口被占用可换端口，例如: PORT=6011 ./start_vad_sensevoice.sh
#
# 为何重启后仍提示 6010 被占用、且 ss/lsof 里像 ssh 在占？
#   - 用 Cursor/VS Code Remote SSH 时，若在「端口转发 / Ports」里转发过 6010，sshd 会在板子上
#     监听该端口（或显示为 ssh 相关），与你想再 bind 的 Web 服务冲突。应在 IDE 里删除该转发，
#     或换用未转发的端口访问（例如只在内网浏览器打开 https://板子IP:6010，不要从本机转发同号端口）。
#   - 需要强制释放本机监听时（会结束占用该端口的进程）: FREE_PORT=1 ./start_vad_sensevoice.sh
PORT="${PORT:-6010}"
echo "监听端口: ${PORT}（可用环境变量 PORT 覆盖）"
if [ "${FREE_PORT:-0}" = "1" ]; then
    echo "FREE_PORT=1: 尝试释放 TCP ${PORT} ..."
    fuser -k "${PORT}"/tcp 2>/dev/null || true
    sleep 0.3
fi
echo ""

# 运行 WebUI（默认偏少「吞句」：略长静音才切段、句首/尾多留上下文；要更碎句可略减 vad-min-silence）
# 若已安装带 neg_threshold 的 sherpa_onnx，可加: --vad-neg-threshold 0.34
python webui_vad_sensevoice.py --port "${PORT}" \
  --vad-threshold 0.38 \
  --vad-min-silence 0.22 \
  --vad-min-speech 0.10 \
  --vad-max-speech 12.0 \
  --min-segment-seconds 0.03 \
  --segment-context-seconds 0.30 \
  --pre-speech-context-seconds 0.34 \
  --partial-interval 0.40 \
  --partial-max-seconds 5.0 \
  --preroll-overlap-search-seconds 0.25
