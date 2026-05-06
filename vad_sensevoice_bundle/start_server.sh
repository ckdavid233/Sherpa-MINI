#!/usr/bin/env bash
# 无 conda 硬编码的启动脚本；参数与仓库根目录 start_vad_sensevoice.sh 对齐。
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-6010}"
echo "安装/检查 Python 依赖..."
pip install -r requirements.txt -q

if [[ "${FREE_PORT:-0}" == "1" ]]; then
  echo "FREE_PORT=1: 尝试释放 TCP ${PORT} ..."
  fuser -k "${PORT}"/tcp 2>/dev/null || true
  sleep 0.3
fi

# CPU 推理示例: SHERPA_ONNX_PROVIDER=cpu ./start_server.sh
export SHERPA_ONNX_PROVIDER="${SHERPA_ONNX_PROVIDER:-cuda}"

exec python webui_vad_sensevoice.py --port "${PORT}" \
  --vad-threshold 0.38 \
  --vad-min-silence 0.22 \
  --vad-min-speech 0.10 \
  --vad-max-speech 12.0 \
  --min-segment-seconds 0.03 \
  --segment-context-seconds 0.30 \
  --pre-speech-context-seconds 0.34 \
  --partial-interval 0.40 \
  --partial-max-seconds 5.0 \
  --preroll-overlap-search-seconds 0.25 \
  "$@"
