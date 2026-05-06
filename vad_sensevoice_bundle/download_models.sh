#!/usr/bin/env bash
# 下载 VAD + SenseVoice 所需模型到本目录 models/ 下（与 webui_vad_sensevoice.py 约定路径一致）
set -euo pipefail
cd "$(dirname "$0")"
MODELS_DIR="models"
mkdir -p "${MODELS_DIR}"
cd "${MODELS_DIR}"

SILERO_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
SV_NAME="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
SV_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${SV_NAME}.tar.bz2"

if [[ ! -f silero_vad.onnx ]]; then
  echo "下载 Silero VAD..."
  curl -fSL -O "${SILERO_URL}"
else
  echo "已存在: silero_vad.onnx"
fi

if [[ ! -f "${SV_NAME}/tokens.txt" ]]; then
  echo "下载 SenseVoice (${SV_NAME})..."
  curl -fSL -O "${SV_URL}"
  tar xf "${SV_NAME}.tar.bz2"
  rm -f "${SV_NAME}.tar.bz2"
else
  echo "已存在目录: ${SV_NAME}/"
fi

echo "完成。请确认存在:"
echo "  ${MODELS_DIR}/silero_vad.onnx"
echo "  ${MODELS_DIR}/${SV_NAME}/tokens.txt"
echo "  ${MODELS_DIR}/${SV_NAME}/model.onnx 或 model.int8.onnx"
