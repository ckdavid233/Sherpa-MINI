#!/bin/bash
# sherpa-onnx HTTPS VAD + SenseVoice 智能分段识别启动脚本
# VAD 自动检测停顿切分句子 + SenseVoice 高准确率识别
#
# 用法:
#   ./start_vad_sensevoice.sh                   # 默认 6010 端口
#   PORT=6011 ./start_vad_sensevoice.sh         # 指定端口
#   FREE_PORT=1 ./start_vad_sensevoice.sh       # 强制释放端口后启动
#   LANGUAGE=en ./start_vad_sensevoice.sh       # 切换识别语言
#
# 环境变量:
#   PORT        监听端口 (默认 6010)
#   FREE_PORT   设为 1 强制释放端口
#   LANGUAGE    识别语言: auto/zh/en/ja/ko/yue (默认 zh)
#   CONDA_ENV   conda 环境名 (默认 sherpa)
#   PROVIDER    ONNX 推理后端: cuda/cpu (默认自动检测)

set -euo pipefail
cd "$(dirname "$0")"

# ============================================
# 1. 检测 conda
# ============================================
CONDA_ENV="${CONDA_ENV:-sherpa}"
CONDA_BASE=""

# 按优先级查找 conda 安装路径
for candidate in \
    "$HOME/anaconda3" \
    "$HOME/miniconda3" \
    "$HOME/miniforge3" \
    /opt/conda \
    /opt/anaconda3 \
    /opt/miniconda3; do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$candidate"
        break
    fi
done

if [ -z "$CONDA_BASE" ]; then
    echo "错误: 未找到 conda 安装。请安装 Anaconda/Miniconda/Miniforge。"
    echo "  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
    echo "  bash Miniforge3-Linux-aarch64.sh"
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

# ============================================
# 2. 检查 / 创建 conda 环境
# ============================================
if conda env list | grep -q "^${CONDA_ENV}\s"; then
    echo "[OK] conda 环境 '${CONDA_ENV}' 已存在"
else
    echo "[创建] conda 环境 '${CONDA_ENV}' 不存在，正在创建..."
    conda create -n "$CONDA_ENV" python=3.10 -y
    echo "[OK] 环境创建完成"
fi

conda activate "$CONDA_ENV"

# ============================================
# 3. 检查 Python 依赖（仅在缺失时安装）
# ============================================
echo "检查 Python 依赖..."
MISSING_PKGS=""

python -c "import flask" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS flask"
python -c "import flask_sock" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS flask-sock"
python -c "import OpenSSL" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS pyopenssl"
python -c "import numpy" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS numpy"
python -c "import sherpa_onnx" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS sherpa-onnx"

if [ -n "$MISSING_PKGS" ]; then
    echo "安装缺失的包:$MISSING_PKGS"
    pip install $MISSING_PKGS -q
    echo "[OK] 依赖安装完成"
else
    echo "[OK] 所有 Python 依赖已满足"
fi

# ============================================
# 4. 检查模型文件
# ============================================
MODEL_DIR="models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
VAD_MODEL="models/silero_vad.onnx"

MISSING_MODELS=""
[ -f "$VAD_MODEL" ] || MISSING_MODELS="$MISSING_MODELS\n  - $VAD_MODEL (Silero VAD)"
[ -f "$MODEL_DIR/model.onnx" ] || [ -f "$MODEL_DIR/model.int8.onnx" ] || \
    MISSING_MODELS="$MISSING_MODELS\n  - $MODEL_DIR/model.onnx (SenseVoice)"
[ -f "$MODEL_DIR/tokens.txt" ] || \
    MISSING_MODELS="$MISSING_MODELS\n  - $MODEL_DIR/tokens.txt"

if [ -n "$MISSING_MODELS" ]; then
    echo ""
    echo "========================================"
    echo "  错误: 缺少模型文件"
    echo "========================================"
    echo -e "缺少:$MISSING_MODELS"
    echo ""
    echo "请下载模型:"
    echo "  Silero VAD:"
    echo "    wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx -O $VAD_MODEL"
    echo ""
    echo "  SenseVoice:"
    echo "    pip install sherpa-onnx -q"
    echo "    python -c \"import sherpa_onnx; print(sherpa_onnx.__file__)\""
    echo "    # 从 https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models 下载"
    echo "    # sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
    echo "    # 解压到 $MODEL_DIR"
    exit 1
fi
echo "[OK] 模型文件已就绪"

# ============================================
# 5. 自动检测推理后端
# ============================================
if [ -z "${PROVIDER:-}" ]; then
    if python -c "import sherpa_onnx; print(sherpa_onnx.__version__)" 2>/dev/null | grep -q "cuda"; then
        PROVIDER="cuda"
        echo "[检测] 推理后端: CUDA"
    else
        PROVIDER="cpu"
        echo "[检测] 推理后端: CPU（未安装 sherpa-onnx CUDA 版本）"
    fi
fi

# ============================================
# 6. 监听端口
# ============================================
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

LANGUAGE="${LANGUAGE:-auto}"

# ============================================
# 7. 启动服务
# ============================================
echo ""
echo "=========================================="
echo "  sherpa-onnx VAD + SenseVoice 智能分段"
echo "=========================================="
echo ""
echo "架构: Silero VAD 自动断句 + SenseVoice 识别"
echo "支持: 中文/英文/日语/韩语/粤语"
echo "语言: ${LANGUAGE}"
echo "后端: ${PROVIDER}"
echo "特点: 自动切分句子，准确率高，不会累积变慢"
echo ""

python webui_vad_sensevoice.py \
  --port "${PORT}" \
  --provider "${PROVIDER}" \
  --language "${LANGUAGE}" \
  --vad-threshold 0.38 \
  --vad-min-silence 0.22 \
  --vad-min-speech 0.10 \
  --vad-max-speech 12.0 \
  --min-segment-seconds 0.03 \
  --segment-context-seconds 0.25 \
  --pre-speech-context-seconds 0.35 \
  --partial-interval 0.40 \
  --partial-max-seconds 5.0 \
  --preroll-overlap-search-seconds 0.50
