本目录应包含（运行 `../download_models.sh` 可自动下载）：

| 路径 | 说明 |
|------|------|
| `silero_vad.onnx` | Silero VAD |
| `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt` | SenseVoice 词表 |
| `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx` 或 `model.int8.onnx` | SenseVoice 权重（有 FP32 则优先用 FP32） |

模型路径写死在 `webui_vad_sensevoice.py` 的 `_MODELS_DIR`（即本包内的 `models/`），复制整包到其他机器时请一并复制本目录或重新执行下载脚本。
