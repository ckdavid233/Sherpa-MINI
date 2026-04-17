# sherpa-onnx：VAD + SenseVoice Web 分段识别（当前固定版）

本目录下的 **`webui_vad_sensevoice.py`** 提供 HTTPS + WebSocket 的浏览器语音识别界面：**Silero VAD 按停顿切段**，每段再送 **SenseVoice** 离线识别，适合中文/英文/日语/韩语/粤语等场景。  
**当前推荐用法以 `start_vad_sensevoice.sh` 中的参数为准**（偏「放宽、少丢句」的一版）。

## 功能概览

- **断句**：VAD 检测语音活动与静音，自动分段。
- **识别**：每段独立调用 SenseVoice，避免流式模型长句越来越慢的问题。
- **实时预览**：浏览器侧通过 WebSocket 推送 `partial` 与已确认 `result`。
- **HTTPS**：默认自签证书 `cert.pem`（不存在时脚本依赖 pyopenssl 生成）；调试可用 `--no-ssl`。

## 环境要求

- **Python**：建议使用 Conda 环境（示例脚本中为 `sherpa`）。
- **依赖**：`flask`、`flask-sock`、`pyopenssl`、`numpy`、`sherpa_onnx`（及 CUDA 相关运行时，若使用 GPU）。
- **模型路径**（代码内写死，请按需改成你本机路径）：
  - VAD：`models/silero_vad.onnx`
  - SenseVoice：`models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/`（含 `model.int8.onnx`、`tokens.txt` 等）

## 一键启动（推荐）

```bash
chmod +x start_vad_sensevoice.sh
./start_vad_sensevoice.sh
```

脚本会：进入项目目录、激活 conda、安装依赖、并用**下面这一组默认参数**启动服务。

浏览器访问（HTTPS 默认）：`https://localhost:6010`（局域网把主机名换成机器 IP）。

## 当前固定默认参数（与 `start_vad_sensevoice.sh` 一致）

| 参数 | 值 | 说明 |
|------|-----|------|
| `--port` | `6010` | 监听端口 |
| `--vad-threshold` | `0.35` | VAD 阈值，略低更易拾取弱起音 |
| `--vad-min-silence` | `0.20` | 判定为停顿所需最短静音（秒），略大减少切碎 |
| `--vad-min-speech` | `0.12` | 最短语音时长（秒），略小减少短音被丢 |
| `--vad-max-speech` | `12.0` | 单段最长语音（秒） |
| `--min-segment-seconds` | `0.03` | 过短分段是否跳过（秒），略小少丢尾音 |
| `--segment-context-seconds` | `0.18` | 分段识别尾部补上下文（秒） |
| `--pre-speech-context-seconds` | `0.22` | 句首 pre-roll（秒） |
| `--partial-interval` | `0.40` | partial 解码最小间隔（秒） |
| `--partial-max-seconds` | `5.0` | partial 只用最近 N 秒音频，控制长句算力 |
| `--preroll-overlap-search-seconds` | `0.25` | pre-roll 与分段重叠去重搜索范围（秒） |

修改体验时，**优先只改 `start_vad_sensevoice.sh` 里对应行**，无需改 Python 源码中的默认常量。

## 手动启动示例

```bash
conda activate sherpa
cd /path/to/sherpa-onnx
pip install flask flask-sock pyopenssl numpy

python webui_vad_sensevoice.py --port 6010 --help   # 查看全部参数

# HTTP 调试（无证书）
python webui_vad_sensevoice.py --no-ssl --port 6010 \
  --vad-threshold 0.35 --vad-min-silence 0.20 --vad-min-speech 0.12
```

## 端口占用说明

若启动报 **Address already in use**，请先检查 `6010` 是否被占用。  
在部分 Linux 上 **`sshd` 的本地转发**会占用 `127.0.0.1:6010`，与监听 `0.0.0.0:6010` 冲突，可换端口：

```bash
python webui_vad_sensevoice.py --port 6011
```

## 常见问题（简）

- **句段偶尔丢字 / 切太碎**：略增大 `--vad-min-silence`（如 `0.22`～`0.25`），或略增大 `--vad-threshold` 减噪声误触发。
- **句首弱音听不清**：略增大 `--pre-speech-context-seconds`、`--segment-context-seconds`（代价是更易句首重复，需折中）。
- **长句实时变慢**：略减小 `--partial-max-seconds` 或略增大 `--partial-interval`。

## 相关文件

| 文件 | 作用 |
|------|------|
| `webui_vad_sensevoice.py` | Web 服务与 WebSocket 逻辑 |
| `start_vad_sensevoice.sh` | 当前固定参数的一键启动 |
| `templates/vad-sensevoice.html` | 由脚本首次运行时生成/覆盖（若使用模板） |

---

以上为当前仓库内「VAD + SenseVoice WebUI」固定可用版本说明；后续若只调参，**以 `start_vad_sensevoice.sh` 为准**即可。
