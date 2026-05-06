# VAD + SenseVoice 可复用后端包

本目录从 sherpa-onnx 仓库中的 `start_vad_sensevoice.sh` / `webui_vad_sensevoice.py` 抽成**可整体复制**到其它项目的一小套文件：Silero VAD 断句 + SenseVoice 离线识别，经 WebSocket 提供实时分段与 partial 预览。

与仓库根目录脚本的区别：本包内 `webui_vad_sensevoice.py` 支持环境变量 `SHERPA_ONNX_PROVIDER`（默认 `cuda`），便于在无 GPU 机器上设为 `cpu`。

---

## 1. 新建 Python 环境需要安装哪些包？

在目标环境执行：

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含：

| 包 | 用途 |
|----|------|
| `flask` | HTTP 服务 |
| `flask-sock` | WebSocket |
| `pyopenssl` | 默认 HTTPS 时生成/读取自签证书 |
| `numpy` | 音频与数值处理 |
| `sherpa-onnx` | VAD + SenseVoice 推理绑定 |

**关于 `sherpa-onnx`：** 需与机器上的 **ONNX Runtime**（CPU 或 GPU）版本匹配。Jetson、CUDA 等环境常用预编译 wheel 或按 [sherpa-onnx 官方文档](https://github.com/k2-fsa/sherpa-onnx) 自行编译；不要假设任意 `pip install sherpa-onnx` 在板子上一定一次成功。

**无 UI 示例客户端**（仅跑示例时需要）：

```bash
pip install websocket-client
python example_ws_client.py --wav /path/to/your_16k_mono.wav
```

---

## 2. 复制本模块时目录里要有什么？模型与参数

### 2.1 目录结构（最小可运行）

```
vad_sensevoice_bundle/
├── README.md                 # 本说明
├── requirements.txt
├── download_models.sh        # 下载模型
├── start_server.sh           # 与原版 start_vad_sensevoice.sh 同参启动
├── webui_vad_sensevoice.py   # 服务主程序（模型路径相对本文件所在目录的 models/）
├── example_ws_client.py      # 无 UI 调用示例
└── models/                   # 见下表；可运行 download_models.sh 填充
    ├── silero_vad.onnx
    └── sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/
        ├── tokens.txt
        └── model.onnx 或 model.int8.onnx
```

首次准备模型：

```bash
chmod +x download_models.sh start_server.sh
./download_models.sh
```

模型官方下载地址（与脚本一致）：

- VAD：`https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx`
- SenseVoice 目录包：`https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2`

### 2.2 推理设备（CPU / CUDA）

默认与原版一致使用 CUDA。若要用 CPU：

```bash
export SHERPA_ONNX_PROVIDER=cpu
./start_server.sh
```

或：

```bash
SHERPA_ONNX_PROVIDER=cpu python webui_vad_sensevoice.py --no-ssl --port 6010 ...
```

### 2.3 与 `start_vad_sensevoice.sh` 对齐的命令行参数

下列为 `start_server.sh` 中固定的一组（偏「少丢句」）；可直接改脚本或改为命令行覆盖：

| 参数 | 当前值 | 含义摘要 |
|------|--------|----------|
| `--port` | `6010`（可用环境变量 `PORT` 覆盖） | 监听端口 |
| `--vad-threshold` | `0.38` | VAD 语音概率阈值 |
| `--vad-min-silence` | `0.22` | 判句末的最短静音（秒） |
| `--vad-min-speech` | `0.10` | 判起说的最短语音（秒） |
| `--vad-max-speech` | `12.0` | 单段最长语音（秒） |
| `--min-segment-seconds` | `0.03` | 过短分段跳过阈值（秒） |
| `--segment-context-seconds` | `0.30` | 分段尾部补上下文（秒） |
| `--pre-speech-context-seconds` | `0.34` | 句首 pre-roll（秒） |
| `--partial-interval` | `0.40` | partial 最小间隔（秒） |
| `--partial-max-seconds` | `5.0` | partial 只用最近 N 秒 |
| `--preroll-overlap-search-seconds` | `0.25` | pre-roll 重叠去重搜索（秒） |

查看全部参数：

```bash
python webui_vad_sensevoice.py --help
```

可选：`--no-ssl` 使用 HTTP（便于内网或其它服务用 `ws://` 连接）；`--no-partial` 关闭 partial，仅 VAD 定稿句输出。

---

## 3. 不要 UI：如何调用

服务仍用 Flask 提供 **HTTP + WebSocket**（无独立 HTML 也能被任意客户端调用）。首页 `/` 仅为浏览器演示页，集成时可忽略。

### 3.1 健康检查（HTTP）

```bash
curl -s http://127.0.0.1:6010/api/status
```

HTTPS 自签时加 `-k`：

```bash
curl -sk https://127.0.0.1:6010/api/status
```

### 3.2 实时识别（WebSocket）

- 路径：`/ws/vad-asr`
- 协议：文本帧，每条为 **JSON**
- 音频：**16 kHz 单声道 float32**，小端字节序经 **base64** 放在字段 `data` 中（与内置网页逻辑一致）

客户端消息类型：

| `type` | 说明 |
|--------|------|
| `start` | 新会话，清空 VAD 缓冲 |
| `audio` | `{"type":"audio","data":"<base64 float32le pcm>"}` |
| `stop` | 结束本轮，刷新尾段并返回带 `final: true` 的 result |
| `ping` | 服务端回复 `pong` |

服务端推送类型（常用）：

| `type` | 说明 |
|--------|------|
| `started` | 已 reset，可开始送 `audio` |
| `partial` | 预览：`text`、`committed_text`、`partial_text`、`segments` |
| `result` | 当前累积全文：`text`、`segments`；最后一包可含 `final: true` |
| `error` | `message` 为错误描述 |
| `pong` | 心跳应答 |

### 3.3 启动服务

```bash
./start_server.sh
# 或 HTTP 调试：
./start_server.sh -- --no-ssl
# 改端口：
PORT=6011 ./start_server.sh
```

### 3.4 在其它项目里复用方式

1. **整目录复制**：把 `vad_sensevoice_bundle/` 拷到目标仓库，按上文安装依赖并下载 `models/`。
2. **进程级集成**：用 systemd / Docker / 父进程 `subprocess` 启动 `start_server.sh` 或 `python webui_vad_sensevoice.py ...`，业务进程只连 WebSocket 与可选 `/api/status`。
3. **库级集成**：当前逻辑与 Flask 路由耦合在同文件；若要在同一进程内 `import`，需要自行拆分模块（本包未做），一般推荐 **独立进程 + WebSocket** 边界更清晰。

命令行示例（与仓库 `start_vad_sensevoice.sh` 一致的一组参数）：

```bash
cd /path/to/vad_sensevoice_bundle
pip install -r requirements.txt
./download_models.sh
SHERPA_ONNX_PROVIDER=cuda ./start_server.sh -- --no-ssl
python example_ws_client.py --url ws://127.0.0.1:6010/ws/vad-asr --wav ./test_16k_mono.wav
```

---

## 4. 证书与端口

- 默认 HTTPS：首次运行会在当前工作目录生成 `cert.pem`（需 `pyopenssl`）。
- 自签 `wss://` 客户端需关闭证书校验或自带信任（见 `example_ws_client.py` 中对 `wss` 的处理）。
- 若 `6010` 被占用（含 SSH 端口转发占用），请换 `PORT` 或见原仓库 `start_vad_sensevoice.sh` 注释。
