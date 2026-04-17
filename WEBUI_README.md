# sherpa-onnx HTTPS WebUI 语音识别服务

通过浏览器访问的实时语音识别服务，支持局域网内其他设备访问。

## 快速开始

### 1. 启动服务

在 Jetson 设备上运行：

```bash
cd /home/tsingwin/apps/sherpa-onnx
./start_webui.sh
```

或者直接运行：

```bash
cd /home/tsingwin/apps/sherpa-onnx
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sherpa
python webui_asr.py --port 6006
```

### 2. 访问 WebUI

服务启动后，会显示访问地址，例如：

```
请在浏览器中访问以下地址之一:
  - 本机: https://localhost:6006
  - 局域网: https://192.168.1.100:6006
```

在您的笔记本或其他设备的浏览器中访问局域网地址。

### 3. 关于安全警告

由于使用自签名证书，浏览器会显示安全警告：

**Chrome/Edge:**
1. 点击"高级"
2. 点击"继续访问 192.168.1.100（不安全）"

**Firefox:**
1. 点击"高级"
2. 点击"接受风险并继续"

### 4. 使用语音识别

1. 允许浏览器访问麦克风
2. 点击"开始录音"按钮
3. 对着麦克风说话
4. 点击"停止录音"
5. 等待识别结果显示

## 配置选项

### 修改端口

默认端口是 6006，可以通过参数修改：

```bash
python webui_asr.py --port 8080
```

### 不使用 HTTPS（仅用于本地测试）

```bash
python webui_asr.py --no-ssl
```

## 功能特性

- ✅ HTTPS 加密传输
- ✅ GPU 加速 (CUDA)
- ✅ 中英文双语识别
- ✅ 实时波形显示
- ✅ 响应式设计，支持手机访问
- ✅ 自动降噪和回声消除

## 技术说明

- **后端框架**: Flask
- **ASR 引擎**: sherpa-onnx
- **模型**: Zipformer 流式中英文双语模型 (INT8)
- **音频格式**: WebM → 16kHz PCM
- **最大文件大小**: 50MB

## 故障排除

### 无法访问服务

1. 检查防火墙设置
2. 确认两台设备在同一局域网
3. 检查 Jetson 的 IP 地址是否正确

### 麦克风无法使用

1. 确保使用 HTTPS 协议（localhost 除外）
2. 检查浏览器是否有权限访问麦克风
3. 尝试使用 Chrome/Edge 浏览器

### 识别速度慢

1. 确认使用的是 GPU 版本（检查服务启动日志）
2. 查看是否有其他进程占用 GPU
3. 使用 `tegrastats` 检查 GPU 使用率

## 文件位置

- 主程序: `/home/tsingwin/apps/sherpa-onnx/webui_asr.py`
- 启动脚本: `/home/tsingwin/apps/sherpa-onnx/start_webui.sh`
- 模型目录: `/home/tsingwin/apps/sherpa-onnx/models/`
- 证书文件: `/home/tsingwin/apps/sherpa-onnx/cert.pem` (自动生成)

## 停止服务

在终端按 `Ctrl+C` 即可停止服务。
