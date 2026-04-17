#!/usr/bin/env python3
"""
sherpa-onnx HTTPS WebSocket 实时流式语音识别服务
真正的实时识别，边说边显示结果
"""

import os
import sys
import json
import threading
import base64
import struct
from pathlib import Path
from datetime import datetime
from io import BytesIO

try:
    from flask import Flask, render_template, request
    from flask_sock import Sock
except ImportError:
    print("请安装: pip install flask flask-sock")
    sys.exit(1)

try:
    import sherpa_onnx
except ImportError:
    print("sherpa_onnx 未安装")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("请安装: pip install numpy")
    sys.exit(1)


# 配置
app = Flask(__name__)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}
sock = Sock(app)

# 全局识别器
recognizer = None
recognizer_lock = threading.Lock()


def init_recognizer():
    """初始化 ASR 识别器"""
    global recognizer

    model_dir = Path("/home/tsingwin/apps/sherpa-onnx/models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20")

    encoder_file = model_dir / "encoder-epoch-99-avg-1.int8.onnx"
    decoder_file = model_dir / "decoder-epoch-99-avg-1.int8.onnx"
    joiner_file = model_dir / "joiner-epoch-99-avg-1.int8.onnx"
    tokens_file = model_dir / "tokens.txt"

    if not all(f.exists() for f in [encoder_file, decoder_file, joiner_file, tokens_file]):
        raise ValueError("模型文件不存在")

    print("正在加载 ASR 模型 (GPU 加速)...")
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(tokens_file),
        encoder=str(encoder_file),
        decoder=str(decoder_file),
        joiner=str(joiner_file),
        num_threads=4,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cuda",
    )
    print("ASR 模型加载完成！")


def decode_pcm_data(data):
    """解码 Web Audio PCM 数据 (Float32LE)"""
    try:
        # 数据是 base64 编码的 Float32LE
        pcm_data = base64.b64decode(data)
        # 转换为 numpy 数组
        samples = np.frombuffer(pcm_data, dtype=np.float32)
        return samples
    except Exception as e:
        print(f"解码 PCM 数据失败: {e}")
        return None


@app.route('/')
def index():
    """主页"""
    return render_template('streaming.html')


@sock.route('/ws/stream')
def stream_asr(ws):
    """WebSocket 流式识别端点"""
    if recognizer is None:
        ws.send(json.dumps({"type": "error", "message": "识别器未初始化"}))
        return

    stream = None
    last_result = ""
    sample_rate = 16000

    try:
        while True:
            message = ws.receive()
            if message is None:
                break

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue

            msg_type = data.get('type')

            if msg_type == 'start':
                # 开始新的识别会话
                with recognizer_lock:
                    stream = recognizer.create_stream()
                last_result = ""
                ws.send(json.dumps({"type": "started"}))

            elif msg_type == 'audio':
                if stream is None:
                    continue

                # 解码音频数据
                pcm_data = data.get('data')
                samples = decode_pcm_data(pcm_data)

                if samples is not None:
                    # 送入识别器
                    stream.accept_waveform(sample_rate, samples)

                    # 识别
                    with recognizer_lock:
                        while recognizer.is_ready(stream):
                            recognizer.decode_stream(stream)

                    # 获取结果
                    result = recognizer.get_result(stream)

                    # 只有结果变化时才发送
                    if result != last_result:
                        last_result = result
                        ws.send(json.dumps({
                            "type": "result",
                            "text": result,
                            "partial": True
                        }))

            elif msg_type == 'stop':
                if stream is not None:
                    # 强制解码剩余数据
                    stream.input_finished()
                    with recognizer_lock:
                        while recognizer.is_ready(stream):
                            recognizer.decode_stream(stream)

                    result = recognizer.get_result(stream)
                    ws.send(json.dumps({
                        "type": "result",
                        "text": result,
                        "partial": False,
                        "final": True
                    }))
                    stream = None
                    last_result = ""

            elif msg_type == 'ping':
                ws.send(json.dumps({"type": "pong"}))

    except Exception as e:
        print(f"WebSocket 错误: {e}")
        try:
            ws.send(json.dumps({"type": "error", "message": str(e)}))
        except:
            pass


@app.route('/api/status', methods=['GET'])
def status():
    """服务状态"""
    return json.dumps({
        "status": "ok",
        "model_loaded": recognizer is not None,
        "provider": "cuda" if recognizer else "none"
    })


def get_server_address():
    """获取服务器地址"""
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"
    return local_ip


def main():
    import argparse

    parser = argparse.ArgumentParser(description="sherpa-onnx HTTPS WebSocket 实时流式 ASR 服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=6007, help="监听端口")
    parser.add_argument("--cert", type=str, default="cert-streaming.pem", help="证书文件")
    parser.add_argument("--no-ssl", action="store_true", help="不使用 HTTPS")

    args = parser.parse_args()

    # 初始化识别器
    try:
        init_recognizer()
    except Exception as e:
        print(f"初始化识别器失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 创建 templates 目录
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

    # 生成 HTML
    index_html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>sherpa-onnx 实时流式语音识别</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 24px;
            box-shadow: 0 25px 80px rgba(0,0,0,0.4);
            max-width: 800px;
            width: 100%;
            padding: 45px;
        }
        h1 {
            color: #1a1a2e;
            text-align: center;
            margin-bottom: 10px;
            font-size: 32px;
            font-weight: 700;
        }
        .subtitle {
            text-align: center;
            color: #6b7280;
            margin-bottom: 35px;
            font-size: 16px;
        }
        .status-bar {
            display: flex;
            align-items: center;
            gap: 12px;
            background: #f8fafc;
            padding: 16px 20px;
            border-radius: 14px;
            margin-bottom: 25px;
        }
        .status-dot {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #9ca3af;
            transition: all 0.3s;
        }
        .status-dot.connected {
            background: #10b981;
            animation: pulse 2s infinite;
        }
        .status-dot.recording {
            background: #ef4444;
            animation: pulse-red 0.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(1.1); }
        }
        @keyframes pulse-red {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .status-text {
            color: #4b5563;
            font-weight: 500;
            flex: 1;
        }
        .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 30px;
        }
        .btn {
            flex: 1;
            padding: 18px 24px;
            font-size: 18px;
            font-weight: 600;
            border: none;
            border-radius: 14px;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
        }
        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(16, 185, 129, 0.35);
        }
        .btn-danger {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
        }
        .btn-danger:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(239, 68, 68, 0.35);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .result-container {
            background: #0f172a;
            border-radius: 18px;
            padding: 28px;
            min-height: 180px;
            position: relative;
            overflow: hidden;
        }
        .result-label {
            color: #94a3b8;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }
        .result-text {
            color: #f1f5f9;
            font-size: 22px;
            line-height: 1.7;
            word-wrap: break-word;
            font-family: 'SF Mono', 'Fira Code', monospace;
        }
        .result-text.empty {
            color: #475569;
            font-style: italic;
        }
        .result-text .partial {
            color: #94a3b8;
        }
        .waveform {
            margin-top: 25px;
            height: 80px;
            background: #f1f5f9;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 5px;
            padding: 15px;
        }
        .wave-bar {
            width: 5px;
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            border-radius: 3px;
            transition: height 0.08s;
        }
        .info-box {
            margin-top: 25px;
            padding: 18px 22px;
            background: #fef3c7;
            border-radius: 12px;
            border-left: 4px solid #f59e0b;
        }
        .info-box p {
            color: #92400e;
            font-size: 14px;
            line-height: 1.6;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 25px;
        }
        .stat-card {
            background: #f8fafc;
            padding: 18px;
            border-radius: 12px;
            text-align: center;
        }
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            color: #1e293b;
        }
        .stat-label {
            font-size: 13px;
            color: #64748b;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 实时流式语音识别</h1>
        <p class="subtitle">基于 sherpa-onnx · GPU 加速 · WebSocket 传输</p>

        <div class="status-bar">
            <div class="status-dot" id="statusDot"></div>
            <span class="status-text" id="statusText">正在连接...</span>
        </div>

        <div class="controls">
            <button class="btn btn-primary" id="startBtn" disabled>
                🎙️ 开始识别
            </button>
            <button class="btn btn-danger" id="stopBtn" disabled>
                ⏹️ 停止识别
            </button>
        </div>

        <div class="waveform" id="waveform">
        </div>

        <div class="stats" id="stats" style="display: none;">
            <div class="stat-card">
                <div class="stat-value" id="audioChunks">0</div>
                <div class="stat-label">音频块</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="processingTime">0ms</div>
                <div class="stat-label">处理延迟</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="isRecording">否</div>
                <div class="stat-label">状态</div>
            </div>
        </div>

        <div class="result-container">
            <div class="result-label">识别结果（实时更新）</div>
            <div class="result-text empty" id="resultText">等待开始识别...</div>
        </div>

        <div class="info-box">
            <p><strong>提示：</strong>点击"开始识别"后立即开始说话，识别结果会实时显示在上方。这是真正的流式识别，边说边出结果！</p>
        </div>
    </div>

    <script>
        let ws = null;
        let mediaStream = null;
        let audioContext = null;
        let processor = null;
        let sourceNode = null;
        let isRecording = false;
        let chunkCount = 0;
        let startTime = 0;

        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const resultText = document.getElementById('resultText');
        const waveform = document.getElementById('waveform');
        const stats = document.getElementById('stats');
        const audioChunksEl = document.getElementById('audioChunks');
        const processingTimeEl = document.getElementById('processingTime');
        const isRecordingEl = document.getElementById('isRecording');

        // 初始化波形条
        function initWaveform() {
            waveform.innerHTML = '';
            for (let i = 0; i < 60; i++) {
                const bar = document.createElement('div');
                bar.className = 'wave-bar';
                bar.style.height = '8px';
                waveform.appendChild(bar);
            }
        }
        initWaveform();

        // 连接 WebSocket
        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${location.host}/ws/stream`;

            statusText.textContent = '正在连接...';
            statusDot.className = 'status-dot';

            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                statusText.textContent = '已连接 - 准备就绪';
                statusDot.className = 'status-dot connected';
                startBtn.disabled = false;
            };

            ws.onclose = () => {
                statusText.textContent = '连接断开';
                statusDot.className = 'status-dot';
                startBtn.disabled = true;
                stopBtn.disabled = true;
                if (isRecording) {
                    stopRecording();
                }
                setTimeout(connect, 3000);
            };

            ws.onerror = (err) => {
                console.error('WebSocket 错误:', err);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
        }

        function handleMessage(data) {
            switch (data.type) {
                case 'started':
                    resultText.textContent = '';
                    resultText.classList.remove('empty');
                    stats.style.display = 'grid';
                    break;

                case 'result':
                    if (data.text) {
                        resultText.textContent = data.text;
                    }
                    if (data.final) {
                        const elapsed = Date.now() - startTime;
                        processingTimeEl.textContent = elapsed + 'ms';
                    }
                    break;

                case 'error':
                    resultText.textContent = '错误: ' + data.message;
                    resultText.classList.add('empty');
                    break;

                case 'pong':
                    break;
            }
        }

        async function startRecording() {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }
                });

                audioContext = new AudioContext({ sampleRate: 16000 });
                sourceNode = audioContext.createMediaStreamSource(mediaStream);

                // 创建 ScriptProcessorNode 处理音频
                const bufferSize = 4096;
                processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

                processor.onaudioprocess = (e) => {
                    if (!isRecording) return;

                    const inputData = e.inputBuffer.getChannelData(0);

                    // 发送音频数据
                    sendAudioData(inputData);

                    // 更新波形
                    updateWaveform(inputData);

                    chunkCount++;
                    audioChunksEl.textContent = chunkCount;
                };

                sourceNode.connect(processor);
                processor.connect(audioContext.destination);

                // 开始识别
                ws.send(JSON.stringify({ type: 'start' }));
                isRecording = true;
                chunkCount = 0;
                startTime = Date.now();

                startBtn.disabled = true;
                stopBtn.disabled = false;
                statusDot.className = 'status-dot recording';
                statusText.textContent = '正在识别中...';
                isRecordingEl.textContent = '是';
                audioChunksEl.textContent = '0';
                processingTimeEl.textContent = '0ms';

            } catch (err) {
                alert('无法访问麦克风: ' + err.message);
                console.error(err);
            }
        }

        function sendAudioData(audioData) {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;

            // 转换为 base64
            const float32Array = new Float32Array(audioData);
            const bytes = new Uint8Array(float32Array.buffer);
            let binary = '';
            for (let i = 0; i < bytes.byteLength; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            const base64 = btoa(binary);

            ws.send(JSON.stringify({
                type: 'audio',
                data: base64
            }));
        }

        function updateWaveform(audioData) {
            const bars = waveform.querySelectorAll('.wave-bar');
            const step = Math.floor(audioData.length / bars.length);

            bars.forEach((bar, i) => {
                let sum = 0;
                for (let j = 0; j < step; j++) {
                    const idx = i * step + j;
                    if (idx < audioData.length) {
                        sum += Math.abs(audioData[idx]);
                    }
                }
                const avg = sum / step;
                const height = Math.max(8, avg * 200);
                bar.style.height = height + 'px';
            });
        }

        function stopRecording() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'stop' }));
            }

            if (processor) {
                processor.disconnect();
                processor = null;
            }
            if (sourceNode) {
                sourceNode.disconnect();
                sourceNode = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }

            isRecording = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            statusDot.className = 'status-dot connected';
            statusText.textContent = '已连接 - 准备就绪';
            isRecordingEl.textContent = '否';

            // 重置波形
            const bars = waveform.querySelectorAll('.wave-bar');
            bars.forEach(bar => bar.style.height = '8px');
        }

        // 事件监听
        startBtn.addEventListener('click', startRecording);
        stopBtn.addEventListener('click', stopRecording);

        // 初始化连接
        connect();

        // 保活
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 20000);
    </script>
</body>
</html>"""

    with open(templates_dir / "streaming.html", "w") as f:
        f.write(index_html)

    # 生成证书（如果需要）
    if not args.no_ssl and not Path(args.cert).exists():
        print(f"证书文件 {args.cert} 不存在，正在生成...")
        try:
            from OpenSSL import crypto

            k = crypto.PKey()
            k.generate_key(crypto.TYPE_RSA, 4096)

            cert = crypto.X509()
            cert.get_subject().C = "CN"
            cert.get_subject().ST = "sherpa"
            cert.get_subject().L = "sherpa"
            cert.get_subject().O = "sherpa"
            cert.get_subject().OU = "sherpa"
            cert.get_subject().CN = "sherpa-onnx-streaming"
            cert.set_serial_number(2000)
            cert.gmtime_adj_notBefore(0)
            cert.gmtime_adj_notAfter(10 * 365 * 24 * 60 * 60)
            cert.set_issuer(cert.get_subject())
            cert.set_pubkey(k)
            cert.sign(k, "sha512")

            with open(args.cert, "wt") as f:
                f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k).decode("utf-8"))
                f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert).decode("utf-8"))
            print(f"证书已生成: {args.cert}")
        except ImportError:
            print("请安装 pyopenssl: pip install pyopenssl")
            sys.exit(1)

    # 显示访问地址
    server_ip = get_server_address()
    print("\n" + "=" * 70)
    print("  sherpa-onnx HTTPS WebSocket 实时流式 ASR 服务已启动")
    print("=" * 70)
    print(f"\n请在浏览器中访问以下地址之一:")
    protocol = "http" if args.no_ssl else "https"
    print(f"  - 本机: {protocol}://localhost:{args.port}")
    print(f"  - 局域网: {protocol}://{server_ip}:{args.port}")
    print(f"\n功能特点:")
    print(f"  ✅ 真正的实时流式识别 - 边说边显示结果")
    print(f"  ✅ WebSocket 低延迟传输")
    print(f"  ✅ GPU 加速 (CUDA)")
    print(f"  ✅ 实时波形显示")
    print(f"\n注意: 这是自签名证书，浏览器会提示安全警告，请")
    print(f"      点击'高级' → '继续访问'来继续使用。")
    print("\n按 Ctrl+C 停止服务\n")

    # 启动服务
    ssl_context = None
    if not args.no_ssl:
        ssl_context = (args.cert, args.cert)

    app.run(host=args.host, port=args.port, ssl_context=ssl_context, debug=False, threaded=True)


if __name__ == "__main__":
    main()
