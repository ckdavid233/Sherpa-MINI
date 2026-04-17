#!/usr/bin/env python3
"""
sherpa-onnx HTTPS WebUI 语音识别服务
支持从其他设备通过浏览器访问
"""

import os
import sys
import json
import wave
import tempfile
import threading
from pathlib import Path
from datetime import datetime

try:
    from flask import Flask, render_template, request, jsonify, send_from_directory
except ImportError:
    print("请安装 flask: pip install flask")
    sys.exit(1)

try:
    import sherpa_onnx
except ImportError:
    print("sherpa_onnx 未安装")
    sys.exit(1)

import numpy as np
import soundfile as sf


# 配置
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 限制

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


def process_audio(audio_data, sample_rate):
    """处理音频数据并返回识别结果"""
    if recognizer is None:
        return {"error": "识别器未初始化"}

    with recognizer_lock:
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio_data)

        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        result = recognizer.get_result(stream)
        return {"text": result}


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """语音识别 API"""
    if 'audio' not in request.files:
        return jsonify({"error": "没有音频文件"}), 400

    audio_file = request.files['audio']

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)

    try:
        # 读取音频
        audio, sr = sf.read(tmp_path, dtype='float32')
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # 只取第一个声道

        # 重采样到 16kHz 如果需要
        if sr != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            except ImportError:
                return jsonify({"error": f"采样率 {sr} 不支持，需要 16000"}), 400

        result = process_audio(audio, sr)
        result["timestamp"] = datetime.now().isoformat()
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route('/api/status', methods=['GET'])
def status():
    """服务状态"""
    return jsonify({
        "status": "ok",
        "model_loaded": recognizer is not None,
        "provider": "cuda" if recognizer else "none"
    })


def get_server_address():
    """获取服务器地址"""
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip


def main():
    import argparse

    parser = argparse.ArgumentParser(description="sherpa-onnx HTTPS WebUI ASR 服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=6006, help="监听端口")
    parser.add_argument("--cert", type=str, default="cert.pem", help="证书文件")
    parser.add_argument("--no-ssl", action="store_true", help="不使用 HTTPS")

    args = parser.parse_args()

    # 初始化识别器
    try:
        init_recognizer()
    except Exception as e:
        print(f"初始化识别器失败: {e}")
        sys.exit(1)

    # 创建 templates 目录
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

    # 生成 index.html
    index_html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>sherpa-onnx 实时语音识别</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 700px;
            width: 100%;
            padding: 40px;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 28px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .status {
            background: #f0f4ff;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #10b981;
            animation: pulse 2s infinite;
        }
        .status-dot.offline {
            background: #ef4444;
            animation: none;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .record-btn {
            width: 100%;
            padding: 20px;
            font-size: 20px;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s;
            background: #10b981;
            color: white;
            font-weight: 600;
        }
        .record-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(16, 185, 129, 0.3);
        }
        .record-btn.recording {
            background: #ef4444;
        }
        .record-btn.recording:hover {
            box-shadow: 0 10px 20px rgba(239, 68, 68, 0.3);
        }
        .result-box {
            margin-top: 25px;
            background: #f9fafb;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 20px;
            min-height: 120px;
        }
        .result-title {
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
            font-weight: 600;
        }
        .result-text {
            color: #1f2937;
            font-size: 18px;
            line-height: 1.6;
            word-wrap: break-word;
        }
        .result-text.empty {
            color: #9ca3af;
            font-style: italic;
        }
        .info {
            margin-top: 20px;
            padding: 15px;
            background: #fffbeb;
            border-radius: 10px;
            border-left: 4px solid #f59e0b;
        }
        .info p {
            color: #92400e;
            font-size: 14px;
            line-height: 1.6;
        }
        .waveform {
            margin-top: 20px;
            height: 60px;
            background: #f3f4f6;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            overflow: hidden;
        }
        .wave-bar {
            width: 4px;
            background: #667eea;
            border-radius: 2px;
            transition: height 0.1s;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 实时语音识别</h1>
        <p class="subtitle">基于 sherpa-onnx · GPU 加速</p>

        <div class="status">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">正在连接...</span>
        </div>

        <button class="record-btn" id="recordBtn">
            🎙️ 开始录音
        </button>

        <div class="waveform" id="waveform" style="display: none;">
        </div>

        <div class="result-box">
            <div class="result-title">识别结果</div>
            <div class="result-text empty" id="resultText">等待输入...</div>
        </div>

        <div class="info">
            <p><strong>提示：</strong>首次访问时浏览器会请求麦克风权限，请点击"允许"。这是自签名证书，浏览器可能会提示安全警告，点击"高级" → "继续访问"即可。</p>
        </div>
    </div>

    <script>
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let audioContext = null;
        let analyser = null;
        let animationId = null;

        const recordBtn = document.getElementById('recordBtn');
        const resultText = document.getElementById('resultText');
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const waveform = document.getElementById('waveform');

        // 检查服务状态
        async function checkStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                if (data.status === 'ok' && data.model_loaded) {
                    statusDot.classList.remove('offline');
                    statusText.textContent = '服务就绪 · ' + (data.provider === 'cuda' ? 'GPU 加速' : 'CPU');
                } else {
                    statusDot.classList.add('offline');
                    statusText.textContent = '模型未加载';
                }
            } catch (e) {
                statusDot.classList.add('offline');
                statusText.textContent = '连接失败';
            }
        }

        // 初始化波形显示
        function initWaveform() {
            waveform.innerHTML = '';
            for (let i = 0; i < 50; i++) {
                const bar = document.createElement('div');
                bar.className = 'wave-bar';
                bar.style.height = '4px';
                waveform.appendChild(bar);
            }
        }

        // 动画波形
        function animateWaveform() {
            const bars = waveform.querySelectorAll('.wave-bar');
            bars.forEach(bar => {
                const height = Math.random() * 50 + 10;
                bar.style.height = height + 'px';
            });
            animationId = requestAnimationFrame(animateWaveform);
        }

        // 开始录音
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });

                audioChunks = [];
                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm'
                });

                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    if (animationId) {
                        cancelAnimationFrame(animationId);
                    }
                    waveform.style.display = 'none';

                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    await sendAudio(audioBlob);
                };

                mediaRecorder.start(100);
                isRecording = true;
                recordBtn.textContent = '⏹️ 停止录音';
                recordBtn.classList.add('recording');

                // 显示波形
                initWaveform();
                waveform.style.display = 'flex';
                animateWaveform();

            } catch (err) {
                alert('无法访问麦克风: ' + err.message);
                console.error(err);
            }
        }

        // 停止录音
        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                recordBtn.textContent = '🎙️ 开始录音';
                recordBtn.classList.remove('recording');
            }
        }

        // 发送音频
        async function sendAudio(blob) {
            resultText.textContent = '正在识别...';
            resultText.classList.remove('empty');

            const formData = new FormData();
            formData.append('audio', blob, 'recording.webm');

            try {
                const response = await fetch('/api/recognize', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.error) {
                    resultText.textContent = '错误: ' + data.error;
                } else if (data.text) {
                    resultText.textContent = data.text;
                    resultText.classList.remove('empty');
                } else {
                    resultText.textContent = '没有识别到语音';
                    resultText.classList.add('empty');
                }
            } catch (err) {
                resultText.textContent = '识别失败: ' + err.message;
            }
        }

        // 事件监听
        recordBtn.addEventListener('click', () => {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        });

        // 初始化
        checkStatus();
        setInterval(checkStatus, 5000);
    </script>
</body>
</html>"""

    with open(templates_dir / "index.html", "w") as f:
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
            cert.get_subject().CN = "sherpa-onnx"
            cert.set_serial_number(1000)
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
    print("\n" + "=" * 60)
    print("  sherpa-onnx HTTPS WebUI ASR 服务已启动")
    print("=" * 60)
    print(f"\n请在浏览器中访问以下地址之一:")
    protocol = "http" if args.no_ssl else "https"
    print(f"  - 本机: {protocol}://localhost:{args.port}")
    print(f"  - 局域网: {protocol}://{server_ip}:{args.port}")
    print(f"\n注意: 这是自签名证书，浏览器会提示安全警告，请")
    print(f"      点击'高级' → '继续访问'来继续使用。")
    print("\n按 Ctrl+C 停止服务\n")

    # 启动服务
    ssl_context = None
    if not args.no_ssl:
        ssl_context = (args.cert, args.cert)

    app.run(host=args.host, port=args.port, ssl_context=ssl_context, debug=False)


if __name__ == "__main__":
    main()
