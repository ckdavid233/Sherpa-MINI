#!/usr/bin/env python3
"""
无浏览器 UI 的调用示例：连接本服务的 WebSocket，发送 16 kHz 单声道 float32 PCM
（协议与浏览器端一致：JSON + base64）。
依赖: pip install websocket-client numpy
"""
from __future__ import annotations

import argparse
import base64
import json
import ssl
import sys
import wave

import numpy as np

try:
    import websocket
except ImportError:
    print("请安装: pip install websocket-client", file=sys.stderr)
    sys.exit(1)


def _pcm_float32_b64(chunk: np.ndarray) -> str:
    return base64.b64encode(chunk.astype("<f4", copy=False).tobytes()).decode("ascii")


def _load_wav_mono_16k(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        ch = wf.getnchannels()
        rate = wf.getframerate()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
    if ch != 1:
        raise SystemExit("需要单声道 WAV")
    if rate != 16000:
        raise SystemExit("需要采样率 16000 Hz（与 VAD / SenseVoice 约定一致）")
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.float32)
    else:
        raise SystemExit(f"不支持的采样宽度: {sw} bytes")
    return x


def main() -> None:
    p = argparse.ArgumentParser(description="VAD+SenseVoice WebSocket 文本客户端示例")
    p.add_argument("--url", default="ws://127.0.0.1:6010/ws/vad-asr", help="WebSocket 地址")
    p.add_argument("--wav", required=True, help="16 kHz 单声道 WAV")
    p.add_argument("--chunk-ms", type=float, default=100.0, help="每包时长（毫秒）")
    args = p.parse_args()

    samples = _load_wav_mono_16k(args.wav)
    chunk = max(1, int(16000 * (args.chunk_ms / 1000.0)))

    sslopt = None
    if args.url.startswith("wss://"):
        sslopt = {"cert_reqs": ssl.CERT_NONE}

    ws = websocket.WebSocket(sslopt=sslopt)
    ws.connect(args.url)
    ws.send(json.dumps({"type": "start"}))

    for i in range(0, len(samples), chunk):
        part = samples[i : i + chunk]
        ws.send(json.dumps({"type": "audio", "data": _pcm_float32_b64(part)}))

    ws.send(json.dumps({"type": "stop"}))

    while True:
        raw = ws.recv()
        if raw is None:
            break
        msg = json.loads(raw)
        typ = msg.get("type")
        if typ == "partial":
            print("[partial]", msg.get("text", ""))
        elif typ == "result":
            print("[result]", msg.get("text", ""), "final=" + str(msg.get("final", False)))
            if msg.get("final"):
                break
        elif typ == "error":
            print("[error]", msg.get("message"), file=sys.stderr)
            break
        elif typ == "pong":
            pass
        else:
            print(msg)

    ws.close()


if __name__ == "__main__":
    main()
