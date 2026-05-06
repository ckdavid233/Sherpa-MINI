#!/usr/bin/env python3
"""
sherpa-onnx HTTPS VAD + SenseVoice 分段识别
VAD 检测停顿切分句子，每句送 SenseVoice 识别，准确率高又不卡
"""

from __future__ import annotations

import errno
import os
import sys
import json
import threading
import base64
import time
import numpy as np
from pathlib import Path

try:
    from flask import Flask, render_template
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

# 语言模式预设：中文用较短静音断句，英文用较长静音容忍句内停顿
LANGUAGE_PRESETS = {
    "zh": {
        "language": "zh",
        "vad_threshold": 0.38,
        "vad_min_silence": 0.22,
        "vad_min_speech": 0.10,
        "vad_max_speech": 12.0,
        "overlap_threshold": 0.76,   # 偏高：中文音节短，避免误删句首
    },
    "en": {
        "language": "en",
        "vad_threshold": 0.35,
        "vad_min_silence": 0.50,
        "vad_min_speech": 0.12,
        "vad_max_speech": 18.0,
        "overlap_threshold": 0.70,   # 偏低：英文重叠长，去重更积极
    },
}

# 全局模型（当前活跃）
vad = None
recognizer = None
vad_window_size = 0
model_lock = threading.Lock()
current_mode = "zh"

# 按语言模式缓存 VAD / Recognizer，切换语言时无需重新加载模型
_vad_cache: dict = {}
_recognizer_cache: dict = {}

# 模型路径相对本脚本所在目录，避免机器/目录结构不同导致硬编码失效
_REPO_ROOT = Path(__file__).resolve().parent
_MODELS_DIR = _REPO_ROOT / "models"

TERMINAL_PUNCTUATION = "。！？!?."
PAUSE_PUNCTUATION = "，、,；;：:"
# 运行时可由 main() 根据命令行参数覆盖
MIN_SEGMENT_SECONDS = 0.03
SEGMENT_CONTEXT_SECONDS = 0.25
PRE_SPEECH_CONTEXT_SECONDS = 0.35
PARTIAL_DECODE_INTERVAL_SECONDS = 0.40
PARTIAL_MAX_SECONDS = 5.0
PREROLL_OVERLAP_SEARCH_SECONDS = 0.50
PARTIAL_ENABLED = True


def _best_suffix_prefix_overlap(preroll: np.ndarray, segment: np.ndarray, max_overlap_samples: int, threshold: float = 0.75) -> int:
    """在 preroll 尾部与 segment 头部之间寻找最佳重叠长度，用于去重拼接。"""
    if preroll.size == 0 or segment.size == 0 or max_overlap_samples <= 0:
        return 0

    pr = preroll.astype(np.float32, copy=False)
    seg = segment.astype(np.float32, copy=False)

    max_k = int(min(pr.size, seg.size, max_overlap_samples))
    best_k = 0
    best_score = -1.0

    # 从大往小找：优先更长的重叠（更像真正的重复边界）
    for k in range(max_k, 0, -1):
        a = pr[-k:]
        b = seg[:k]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            continue
        score = float(np.dot(a, b) / (na * nb))
        if score > best_score:
            best_score = score
            best_k = k
        # 足够好就提前结束，避免每次都扫满
        if best_score >= 0.92 and k <= max_k // 2:
            break

    if best_score < threshold:
        return 0
    return best_k


def attach_preroll_and_pad(segment_audio: np.ndarray, pending_preroll: np.ndarray, sample_rate: int, overlap_threshold: float = 0.75) -> np.ndarray:
    """拼接句首上下文并做轻量 padding，尽量避免重复喂同一段音频。"""
    audio = segment_audio
    if pending_preroll.size > 0:
        max_overlap = int(PREROLL_OVERLAP_SEARCH_SECONDS * sample_rate)
        overlap = _best_suffix_prefix_overlap(pending_preroll, audio, max_overlap_samples=max_overlap, threshold=overlap_threshold)
        if overlap > 0:
            audio = np.concatenate([pending_preroll[:-overlap], audio])
        else:
            audio = np.concatenate([pending_preroll, audio])

    context = int(SEGMENT_CONTEXT_SECONDS * sample_rate)
    if context > 0:
        # 左侧用极小幅值填充，避免全零导致模型把句首能量判成噪声
        pad_left = float(np.mean(np.abs(audio)) * 1e-4) if audio.size else 0.0
        audio = np.pad(audio, (context, 0), mode="constant", constant_values=pad_left)
        audio = np.pad(audio, (0, context), mode="edge")
    return audio


def init_vad(
    *,
    vad_threshold: float,
    min_silence_duration: float,
    min_speech_duration: float,
    max_speech_duration: float,
    vad_neg_threshold: float | None = None,
    provider: str = "cuda",
):
    """初始化 VAD 语音活动检测"""
    global vad, vad_window_size

    model_path = _MODELS_DIR / "silero_vad.onnx"
    if not model_path.exists():
        raise ValueError(f"VAD 模型不存在: {model_path}\n请先下载 silero_vad.onnx 放到 models 目录")

    print(f"加载 Silero VAD 模型: {model_path}")
    print(
        "VAD 参数: "
        f"threshold={vad_threshold}, "
        f"min_silence_duration={min_silence_duration}, "
        f"min_speech_duration={min_speech_duration}, "
        f"max_speech_duration={max_speech_duration}"
    )

    silero_config = sherpa_onnx.SileroVadModelConfig(
        model=str(model_path),
        threshold=vad_threshold,
        min_silence_duration=min_silence_duration,
        min_speech_duration=min_speech_duration,
        max_speech_duration=max_speech_duration,
    )

    vad_config = sherpa_onnx.VadModelConfig(
        silero_vad=silero_config,
        sample_rate=16000,
        num_threads=6,
        provider=provider,
        debug=False,
    )

    vad = sherpa_onnx.VoiceActivityDetector(
        vad_config,
        buffer_size_in_seconds=60,
    )
    vad_window_size = vad_config.silero_vad.window_size
    print("VAD 模型加载完成！")
    return vad


def init_recognizer(language: str = "auto", provider: str = "cuda"):
    """初始化 SenseVoice 识别器"""
    global recognizer

    model_dir = _MODELS_DIR / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    tokens_file = model_dir / "tokens.txt"
    fp32 = model_dir / "model.onnx"
    int8 = model_dir / "model.int8.onnx"
    # 全精度通常更稳；仅有 int8 时自动回退
    if fp32.is_file():
        model_file = fp32
        print("SenseVoice: 使用 model.onnx（全精度）")
    elif int8.is_file():
        model_file = int8
        print("SenseVoice: 使用 model.int8.onnx（量化）")
    else:
        model_file = int8

    if not model_file.exists() or not tokens_file.exists():
        raise ValueError(f"模型文件不存在，请检查: {model_dir}")

    print(f"正在加载 SenseVoice 模型 (GPU 加速, language={language})...")

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model_file),
        tokens=str(tokens_file),
        use_itn=True,
        debug=False,
        provider=provider,
        num_threads=6,
        language=language,
    )
    print("SenseVoice 模型加载完成！支持：中文/英文/日语/韩语/粤语")
    return recognizer


def switch_language_mode(mode: str):
    """切换语言模式：从缓存取或新建 VAD / Recognizer。调用方须持有 model_lock。"""
    global vad, recognizer, vad_window_size, current_mode

    if mode not in LANGUAGE_PRESETS:
        print(f"未知语言模式 {mode}，回退到 zh")
        mode = "zh"

    if mode == current_mode and vad is not None and recognizer is not None:
        return

    preset = LANGUAGE_PRESETS[mode]
    print(f"切换语言模式: {current_mode} -> {mode}")

    # VAD：查缓存，未命中则新建
    if mode in _vad_cache:
        vad = _vad_cache[mode]
        vad_window_size = vad.config.silero_vad.window_size if hasattr(vad, 'config') else vad_window_size
        print(f"[缓存] 使用已缓存的 {mode} VAD")
    else:
        init_vad(
            vad_threshold=preset["vad_threshold"],
            min_silence_duration=preset["vad_min_silence"],
            min_speech_duration=preset["vad_min_speech"],
            max_speech_duration=preset["vad_max_speech"],
        )
        _vad_cache[mode] = vad
        print(f"[新建] {mode} VAD 已缓存")

    # Recognizer：查缓存，未命中则新建
    if mode in _recognizer_cache:
        recognizer = _recognizer_cache[mode]
        print(f"[缓存] 使用已缓存的 {mode} Recognizer")
    else:
        recognizer = None
        init_recognizer(language=preset["language"])
        _recognizer_cache[mode] = recognizer
        print(f"[新建] {mode} Recognizer 已缓存")

    current_mode = mode


def _decode_sense_voice_once(sample_rate: int, waveform: np.ndarray) -> str:
    """对单段波形做一次 SenseVoice 解码（调用方已持 model_lock）。"""
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, waveform.astype(np.float32, copy=False))
    recognizer.decode_stream(stream)
    return normalize_recognition_text(stream.result.text)


def decode_sense_voice_segment(sample_rate: int, enhanced: np.ndarray, raw: np.ndarray) -> str:
    """
    解码 VAD 分段；多路重试减轻「空识别→整段被跳过」的吞句现象。
    调用方须已持有 model_lock。
    """
    raw_f = raw.astype(np.float32, copy=False)

    for w in (enhanced, raw_f):
        text = _decode_sense_voice_once(sample_rate, w)
        if text:
            return text

    if raw_f.size == 0:
        return ""

    pad_r = int(0.18 * sample_rate)
    padded_r = np.pad(np.array(raw_f, copy=True), (0, pad_r), mode="constant", constant_values=0.0)
    text = _decode_sense_voice_once(sample_rate, padded_r)
    if text:
        return text

    pad_l = int(0.08 * sample_rate)
    pad_r2 = int(0.12 * sample_rate)
    padded_lr = np.pad(np.array(raw_f, copy=True), (pad_l, pad_r2), mode="constant", constant_values=0.0)
    text = _decode_sense_voice_once(sample_rate, padded_lr)
    if text:
        return text

    # 极轻音量尾音时偶发全空：略提升幅度再试（限幅避免爆音）
    boosted = np.clip(np.array(raw_f, copy=True) * 1.75, -1.0, 1.0)
    text = _decode_sense_voice_once(sample_rate, boosted)
    return text


def decode_pcm_data(data):
    """解码 Web Audio PCM 数据 (Float32LE)"""
    try:
        pcm_data = base64.b64decode(data)
        samples = np.frombuffer(pcm_data, dtype=np.float32)
        return samples
    except Exception as e:
        print(f"解码 PCM 数据失败: {e}")
        return None


def normalize_recognition_text(text):
    """清理识别结果中的首尾空白。"""
    return text.strip()


def merge_segment_text(full_result, segment_text, is_final=False):
    """拼接 VAD 片段，尽量保留模型自身标点。"""
    segment_text = normalize_recognition_text(segment_text)
    if not segment_text:
        return full_result.strip()

    if not full_result:
        return segment_text

    full_result = full_result.rstrip()
    if not full_result:
        return segment_text

    # 如果边界处已经有标点，直接拼接，避免“标点乱改”
    if full_result[-1] in TERMINAL_PUNCTUATION + PAUSE_PUNCTUATION:
        return f"{full_result}{segment_text}".strip()

    if segment_text[0] in TERMINAL_PUNCTUATION + PAUSE_PUNCTUATION:
        return f"{full_result}{segment_text}".strip()

    # 尽量不打断模型输出的顿号/冒号等：中文连续文本默认直接拼接
    def _is_latin(ch: str) -> bool:
        return ("A" <= ch <= "Z") or ("a" <= ch <= "z")

    prev = full_result[-1]
    nxt = segment_text[0]
    if _is_latin(prev) and _is_latin(nxt):
        return f"{full_result} {segment_text}".strip()
    if prev.isdigit() and nxt.isdigit():
        return f"{full_result}{segment_text}".strip()

    return f"{full_result}{segment_text}".strip()


@app.route('/')
def index():
    """主页"""
    return render_template('vad-sensevoice.html')


@app.route('/api/status', methods=['GET'])
def status():
    """服务状态"""
    return json.dumps({
        "status": "ok",
        "vad_loaded": vad is not None,
        "recognizer_loaded": recognizer is not None,
        "provider": "cuda",
        "model_name": "VAD + SenseVoice"
    })


@sock.route('/ws/vad-asr')
def vad_asr(ws):
    """VAD + SenseVoice 识别端点"""
    if vad is None or recognizer is None or vad_window_size <= 0:
        ws.send(json.dumps({"type": "error", "message": "模型未初始化"}))
        return

    sample_rate = 16000
    max_live_buffer_samples = int(6.0 * sample_rate)
    buffer = np.array([], dtype=np.float32)  # 累积待送入 VAD 的音频
    live_buffer = np.array([], dtype=np.float32)  # 当前句子的实时音频，用于生成预览结果
    rolling_audio_tail = np.array([], dtype=np.float32)  # 语音开始前的上下文
    pending_segment_preroll = np.array([], dtype=np.float32)
    speech_active = False
    full_result = ""  # 累积识别结果
    last_segment_count = 0
    last_partial_text = ""
    last_partial_decode_time = 0.0

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
                # 开始新的识别会话，支持语言模式切换
                req_mode = data.get('mode', current_mode)
                with model_lock:
                    switch_language_mode(req_mode)
                    if hasattr(vad, "reset"):
                        vad.reset()
                    else:
                        vad.clear()
                buffer = np.array([], dtype=np.float32)
                live_buffer = np.array([], dtype=np.float32)
                rolling_audio_tail = np.array([], dtype=np.float32)
                pending_segment_preroll = np.array([], dtype=np.float32)
                speech_active = False
                full_result = ""
                last_segment_count = 0
                last_partial_text = ""
                last_partial_decode_time = 0.0
                ws.send(json.dumps({"type": "started", "mode": current_mode}))

            elif msg_type == 'audio':
                # 接收音频数据
                pcm_data = data.get('data')
                samples = decode_pcm_data(pcm_data)
                if samples is not None:
                    buffer = np.concatenate([buffer, samples])
                    live_buffer = np.concatenate([live_buffer, samples])
                    rolling_audio_tail = np.concatenate([rolling_audio_tail, samples])
                    max_tail_samples = int(PRE_SPEECH_CONTEXT_SECONDS * sample_rate)
                    if len(rolling_audio_tail) > max_tail_samples:
                        rolling_audio_tail = rolling_audio_tail[-max_tail_samples:]
                    segments_to_decode = []
                    speech_detected = False

                    # 按 VAD 期望的 window_size 分帧送入；pending 的“新句 rolling”须在 pop 之后更新，
                    # 否则同一块音频里刚结束的旧段会误用新句开头的上下文作 preroll。
                    with model_lock:
                        while len(buffer) >= vad_window_size:
                            vad.accept_waveform(buffer[:vad_window_size])
                            buffer = buffer[vad_window_size:]

                        speech_detected = vad.is_speech_detected()
                        while not vad.empty():
                            segment = vad.front
                            vad.pop()
                            segments_to_decode.append(np.array(segment.samples, dtype=np.float32))

                    speech_just_started = speech_detected and not speech_active

                    if speech_detected and PARTIAL_ENABLED:
                        now = time.time()
                        enough_audio = len(live_buffer) >= int(0.6 * sample_rate)
                        should_decode_partial = now - last_partial_decode_time >= PARTIAL_DECODE_INTERVAL_SECONDS
                        if enough_audio and should_decode_partial:
                            max_partial_samples = int(PARTIAL_MAX_SECONDS * sample_rate)
                            partial_audio = live_buffer[-max_partial_samples:]
                            with model_lock:
                                stream = recognizer.create_stream()
                                stream.accept_waveform(sample_rate, partial_audio)
                                recognizer.decode_stream(stream)
                                partial_text = normalize_recognition_text(stream.result.text)

                            last_partial_decode_time = now
                            if partial_text and partial_text != last_partial_text:
                                last_partial_text = partial_text
                                preview_text = merge_segment_text(full_result, partial_text, is_final=False)
                                ws.send(json.dumps({
                                    "type": "partial",
                                    "text": preview_text,
                                    "committed_text": full_result,
                                    "partial_text": partial_text,
                                    "segments": last_segment_count,
                                }))
                    elif len(live_buffer) > max_live_buffer_samples:
                        # 没检测到语音时，只保留少量上下文，避免缓冲区无限增长
                        live_buffer = live_buffer[-max_live_buffer_samples:]

                    # 同一次 VAD 弹出多段时，除首段外须带上一段尾部作上下文，否则易丢句首/丢短句
                    preroll_carry = (
                        pending_segment_preroll.copy()
                        if pending_segment_preroll.size
                        else np.array([], dtype=np.float32)
                    )
                    for segment_audio in segments_to_decode:
                        if len(segment_audio) < int(MIN_SEGMENT_SECONDS * sample_rate):
                            continue

                        raw_segment = np.array(segment_audio, dtype=np.float32, copy=False)
                        enhanced_segment = attach_preroll_and_pad(raw_segment, preroll_carry, sample_rate, overlap_threshold=LANGUAGE_PRESETS[current_mode]["overlap_threshold"])

                        with model_lock:
                            text = decode_sense_voice_segment(
                                sample_rate, enhanced_segment, raw_segment
                            )

                        if not text:
                            continue

                        full_result = merge_segment_text(full_result, text, is_final=False)

                        last_segment_count += 1

                        ws.send(json.dumps({
                            "type": "result",
                            "text": full_result,
                            "segments": last_segment_count
                        }))

                        live_buffer = np.array([], dtype=np.float32)
                        last_partial_text = ""
                        last_partial_decode_time = 0.0

                        tail_n = int(max(PRE_SPEECH_CONTEXT_SECONDS, 0.15) * sample_rate)
                        if raw_segment.size > 0:
                            preroll_carry = raw_segment[-min(tail_n, raw_segment.size) :].copy()
                        else:
                            preroll_carry = np.array([], dtype=np.float32)

                    if speech_just_started:
                        pending_segment_preroll = rolling_audio_tail.copy()
                    else:
                        pending_segment_preroll = preroll_carry
                    speech_active = speech_detected

            elif msg_type == 'stop':
                # 停止录音，处理剩余语音
                segments_to_decode = []
                with model_lock:
                    while len(buffer) >= vad_window_size:
                        vad.accept_waveform(buffer[:vad_window_size])
                        buffer = buffer[vad_window_size:]

                    vad.flush()
                    while not vad.empty():
                        segment = vad.front
                        vad.pop()
                        segments_to_decode.append(np.array(segment.samples, dtype=np.float32))

                preroll_carry = (
                    pending_segment_preroll.copy()
                    if pending_segment_preroll.size
                    else np.array([], dtype=np.float32)
                )
                for segment_audio in segments_to_decode:
                    if len(segment_audio) < int(MIN_SEGMENT_SECONDS * sample_rate):
                        continue

                    raw_segment = np.array(segment_audio, dtype=np.float32, copy=False)
                    enhanced_segment = attach_preroll_and_pad(raw_segment, preroll_carry, sample_rate, overlap_threshold=LANGUAGE_PRESETS[current_mode]["overlap_threshold"])

                    with model_lock:
                        text = decode_sense_voice_segment(
                            sample_rate, enhanced_segment, raw_segment
                        )

                    if not text:
                        continue

                    full_result = merge_segment_text(full_result, text, is_final=True)

                    last_segment_count += 1

                    tail_n = int(max(PRE_SPEECH_CONTEXT_SECONDS, 0.15) * sample_rate)
                    if raw_segment.size > 0:
                        preroll_carry = raw_segment[-min(tail_n, raw_segment.size) :].copy()
                    else:
                        preroll_carry = np.array([], dtype=np.float32)

                pending_segment_preroll = preroll_carry

                # 发送最终结果
                ws.send(json.dumps({
                    "type": "result",
                    "text": full_result.strip(),
                    "segments": last_segment_count,
                    "final": True
                }))

                buffer = np.array([], dtype=np.float32)
                live_buffer = np.array([], dtype=np.float32)
                rolling_audio_tail = np.array([], dtype=np.float32)
                pending_segment_preroll = np.array([], dtype=np.float32)
                speech_active = False
                full_result = ""
                last_partial_text = ""
                last_partial_decode_time = 0.0

            elif msg_type == 'ping':
                ws.send(json.dumps({"type": "pong"}))

    except Exception as e:
        print(f"WebSocket 错误: {e}")
        import traceback
        traceback.print_exc()
        try:
            ws.send(json.dumps({"type": "error", "message": str(e)}))
        except:
            pass


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

    parser = argparse.ArgumentParser(description="sherpa-onnx HTTPS VAD + SenseVoice 分段识别")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=6020, help="监听端口")
    parser.add_argument("--cert", type=str, default="cert.pem", help="证书文件")
    parser.add_argument("--no-ssl", action="store_true", help="不使用 HTTPS")
    parser.add_argument("--provider", type=str, default="cuda", help="ONNX 推理后端: cuda / cpu（默认 cuda）")
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.38,
        help="Silero VAD 语音概率阈值；过高易把句尾当静音(吞字)，过低底噪易拖着不断句",
    )
    parser.add_argument(
        "--vad-min-silence",
        type=float,
        default=0.22,
        help="判定为一句结束的最短静音(秒)；过小句内短停顿也会切段→短段空识别易吞句，略大更整句",
    )
    parser.add_argument(
        "--vad-min-speech",
        type=float,
        default=0.10,
        help="判为「开始说话」前至少多长语音(秒)；过大易吃掉句首短音，过小易把噪声当头",
    )
    parser.add_argument(
        "--vad-max-speech",
        type=float,
        default=12.0,
        help="单段最长语音（秒），超过后 VAD 内部会提高阈值分段",
    )
    parser.add_argument(
        "--vad-neg-threshold",
        type=float,
        default=None,
        help="[当前 sherpa_onnx 版本不支持，参数被忽略] Silero 退出滞回阈值",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        help="SenseVoice 语言 bias: auto, zh, en, ja, ko, yue（默认 auto 自动检测）",
    )
    parser.add_argument(
        "--no-partial",
        action="store_true",
        help="关闭实时 partial 解码，仅输出 VAD 定稿分句，减轻“预览与定稿不一致”的错觉",
    )
    parser.add_argument(
        "--min-segment-seconds",
        type=float,
        default=0.03,
        help="过短的 VAD 分段直接跳过（秒）",
    )
    parser.add_argument(
        "--segment-context-seconds",
        type=float,
        default=0.25,
        help="分段识别时在尾部补的上下文（秒），过大容易造成回声/字重复",
    )
    parser.add_argument(
        "--pre-speech-context-seconds",
        type=float,
        default=0.35,
        help="语音段开始前保留的前置上下文（秒），过小易丢句首，过大易回声",
    )
    parser.add_argument(
        "--partial-interval",
        type=float,
        default=0.40,
        help="实时 partial 解码最小间隔（秒）",
    )
    parser.add_argument(
        "--partial-max-seconds",
        type=float,
        default=5.0,
        help="实时 partial 解码最多使用最近多少秒音频（秒），避免长句越来越慢",
    )
    parser.add_argument(
        "--preroll-overlap-search-seconds",
        type=float,
        default=0.50,
        help="拼接句首 pre-roll 时，在尾部/头部搜索重复重叠的最大长度（秒）",
    )

    args = parser.parse_args()

    global MIN_SEGMENT_SECONDS, SEGMENT_CONTEXT_SECONDS, PRE_SPEECH_CONTEXT_SECONDS, PARTIAL_DECODE_INTERVAL_SECONDS
    global PARTIAL_MAX_SECONDS, PREROLL_OVERLAP_SEARCH_SECONDS, PARTIAL_ENABLED
    MIN_SEGMENT_SECONDS = args.min_segment_seconds
    SEGMENT_CONTEXT_SECONDS = args.segment_context_seconds
    PRE_SPEECH_CONTEXT_SECONDS = args.pre_speech_context_seconds
    PARTIAL_DECODE_INTERVAL_SECONDS = args.partial_interval
    PARTIAL_MAX_SECONDS = args.partial_max_seconds
    PREROLL_OVERLAP_SEARCH_SECONDS = args.preroll_overlap_search_seconds
    PARTIAL_ENABLED = not args.no_partial

    # 初始化模型
    try:
        init_vad(
            vad_threshold=args.vad_threshold,
            min_silence_duration=args.vad_min_silence,
            min_speech_duration=args.vad_min_speech,
            max_speech_duration=args.vad_max_speech,
            vad_neg_threshold=args.vad_neg_threshold,
            provider=args.provider,
        )
        init_recognizer(language=args.language, provider=args.provider)
    except Exception as e:
        print(f"初始化模型失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 确保模板目录存在。页面样式与交互统一维护在 templates/vad-sensevoice.html 中，
    # 避免启动时被脚本动态覆盖，便于后续 UI 定制与迭代。
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

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
            cert.get_subject().CN = "sherpa-onnx-vad-sensevoice"
            cert.set_serial_number(5000)
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
    print("  sherpa-onnx HTTPS VAD + SenseVoice 智能分段识别已启动")
    print("=" * 70)
    print(f"\n架构: Silero VAD 自动断句 + SenseVoice 识别")
    print(f"Supported languages: 中文/英文/日语/韩语/粤语")
    print(f"\n请在浏览器中访问以下地址之一:")
    protocol = "http" if args.no_ssl else "https"
    print(f"  - 本机: {protocol}://localhost:{args.port}")
    print(f"  - 局域网: {protocol}://{server_ip}:{args.port}")
    print(f"\n✅ 特点:")
    print(f"  - 👍 SenseVoice 高准确率")
    print(f"  - 🔍 VAD 自动检测停顿切分句子")
    print(f"  - ⚡ 不会累积变慢，每句单独识别")
    print(f"  - 🎯 一句话说完就出结果")
    print(f"  - GPU 加速 (CUDA)")
    print(f"\n使用方法:")
    print(f"  1. 点击'开始识别'")
    print(f"  2. 正常说话，句子之间自然停顿")
    print(f"  3. VAD 检测到停顿自动切分识别")
    print(f"  4. 说完点击'停止识别'")
    print("\n按 Ctrl+C 停止服务\n")

    # 让 Werkzeug 自己创建监听 socket，避免重复 bind 导致误报端口占用
    from werkzeug.serving import run_simple

    ssl_context = None
    if not args.no_ssl:
        ssl_context = (args.cert, args.cert)

    print(f"* Listening on {args.host}:{args.port}")
    try:
        run_simple(args.host, args.port, app, threaded=True, ssl_context=ssl_context)
    except OSError as e:
        if e.errno in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", -1)):
            print(
                "\n端口已被占用 (EADDRINUSE)。常见原因:\n"
                "  1) Cursor/VS Code「端口转发」仍把本机 TCP 6010 留给 ssh/sshd 进程显示为占用——"
                "在 IDE 的 Ports 面板里删掉旧转发，或断开重连 SSH 后再启动。\n"
                "  2) 上次 python 未退出（较少发生在整机重启后）；可用: "
                f"sudo ss -tlnp 'sport = :{args.port}' 或 sudo lsof -iTCP:{args.port} -sTCP:LISTEN 查看进程。\n"
                "  3) 开发时可在启动脚本里设 FREE_PORT=1 尝试释放端口（会 kill 占用该端口的进程，慎用）。\n"
            )
        raise


if __name__ == "__main__":
    main()
