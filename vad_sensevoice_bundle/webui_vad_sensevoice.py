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

# 全局模型
vad = None
recognizer = None
vad_window_size = 0
model_lock = threading.Lock()

# 模型路径相对本脚本所在目录，避免机器/目录结构不同导致硬编码失效
_REPO_ROOT = Path(__file__).resolve().parent
_MODELS_DIR = _REPO_ROOT / "models"

TERMINAL_PUNCTUATION = "。！？!?."
PAUSE_PUNCTUATION = "，、,；;：:"
# 运行时可由 main() 根据命令行参数覆盖
MIN_SEGMENT_SECONDS = 0.03
SEGMENT_CONTEXT_SECONDS = 0.30
PRE_SPEECH_CONTEXT_SECONDS = 0.34
PARTIAL_DECODE_INTERVAL_SECONDS = 0.40
PARTIAL_MAX_SECONDS = 5.0
PREROLL_OVERLAP_SEARCH_SECONDS = 0.25
PARTIAL_ENABLED = True


def _best_suffix_prefix_overlap(preroll: np.ndarray, segment: np.ndarray, max_overlap_samples: int) -> int:
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

    # 阈值：必须是“高度相似”的重叠，否则宁可不做去重（避免把音频拼坏）
    if best_score < 0.90:
        return 0
    return best_k


def attach_preroll_and_pad(segment_audio: np.ndarray, pending_preroll: np.ndarray, sample_rate: int) -> np.ndarray:
    """拼接句首上下文并做轻量 padding，尽量避免重复喂同一段音频。"""
    audio = segment_audio
    if pending_preroll.size > 0:
        max_overlap = int(PREROLL_OVERLAP_SEARCH_SECONDS * sample_rate)
        overlap = _best_suffix_prefix_overlap(pending_preroll, audio, max_overlap_samples=max_overlap)
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
):
    """初始化 VAD 语音活动检测"""
    global vad, vad_window_size

    model_path = _MODELS_DIR / "silero_vad.onnx"
    if not model_path.exists():
        raise ValueError(f"VAD 模型不存在: {model_path}\n请先下载 silero_vad.onnx 放到 models 目录")

    _provider = os.environ.get("SHERPA_ONNX_PROVIDER", "cuda")
    print(f"加载 Silero VAD 模型: {model_path} (provider={_provider})")
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
    # 新版 Python 绑定可写；略抬高 neg_threshold 可收窄滞回，句尾在底噪下更易“收束”（默认 -1 由 C++ 用 threshold-0.15）
    if vad_neg_threshold is not None and hasattr(silero_config, "neg_threshold"):
        silero_config.neg_threshold = float(vad_neg_threshold)
        print(f"Silero VAD neg_threshold={vad_neg_threshold}（自定义退出滞回）")

    vad_config = sherpa_onnx.VadModelConfig(
        silero_vad=silero_config,
        sample_rate=16000,
        num_threads=4,
        provider=_provider,
        debug=False,
    )

    vad = sherpa_onnx.VoiceActivityDetector(
        vad_config,
        buffer_size_in_seconds=60,
    )
    vad_window_size = vad_config.silero_vad.window_size
    print("VAD 模型加载完成！")
    return vad


def init_recognizer():
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

    _provider = os.environ.get("SHERPA_ONNX_PROVIDER", "cuda")
    print(f"正在加载 SenseVoice 模型 (provider={_provider})...")

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model_file),
        tokens=str(tokens_file),
        use_itn=True,
        debug=False,
        provider=_provider,
        num_threads=4
    )
    print("SenseVoice 模型加载完成！支持：中文/英文/日语/韩语/粤语")
    return recognizer


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
        "provider": os.environ.get("SHERPA_ONNX_PROVIDER", "cuda"),
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
                # 开始新的识别会话
                buffer = np.array([], dtype=np.float32)
                live_buffer = np.array([], dtype=np.float32)
                rolling_audio_tail = np.array([], dtype=np.float32)
                pending_segment_preroll = np.array([], dtype=np.float32)
                speech_active = False
                full_result = ""
                last_segment_count = 0
                last_partial_text = ""
                last_partial_decode_time = 0.0
                with model_lock:
                    if hasattr(vad, "reset"):
                        vad.reset()
                    else:
                        vad.clear()
                ws.send(json.dumps({"type": "started"}))

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
                        enhanced_segment = attach_preroll_and_pad(raw_segment, preroll_carry, sample_rate)

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
                    enhanced_segment = attach_preroll_and_pad(raw_segment, preroll_carry, sample_rate)

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
    parser.add_argument("--port", type=int, default=6010, help="监听端口")
    parser.add_argument("--cert", type=str, default="cert.pem", help="证书文件")
    parser.add_argument("--no-ssl", action="store_true", help="不使用 HTTPS")
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
        help="Silero 退出滞回阈值(需 sherpa_onnx 暴露 neg_threshold)；如 0.38 配合 threshold=0.45，句尾在杂音下更易断",
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
        default=0.30,
        help="分段识别时在尾部补的上下文（秒），减轻截断与吞尾",
    )
    parser.add_argument(
        "--pre-speech-context-seconds",
        type=float,
        default=0.34,
        help="语音段开始前保留的前置上下文（秒），减轻句首被吃",
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
        default=0.25,
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
        )
        init_recognizer()
    except Exception as e:
        print(f"初始化模型失败: {e}")
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
    <title>sherpa-onnx VAD + SenseVoice 智能分段识别</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
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
            color: #1e1b4b;
            text-align: center;
            margin-bottom: 10px;
            font-size: 30px;
            font-weight: 700;
        }
        .subtitle {
            text-align: center;
            color: #64748b;
            margin-bottom: 15px;
            font-size: 16px;
        }
        .model-badge {
            text-align: center;
            margin-bottom: 25px;
        }
        .model-badge span {
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
            color: white;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
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
            color: #475569;
            font-weight: 500;
            flex: 1;
        }
        .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 25px;
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
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
            color: white;
        }
        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(139, 92, 246, 0.35);
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
            min-height: 220px;
            max-height: 400px;
            overflow-y: auto;
        }
        .result-label {
            color: #94a3b8;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .segment-count {
            background: #334155;
            color: #e2e8f0;
            padding: 2px 10px;
            border-radius: 10px;
            font-size: 12px;
        }
        .result-text {
            color: #f1f5f9;
            font-size: 22px;
            line-height: 1.8;
            word-wrap: break-word;
            font-family: 'SF Mono', 'Fira Code', monospace;
        }
        .result-text.empty {
            color: #475569;
            font-style: italic;
        }
        .waveform {
            margin-top: 25px;
            height: 80px;
            background: #f1f5f9;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            padding: 15px;
        }
        .wave-bar {
            width: 5px;
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
            border-radius: 3px;
            transition: height 0.08s;
        }
        .languages {
            margin-top: 15px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
        }
        .lang-tag {
            background: #f3f4f6;
            color: #374151;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 500;
        }
        .info-box {
            margin-top: 25px;
            padding: 18px 22px;
            background: #ede9fe;
            border-radius: 12px;
            border-left: 4px solid #8b5cf6;
        }
        .info-box p {
            color: #581c87;
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
        <h1>🎯 智能分段语音识别</h1>
        <p class="subtitle">VAD 自动断句 + SenseVoice 高准确率识别</p>

        <div class="model-badge">
            <span>✓ VAD自动断句 ✓ SenseVoice 高准确率</span>
        </div>

        <div class="languages">
            <span class="lang-tag">中文</span>
            <span class="lang-tag">English</span>
            <span class="lang-tag">日本語</span>
            <span class="lang-tag">한국어</span>
            <span class="lang-tag">粤语</span>
        </div>

        <div class="status-bar" style="margin-top: 25px;">
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

        <div class="waveform" id="waveform"></div>

        <div class="stats" id="stats" style="display: none;">
            <div class="stat-card">
                <div class="stat-value" id="segmentCount">0</div>
                <div class="stat-label">已识别句子</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="audioLength">0s</div>
                <div class="stat-label">音频时长</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="isRecording">否</div>
                <div class="stat-label">录音状态</div>
            </div>
        </div>

        <div class="result-container" style="margin-top: 25px;">
            <div class="result-label">
                识别结果
                <span class="segment-count" id="segmentCountBadge">0 个句子</span>
            </div>
            <div class="result-text empty" id="resultText">点击"开始识别"开始说话...<br>VAD 会自动检测停顿切分句子，每句实时识别</div>
        </div>

        <div class="info-box">
            <p><strong>工作原理：</strong>Silero VAD 检测语音活动，自动根据停顿切分成句子，每个句子送给 SenseVoice 识别。结合了 VAD 切分能力和 SenseVoice 高准确率，既不会累积变慢，识别效果又比纯流式 Zipformer 好。</p>
        </div>
    </div>

    <script>
        let ws = null;
        let mediaStream = null;
        let audioContext = null;
        let processor = null;
        let sourceNode = null;
        let isRecording = false;
        let startTime = 0;
        let totalSamples = 0;
        let committedText = '';
        let partialText = '';
        let frozenPartialPrefix = '';
        let lastRawPartialText = '';

        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const resultText = document.getElementById('resultText');
        const waveform = document.getElementById('waveform');
        const stats = document.getElementById('stats');
        const segmentCountEl = document.getElementById('segmentCount');
        const segmentCountBadge = document.getElementById('segmentCountBadge');
        const audioLengthEl = document.getElementById('audioLength');
        const isRecordingEl = document.getElementById('isRecording');

        // 初始化波形条
        function initWaveform() {
            waveform.innerHTML = '';
            for (let i = 0; i < 70; i++) {
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
            const wsUrl = `${protocol}//${location.host}/ws/vad-asr`;

            statusText.textContent = '正在连接...';
            statusDot.className = 'status-dot';

            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                statusText.textContent = '已连接 - VAD+SenseVoice 就绪';
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

        function longestCommonPrefix(a, b) {
            const maxLen = Math.min(a.length, b.length);
            let i = 0;
            while (i < maxLen && a[i] === b[i]) {
                i += 1;
            }
            return a.slice(0, i);
        }

        function mergeOverlappingText(previousText, nextText) {
            if (!previousText) return nextText || '';
            if (!nextText) return previousText || '';
            if (nextText.startsWith(previousText)) return nextText;
            if (previousText.includes(nextText)) return previousText;

            const maxOverlap = Math.min(previousText.length, nextText.length);
            for (let overlap = maxOverlap; overlap > 0; overlap -= 1) {
                if (previousText.slice(-overlap) === nextText.slice(0, overlap)) {
                    return previousText + nextText.slice(overlap);
                }
            }

            return nextText;
        }

        function buildStablePreview() {
            const previewTail = partialText.startsWith(frozenPartialPrefix)
                ? partialText.slice(frozenPartialPrefix.length)
                : partialText;

            const previewText = committedText
                ? `${committedText} ${(frozenPartialPrefix + previewTail).trim()}`.trim()
                : (frozenPartialPrefix + previewTail).trim();

            return previewText.trim();
        }

        function refreshResultDisplay() {
            const preview = buildStablePreview();
            if (preview) {
                resultText.textContent = preview;
            }
        }

        function handleMessage(data) {
            switch (data.type) {
                case 'started':
                    committedText = '';
                    partialText = '';
                    frozenPartialPrefix = '';
                    lastRawPartialText = '';
                    resultText.textContent = '';
                    resultText.classList.remove('empty');
                    stats.style.display = 'grid';
                    segmentCountEl.textContent = '0';
                    segmentCountBadge.textContent = '0 个句子';
                    audioLengthEl.textContent = '0s';
                    startTime = Date.now();
                    totalSamples = 0;
                    break;

                case 'result':
                    if (data.text) {
                        committedText = data.text;
                        partialText = '';
                        frozenPartialPrefix = '';
                        lastRawPartialText = '';
                        resultText.textContent = committedText;
                    } else {
                        // 避免空消息把界面刷成空白；仍保留最后一次预览
                        refreshResultDisplay();
                    }
                    if (data.segments !== undefined) {
                        segmentCountEl.textContent = data.segments;
                        segmentCountBadge.textContent = data.segments + ' 个句子';
                    }
                    if (data.final) {
                        resultText.style.fontWeight = 'bold';
                    } else {
                        resultText.style.fontWeight = 'normal';
                    }
                    break;

                case 'partial':
                    if (data.committed_text !== undefined) {
                        if (data.committed_text.length >= committedText.length) {
                            committedText = data.committed_text;
                        }
                    }

                    if (data.partial_text) {
                        const commonPrefix = longestCommonPrefix(lastRawPartialText, data.partial_text);
                        if (commonPrefix.length > frozenPartialPrefix.length) {
                            frozenPartialPrefix = commonPrefix;
                        }
                        partialText = mergeOverlappingText(partialText, data.partial_text);
                        lastRawPartialText = data.partial_text;
                    }

                    if (committedText || partialText) {
                        refreshResultDisplay();
                        resultText.style.fontWeight = 'normal';
                    } else if (data.text) {
                        // 兼容旧消息格式
                        resultText.textContent = data.text;
                        resultText.style.fontWeight = 'normal';
                    }
                    if (data.segments !== undefined) {
                        segmentCountEl.textContent = data.segments;
                        segmentCountBadge.textContent = data.segments + ' 个句子';
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
                // 开启浏览器噪声抑制/回声消除：关闭时环境底噪易被 Silero 判为“持续在说话”，句尾无法断句
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
                const bufferSize = 2048;
                processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

                processor.onaudioprocess = (e) => {
                    if (!isRecording) return;

                    const inputData = e.inputBuffer.getChannelData(0);
                    totalSamples += inputData.length;
                    sendAudioData(inputData);
                    updateWaveform(inputData);

                    const duration = totalSamples / 16000;
                    audioLengthEl.textContent = duration.toFixed(1) + 's';
                };

                sourceNode.connect(processor);
                processor.connect(audioContext.destination);

                // 开始识别会话
                ws.send(JSON.stringify({ type: 'start' }));
                isRecording = true;

                startBtn.disabled = true;
                stopBtn.disabled = false;
                statusDot.className = 'status-dot recording';
                statusText.textContent = '正在录音 - VAD自动断句识别中';
                isRecordingEl.textContent = '是';

            } catch (err) {
                alert('无法访问麦克风: ' + err.message);
                console.error(err);
            }
        }

        function sendAudioData(audioData) {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;

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
            // 停录后先保留最后一次 partial 预览，等服务端最终 result 到来再清空，
            // 避免出现“识别过程有字，停录瞬间整段消失”的错觉
            startBtn.disabled = false;
            stopBtn.disabled = true;
            statusDot.className = 'status-dot connected';
            statusText.textContent = '已连接 - 等待开始';
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

    with open(templates_dir / "vad-sensevoice.html", "w") as f:
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
