#!/usr/bin/env python3
"""
专门测试 GPU 内存使用情况
"""

import os
import sys
import time
import psutil
from pathlib import Path

try:
    import sherpa_onnx
except ImportError:
    print("sherpa_onnx not found")
    sys.exit(1)


def get_gpu_memory():
    """获取 GPU 内存使用"""
    try:
        import subprocess
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                                       encoding='utf-8')
        used, total = result.strip().split(', ')
        return float(used), float(total)
    except Exception as e:
        print(f"获取 GPU 信息失败: {e}")
        return None


def create_recognizer(use_gpu=True, use_int8=False):
    """创建流式识别器"""
    model_dir = Path("/home/tsingwin/apps/sherpa-onnx/models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20")

    suffix = ".int8.onnx" if use_int8 else ".onnx"
    encoder_file = model_dir / f"encoder-epoch-99-avg-1{suffix}"
    decoder_file = model_dir / f"decoder-epoch-99-avg-1{suffix}"
    joiner_file = model_dir / f"joiner-epoch-99-avg-1{suffix}"
    tokens_file = model_dir / "tokens.txt"

    provider = "cuda" if use_gpu else "cpu"

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(tokens_file),
        encoder=str(encoder_file),
        decoder=str(decoder_file),
        joiner=str(joiner_file),
        num_threads=4,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        provider=provider,
    )

    return recognizer


def main():
    print("=" * 60)
    print("GPU 内存使用测试")
    print("=" * 60)

    # 初始 GPU 内存
    print("\n初始状态:")
    initial_gpu = get_gpu_memory()
    if initial_gpu:
        used, total = initial_gpu
        print(f"  GPU 内存: {used:.0f} MB / {total:.0f} MB")

    # 测试 GPU FP32
    print("\n--- 加载 FP32 模型 ---")
    try:
        recognizer = create_recognizer(use_gpu=True, use_int8=False)
        time.sleep(1)  # 等待稳定

        fp32_gpu = get_gpu_memory()
        if fp32_gpu:
            used, total = fp32_gpu
            print(f"  GPU 内存: {used:.0f} MB / {total:.0f} MB")
            if initial_gpu:
                print(f"  额外使用: {used - initial_gpu[0]:.0f} MB")

        # 做一次推理
        import numpy as np
        sample_rate = 16000
        audio = np.zeros(int(sample_rate * 1), dtype=np.float32)
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        del recognizer
        time.sleep(1)

    except Exception as e:
        print(f"失败: {e}")

    # 测试 GPU INT8
    print("\n--- 加载 INT8 模型 ---")
    try:
        recognizer_int8 = create_recognizer(use_gpu=True, use_int8=True)
        time.sleep(1)

        int8_gpu = get_gpu_memory()
        if int8_gpu:
            used, total = int8_gpu
            print(f"  GPU 内存: {used:.0f} MB / {total:.0f} MB")
            if initial_gpu:
                print(f"  额外使用: {used - initial_gpu[0]:.0f} MB")

        del recognizer_int8
        time.sleep(1)

    except Exception as e:
        print(f"失败: {e}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n实时 ASR 功能说明:")
    print("  - 使用 'speech-recognition-from-microphone.py' 进行实时识别")
    print("  - 设置 --provider=cuda 启用 GPU 加速")
    print("  - 推荐使用 INT8 模型以节省内存")


if __name__ == "__main__":
    main()
