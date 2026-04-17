#!/usr/bin/env python3
"""
测试 sherpa-onnx 流式 GPU 加速的 ASR 功能
并监控资源占用情况
"""

import os
import sys
import time
import psutil
from pathlib import Path

try:
    import sherpa_onnx
except ImportError:
    print("sherpa_onnx not found, please install it first")
    sys.exit(1)


def get_resource_usage():
    """获取 CPU 和内存使用情况"""
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=0.1)
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024

    # 尝试获取 GPU 内存使用
    gpu_memory = None
    try:
        import subprocess
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                                       encoding='utf-8')
        used, total = result.strip().split(', ')
        gpu_memory = (float(used), float(total))
    except Exception as e:
        pass

    return {
        'cpu_percent': cpu_percent,
        'memory_mb': memory_mb,
        'gpu_memory_mb': gpu_memory
    }


def create_recognizer(use_gpu=True, use_int8=False):
    """创建流式识别器"""
    model_dir = Path("/home/tsingwin/apps/sherpa-onnx/models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20")

    suffix = ".int8.onnx" if use_int8 else ".onnx"
    encoder_file = model_dir / f"encoder-epoch-99-avg-1{suffix}"
    decoder_file = model_dir / f"decoder-epoch-99-avg-1{suffix}"
    joiner_file = model_dir / f"joiner-epoch-99-avg-1{suffix}"
    tokens_file = model_dir / "tokens.txt"

    if not encoder_file.exists() or not decoder_file.exists() or not joiner_file.exists() or not tokens_file.exists():
        raise ValueError(f"Model files not found in {model_dir}")

    provider = "cuda" if use_gpu else "cpu"

    print(f"Creating recognizer with provider: {provider}, {'INT8' if use_int8 else 'FP32'}")

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


def test_recognizer(recognizer):
    """测试识别器，生成一些测试音频"""
    import numpy as np

    sample_rate = 16000
    duration = 3.0  # 3秒
    num_samples = int(sample_rate * duration)

    # 生成静音音频
    audio = np.zeros(num_samples, dtype=np.float32)

    print(f"Testing with {duration}s of audio...")

    # 第一次推理（预热）
    print("Warm-up run...")
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    _ = recognizer.get_result(stream)

    # 正式测试
    print("Benchmark run...")
    start_time = time.time()

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    result = recognizer.get_result(stream)

    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Inference time: {inference_time:.3f}s for {duration}s audio")
    print(f"Real-time factor: {inference_time/duration:.3f}x")
    print(f"Result: {result}")

    return inference_time, duration


def main():
    print("=" * 60)
    print("sherpa-onnx 流式 GPU ASR 测试")
    print("=" * 60)

    # 初始资源使用
    print("\n初始资源使用:")
    initial_usage = get_resource_usage()
    print(f"  CPU: {initial_usage['cpu_percent']}%")
    print(f"  内存: {initial_usage['memory_mb']:.2f} MB")
    if initial_usage['gpu_memory_mb']:
        used, total = initial_usage['gpu_memory_mb']
        print(f"  GPU 内存: {used:.0f} MB / {total:.0f} MB")

    # 测试 GPU FP32 版本
    print("\n" + "=" * 60)
    print("测试 GPU FP32 版本")
    print("=" * 60)

    try:
        recognizer_gpu = create_recognizer(use_gpu=True, use_int8=False)

        print("\n加载模型后资源使用:")
        after_load_usage = get_resource_usage()
        print(f"  CPU: {after_load_usage['cpu_percent']}%")
        print(f"  内存: {after_load_usage['memory_mb']:.2f} MB")
        if after_load_usage['gpu_memory_mb']:
            used, total = after_load_usage['gpu_memory_mb']
            print(f"  GPU 内存: {used:.0f} MB / {total:.0f} MB")

        inference_time_gpu, duration = test_recognizer(recognizer_gpu)

        print("\n推理后资源使用:")
        after_infer_usage = get_resource_usage()
        print(f"  CPU: {after_infer_usage['cpu_percent']}%")
        print(f"  内存: {after_infer_usage['memory_mb']:.2f} MB")
        if after_infer_usage['gpu_memory_mb']:
            used, total = after_infer_usage['gpu_memory_mb']
            print(f"  GPU 内存: {used:.0f} MB / {total:.0f} MB")

        # 保存 GPU 版本的资源使用
        gpu_memory_usage = None
        if after_infer_usage['gpu_memory_mb']:
            gpu_memory_usage = after_infer_usage['gpu_memory_mb'][0]

        del recognizer_gpu
        # 给一点时间释放资源
        time.sleep(1)

    except Exception as e:
        print(f"GPU 测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 测试 GPU INT8 版本
    print("\n" + "=" * 60)
    print("测试 GPU INT8 版本")
    print("=" * 60)

    try:
        recognizer_gpu_int8 = create_recognizer(use_gpu=True, use_int8=True)

        print("\n加载模型后资源使用 (INT8):")
        after_load_usage_int8 = get_resource_usage()
        print(f"  CPU: {after_load_usage_int8['cpu_percent']}%")
        print(f"  内存: {after_load_usage_int8['memory_mb']:.2f} MB")
        if after_load_usage_int8['gpu_memory_mb']:
            used, total = after_load_usage_int8['gpu_memory_mb']
            print(f"  GPU 内存: {used:.0f} MB / {total:.0f} MB")

        inference_time_gpu_int8, duration = test_recognizer(recognizer_gpu_int8)

        del recognizer_gpu_int8
        time.sleep(1)

    except Exception as e:
        print(f"GPU INT8 测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 测试 CPU 版本（用于对比）
    print("\n" + "=" * 60)
    print("测试 CPU 版本（对比）")
    print("=" * 60)

    try:
        recognizer_cpu = create_recognizer(use_gpu=False, use_int8=True)

        print("\n加载模型后资源使用 (CPU):")
        after_load_usage_cpu = get_resource_usage()
        print(f"  CPU: {after_load_usage_cpu['cpu_percent']}%")
        print(f"  内存: {after_load_usage_cpu['memory_mb']:.2f} MB")

        inference_time_cpu, duration = test_recognizer(recognizer_cpu)

        del recognizer_cpu

    except Exception as e:
        print(f"CPU 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n资源占用总结:")
    if 'gpu_memory_usage' in locals():
        print(f"  GPU 内存使用: ~{gpu_memory_usage:.0f} MB")


if __name__ == "__main__":
    main()
