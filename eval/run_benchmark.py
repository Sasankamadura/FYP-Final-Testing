"""Latency & Throughput Benchmarking Script.

Measures inference speed (latency, FPS, P95/P99) of each ONNX model
on the current GPU. Results are saved per-GPU for cross-environment comparison.

Usage:
    python run_benchmark.py
    python run_benchmark.py --config eval/config.yaml
    python run_benchmark.py --model baseline_rtdetr_r18
    python run_benchmark.py --e2e    # include end-to-end pipeline timing
"""

import os
import sys
import json
import time
import argparse
import csv
import numpy as np
import yaml
from pathlib import Path
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.onnx_inference import OnnxDetector
from utils.gpu_info import (
    get_gpu_info, get_gpu_tag, print_gpu_info, save_gpu_info, get_gpu_memory_usage
)


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def benchmark_model(detector, warmup_iters, measure_iters, input_size):
    """Benchmark a model's raw inference latency.

    Uses random dummy inputs to isolate model forward-pass time from
    I/O and preprocessing overhead.

    Returns:
        dict with latency statistics (ms), FPS, GPU memory.
    """
    dummy_input = np.random.randn(1, 3, *input_size).astype(np.float32)

    # Warmup phase
    print(f"  Warmup ({warmup_iters} iterations)...")
    for _ in range(warmup_iters):
        detector.inference_raw(dummy_input, orig_size=input_size)

    gpu_mem_before = get_gpu_memory_usage()

    # Measurement phase
    print(f"  Measuring ({measure_iters} iterations)...")
    latencies = []
    for i in range(measure_iters):
        t0 = time.perf_counter()
        detector.inference_raw(dummy_input, orig_size=input_size)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # Convert to ms

    gpu_mem_after = get_gpu_memory_usage()

    latencies = np.array(latencies)

    return {
        "mean_latency_ms": round(float(np.mean(latencies)), 3),
        "std_latency_ms": round(float(np.std(latencies)), 3),
        "min_latency_ms": round(float(np.min(latencies)), 3),
        "max_latency_ms": round(float(np.max(latencies)), 3),
        "median_latency_ms": round(float(np.median(latencies)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 3),
        "fps": round(1000.0 / float(np.mean(latencies)), 2),
        "gpu_memory_mb": gpu_mem_after if gpu_mem_after > 0 else -1,
        "warmup_iterations": warmup_iters,
        "measure_iterations": measure_iters,
    }


def benchmark_e2e(detector, image_path, measure_iters, config):
    """Benchmark end-to-end pipeline (preprocess + inference + postprocess) on a real image.

    Args:
        detector: OnnxDetector instance.
        image_path: path to a sample image.
        measure_iters: number of timed iterations.
        config: full config dict.

    Returns:
        dict with e2e latency and FPS, or None if image can't be loaded.
    """
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        return None

    eval_cfg = config["evaluation"]
    conf_threshold = eval_cfg.get("vis_conf_threshold", 0.25)
    nms_threshold = eval_cfg["nms_threshold"]
    max_detections = eval_cfg["max_detections"]

    # Warmup
    for _ in range(min(10, measure_iters)):
        detector.predict(image, conf_threshold, nms_threshold, max_detections)

    # Measure
    latencies = []
    for _ in range(measure_iters):
        t0 = time.perf_counter()
        detector.predict(image, conf_threshold, nms_threshold, max_detections)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    latencies = np.array(latencies)

    return {
        "e2e_mean_latency_ms": round(float(np.mean(latencies)), 3),
        "e2e_p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
        "e2e_fps": round(1000.0 / float(np.mean(latencies)), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="VisDrone Model Benchmarking")
    parser.add_argument("--config", type=str, default="eval/config.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Benchmark single model by config key")
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--e2e", action="store_true",
                        help="Also benchmark end-to-end pipeline on a real image")
    args = parser.parse_args()

    if args.workspace:
        workspace_root = args.workspace
    else:
        workspace_root = str(Path(args.config).resolve().parent.parent)

    config = load_config(args.config)

    # ---- GPU Info ----
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
    gpu_tag = get_gpu_tag(gpu_info)

    # ---- Output Directory ----
    output_dir = os.path.join(workspace_root, config["output_dir"], gpu_tag, "benchmark")
    os.makedirs(output_dir, exist_ok=True)
    save_gpu_info(output_dir, gpu_info)

    # ---- Benchmark Settings ----
    bench_cfg = config["benchmark"]
    warmup_iters = bench_cfg["warmup_iterations"]
    measure_iters = bench_cfg["measure_iterations"]
    input_size = tuple(config["evaluation"]["input_size"])

    # ---- Find Sample Image for E2E Benchmark ----
    sample_image = None
    if args.e2e:
        val_dir = os.path.join(workspace_root, config["datasets"]["val"]["images_dir"])
        if os.path.exists(val_dir):
            for f in os.listdir(val_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    sample_image = os.path.join(val_dir, f)
                    break
        if sample_image:
            print(f"\nE2E sample image: {sample_image}")
        else:
            print("\n[WARN] No sample image found for E2E benchmark.")

    # ---- Select Models ----
    models = config["models"]
    if args.model:
        if args.model not in models:
            print(f"\nModel '{args.model}' not found. Available: {list(models.keys())}")
            return
        models = {args.model: models[args.model]}

    # ---- Benchmark Each Model ----
    all_results = OrderedDict()

    for model_key, model_cfg in models.items():
        model_path = os.path.join(workspace_root, model_cfg["path"])
        model_name = model_cfg["name"]

        print(f"\n{'=' * 80}")
        print(f"  Benchmarking: {model_name}")
        print(f"{'=' * 80}")

        if not os.path.exists(model_path):
            print(f"  [ERROR] Model file not found, skipping.")
            continue

        try:
            detector = OnnxDetector(model_path, input_size=input_size)
            model_info = detector.get_model_info()

            # Raw inference benchmark
            bench_results = benchmark_model(detector, warmup_iters, measure_iters, input_size)

            # End-to-end benchmark
            if sample_image:
                print("  Benchmarking end-to-end pipeline...")
                e2e_results = benchmark_e2e(detector, sample_image, min(50, measure_iters), config)
                if e2e_results:
                    bench_results.update(e2e_results)

            bench_results["model_size_mb"] = model_info["file_size_mb"]
            bench_results["providers"] = model_info["providers"]

            all_results[model_key] = {
                "name": model_name,
                "decoder_layers": model_cfg.get("decoder_layers", "N/A"),
                "backbone": model_cfg.get("backbone", "N/A"),
                **bench_results,
            }

            print(
                f"  >> Mean: {bench_results['mean_latency_ms']:.2f}ms | "
                f"FPS: {bench_results['fps']:.1f} | "
                f"P95: {bench_results['p95_latency_ms']:.2f}ms | "
                f"Size: {model_info['file_size_mb']:.1f}MB"
            )

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            continue

    # ---- Print Summary Table ----
    if all_results:
        print(f"\n{'=' * 130}")
        print("  BENCHMARK RESULTS SUMMARY")
        print(f"{'=' * 130}")
        print(f"  GPU: {gpu_info['gpu_name']} | CUDA: {gpu_info['cuda_version']} | ORT: {gpu_info['onnxruntime_version']}")
        print(f"{'=' * 130}")

        header = (
            f"{'Model':<40} {'Dec':>4} {'Mean(ms)':>9} {'P95(ms)':>9} "
            f"{'FPS':>8} {'GPU Mem':>8} {'Size(MB)':>9}"
        )
        print(header)
        print("-" * 130)

        for key, res in all_results.items():
            gpu_mem_str = f"{res.get('gpu_memory_mb', -1)}M" if res.get('gpu_memory_mb', -1) > 0 else "N/A"
            row = (
                f"{res['name']:<40} "
                f"{res['decoder_layers']:>4} "
                f"{res['mean_latency_ms']:>9.2f} "
                f"{res['p95_latency_ms']:>9.2f} "
                f"{res['fps']:>8.1f} "
                f"{gpu_mem_str:>8} "
                f"{res.get('model_size_mb', 0):>9.1f}"
            )
            print(row)

        print(f"{'=' * 130}")

        # ---- Save Results ----
        # JSON
        results_json = os.path.join(output_dir, "benchmark_results.json")
        with open(results_json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # CSV
        results_csv = os.path.join(output_dir, "benchmark_results.csv")
        with open(results_csv, "w", newline="") as f:
            writer = csv.writer(f)
            first = True
            for key, res in all_results.items():
                if first:
                    writer.writerow(["model_key"] + list(res.keys()))
                    first = False
                writer.writerow([key] + [str(v) for v in res.values()])

        print(f"\nResults saved to: {output_dir}")
    else:
        print("\nNo models were benchmarked successfully.")


if __name__ == "__main__":
    main()
