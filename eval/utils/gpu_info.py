"""GPU information detection and logging utility.

Auto-detects GPU hardware, CUDA version, driver info, and ONNX Runtime
configuration. Used to tag results per-GPU environment.
"""

import platform
import subprocess
import json
import os
import re
from datetime import datetime


def get_gpu_info():
    """Detect GPU information and return as dictionary."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "gpu_name": "Unknown",
        "gpu_count": 0,
        "gpu_memory_total_mb": 0,
        "cuda_version": "N/A",
        "cudnn_version": "N/A",
        "driver_version": "N/A",
        "onnxruntime_version": "N/A",
        "onnxruntime_providers": [],
    }

    # ONNX Runtime info
    try:
        import onnxruntime as ort
        info["onnxruntime_version"] = ort.__version__
        info["onnxruntime_providers"] = ort.get_available_providers()
    except ImportError:
        pass

    # GPU info via nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines and lines[0].strip():
                parts = lines[0].split(", ")
                info["gpu_name"] = parts[0].strip()
                info["gpu_memory_total_mb"] = int(float(parts[1].strip()))
                info["driver_version"] = parts[2].strip()
                info["gpu_count"] = len(lines)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # CUDA version
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            match = re.search(r"CUDA Version:\s*([\d.]+)", result.stdout)
            if match:
                info["cuda_version"] = match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # cuDNN version via torch if available (skip if numpy incompatible)
    try:
        import torch
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            info["cudnn_version"] = str(torch.backends.cudnn.version())
    except Exception:
        pass  # Skip torch entirely if any error (numpy compat, import, etc.)

    return info


def get_gpu_tag(info=None):
    """Generate a short tag for the GPU, used for result folder naming.

    Example: 'GPU_NVIDIA_GeForce_RTX_3090'
    """
    if info is None:
        info = get_gpu_info()

    gpu_name = info.get("gpu_name", "Unknown")
    # Clean up GPU name for safe folder naming
    tag = gpu_name.replace(" ", "_").replace("/", "_")
    tag = re.sub(r"[^\w\-]", "", tag)
    return f"GPU_{tag}"


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB. Returns -1 if unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return -1


def print_gpu_info(info=None):
    """Pretty-print GPU environment information."""
    if info is None:
        info = get_gpu_info()

    print("=" * 60)
    print("  GPU Environment Information")
    print("=" * 60)
    print(f"  Platform:          {info['platform']}")
    print(f"  Python:            {info['python_version']}")
    print(f"  GPU:               {info['gpu_name']}")
    print(f"  GPU Count:         {info['gpu_count']}")
    print(f"  GPU Memory:        {info['gpu_memory_total_mb']} MB")
    print(f"  CUDA Version:      {info['cuda_version']}")
    print(f"  cuDNN Version:     {info['cudnn_version']}")
    print(f"  Driver Version:    {info['driver_version']}")
    print(f"  ONNX Runtime:      {info['onnxruntime_version']}")
    print(f"  ORT Providers:     {', '.join(info['onnxruntime_providers'])}")
    print("=" * 60)


def save_gpu_info(output_dir, info=None):
    """Save GPU info to a JSON file in the output directory."""
    if info is None:
        info = get_gpu_info()
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "gpu_info.json")
    with open(filepath, "w") as f:
        json.dump(info, f, indent=2)
    return filepath
