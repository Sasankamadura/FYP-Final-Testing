"""Model Parameter Extraction Script.

Counts trainable parameters for each ONNX model and optionally provides
a component-level breakdown (backbone, encoder, decoder, etc.).

Results are printed as a summary table and saved to:
  eval/results/reports/model_parameters.csv
  eval/results/reports/model_parameters.json

Usage:
    python get_model_params.py
    python get_model_params.py --config eval/config.yaml
    python get_model_params.py --model baseline_rtdetr_r18
    python get_model_params.py --breakdown          # show per-component breakdown
"""

import os
import sys
import json
import csv
import argparse
import yaml
from pathlib import Path
from collections import OrderedDict

# Ensure onnx can be found (Windows long-path workaround)
_onnx_fallback = r"D:\onnx_lib"
if os.path.isdir(_onnx_fallback) and _onnx_fallback not in sys.path:
    sys.path.insert(0, _onnx_fallback)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.onnx_inference import OnnxDetector


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_params(count):
    """Format a parameter count as a human-readable string."""
    if count is None:
        return "N/A"
    if count >= 1e6:
        return f"{count / 1e6:.2f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.2f}K"
    return str(count)


def main():
    parser = argparse.ArgumentParser(description="VisDrone Model Parameter Extraction")
    parser.add_argument("--config", type=str, default="eval/config.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Extract params for a single model by config key")
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--breakdown", action="store_true",
                        help="Print per-component parameter breakdown for each model")
    args = parser.parse_args()

    if args.workspace:
        workspace_root = args.workspace
    else:
        workspace_root = str(Path(args.config).resolve().parent.parent)

    config = load_config(args.config)
    input_size = tuple(config["evaluation"]["input_size"])

    # Output directory
    report_dir = os.path.join(workspace_root, config["output_dir"], "reports")
    os.makedirs(report_dir, exist_ok=True)

    # Select models
    models = config["models"]
    if args.model:
        if args.model not in models:
            print(f"Model '{args.model}' not found. Available: {list(models.keys())}")
            return
        models = {args.model: models[args.model]}

    # ---- Extract Parameters ----
    all_results = OrderedDict()

    print(f"\n{'=' * 100}")
    print("  ONNX MODEL PARAMETER COUNT")
    print(f"{'=' * 100}")
    print(
        f"{'Model':<42} {'Dec':>4} {'Params':>12} {'Size(MB)':>10} "
        f"{'Params/MB':>10}"
    )
    print("-" * 100)

    for model_key, model_cfg in models.items():
        model_path = os.path.join(workspace_root, model_cfg["path"])
        model_name = model_cfg["name"]

        if not os.path.exists(model_path):
            print(f"  [ERROR] Not found: {model_path}")
            continue

        try:
            detector = OnnxDetector(model_path, input_size=input_size)
            param_info = detector.get_param_count()
            model_info = detector.get_model_info()

            total = param_info["total_params"]
            size_mb = model_info["file_size_mb"]
            dec = model_cfg.get("decoder_layers", "N/A")

            # Params per MB ratio (rough density metric)
            params_per_mb = (total / size_mb / 1e6) if (total and size_mb) else 0

            print(
                f"{model_name:<42} {dec:>4} "
                f"{format_params(total):>12} {size_mb:>10.1f} "
                f"{params_per_mb:>9.2f}M"
            )

            result = {
                "name": model_name,
                "decoder_layers": dec,
                "backbone": model_cfg.get("backbone", "N/A"),
                "category": model_cfg.get("category", "N/A"),
                "total_params": total,
                "param_str": param_info["param_str"],
                "file_size_mb": size_mb,
            }

            if args.breakdown and param_info["param_breakdown"]:
                result["breakdown"] = param_info["param_breakdown"]
                print(f"  {'Component':<30} {'Params':>14} {'% of Total':>10}")
                print(f"  {'-' * 56}")
                for comp, count in param_info["param_breakdown"].items():
                    pct = (count / total * 100) if total else 0
                    print(
                        f"  {comp:<30} {format_params(count):>14} "
                        f"{pct:>9.1f}%"
                    )
                print()

            all_results[model_key] = result

        except Exception as e:
            print(f"  [ERROR] {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"{'=' * 100}")

    if not all_results:
        print("No models processed.")
        return

    # ---- Summary Statistics ----
    param_values = [
        r["total_params"] for r in all_results.values() if r["total_params"]
    ]
    if param_values:
        print(f"\n  Models analyzed: {len(param_values)}")
        print(f"  Min params:  {format_params(min(param_values))} ({min(param_values):,})")
        print(f"  Max params:  {format_params(max(param_values))} ({max(param_values):,})")
        print(f"  Mean params: {format_params(int(sum(param_values) / len(param_values)))}")

    # ---- Save JSON ----
    json_path = os.path.join(report_dir, "model_parameters.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    # ---- Save CSV ----
    csv_path = os.path.join(report_dir, "model_parameters.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "model_key", "model_name", "decoder_layers", "backbone",
            "category", "total_params", "param_str", "file_size_mb",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, res in all_results.items():
            row = {
                "model_key": key,
                "model_name": res["name"],
                "decoder_layers": res["decoder_layers"],
                "backbone": res["backbone"],
                "category": res["category"],
                "total_params": res["total_params"],
                "param_str": res["param_str"],
                "file_size_mb": res["file_size_mb"],
            }
            writer.writerow(row)
    print(f"  Saved: {csv_path}")

    print(f"\n{'=' * 60}")
    print(f"  Parameter extraction complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
