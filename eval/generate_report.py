"""Report Generation Script.

Aggregates results from validation and benchmark runs across GPU environments.
Generates comparison tables (CSV), plots (PNG), and cross-GPU analysis.

Usage:
    python generate_report.py
    python generate_report.py --config eval/config.yaml
    python generate_report.py --gpu GPU_NVIDIA_GeForce_RTX_3090
    python generate_report.py --no-plots
"""

import os
import sys
import json
import argparse
import csv
import yaml
import numpy as np
from pathlib import Path
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_json(filepath):
    """Load JSON file, return None if not found."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        return json.load(f)


def find_gpu_dirs(results_root):
    """Find all GPU result directories under the results root."""
    if not os.path.exists(results_root):
        return []
    return sorted([
        d for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d)) and d.startswith("GPU_")
    ])


# ============================================================
#  TABLE GENERATORS
# ============================================================

def generate_accuracy_table(val_results, output_dir):
    """Generate accuracy comparison table with delta vs baseline."""
    if not val_results:
        print("  No validation results found.")
        return

    print(f"\n{'=' * 140}")
    print("  ACCURACY COMPARISON TABLE")
    print(f"{'=' * 140}")

    header = (
        f"{'Model':<40} {'Dec':>4} {'mAP@50':>8} {'mAP@50:95':>10} "
        f"{'AP-S':>8} {'AP-M':>8} {'AP-L':>8} {'delta mAP@50':>12}"
    )
    print(header)
    print("-" * 140)

    # Find baseline for delta calculation
    baseline_map50 = None
    for key, res in val_results.items():
        if "baseline" in key.lower():
            baseline_map50 = res["metrics"]["mAP_50"]
            break

    rows = []
    for key, res in val_results.items():
        m = res["metrics"]
        dec = res.get("config", {}).get("decoder_layers", "N/A")
        delta = (m["mAP_50"] - baseline_map50) if baseline_map50 is not None else 0

        delta_str = f"{delta:>+12.4f}" if baseline_map50 is not None else "         N/A"

        print(
            f"{res['name']:<40} {dec:>4} "
            f"{m['mAP_50']:>8.4f} {m['mAP_50_95']:>10.4f} "
            f"{m['mAP_small']:>8.4f} {m['mAP_medium']:>8.4f} {m['mAP_large']:>8.4f} "
            f"{delta_str}"
        )

        rows.append({
            "model_key": key,
            "model_name": res["name"],
            "decoder_layers": dec,
            "mAP_50": m["mAP_50"],
            "mAP_50_95": m["mAP_50_95"],
            "mAP_75": m.get("mAP_75", ""),
            "mAP_small": m["mAP_small"],
            "mAP_medium": m["mAP_medium"],
            "mAP_large": m["mAP_large"],
            "AR_100": m.get("AR_100", ""),
            "delta_mAP_50": round(delta, 4) if baseline_map50 else "",
        })

    print(f"{'=' * 140}")

    # Save CSV
    csv_path = os.path.join(output_dir, "accuracy_comparison.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: {csv_path}")


def generate_per_class_table(val_results, output_dir):
    """Generate per-class AP@50 table across all models."""
    if not val_results:
        return

    # Collect all class names from results
    all_classes = set()
    for res in val_results.values():
        per_class = res.get("metrics", {}).get("per_class_ap", {})
        all_classes.update(per_class.keys())
    all_classes = sorted(all_classes)

    if not all_classes:
        print("  No per-class AP data found.")
        return

    print(f"\n{'=' * 160}")
    print("  PER-CLASS AP@50 TABLE")
    print(f"{'=' * 160}")

    class_cols = "".join(f"{c:>14}" for c in all_classes)
    print(f"{'Model':<35}{class_cols}")
    print("-" * 160)

    rows = []
    for key, res in val_results.items():
        pc = res.get("metrics", {}).get("per_class_ap", {})
        val_strs = "".join(
            f"{pc.get(c, {}).get('AP_50', 0):>14.4f}" for c in all_classes
        )
        print(f"{res['name']:<35}{val_strs}")

        row = {"model_key": key, "model_name": res["name"]}
        for c in all_classes:
            row[f"AP50_{c}"] = pc.get(c, {}).get("AP_50", 0)
        rows.append(row)

    print(f"{'=' * 160}")

    csv_path = os.path.join(output_dir, "per_class_ap50.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: {csv_path}")


def generate_speed_accuracy_table(val_results, bench_results, output_dir):
    """Generate combined speed-accuracy comparison table."""
    if not val_results or not bench_results:
        print("  Need both validation and benchmark results for this analysis.")
        return

    print(f"\n{'=' * 150}")
    print("  SPEED vs ACCURACY COMPARISON")
    print(f"{'=' * 150}")

    header = (
        f"{'Model':<40} {'mAP@50':>8} {'mAP@50:95':>10} {'AP-S':>8} "
        f"{'Latency(ms)':>12} {'FPS':>8} {'Size(MB)':>9} {'Efficiency':>10}"
    )
    print(header)
    print("-" * 150)

    rows = []
    for key in val_results:
        if key not in bench_results:
            continue

        v = val_results[key]
        b = bench_results[key]
        m = v["metrics"]
        latency = b["mean_latency_ms"]
        fps = b["fps"]
        size = b.get("model_size_mb", 0)

        # Efficiency = mAP@50 * FPS (higher is better)
        efficiency = m["mAP_50"] * fps

        print(
            f"{v['name']:<40} "
            f"{m['mAP_50']:>8.4f} {m['mAP_50_95']:>10.4f} {m['mAP_small']:>8.4f} "
            f"{latency:>11.2f}  {fps:>8.1f} {size:>9.1f} {efficiency:>10.2f}"
        )

        rows.append({
            "model": v["name"],
            "mAP_50": m["mAP_50"],
            "mAP_50_95": m["mAP_50_95"],
            "mAP_small": m["mAP_small"],
            "latency_ms": latency,
            "fps": fps,
            "size_mb": size,
            "efficiency_score": round(efficiency, 2),
        })

    print(f"{'=' * 150}")
    print("  Efficiency = mAP@50 x FPS (higher is better)")

    csv_path = os.path.join(output_dir, "speed_accuracy_comparison.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: {csv_path}")


def generate_decoder_comparison(val_results, bench_results, output_dir):
    """Generate 3-decoder (CRR) vs 6-decoder comparison table."""
    pairs = [
        ("gnconv_p2", "gnconv_p2_crr", "gnConv + P2"),
        ("gnconv_p2_repvgg", "gnconv_p2_repvgg_crr", "gnConv + P2 + RepVGG"),
        ("gnconv_slim_p2_repvgg", "gnconv_slim_p2_repvgg_crr", "gnConv + SLIM-P2 + RepVGG"),
    ]

    print(f"\n{'=' * 130}")
    print("  3-DECODER (CRR) vs 6-DECODER COMPARISON")
    print(f"{'=' * 130}")

    header = (
        f"{'Architecture':<35} {'Config':>8} {'mAP@50':>8} {'mAP@50:95':>10} "
        f"{'AP-S':>8} {'Latency(ms)':>12} {'FPS':>8}"
    )
    print(header)
    print("-" * 130)

    for key_6dec, key_3dec, arch_name in pairs:
        for key, dec_label in [(key_6dec, "6-dec"), (key_3dec, "3-dec (CRR)")]:
            if key not in val_results:
                continue
            v = val_results[key]
            m = v["metrics"]
            lat = bench_results[key]["mean_latency_ms"] if bench_results and key in bench_results else -1
            fps = bench_results[key]["fps"] if bench_results and key in bench_results else -1

            lat_str = f"{lat:>11.2f}" if lat > 0 else "        N/A"
            fps_str = f"{fps:>8.1f}" if fps > 0 else "     N/A"

            print(
                f"{arch_name:<35} {dec_label:>8} "
                f"{m['mAP_50']:>8.4f} {m['mAP_50_95']:>10.4f} {m['mAP_small']:>8.4f} "
                f"{lat_str}  {fps_str}"
            )
        print("-" * 130)

    print(f"{'=' * 130}")


# ============================================================
#  PLOT GENERATORS
# ============================================================

def generate_plots(val_results, bench_results, output_dir):
    """Generate all comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed. Skipping plot generation.")
        print("         Install with: pip install matplotlib")
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ---- Plot 1: mAP@50 Horizontal Bar Chart ----
    if val_results:
        names = [r["name"] for r in val_results.values()]
        map50 = [r["metrics"]["mAP_50"] for r in val_results.values()]

        fig, ax = plt.subplots(figsize=(16, max(8, len(names) * 0.5)))
        colors = ["#e74c3c" if "Baseline" in n else "#3498db" for n in names]
        bars = ax.barh(range(len(names)), map50, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("mAP@50", fontsize=12)
        ax.set_title("mAP@50 Comparison Across All Models", fontsize=14)
        ax.invert_yaxis()

        for bar, val in zip(bars, map50):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "mAP50_comparison.png"), dpi=150)
        plt.close()
        print(f"  Saved: plots/mAP50_comparison.png")

    # ---- Plot 2: Accuracy vs Latency Scatter (Pareto Analysis) ----
    if val_results and bench_results:
        common_keys = set(val_results.keys()) & set(bench_results.keys())

        if common_keys:
            fig, ax = plt.subplots(figsize=(12, 8))

            for key in common_keys:
                v = val_results[key]
                b = bench_results[key]
                map50_val = v["metrics"]["mAP_50"]
                latency = b["mean_latency_ms"]
                dec = v.get("config", {}).get("decoder_layers", "?")

                color = "#e74c3c" if dec == 3 else "#3498db" if dec == 6 else "#95a5a6"
                marker = "s" if "baseline" in key.lower() else "o"
                size = 150 if "baseline" in key.lower() else 100

                ax.scatter(latency, map50_val, c=color, marker=marker, s=size, zorder=5, edgecolors="black", linewidths=0.5)
                ax.annotate(v["name"], (latency, map50_val), fontsize=7,
                            xytext=(5, 5), textcoords="offset points")

            ax.set_xlabel("Latency (ms)", fontsize=12)
            ax.set_ylabel("mAP@50", fontsize=12)
            ax.set_title("Accuracy vs Speed — Pareto Analysis", fontsize=14)
            ax.grid(True, alpha=0.3)

            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label="3 decoders (CRR)"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=10, label="6 decoders"),
                Line2D([0], [0], marker="s", color="w", markerfacecolor="#95a5a6", markersize=10, label="Baseline"),
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "accuracy_vs_latency.png"), dpi=150)
            plt.close()
            print(f"  Saved: plots/accuracy_vs_latency.png")

    # ---- Plot 3: AP by Object Size (Grouped Bar) ----
    if val_results:
        names = [r["name"] for r in val_results.values()]
        ap_small = [r["metrics"]["mAP_small"] for r in val_results.values()]
        ap_medium = [r["metrics"]["mAP_medium"] for r in val_results.values()]
        ap_large = [r["metrics"]["mAP_large"] for r in val_results.values()]

        x = np.arange(len(names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(16, max(8, len(names) * 0.4)))
        ax.bar(x - width, ap_small, width, label="AP-Small", color="#e74c3c", alpha=0.85)
        ax.bar(x, ap_medium, width, label="AP-Medium", color="#3498db", alpha=0.85)
        ax.bar(x + width, ap_large, width, label="AP-Large", color="#2ecc71", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Average Precision", fontsize=12)
        ax.set_title("AP by Object Size Across Models", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "ap_by_object_size.png"), dpi=150)
        plt.close()
        print(f"  Saved: plots/ap_by_object_size.png")

    # ---- Plot 4: Per-Class AP Heatmap ----
    if val_results:
        all_classes = set()
        for res in val_results.values():
            pc = res.get("metrics", {}).get("per_class_ap", {})
            all_classes.update(pc.keys())
        all_classes = sorted(all_classes)

        if all_classes:
            model_names = [r["name"] for r in val_results.values()]
            data = []
            for res in val_results.values():
                pc = res.get("metrics", {}).get("per_class_ap", {})
                row = [pc.get(c, {}).get("AP_50", 0) for c in all_classes]
                data.append(row)
            data = np.array(data)

            fig, ax = plt.subplots(figsize=(14, max(8, len(model_names) * 0.5)))
            im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

            ax.set_xticks(range(len(all_classes)))
            ax.set_xticklabels(all_classes, rotation=45, ha="right", fontsize=9)
            ax.set_yticks(range(len(model_names)))
            ax.set_yticklabels(model_names, fontsize=8)

            # Value annotations
            for i in range(len(model_names)):
                for j in range(len(all_classes)):
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if data[i, j] > 0.5 else "black")

            ax.set_title("Per-Class AP@50 Heatmap", fontsize=14)
            plt.colorbar(im, ax=ax, label="AP@50", shrink=0.8)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "per_class_ap_heatmap.png"), dpi=150)
            plt.close()
            print(f"  Saved: plots/per_class_ap_heatmap.png")

    # ---- Plot 5: FPS Bar Chart ----
    if bench_results:
        names = [r["name"] for r in bench_results.values()]
        fps_vals = [r["fps"] for r in bench_results.values()]

        fig, ax = plt.subplots(figsize=(16, max(8, len(names) * 0.5)))
        colors = ["#2ecc71" if fps > 30 else "#f39c12" if fps > 15 else "#e74c3c" for fps in fps_vals]
        bars = ax.barh(range(len(names)), fps_vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Frames Per Second (FPS)", fontsize=12)
        ax.set_title("Inference Speed Comparison", fontsize=14)
        ax.invert_yaxis()
        ax.axvline(x=30, color="green", linestyle="--", alpha=0.5, label="30 FPS (real-time)")
        ax.legend(fontsize=10)

        for bar, val in zip(bars, fps_vals):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fps_comparison.png"), dpi=150)
        plt.close()
        print(f"  Saved: plots/fps_comparison.png")


# ============================================================
#  CROSS-GPU COMPARISON
# ============================================================

def generate_cross_gpu_comparison(results_root, output_dir):
    """Compare model performance across different GPU environments."""
    gpu_dirs = find_gpu_dirs(results_root)

    if len(gpu_dirs) < 2:
        print("  Need results from at least 2 GPUs for cross-GPU comparison.")
        return

    print(f"\n{'=' * 130}")
    print(f"  CROSS-GPU LATENCY COMPARISON ({len(gpu_dirs)} GPUs)")
    print(f"{'=' * 130}")

    # Load benchmark results from each GPU
    all_gpu_bench = {}
    gpu_names = {}

    for gpu_dir in gpu_dirs:
        bench_file = os.path.join(results_root, gpu_dir, "benchmark", "benchmark_results.json")
        gpu_info_file = os.path.join(results_root, gpu_dir, "benchmark", "gpu_info.json")

        bench_data = load_json(bench_file)
        gpu_info = load_json(gpu_info_file)

        if bench_data:
            all_gpu_bench[gpu_dir] = bench_data
            gpu_names[gpu_dir] = gpu_info.get("gpu_name", gpu_dir) if gpu_info else gpu_dir

    if len(all_gpu_bench) < 2:
        print("  Insufficient benchmark data across GPUs.")
        return

    # Get union of all model keys
    all_model_keys = set()
    for bench in all_gpu_bench.values():
        all_model_keys.update(bench.keys())

    # Print header
    gpu_col_names = [gpu_names.get(g, g) for g in sorted(all_gpu_bench.keys())]
    gpu_cols = "".join(f"{n:>20}" for n in gpu_col_names)
    print(f"{'Model':<35}{gpu_cols}")
    print("-" * 130)

    rows = []
    for model_key in sorted(all_model_keys):
        values = {}
        model_name = model_key
        for gpu_dir in sorted(all_gpu_bench.keys()):
            bench = all_gpu_bench[gpu_dir]
            if model_key in bench:
                values[gpu_dir] = bench[model_key]["fps"]
                model_name = bench[model_key].get("name", model_key)

        if values:
            val_cols = "".join(
                f"{values.get(g, -1):>19.1f} " for g in sorted(all_gpu_bench.keys())
            )
            print(f"{model_name:<35}{val_cols}")

            row = {"model": model_name}
            for g in sorted(all_gpu_bench.keys()):
                row[f"FPS_{gpu_names.get(g, g)}"] = values.get(g, None)
            rows.append(row)

    print(f"{'=' * 130}")
    print("  (Values are FPS)")

    if rows:
        csv_path = os.path.join(output_dir, "cross_gpu_fps_comparison.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: {csv_path}")


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Evaluation Report")
    parser.add_argument("--config", type=str, default="eval/config.yaml")
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--gpu", type=str, default=None,
                        help="Specific GPU tag to analyze (e.g., GPU_NVIDIA_GeForce_RTX_3090)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    if args.workspace:
        workspace_root = args.workspace
    else:
        workspace_root = str(Path(args.config).resolve().parent.parent)

    config = load_config(args.config)
    results_root = os.path.join(workspace_root, config["output_dir"])

    # Find GPU result directories
    gpu_dirs = find_gpu_dirs(results_root)

    if not gpu_dirs:
        print("No results found. Run validation and benchmark scripts first.")
        print(f"  Expected results in: {results_root}")
        return

    print(f"Found results for {len(gpu_dirs)} GPU(s): {gpu_dirs}")

    # Select GPU directory to analyze
    if args.gpu:
        if args.gpu in gpu_dirs:
            target_gpu = args.gpu
        else:
            print(f"GPU '{args.gpu}' not found. Available: {gpu_dirs}")
            return
    else:
        target_gpu = gpu_dirs[0]

    gpu_results_dir = os.path.join(results_root, target_gpu)
    report_dir = os.path.join(results_root, "reports")
    os.makedirs(report_dir, exist_ok=True)

    print(f"\nAnalyzing results for: {target_gpu}")
    print(f"Report output: {report_dir}")

    # ---- Load Results ----
    val_file = os.path.join(gpu_results_dir, "validation", "all_validation_results.json")
    val_results = load_json(val_file)
    if val_results:
        print(f"  Loaded validation results: {len(val_results)} models")
    else:
        print(f"  [WARN] No validation results found at: {val_file}")

    bench_file = os.path.join(gpu_results_dir, "benchmark", "benchmark_results.json")
    bench_results = load_json(bench_file)
    if bench_results:
        print(f"  Loaded benchmark results: {len(bench_results)} models")
    else:
        print(f"  [WARN] No benchmark results found at: {bench_file}")

    # ---- Generate Tables ----
    if val_results:
        generate_accuracy_table(val_results, report_dir)
        generate_per_class_table(val_results, report_dir)

    if val_results and bench_results:
        generate_speed_accuracy_table(val_results, bench_results, report_dir)
        generate_decoder_comparison(val_results, bench_results, report_dir)

    # ---- Generate Plots ----
    if not args.no_plots:
        print("\nGenerating plots...")
        generate_plots(val_results, bench_results, report_dir)

    # ---- Cross-GPU Comparison ----
    if len(gpu_dirs) >= 2:
        generate_cross_gpu_comparison(results_root, report_dir)

    print(f"\n{'=' * 60}")
    print(f"  Report generation complete!")
    print(f"  Output: {report_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
