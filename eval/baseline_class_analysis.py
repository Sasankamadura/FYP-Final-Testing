"""Baseline Per-Class Analysis & Heatmap Generator.

Produces:
  1. Detailed baseline per-class breakdown (AP@50, AP@50:95)
  2. Baseline-only class heatmap (multi-metric)
  3. Baseline vs all models per-class comparison heatmap (delta from baseline)

Outputs saved to: eval/results/reports/plots/
"""

import os
import sys
import json
import argparse
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _init_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return matplotlib, plt


def main():
    parser = argparse.ArgumentParser(description="Baseline Per-Class Analysis")
    parser.add_argument("--config", type=str, default="eval/config.yaml")
    parser.add_argument("--workspace", type=str, default=None)
    args = parser.parse_args()

    if args.workspace:
        workspace_root = args.workspace
    else:
        workspace_root = str(Path(args.config).resolve().parent.parent)

    config = load_config(args.config)
    results_root = os.path.join(workspace_root, config["output_dir"])

    # Find GPU dir
    gpu_dirs = sorted([
        d for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d)) and d.startswith("GPU_")
    ])
    if not gpu_dirs:
        print("No GPU results found.")
        return
    gpu_dir = gpu_dirs[0]

    # Load all validation results
    val_file = os.path.join(results_root, gpu_dir, "validation", "all_validation_results.json")
    all_results = load_json(val_file)

    # Extract baseline
    baseline_key = "baseline_rtdetr_r18"
    if baseline_key not in all_results:
        print(f"Baseline key '{baseline_key}' not found in results.")
        return

    baseline = all_results[baseline_key]
    baseline_metrics = baseline["metrics"]
    baseline_pc = baseline_metrics["per_class_ap"]

    # Sort classes by AP@50 descending for better readability
    classes_sorted = sorted(baseline_pc.keys(), key=lambda c: baseline_pc[c]["AP_50"], reverse=True)

    # ================================================================
    #  1. PRINT DETAILED BASELINE PER-CLASS BREAKDOWN
    # ================================================================
    print("\n" + "=" * 90)
    print("  BASELINE RT-DETR R18 — PER-CLASS ANALYSIS")
    print("=" * 90)
    print(f"  Overall  mAP@50: {baseline_metrics['mAP_50']:.4f}")
    print(f"  Overall  mAP@50:95: {baseline_metrics['mAP_50_95']:.4f}")
    print(f"  AP-Small: {baseline_metrics['mAP_small']:.4f}  |  "
          f"AP-Medium: {baseline_metrics['mAP_medium']:.4f}  |  "
          f"AP-Large: {baseline_metrics['mAP_large']:.4f}")
    print("-" * 90)
    print(f"  {'Class':<20} {'AP@50':>10} {'AP@50:95':>12} {'Δ from mean':>14} {'Rating':>10}")
    print("-" * 90)

    ap50_values = [baseline_pc[c]["AP_50"] for c in classes_sorted]
    mean_ap50 = np.mean(ap50_values)

    for cls in classes_sorted:
        ap50 = baseline_pc[cls]["AP_50"]
        ap5095 = baseline_pc[cls]["AP_50_95"]
        delta = ap50 - mean_ap50

        if ap50 >= 0.5:
            rating = "★★★ Good"
        elif ap50 >= 0.3:
            rating = "★★ Fair"
        elif ap50 >= 0.1:
            rating = "★ Weak"
        else:
            rating = "✗ Poor"

        sign = "+" if delta >= 0 else ""
        print(f"  {cls:<20} {ap50:>10.4f} {ap5095:>12.4f} {sign}{delta:>13.4f} {rating:>10}")

    print("-" * 90)
    print(f"  {'MEAN':<20} {mean_ap50:>10.4f}")
    print("=" * 90)

    # Identify strengths and weaknesses
    best_cls = classes_sorted[0]
    worst_cls = classes_sorted[-1]
    print(f"\n  ✅ Strongest class: {best_cls} (AP@50 = {baseline_pc[best_cls]['AP_50']:.4f})")
    print(f"  ❌ Weakest class:   {worst_cls} (AP@50 = {baseline_pc[worst_cls]['AP_50']:.4f})")

    weak_classes = [c for c in classes_sorted if baseline_pc[c]["AP_50"] < 0.2]
    if weak_classes:
        print(f"  ⚠️  Classes below 0.20 AP@50: {', '.join(weak_classes)}")

    # ================================================================
    #  2. BASELINE-ONLY MULTI-METRIC HEATMAP
    # ================================================================
    _, plt = _init_matplotlib()
    plots_dir = os.path.join(results_root, "reports", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- Heatmap 1: Baseline multi-metric per class ---
    metrics_labels = ["AP@50", "AP@50:95"]
    data_baseline = np.array([
        [baseline_pc[c]["AP_50"] for c in classes_sorted],
        [baseline_pc[c]["AP_50_95"] for c in classes_sorted],
    ])

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(data_baseline, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(classes_sorted)))
    ax.set_xticklabels(classes_sorted, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(metrics_labels)))
    ax.set_yticklabels(metrics_labels, fontsize=11)

    for i in range(len(metrics_labels)):
        for j in range(len(classes_sorted)):
            val = data_baseline[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=10,
                    fontweight="bold", color=color)

    ax.set_title("Baseline RT-DETR R18 — Per-Class Performance", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="AP Score", shrink=0.8, pad=0.02)
    plt.tight_layout()
    path1 = os.path.join(plots_dir, "baseline_class_heatmap.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {path1}")

    # ================================================================
    #  3. BASELINE vs ALL MODELS — DELTA HEATMAP
    # ================================================================
    # Use alphabetical class order for the comparison heatmap
    classes_alpha = sorted(baseline_pc.keys())

    model_names = []
    delta_data = []

    for key, res in all_results.items():
        if key == baseline_key:
            continue
        pc = res.get("metrics", {}).get("per_class_ap", {})
        if not pc:
            continue
        model_names.append(res["name"])
        row = []
        for cls in classes_alpha:
            model_ap = pc.get(cls, {}).get("AP_50", 0)
            base_ap = baseline_pc.get(cls, {}).get("AP_50", 0)
            row.append(model_ap - base_ap)
        delta_data.append(row)

    if delta_data:
        delta_arr = np.array(delta_data)
        vmax = max(abs(delta_arr.min()), abs(delta_arr.max()), 0.15)

        fig, ax = plt.subplots(figsize=(15, max(8, len(model_names) * 0.55)))
        im = ax.imshow(delta_arr, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(len(classes_alpha)))
        ax.set_xticklabels(classes_alpha, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontsize=8)

        for i in range(len(model_names)):
            for j in range(len(classes_alpha)):
                val = delta_arr[i, j]
                sign = "+" if val >= 0 else ""
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{sign}{val:.3f}", ha="center", va="center",
                        fontsize=7, color=color)

        ax.set_title("Per-Class AP@50 Δ vs Baseline (Blue = Better, Red = Worse)",
                      fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Δ AP@50 vs Baseline", shrink=0.8)
        plt.tight_layout()
        path2 = os.path.join(plots_dir, "baseline_vs_models_class_delta.png")
        plt.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path2}")

    # ================================================================
    #  4. BASELINE CLASS DISTRIBUTION BAR CHART
    # ================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    ap50_vals = [baseline_pc[c]["AP_50"] for c in classes_sorted]
    ap5095_vals = [baseline_pc[c]["AP_50_95"] for c in classes_sorted]

    x = np.arange(len(classes_sorted))
    width = 0.35

    bars1 = ax.bar(x - width / 2, ap50_vals, width, label="AP@50",
                   color="#3498db", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, ap5095_vals, width, label="AP@50:95",
                   color="#e74c3c", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Average Precision", fontsize=12)
    ax.set_title("Baseline RT-DETR R18 — Per-Class AP Breakdown", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes_sorted, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=mean_ap50, color="gray", linestyle="--", alpha=0.7, label=f"Mean AP@50 ({mean_ap50:.3f})")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path3 = os.path.join(plots_dir, "baseline_class_bar_chart.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path3}")

    print(f"\n{'=' * 60}")
    print(f"  Baseline class analysis complete!")
    print(f"  Plots saved to: {plots_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
