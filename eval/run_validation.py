"""Validation Evaluation Script.

Runs ONNX model inference on the VisDrone validation set and computes
COCO detection metrics (mAP@50, mAP@50:95, AP-small/medium/large, per-class AP).

Usage:
    python run_validation.py
    python run_validation.py --config eval/config.yaml
    python run_validation.py --model baseline_rtdetr_r18     # single model only
    python run_validation.py --workspace "D:/Final Testing"  # explicit root
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import cv2
import yaml
from pathlib import Path
from collections import OrderedDict

# Add eval directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.onnx_inference import OnnxDetector
from utils.postprocess import predictions_to_coco_format
from utils.gpu_info import get_gpu_info, get_gpu_tag, print_gpu_info, save_gpu_info


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_category_mapping(coco_data):
    """Build mapping from model output label to COCO category ID.

    PaddleDetection RT-DETR ONNX exports add +1 to internal 0-based class
    indices, producing labels 1–N. VisDrone annotations use cat_ids 1–10.
    Empirically verified: model_label N → cat_id N+1.

    The original 0-based mapping {0→1, 1→2, ..., 9→10} handles this
    correctly because model labels 1–9 look up keys 1–9 and get cat_ids 2–10.
    Label 0 (pedestrian) is never produced at meaningful confidence.
    """
    categories = sorted(coco_data["categories"], key=lambda x: x["id"])
    mapping = {}
    for idx, cat in enumerate(categories):
        mapping[idx] = cat["id"]  # 0→1, 1→2, ..., 9→10
    return mapping


def evaluate_model(detector, coco_data, images_dir, config, category_mapping):
    """Run inference on all validation images and collect COCO-format predictions.

    Args:
        detector: OnnxDetector instance.
        coco_data: COCO annotation dict.
        images_dir: path to validation images folder.
        config: full config dict.
        category_mapping: dict mapping model index -> COCO category ID.

    Returns:
        list of COCO prediction dicts.
    """
    eval_cfg = config["evaluation"]
    conf_threshold = eval_cfg["conf_threshold"]
    nms_threshold = eval_cfg["nms_threshold"]
    max_detections = eval_cfg["max_detections"]

    all_predictions = []
    image_list = coco_data["images"]
    total = len(image_list)

    for idx, img_info in enumerate(image_list):
        img_path = os.path.join(images_dir, img_info["file_name"])

        if not os.path.exists(img_path):
            print(f"  [WARN] Image not found: {img_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"  [WARN] Failed to read: {img_path}")
            continue

        # Run prediction
        boxes, scores, class_ids = detector.predict(
            image,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            max_detections=max_detections,
        )

        # Convert to COCO format
        preds = predictions_to_coco_format(
            img_info["id"], boxes, scores, class_ids, category_mapping
        )
        all_predictions.extend(preds)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"  Progress: {idx + 1}/{total} images processed", end="\r")

    print()
    return all_predictions


def run_coco_evaluation(coco_data, predictions, output_dir, model_key):
    """Run COCO evaluation and return structured metrics.

    Args:
        coco_data: COCO annotation dict.
        predictions: list of COCO prediction dicts.
        output_dir: directory to save prediction JSON files.
        model_key: string identifier for this model.

    Returns:
        dict with mAP, AR, per-class AP metrics.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # Save predictions to file (required by COCO API)
    pred_file = os.path.join(output_dir, f"{model_key}_predictions.json")
    with open(pred_file, "w") as f:
        json.dump(predictions, f)

    # Save annotations to temp file for COCO API
    anno_file = os.path.join(output_dir, "_temp_annotations.json")
    with open(anno_file, "w") as f:
        json.dump(coco_data, f)

    coco_gt = COCO(anno_file)

    if len(predictions) == 0:
        print("  [WARN] No predictions generated! All metrics will be 0.")
        return {
            "mAP_50": 0.0, "mAP_50_95": 0.0, "mAP_75": 0.0,
            "mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0,
            "AR_1": 0.0, "AR_10": 0.0, "AR_100": 0.0,
            "AR_small": 0.0, "AR_medium": 0.0, "AR_large": 0.0,
            "per_class_ap": {},
        }

    coco_dt = coco_gt.loadRes(pred_file)

    # ---- Overall Evaluation ----
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats

    results = {
        "mAP_50_95": round(float(stats[0]), 4),   # AP @ IoU=0.50:0.95
        "mAP_50": round(float(stats[1]), 4),       # AP @ IoU=0.50
        "mAP_75": round(float(stats[2]), 4),       # AP @ IoU=0.75
        "mAP_small": round(float(stats[3]), 4),    # AP for small objects
        "mAP_medium": round(float(stats[4]), 4),   # AP for medium objects
        "mAP_large": round(float(stats[5]), 4),    # AP for large objects
        "AR_1": round(float(stats[6]), 4),          # AR given 1 det per image
        "AR_10": round(float(stats[7]), 4),         # AR given 10 dets per image
        "AR_100": round(float(stats[8]), 4),        # AR given 100 dets per image
        "AR_small": round(float(stats[9]), 4),      # AR for small objects
        "AR_medium": round(float(stats[10]), 4),    # AR for medium objects
        "AR_large": round(float(stats[11]), 4),     # AR for large objects
    }

    # ---- Per-Class AP@50 ----
    per_class_ap = {}
    cat_ids = coco_gt.getCatIds()
    cat_names = [c["name"] for c in coco_gt.loadCats(cat_ids)]

    for cat_id, cat_name in zip(cat_ids, cat_names):
        coco_eval_cls = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval_cls.params.catIds = [cat_id]
        coco_eval_cls.evaluate()
        coco_eval_cls.accumulate()
        # Suppress per-class printout
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval_cls.summarize()

        per_class_ap[cat_name] = {
            "AP_50": round(float(coco_eval_cls.stats[1]), 4),
            "AP_50_95": round(float(coco_eval_cls.stats[0]), 4),
        }

    results["per_class_ap"] = per_class_ap

    # Cleanup temp file
    if os.path.exists(anno_file):
        os.remove(anno_file)

    return results


def print_summary_table(all_results):
    """Print a formatted summary table of all model validation results."""
    print("\n" + "=" * 120)
    print("  VALIDATION RESULTS SUMMARY")
    print("=" * 120)

    header = (
        f"{'Model':<40} {'mAP@50':>8} {'mAP@50:95':>10} "
        f"{'AP-S':>8} {'AP-M':>8} {'AP-L':>8} {'AR@100':>8}"
    )
    print(header)
    print("-" * 120)

    for model_key, res in all_results.items():
        metrics = res["metrics"]
        name = res.get("name", model_key)
        row = (
            f"{name:<40} "
            f"{metrics['mAP_50']:>8.4f} "
            f"{metrics['mAP_50_95']:>10.4f} "
            f"{metrics['mAP_small']:>8.4f} "
            f"{metrics['mAP_medium']:>8.4f} "
            f"{metrics['mAP_large']:>8.4f} "
            f"{metrics['AR_100']:>8.4f}"
        )
        print(row)

    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="VisDrone Validation Evaluation")
    parser.add_argument("--config", type=str, default="eval/config.yaml",
                        help="Path to config file")
    parser.add_argument("--model", type=str, default=None,
                        help="Evaluate a single model by its config key")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Workspace root directory")
    args = parser.parse_args()

    # Determine workspace root
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
    output_dir = os.path.join(workspace_root, config["output_dir"], gpu_tag, "validation")
    os.makedirs(output_dir, exist_ok=True)
    save_gpu_info(output_dir, gpu_info)

    # ---- Load Annotations ----
    anno_path = os.path.join(workspace_root, config["datasets"]["val"]["annotations"])
    print(f"\nLoading annotations from: {anno_path}")

    with open(anno_path, "r") as f:
        coco_data = json.load(f)

    images_dir = os.path.join(workspace_root, config["datasets"]["val"]["images_dir"])
    print(f"Images directory: {images_dir}")
    print(f"Total validation images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Categories: {[c['name'] for c in coco_data['categories']]}")

    # ---- Category Mapping ----
    category_mapping = build_category_mapping(coco_data)
    print(f"Category mapping (model_idx -> coco_id): {category_mapping}")

    # ---- Select Models ----
    models = config["models"]
    if args.model:
        if args.model not in models:
            print(f"\nModel '{args.model}' not found in config.")
            print(f"Available models: {list(models.keys())}")
            return
        models = {args.model: models[args.model]}

    input_size = tuple(config["evaluation"]["input_size"])

    # ---- Evaluate Each Model (with resume support) ----
    all_results = OrderedDict()

    # Load any previously saved results for resume
    for model_key in models:
        prev_result_file = os.path.join(output_dir, f"{model_key}_results.json")
        if os.path.exists(prev_result_file):
            with open(prev_result_file, "r") as f:
                all_results[model_key] = json.load(f)

    for model_key, model_cfg in models.items():
        model_path = os.path.join(workspace_root, model_cfg["path"])
        model_name = model_cfg["name"]

        # Skip already completed models
        if model_key in all_results:
            m = all_results[model_key].get("metrics", {})
            print(f"\n  [SKIP] {model_name} — already evaluated (mAP@50={m.get('mAP_50', 'N/A')})")
            continue

        print(f"\n{'=' * 80}")
        print(f"  Evaluating: {model_name}")
        print(f"  Path: {model_path}")
        print(f"  Decoder layers: {model_cfg.get('decoder_layers', 'N/A')}")
        print(f"{'=' * 80}")

        if not os.path.exists(model_path):
            print(f"  [ERROR] Model file not found, skipping.")
            continue

        try:
            # Load model
            detector = OnnxDetector(model_path, input_size=input_size)
            model_info = detector.get_model_info()
            print(f"  Model size: {model_info['file_size_mb']:.2f} MB")
            print(f"  Input shape: {model_info['input_shape']}")
            print(f"  Output format: {model_info['output_format']}")
            print(f"  Providers: {model_info['providers']}")

            # Warmup
            print("  Warming up...")
            detector.warmup(iterations=5)

            # Run inference on validation set
            print("  Running inference on validation set...")
            t_start = time.time()
            predictions = evaluate_model(
                detector, coco_data, images_dir, config, category_mapping
            )
            t_elapsed = time.time() - t_start
            print(f"  Inference complete: {len(predictions)} detections in {t_elapsed:.1f}s")

            # Compute COCO metrics
            print("  Computing COCO metrics...")
            metrics = run_coco_evaluation(coco_data, predictions, output_dir, model_key)

            all_results[model_key] = {
                "name": model_name,
                "model_info": model_info,
                "config": model_cfg,
                "metrics": metrics,
                "num_predictions": len(predictions),
                "inference_time_s": round(t_elapsed, 2),
            }

            # Save individual model result
            result_file = os.path.join(output_dir, f"{model_key}_results.json")
            with open(result_file, "w") as f:
                json.dump(all_results[model_key], f, indent=2)

            print(
                f"  >> mAP@50: {metrics['mAP_50']:.4f} | "
                f"mAP@50:95: {metrics['mAP_50_95']:.4f} | "
                f"AP-small: {metrics['mAP_small']:.4f}"
            )

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            continue

    # ---- Summary ----
    if all_results:
        print_summary_table(all_results)

        # Save aggregated results
        summary_file = os.path.join(output_dir, "all_validation_results.json")
        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results saved to: {output_dir}")
    else:
        print("\nNo models were evaluated successfully.")


if __name__ == "__main__":
    main()
