"""Test Set Inference Script.

Runs ONNX model inference on VisDrone test images (no ground truth)
and saves predictions in COCO JSON format. Optionally saves visualization images.

Usage:
    python run_test_inference.py
    python run_test_inference.py --model baseline_rtdetr_r18
    python run_test_inference.py --visualize --max_vis 50
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.onnx_inference import OnnxDetector
from utils.postprocess import predictions_to_coco_format
from utils.visualization import save_visualization
from utils.gpu_info import get_gpu_info, get_gpu_tag, print_gpu_info, save_gpu_info


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="VisDrone Test Set Inference")
    parser.add_argument("--config", type=str, default="eval/config.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Run single model by config key")
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualization images with drawn detections")
    parser.add_argument("--max_vis", type=int, default=50,
                        help="Maximum number of images to visualize")
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
    output_dir = os.path.join(workspace_root, config["output_dir"], gpu_tag, "test_predictions")
    os.makedirs(output_dir, exist_ok=True)
    save_gpu_info(output_dir, gpu_info)

    # ---- Test Images ----
    test_dir = os.path.join(workspace_root, config["datasets"]["test"]["images_dir"])

    if not os.path.exists(test_dir):
        print(f"[ERROR] Test images directory not found: {test_dir}")
        return

    image_files = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])
    print(f"Found {len(image_files)} test images in: {test_dir}")

    # ---- Evaluation Settings ----
    eval_cfg = config["evaluation"]
    input_size = tuple(eval_cfg["input_size"])
    conf_threshold = eval_cfg["conf_threshold"]
    nms_threshold = eval_cfg["nms_threshold"]
    max_detections = eval_cfg["max_detections"]
    vis_conf = eval_cfg.get("vis_conf_threshold", 0.25)

    # ---- Select Models ----
    models = config["models"]
    if args.model:
        if args.model not in models:
            print(f"\nModel '{args.model}' not found. Available: {list(models.keys())}")
            return
        models = {args.model: models[args.model]}

    # ---- Run Inference Per Model ----
    for model_key, model_cfg in models.items():
        model_path = os.path.join(workspace_root, model_cfg["path"])
        model_name = model_cfg["name"]

        print(f"\n{'=' * 80}")
        print(f"  Test Inference: {model_name}")
        print(f"{'=' * 80}")

        if not os.path.exists(model_path):
            print(f"  [ERROR] Model file not found, skipping.")
            continue

        try:
            detector = OnnxDetector(model_path, input_size=input_size)
            detector.warmup(iterations=5)

            model_output_dir = os.path.join(output_dir, model_key)
            os.makedirs(model_output_dir, exist_ok=True)

            all_predictions = []
            detection_counts = []
            total = len(image_files)
            t_start = time.time()

            for idx, img_file in enumerate(image_files):
                img_path = os.path.join(test_dir, img_file)
                image = cv2.imread(img_path)

                if image is None:
                    print(f"  [WARN] Failed to read: {img_path}")
                    continue

                boxes, scores, class_ids = detector.predict(
                    image,
                    conf_threshold=conf_threshold,
                    nms_threshold=nms_threshold,
                    max_detections=max_detections,
                )

                # Use sequential image ID
                image_id = idx + 1

                preds = predictions_to_coco_format(image_id, boxes, scores, class_ids)
                all_predictions.extend(preds)

                # Detection statistics
                high_conf = int((scores > vis_conf).sum()) if len(scores) > 0 else 0
                detection_counts.append({
                    "file": img_file,
                    "image_id": image_id,
                    "total_detections": len(scores),
                    "high_conf_detections": high_conf,
                })

                # Save visualization for a subset of images
                if args.visualize and idx < args.max_vis:
                    vis_dir = os.path.join(model_output_dir, "visualizations")
                    vis_path = os.path.join(vis_dir, img_file)
                    save_visualization(image, boxes, scores, class_ids, vis_path, vis_conf)

                if (idx + 1) % 100 == 0 or (idx + 1) == total:
                    print(f"  Progress: {idx + 1}/{total}", end="\r")

            t_elapsed = time.time() - t_start
            print()

            # ---- Save Predictions ----
            pred_file = os.path.join(model_output_dir, "test_predictions.json")
            with open(pred_file, "w") as f:
                json.dump(all_predictions, f)

            # Save detection counts
            counts_file = os.path.join(model_output_dir, "detection_counts.json")
            with open(counts_file, "w") as f:
                json.dump(detection_counts, f, indent=2)

            # Summary
            avg_dets = np.mean([d["high_conf_detections"] for d in detection_counts])
            total_dets = len(all_predictions)
            print(
                f"  >> {total_dets} total detections | "
                f"Avg {avg_dets:.1f}/image (conf>{vis_conf}) | "
                f"{t_elapsed:.1f}s total"
            )

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print(f"\nTest predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()
