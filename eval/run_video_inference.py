"""Video Inference Script.

Runs ONNX model inference on a video file and saves an annotated output
video with bounding boxes drawn on each frame.

Usage:
    python eval/run_video_inference.py --video path/to/video.mp4
    python eval/run_video_inference.py --video path/to/video.mp4 --model baseline_rtdetr_r18
    python eval/run_video_inference.py --video path/to/video.mp4 --model baseline_rtdetr_r18 --output out.mp4
    python eval/run_video_inference.py --video path/to/video.mp4 --model_path model.onnx --conf 0.3
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import yaml
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.onnx_inference import OnnxDetector
from utils.visualization import draw_detections


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="ONNX Model Video Inference")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the input video file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to the output annotated video file. "
                             "Defaults to <input_name>_detected.<ext>")
    parser.add_argument("--model", type=str, default=None,
                        help="Model key from config.yaml (e.g. baseline_rtdetr_r18)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Direct path to an ONNX model file (overrides --model)")
    parser.add_argument("--config", type=str, default="eval/config.yaml",
                        help="Path to the evaluation config YAML")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Workspace root directory (auto-detected if omitted)")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence threshold for drawing detections. "
                             "Defaults to vis_conf_threshold from config.")
    parser.add_argument("--input_size", type=int, nargs=2, default=None,
                        metavar=("H", "W"),
                        help="Model input size as height width, e.g. --input_size 640 640")
    args = parser.parse_args()

    # ---- Resolve workspace root ----
    if args.workspace:
        workspace_root = args.workspace
    else:
        workspace_root = str(Path(args.config).resolve().parent.parent)

    # ---- Load config ----
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(workspace_root, config_path)

    config = load_config(config_path)
    eval_cfg = config["evaluation"]

    conf_threshold = args.conf if args.conf is not None else eval_cfg.get("vis_conf_threshold", 0.25)
    nms_threshold = eval_cfg["nms_threshold"]
    max_detections = eval_cfg["max_detections"]
    input_size = tuple(args.input_size) if args.input_size else tuple(eval_cfg["input_size"])

    # ---- Resolve model path ----
    if args.model_path:
        model_path = args.model_path
        model_name = os.path.basename(model_path)
    elif args.model:
        models = config["models"]
        if args.model not in models:
            print(f"[ERROR] Model '{args.model}' not found in config.")
            print(f"  Available models: {list(models.keys())}")
            return
        model_cfg = models[args.model]
        model_path = os.path.join(workspace_root, model_cfg["path"])
        model_name = model_cfg["name"]
    else:
        # Default to the first model in config
        first_key = next(iter(config["models"]))
        model_cfg = config["models"][first_key]
        model_path = os.path.join(workspace_root, model_cfg["path"])
        model_name = model_cfg["name"]
        print(f"[INFO] No model specified, using default: {first_key} ({model_name})")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return

    # ---- Resolve input video ----
    if not os.path.exists(args.video):
        print(f"[ERROR] Input video not found: {args.video}")
        return

    # ---- Resolve output path ----
    if args.output:
        output_path = args.output
    else:
        video_stem = Path(args.video).stem
        video_ext = Path(args.video).suffix or ".mp4"
        output_path = str(Path(args.video).parent / f"{video_stem}_detected{video_ext}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ---- Open input video ----
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video : {args.video}")
    print(f"Resolution  : {frame_w}x{frame_h}  |  FPS: {fps:.2f}  |  Frames: {total_frames}")
    print(f"Model       : {model_name}")
    print(f"Output video: {output_path}")

    # ---- Load model ----
    detector = OnnxDetector(model_path, input_size=input_size)
    detector.warmup(iterations=5)

    # ---- Open output video writer ----
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    # ---- Process frames ----
    frame_idx = 0
    total_detections = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, scores, class_ids = detector.predict(
            frame,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            max_detections=max_detections,
        )

        annotated = draw_detections(frame, boxes, scores, class_ids, conf_threshold)

        # Overlay FPS and detection count
        n_det = int((scores >= conf_threshold).sum()) if len(scores) > 0 else 0
        total_detections += n_det
        elapsed = time.time() - t_start
        current_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0

        cv2.putText(
            annotated,
            f"FPS: {current_fps:.1f}  Dets: {n_det}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            annotated,
            f"FPS: {current_fps:.1f}  Dets: {n_det}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        out.write(annotated)
        frame_idx += 1

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            pct = f"{frame_idx / total_frames * 100:.1f}%" if total_frames > 0 else f"{frame_idx} frames"
            print(f"  Progress: {frame_idx}/{total_frames} ({pct})  |  "
                  f"FPS: {current_fps:.1f}", end="\r")

    cap.release()
    out.release()

    elapsed = time.time() - t_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    avg_dets = total_detections / frame_idx if frame_idx > 0 else 0.0

    print()
    print(f"Done. Processed {frame_idx} frames in {elapsed:.1f}s  "
          f"({avg_fps:.1f} FPS avg)")
    print(f"Total detections (conf>{conf_threshold}): {total_detections}  "
          f"({avg_dets:.1f}/frame avg)")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
