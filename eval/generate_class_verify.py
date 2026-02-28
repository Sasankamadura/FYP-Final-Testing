"""Generate class-verification annotated images.
For each chosen image:  Ground Truth  |  Baseline Prediction  (side-by-side)
with class names + colours clearly labelled so you can visually confirm the
model's class mapping is correct.

Output → eval/results/reports/plots/class_verify_*.png
"""
import json
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils.onnx_inference import OnnxDetector

# ── Paths ──────────────────────────────────────────────────────────────
WORKSPACE  = r"D:\Final Testing"
ANN_PATH   = os.path.join(WORKSPACE, "VisDrone Val image set", "annotations_VisDrone_val.json")
IMG_DIR    = os.path.join(WORKSPACE, "VisDrone Val image set")
MODEL_PATH = os.path.join(WORKSPACE, "Checkpoints - Baseline_Visdrone2019",
                          "RT-DETR Resnet 18", "base_rtdetr.onnx")
OUT_DIR    = os.path.join(WORKSPACE, "eval", "results", "reports", "plots")

CONF_THRESH = 0.40  # Draw only confident predictions for clarity

# ── Class mapping ──────────────────────────────────────────────────────
# Model outputs 1-based labels: 1=pedestrian, 2=people, ... 10=motor
# (PaddleDetection internally corrects the +1 shift during training)
RAW_TO_NAME = {
    1: "pedestrian", 2: "people",   3: "bicycle",     4: "car",    5: "van",
    6: "truck",      7: "tricycle", 8: "awning-tri",  9: "bus",   10: "motor",
}
# GT annotations are shifted +1 (COCO conversion included VisDrone
# "ignored regions" class 0 as cat_id 1, pushing real classes to 2-10)
CAT_TO_NAME = {
    # 1 = ignored regions (skip)
    2: "pedestrian", 3: "people",   4: "bicycle",     5: "car",    6: "van",
    7: "truck",      8: "tricycle", 9: "awning-tri", 10: "bus",
}

# ── Distinct colours per class (BGR) ──────────────────────────────────
COLORS = {
    "pedestrian": (0,   0,   255),   # red
    "people":     (0,   128, 255),   # orange
    "bicycle":    (0,   255, 255),   # yellow
    "car":        (255, 50,  50),    # blue
    "van":        (255, 180, 0),     # cyan-ish blue
    "truck":      (255, 255, 0),     # cyan
    "tricycle":   (0,   255, 0),     # green
    "awning-tri": (128, 0,   255),   # purple
    "bus":        (255, 0,   255),   # magenta
    "motor":      (0,   255, 128),   # spring green
}

# ── 6 images covering ALL 10 classes with various densities ───────────
VERIFY_IMAGES = [
    # (filename, description) — all have 10 classes present
    ("0000291_03201_d_0000884.jpg", "Dense street scene — all 10 classes (226 anns)"),
    ("0000244_02000_d_0000005.jpg", "Medium density — all 10 classes (124 anns)"),
    ("0000327_04201_d_0000732.jpg", "Mixed traffic — all 10 classes (99 anns)"),
    ("0000213_02500_d_0000240.jpg", "Urban road — all 10 classes (92 anns)"),
    ("0000249_02900_d_0000009.jpg", "Intersection — all 10 classes (90 anns)"),
    ("0000154_01601_d_0000001.jpg", "Wide view — all 10 classes (81 anns)"),
]


# ── Drawing helpers ────────────────────────────────────────────────────
def draw_boxes(img, boxes, label_func, show_score=False):
    """Draw bounding boxes with class-coloured labels."""
    vis = img.copy()
    overlay = img.copy()

    for b in boxes:
        name = label_func(b)
        if name is None:
            continue
        color = COLORS.get(name, (200, 200, 200))

        # Support both xywh (COCO GT) and xyxy (predictions)
        if "bbox" in b:
            x, y, w, h = [int(v) for v in b["bbox"]]
            x2, y2 = x + w, y + h
        else:
            x, y, x2, y2 = [int(v) for v in b["box_xyxy"]]

        # Semi-transparent fill + solid border
        cv2.rectangle(overlay, (x, y), (x2, y2), color, -1)
        cv2.rectangle(vis, (x, y), (x2, y2), color, 2)

        # Label text
        if show_score and "score" in b:
            label = f"{name} {b['score']:.2f}"
        else:
            label = name

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.42
        thick = 1
        (tw, th), _ = cv2.getTextSize(label, font, fs, thick)
        cv2.rectangle(vis, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(vis, label, (x + 2, y - 4), font, fs,
                    (255, 255, 255), thick, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
    return vis


def add_title(img, text, bg=(30, 30, 30)):
    """Add a dark title bar on top."""
    h, w = img.shape[:2]
    bar_h = 48
    bar = np.full((bar_h, w, 3), bg, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.70, 2)
    cv2.putText(bar, text, ((w - tw) // 2, (bar_h + th) // 2),
                font, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def add_legend(width):
    """Create a class-colour legend bar."""
    legend = np.full((42, width, 3), (30, 30, 30), dtype=np.uint8)
    x = 12
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name in ["pedestrian", "people", "bicycle", "car", "van",
                 "truck", "tricycle", "awning-tri", "bus", "motor"]:
        color = COLORS[name]
        cv2.rectangle(legend, (x, 10), (x + 18, 30), color, -1)
        cv2.rectangle(legend, (x, 10), (x + 18, 30), (200, 200, 200), 1)
        x += 22
        cv2.putText(legend, name, (x, 27), font, 0.42,
                    (220, 220, 220), 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(name, font, 0.42, 1)
        x += tw + 16
    return legend


def class_summary(annotations, cat_map):
    """Return per-class count string for title."""
    from collections import Counter
    counts = Counter(cat_map.get(a["category_id"], "?") for a in annotations)
    parts = [f"{n}:{c}" for n, c in sorted(counts.items())]
    return "  ".join(parts)


# ── Main ───────────────────────────────────────────────────────────────
def main():
    # Load COCO annotations
    print("Loading annotations...")
    with open(ANN_PATH) as f:
        coco = json.load(f)
    gt_by_img = {}
    for ann in coco["annotations"]:
        gt_by_img.setdefault(ann["image_id"], []).append(ann)
    id2img = {img["id"]: img for img in coco["images"]}
    file2id = {img["file_name"]: img["id"] for img in coco["images"]}

    # Load baseline model
    print("Loading baseline RT-DETR model...")
    detector = OnnxDetector(MODEL_PATH)
    detector.warmup(5)
    print(f"  Model loaded. Provider: {detector.session.get_providers()[0]}")

    os.makedirs(OUT_DIR, exist_ok=True)
    generated = []

    for idx, (img_name, desc) in enumerate(VERIFY_IMAGES, 1):
        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"  SKIP (not found): {img_name}")
            continue

        image = cv2.imread(img_path)
        img_id = file2id.get(img_name)
        print(f"\n[{idx}/{len(VERIFY_IMAGES)}] {img_name}  ({desc})")

        # ── Ground truth ───────────────────────────────────────────
        gt_anns = gt_by_img.get(img_id, [])
        n_gt = len(gt_anns)
        n_gt_cls = len(set(a["category_id"] for a in gt_anns))
        gt_vis = draw_boxes(image, gt_anns,
                            lambda b: CAT_TO_NAME.get(b["category_id"]))

        # ── Model predictions ──────────────────────────────────────
        input_tensor, orig_size = detector.preprocess(image)
        outputs = detector.inference_raw(input_tensor, orig_size)
        labels = outputs["labels"][0]
        boxes_xyxy = outputs["boxes"][0]
        scores = outputs["scores"][0]

        pred_list = []
        for i in range(len(scores)):
            if scores[i] >= CONF_THRESH:
                raw_label = int(labels[i])
                name = RAW_TO_NAME.get(raw_label, f"raw{raw_label}")
                pred_list.append({
                    "box_xyxy": boxes_xyxy[i],
                    "score": float(scores[i]),
                    "name": name,
                })

        n_pred = len(pred_list)
        pred_vis = draw_boxes(image, pred_list,
                              lambda b: b["name"], show_score=True)

        # Print per-class breakdown
        gt_summary = class_summary(gt_anns, CAT_TO_NAME)
        pred_names = [p["name"] for p in pred_list]
        from collections import Counter
        pred_counts = Counter(pred_names)
        pred_summary = "  ".join(f"{n}:{c}" for n, c in sorted(pred_counts.items()))
        print(f"  GT:   {n_gt} anns, {n_gt_cls} classes  |  {gt_summary}")
        print(f"  Pred: {n_pred} dets (conf>={CONF_THRESH})  |  {pred_summary}")

        # ── Compose side-by-side ───────────────────────────────────
        panel_gt = add_title(gt_vis,
            f"GROUND TRUTH  ({n_gt} anns, {n_gt_cls} classes)")
        panel_pred = add_title(pred_vis,
            f"BASELINE PREDICTION  ({n_pred} dets, conf >= {CONF_THRESH})")

        # Match heights (should already match, but just in case)
        if panel_gt.shape[0] != panel_pred.shape[0]:
            h = max(panel_gt.shape[0], panel_pred.shape[0])
            panel_gt = cv2.copyMakeBorder(panel_gt, 0, h - panel_gt.shape[0],
                                          0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            panel_pred = cv2.copyMakeBorder(panel_pred, 0, h - panel_pred.shape[0],
                                            0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))

        sep = np.full((panel_gt.shape[0], 4, 3), 255, dtype=np.uint8)
        composite = np.hstack([panel_gt, sep, panel_pred])

        # Add legend + description
        legend = add_legend(composite.shape[1])
        desc_bar = np.full((30, composite.shape[1], 3), (40, 40, 40), dtype=np.uint8)
        cv2.putText(desc_bar, f"Image {idx}: {desc}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
        composite = np.vstack([composite, legend, desc_bar])

        # Save
        out_name = f"class_verify_{idx}_{img_name.replace('.jpg', '.png')}"
        out_path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(out_path, composite, [cv2.IMWRITE_PNG_COMPRESSION, 5])
        print(f"  -> Saved: {out_path}")
        generated.append(out_path)

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Generated {len(generated)} class verification images")
    print(f"  Output folder: {OUT_DIR}")
    for p in generated:
        print(f"    {os.path.basename(p)}")
    print(f"{'='*60}")
    print("\nCompare LEFT (Ground Truth) vs RIGHT (Prediction):")
    print("  - Same colour + label = correct class mapping")
    print("  - Different label for same object = class mismatch")
    print("  - Missing detections = low recall (not a class issue)")


if __name__ == "__main__":
    main()
