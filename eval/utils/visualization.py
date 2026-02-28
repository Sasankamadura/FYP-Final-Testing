"""Visualization utilities for detection results.

Provides functions to draw bounding boxes, create side-by-side model
comparisons, and save annotated images.
"""

import numpy as np
import cv2
import os


# VisDrone class color palette (BGR format for OpenCV)
COLORS = [
    (0, 255, 0),       # pedestrian - green
    (0, 200, 0),       # people - dark green
    (255, 0, 0),       # bicycle - blue
    (0, 0, 255),       # car - red
    (255, 255, 0),     # van - cyan
    (0, 165, 255),     # truck - orange
    (128, 0, 128),     # tricycle - purple
    (203, 192, 255),   # awning-tricycle - pink
    (0, 255, 255),     # bus - yellow
    (255, 0, 255),     # motor - magenta
]

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def draw_detections(image, boxes, scores, class_ids, conf_threshold=0.25):
    """Draw bounding boxes and labels on an image.

    Args:
        image: BGR numpy array.
        boxes: [N, 4] xyxy format.
        scores: [N] confidence scores.
        class_ids: [N] class indices.
        conf_threshold: minimum score to draw.

    Returns:
        Annotated image copy.
    """
    img = image.copy()

    for i in range(len(boxes)):
        if scores[i] < conf_threshold:
            continue

        x1, y1, x2, y2 = boxes[i].astype(int)
        cls_id = int(class_ids[i])
        score = scores[i]

        color = COLORS[cls_id % len(COLORS)]
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label background + text
        label = f"{cls_name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(
            img, label, (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    return img


def create_comparison(image, results_dict, conf_threshold=0.25, max_cols=3):
    """Create a grid comparison of detections from multiple models.

    Args:
        image: original BGR image.
        results_dict: dict of {model_name: (boxes, scores, class_ids)}.
        conf_threshold: minimum score to draw.
        max_cols: max columns in the comparison grid.

    Returns:
        Grid image with all model detections.
    """
    panels = []

    for name, (boxes, scores, class_ids) in results_dict.items():
        panel = draw_detections(image, boxes, scores, class_ids, conf_threshold)
        # Overlay model name
        cv2.putText(panel, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(panel, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        panels.append(panel)

    if not panels:
        return image

    # Create grid layout
    h, w = panels[0].shape[:2]
    n = len(panels)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    # Resize panels to uniform size
    target_w = min(w, 640)
    target_h = int(target_w * h / w)
    panels = [cv2.resize(p, (target_w, target_h)) for p in panels]

    grid = np.full((rows * target_h, cols * target_w, 3), 255, dtype=np.uint8)

    for idx, panel in enumerate(panels):
        r, c = divmod(idx, cols)
        grid[r * target_h : (r + 1) * target_h, c * target_w : (c + 1) * target_w] = panel

    return grid


def save_visualization(image, boxes, scores, class_ids, output_path, conf_threshold=0.25):
    """Draw detections and save the annotated image.

    Args:
        image: BGR numpy array.
        boxes: [N, 4] xyxy format.
        scores: [N] confidence scores.
        class_ids: [N] class indices.
        output_path: path to save the annotated image.
        conf_threshold: minimum score to draw.
    """
    vis = draw_detections(image, boxes, scores, class_ids, conf_threshold)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis)
    return vis
