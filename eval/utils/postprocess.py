"""Post-processing utilities for RT-DETR ONNX model outputs.

Handles bbox format conversion, NMS, score filtering, and
conversion to COCO prediction format.
"""

import numpy as np


def xywh_to_xyxy(boxes):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2].

    Args:
        boxes: numpy array of shape [N, 4] in center format.

    Returns:
        numpy array of shape [N, 4] in corner format.
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def xyxy_to_xywh(boxes):
    """Convert [x1, y1, x2, y2] to [x, y, w, h] (COCO format: top-left origin).

    Args:
        boxes: numpy array of shape [N, 4] in corner format.

    Returns:
        numpy array of shape [N, 4] in COCO bbox format.
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)


def nms(boxes, scores, iou_threshold=0.7):
    """Apply Non-Maximum Suppression.

    Args:
        boxes: [N, 4] in xyxy format.
        scores: [N] confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        keep: numpy array of indices of kept detections.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return np.array(keep, dtype=int)


def multiclass_nms(boxes, scores, conf_threshold=0.001, nms_threshold=0.7, max_detections=300):
    """Apply multiclass NMS on detections.

    Args:
        boxes: [N, 4] in xyxy format.
        scores: [N, num_classes] class scores.
        conf_threshold: minimum confidence threshold.
        nms_threshold: NMS IoU threshold.
        max_detections: maximum number of detections to keep.

    Returns:
        final_boxes: [M, 4] in xyxy format.
        final_scores: [M] confidence scores.
        final_class_ids: [M] class indices (0-based).
    """
    num_classes = scores.shape[1]

    all_boxes = []
    all_scores = []
    all_class_ids = []

    for cls_id in range(num_classes):
        cls_scores = scores[:, cls_id]
        mask = cls_scores > conf_threshold

        if not mask.any():
            continue

        cls_boxes = boxes[mask]
        cls_scores_filtered = cls_scores[mask]

        keep = nms(cls_boxes, cls_scores_filtered, nms_threshold)

        all_boxes.append(cls_boxes[keep])
        all_scores.append(cls_scores_filtered[keep])
        all_class_ids.append(np.full(len(keep), cls_id, dtype=int))

    if len(all_boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    final_boxes = np.concatenate(all_boxes, axis=0)
    final_scores = np.concatenate(all_scores, axis=0)
    final_class_ids = np.concatenate(all_class_ids, axis=0)

    # Keep top-k by score
    if len(final_scores) > max_detections:
        top_k = np.argsort(final_scores)[::-1][:max_detections]
        final_boxes = final_boxes[top_k]
        final_scores = final_scores[top_k]
        final_class_ids = final_class_ids[top_k]

    return final_boxes, final_scores, final_class_ids


def scale_boxes_to_original(boxes, input_size, original_size, letterbox_info=None):
    """Scale boxes from input coordinates back to original image coordinates.

    Args:
        boxes: [N, 4] in xyxy format, in input_size coordinates.
        input_size: (height, width) of model input.
        original_size: (height, width) of original image.
        letterbox_info: dict with 'scale', 'pad_h', 'pad_w' from letterbox preprocessing.

    Returns:
        scaled_boxes: [N, 4] in xyxy format, in original image coordinates.
    """
    if len(boxes) == 0:
        return boxes

    boxes = boxes.copy()

    if letterbox_info is not None:
        # Remove letterbox padding
        boxes[:, 0] -= letterbox_info["pad_w"]
        boxes[:, 1] -= letterbox_info["pad_h"]
        boxes[:, 2] -= letterbox_info["pad_w"]
        boxes[:, 3] -= letterbox_info["pad_h"]

        # Rescale to original
        scale = letterbox_info["scale"]
        boxes /= scale
    else:
        # Simple resize scaling
        h_scale = original_size[0] / input_size[0]
        w_scale = original_size[1] / input_size[1]
        boxes[:, [0, 2]] *= w_scale
        boxes[:, [1, 3]] *= h_scale

    # Clip to image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_size[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_size[0])

    return boxes


def predictions_to_coco_format(image_id, boxes_xyxy, scores, class_ids, category_mapping=None):
    """Convert detection predictions to COCO results format.

    Args:
        image_id: COCO image ID.
        boxes_xyxy: [N, 4] in xyxy format (original image coordinates).
        scores: [N] confidence scores.
        class_ids: [N] class indices (1-based from model output for VisDrone RT-DETR).
        category_mapping: dict mapping model class index -> COCO category ID.

    Returns:
        list of COCO prediction dicts with keys: image_id, category_id, bbox, score.
    """
    if len(scores) == 0:
        return []

    results = []
    boxes_xywh = xyxy_to_xywh(boxes_xyxy)

    for i in range(len(scores)):
        cat_id = int(class_ids[i])
        if category_mapping is not None:
            cat_id = category_mapping.get(cat_id, cat_id)

        results.append(
            {
                "image_id": int(image_id),
                "category_id": cat_id,
                "bbox": [round(float(x), 2) for x in boxes_xywh[i]],
                "score": round(float(scores[i]), 5),
            }
        )

    return results
