"""Quick debug script to check category mapping and prediction format."""
import json
import sys
import os
import numpy as np

# 1. Check annotation categories
ann_path = r"D:\Final Testing\VisDrone Val image set\annotations_VisDrone_val.json"
with open(ann_path) as f:
    data = json.load(f)

cats = {c['id']: c['name'] for c in data['categories']}
print("=== Ground Truth Categories ===")
for cid, name in sorted(cats.items()):
    print(f"  ID {cid}: {name}")

print(f"\n=== Sample Annotations (first 5) ===")
for ann in data['annotations'][:5]:
    print(f"  cat_id={ann['category_id']}, bbox(xywh)={ann['bbox']}, area={ann.get('area', 'N/A')}")

print(f"\n=== Sample Images (first 3) ===")
for img in data['images'][:3]:
    print(f"  id={img['id']}, w={img['width']}, h={img['height']}, file={img['file_name']}")

# 2. Run model on one image and check raw outputs
print("\n=== Model Raw Output Check ===")
sys.path.insert(0, r"D:\Final Testing\eval")
from utils.onnx_inference import OnnxDetector
import cv2

model_path = r"D:\Final Testing\Checkpoints - Baseline_Visdrone2019\RT-DETR Resnet 18\base_rtdetr.onnx"
detector = OnnxDetector(model_path, input_size=(640, 640))

# Pick first image
img_info = data['images'][0]
img_dir = r"D:\Final Testing\VisDrone Val image set"
# Try to find the image
for subfolder in ['images', '']:
    img_path = os.path.join(img_dir, subfolder, img_info['file_name'])
    if os.path.exists(img_path):
        break

print(f"  Image: {img_path}")
print(f"  Image exists: {os.path.exists(img_path)}")

if os.path.exists(img_path):
    img = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]
    print(f"  Original size: {orig_w}x{orig_h}")

    # Preprocess
    input_tensor, scale_info = detector.preprocess(img)
    print(f"  Input tensor shape: {input_tensor.shape}")

    # Run raw inference - test both [h,w] and [w,h] orders
    orig_size_hw = np.array([[orig_h, orig_w]], dtype=np.int64)
    outputs = detector.inference_raw(input_tensor, orig_size=orig_size_hw)
    
    # Also test [w,h] order
    orig_size_wh = np.array([[orig_w, orig_h]], dtype=np.int64)
    outputs_wh = detector.inference_raw(input_tensor, orig_size=orig_size_wh)
    
    print(f"\n  === Raw Outputs ===")
    for name, arr in outputs.items():
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}")
    
    # Check labels distribution
    labels = outputs['labels'][0]
    scores = outputs['scores'][0]
    boxes = outputs['boxes'][0]
    
    print(f"\n  === Label Distribution (all 300 detections) ===")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Label {u}: {c} detections")
    
    print(f"\n  === Top-10 detections by score ===")
    top_idx = np.argsort(scores)[::-1][:10]
    for i in top_idx:
        print(f"    score={scores[i]:.4f}, label={labels[i]}, box(xyxy)={boxes[i]}")
    
    # Check high-confidence detections
    high_conf = scores > 0.5
    print(f"\n  === Detections with score > 0.5: {high_conf.sum()} ===")
    high_conf_02 = scores > 0.2
    print(f"  === Detections with score > 0.2: {high_conf_02.sum()} ===")
    high_conf_01 = scores > 0.1
    print(f"  === Detections with score > 0.1: {high_conf_01.sum()} ===")
    
    # Check corresponding GT annotations for this image
    gt_anns = [a for a in data['annotations'] if a['image_id'] == img_info['id']]
    print(f"\n  === GT annotations for image {img_info['id']}: {len(gt_anns)} ===")
    gt_cats = set(a['category_id'] for a in gt_anns)
    print(f"  GT category IDs present: {sorted(gt_cats)}")
    for ann in gt_anns[:5]:
        x, y, w, h = ann['bbox']
        print(f"    cat_id={ann['category_id']}, bbox(xywh)={ann['bbox']}, xyxy=[{x},{y},{x+w},{y+h}]")
    
    # Compare [h,w] vs [w,h] boxes
    boxes_hw = outputs['boxes'][0]
    boxes_wh = outputs_wh['boxes'][0]
    scores_hw = outputs['scores'][0]
    top5 = np.argsort(scores_hw)[::-1][:5]
    print(f"\n  === Box comparison [h,w] vs [w,h] (top 5) ===")
    for i in top5:
        print(f"    [h,w]: {boxes_hw[i]}  |  [w,h]: {boxes_wh[i]}  score={scores_hw[i]:.4f} label={labels[i]}")
