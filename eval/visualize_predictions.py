import os
import json
import cv2
import argparse
import numpy as np
from pycocotools.coco import COCO

def draw_boxes(img, anns, coco, color=(0, 255, 0)):
    # Create a copy so we don't draw on original
    img_draw = img.copy()
    for ann in anns:
        x, y, w, h = [int(v) for v in ann['bbox']]
        cat_id = ann['category_id']
        # Try to get category name
        try:
            cat_name = coco.loadCats([cat_id])[0]['name']
        except Exception:
            cat_name = f"ID: {cat_id}"
            
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img_draw, f'{cat_name}', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_draw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions JSON')
    parser.add_argument('--annotations', type=str, default='VisDrone Val image set/annotations_VisDrone_val.json', help='Path to ground truth JSON')
    parser.add_argument('--images', type=str, default='VisDrone Val image set', help='Path to validation images')
    parser.add_argument('--output', type=str, default='eval/visualizations', help='Output directory')
    parser.add_argument('--num-images', type=int, default=5, help='Number of images to visualize')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load ground truth
    print(f"Loading annotations from {args.annotations}...")
    coco_gt = COCO(args.annotations)

    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    with open(args.predictions, 'r') as f:
        preds = json.load(f)
    print(f"Loaded {len(preds)} predictions")

    # Group predictions by image_id
    preds_by_img = {}
    for p in preds:
        img_id = p['image_id']
        # Only keep confident predictions for visualization
        if p.get('score', 1.0) > 0.3:
            if img_id not in preds_by_img:
                preds_by_img[img_id] = []
            preds_by_img[img_id].append(p)

    # Get sample images
    img_ids = list(coco_gt.imgs.keys())
    # Pick a few that have predictions
    sample_ids = []
    for img_id in img_ids:
        if img_id in preds_by_img:
            sample_ids.append(img_id)
            if len(sample_ids) >= args.num_images:
                break

    if not sample_ids:
        sample_ids = img_ids[:args.num_images]

    print(f"Visualizing {len(sample_ids)} images...")

    for img_id in sample_ids:
        img_info = coco_gt.loadImgs([img_id])[0]
        img_path = os.path.join(args.images, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        # Get annotations
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        
        # Get predictions
        img_preds = preds_by_img.get(img_id, [])

        # Draw ground truth (Green)
        img_gt = draw_boxes(img, anns, coco_gt, color=(0, 255, 0))
        cv2.putText(img_gt, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw predictions (Red)
        img_pred = draw_boxes(img, img_preds, coco_gt, color=(0, 0, 255))
        cv2.putText(img_pred, "Predictions", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Concatenate side by side
        vis_img = np.concatenate((img_gt, img_pred), axis=1)
        
        # Resize if too large
        max_width = 1920
        if vis_img.shape[1] > max_width:
            scale = max_width / vis_img.shape[1]
            vis_img = cv2.resize(vis_img, (0, 0), fx=scale, fy=scale)
            
        out_path = os.path.join(args.output, f"vis_{img_id}.jpg")
        cv2.imwrite(out_path, vis_img)
        print(f"Saved {out_path}")

    print(f"\nVisualization complete. Check the '{args.output}' folder.")

if __name__ == "__main__":
    main()
