import json
import os
import torch
import cv2  # Added for image saving
from tqdm import tqdm
from ultralytics import YOLO

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes. Box format: [x_min, y_min, x_max, y_max]"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def validate_yoloworld_on_vg(json_path, images_dir, conf_thresh=0.1, iou_thresh=0.5):
    # 1. Load the model
    model = YOLO("ultralytics/yolov8s-worldv2.pt")
    model.to("cuda")

    # 2. Load annotation data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Group annotations by image_id
    img_to_anns = {}
    for item in tqdm(data, desc='Loading JSON Doc'):
        img_id = str(item['image_id'])
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(item)
        
    total_targets = 0
    matched_targets = 0
    
    print(f"Validation Start! {len(img_to_anns)} Pics in total...")
    
    last_result = None
    last_img_id = None
    
    # 3. Dynamic validation per image
    for img_id, anns in tqdm(img_to_anns.items()):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
            
        # Get all unique phrases for the current image
        phrases = list(set([ann['phrase'] for ann in anns]))
        # Create a phrase-to-index mapping for easier matching later
        phrase_to_idx = {p: i for i, p in enumerate(phrases)}
        
        # [Key step] Dynamic injection! Set the classes to only the phrases present in this image
        model.set_classes(phrases)
        
        # Perform prediction
        results = model.predict(img_path, conf=conf_thresh, verbose=False)
        result = results[0]
        
        # Track the result and ID of the last processed image for visualization
        last_result = result
        last_img_id = img_id
        
        # Get predicted bounding boxes (xyxy) and class IDs
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_classes = result.boxes.cls.cpu().numpy()
        
        # Iterate over ground truth annotations to check for matches
        for ann in anns:
            total_targets += 1
            
            # Ground truth box (converted to xyxy)
            x, y, w, h = ann['x'], ann['y'], ann['width'], ann['height']
            gt_box = [x, y, x + w, y + h]
            gt_phrase = ann['phrase']
            gt_cls_id = phrase_to_idx[gt_phrase]
            
            # Search for predicted boxes of the corresponding class
            is_matched = False
            for p_box, p_cls in zip(pred_boxes, pred_classes):
                if int(p_cls) == gt_cls_id:
                    iou = calculate_iou(gt_box, p_box)
                    if iou >= iou_thresh:
                        is_matched = True
                        break # This ground truth target was successfully recalled (matched)
            
            if is_matched:
                matched_targets += 1
                
    # 4. Calculate and output final metrics
    recall = matched_targets / total_targets if total_targets > 0 else 0
    print("\n" + "="*40)
    print("Validation Succeeded!")
    print(f"Ground Truths: {total_targets}")
    print(f"IoU >= {iou_thresh}: {matched_targets}")
    print(f"Recall@{iou_thresh}: {recall:.4f} ({recall*100:.2f}%)")
    print("="*40)

    # 5. Visualize the result of the last processed image
    if last_result is not None:
        # result.plot() returns a BGR numpy array with drawn bounding boxes and labels
        img_with_boxes = last_result.plot()
        output_filename = f"last_val_result_{last_img_id}.jpg"
        cv2.imwrite(output_filename, img_with_boxes)
        print(f"\n🖼️ Saved prediction visualization for the last image ({last_img_id}.jpg) as: {output_filename}")

if __name__ == "__main__":
    # --- Modify paths according to your actual setup ---
    JSON_FILE = "yolo_dataset/home_ovd_samples.json"    
    IMAGES_DIR = "yolo_dataset/images/val"                
    
    validate_yoloworld_on_vg(JSON_FILE, IMAGES_DIR)