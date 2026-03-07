import json
import os
import torch
import cv2
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModel

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

def siglip_verify(crop_img, phrase, processor, model, device):
    """
    Use SigLIP to verify if the cropped image matches the given phrase.
    It compares the target phrase against negative prompts.
    """
    # Texts to compare: target phrase vs negative prompts
    texts = [phrase, "blank background", "a picture of a random object"]
    
    # Process inputs for SigLIP
    inputs = processor(text=texts, images=crop_img, padding="max_length", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits_per_image = outputs.logits_per_image
    # SigLIP uses sigmoid rather than softmax
    probs = torch.sigmoid(logits_per_image).squeeze() 
    
    # Check if the target phrase has the highest probability among the three
    max_idx = torch.argmax(probs).item()
    return max_idx == 0

def validate_yoloworld_siglip_on_vg(json_path, images_dir, conf_thresh=0.1, iou_thresh=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load YOLO-World model
    print("Loading YOLO-World...")
    yolo_model = YOLO("ultralytics/yolov8s-worldv2.pt")
    yolo_model.to(device)

    # 2. Load SigLIP model
    print("Loading SigLIP...")
    siglip_processor = AutoProcessor.from_pretrained("/home/chen/workplace/LangTrack-Realtime/ultralytics/SigLIP/siglip2-base-patch16-224")
    siglip_model = AutoModel.from_pretrained("/home/chen/workplace/LangTrack-Realtime/ultralytics/SigLIP/siglip2-base-patch16-224").to(device)
    siglip_model.eval()

    # 3. Load annotation data
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
    
    # 4. Dynamic validation per image with dual filtering
    for img_id, anns in tqdm(img_to_anns.items(), desc="Evaluating"):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
            
        # Get all unique phrases for the current image
        phrases = list(set([ann['phrase'] for ann in anns]))
        phrase_to_idx = {p: i for i, p in enumerate(phrases)}
        
        # Open PIL image for SigLIP cropping later
        try:
            img_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        # [Stage 1] YOLO-World prediction
        yolo_model.set_classes(phrases)
        results = yolo_model.predict(img_path, conf=conf_thresh, verbose=False)
        result = results[0]
        
        raw_pred_boxes = result.boxes.xyxy.cpu().numpy()
        raw_pred_classes = result.boxes.cls.cpu().numpy()
        
        # [Stage 2] SigLIP Filtering
        filtered_pred_boxes = []
        filtered_pred_classes = []
        
        for p_box, p_cls in zip(raw_pred_boxes, raw_pred_classes):
            phrase = phrases[int(p_cls)]
            
            # Get integer coordinates for cropping
            x1, y1, x2, y2 = map(int, p_box)
            
            # Basic sanity check to avoid invalid crops
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop_img = img_pil.crop((x1, y1, x2, y2))
            
            # Verify via SigLIP
            is_valid = siglip_verify(crop_img, phrase, siglip_processor, siglip_model, device)
            
            if is_valid:
                filtered_pred_boxes.append(p_box)
                filtered_pred_classes.append(p_cls)

        # [Stage 3] Evaluation using filtered bounding boxes
        for ann in anns:
            total_targets += 1
            
            # Ground truth box
            x, y, w, h = ann['x'], ann['y'], ann['width'], ann['height']
            gt_box = [x, y, x + w, y + h]
            gt_phrase = ann['phrase']
            gt_cls_id = phrase_to_idx[gt_phrase]
            
            # Search for matches in the SigLIP-filtered predictions
            is_matched = False
            for p_box, p_cls in zip(filtered_pred_boxes, filtered_pred_classes):
                if int(p_cls) == gt_cls_id:
                    iou = calculate_iou(gt_box, p_box)
                    if iou >= iou_thresh:
                        is_matched = True
                        break # This ground truth target was successfully matched
            
            if is_matched:
                matched_targets += 1
                
    # 5. Calculate and output final metrics
    recall = matched_targets / total_targets if total_targets > 0 else 0
    print("\n" + "="*40)
    print("Validation with SigLIP Filtering Succeeded!")
    print(f"Ground Truths: {total_targets}")
    print(f"Matched (IoU >= {iou_thresh}): {matched_targets}")
    print(f"Recall@{iou_thresh}: {recall:.4f} ({recall*100:.2f}%)")
    print("="*40)

if __name__ == "__main__":
    # --- Modify paths according to your actual setup ---
    JSON_FILE = "yolo_dataset/home_ovd_filtered.json" # Recommend using your VLM-filtered JSON here
    IMAGES_DIR = "yolo_dataset/filtered_images"                
    
    validate_yoloworld_siglip_on_vg(JSON_FILE, IMAGES_DIR)