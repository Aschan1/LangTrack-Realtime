import json
import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLOE
from transformers import AutoProcessor, AutoModel
from sklearn.linear_model import LogisticRegression

# ================= Helper Functions =================
def expand_crop(img_pil, box, expansion_factor=1.5):
    width, height = img_pil.size
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    new_w, new_h = w * expansion_factor, h * expansion_factor
    new_x1 = max(0, int(cx - new_w / 2))
    new_y1 = max(0, int(cy - new_h / 2))
    new_x2 = min(width, int(cx + new_w / 2))
    new_y2 = min(height, int(cy + new_h / 2))
    return img_pil.crop((new_x1, new_y1, new_x2, new_y2))

def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    if inter_area == 0:
        return 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)

def get_siglip_logits(crop_img, target_phrase, processor, model, device):
    """返回目标词和几个负面词的原始 Logits (不经过 Sigmoid)"""
    negative_phrases = ["a blank background", "a photo of a random object", 
        "noise and meaningless texture", "cropped irrelevant background"]
    texts = [target_phrase] + negative_phrases
    
    inputs = processor(
        text=texts,
        images=crop_img,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # 获取 Logits
    logits = outputs.logits_per_image[0].cpu().numpy()
    
    target_logit = float(logits[0])
    negative_logits = [float(l) for l in logits[1:]]
    
    return target_logit, negative_logits

# ================= Main Calibration Logic =================
def run_calibration(json_path, images_dir, conf_thresh=0.01, iou_thresh=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Models...")
    yolo_model = YOLOE("yoloe-26l-seg.pt").to(device)
    siglip_processor = AutoProcessor.from_pretrained("/home/chen/workplace/LangTrack-Realtime/ultralytics/SigLIP/siglip2-base-patch16-224")
    siglip_model = AutoModel.from_pretrained("/home/chen/workplace/LangTrack-Realtime/ultralytics/SigLIP/siglip2-base-patch16-224").to(device)
    siglip_model.eval()

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    img_to_anns = {}
    for item in data:
        img_id = str(item['image_id'])
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(item)
        
    print(f"Extraction Start! {len(img_to_anns)} Pics in total...")
    
    X_logits = []
    y_labels = []

    for img_id, anns in tqdm(img_to_anns.items(), desc="Extracting Samples"):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        
        # --- Debug 提示：如果图片没找到，打印出来 ---
        if not os.path.exists(img_path):
            print(f"\n[Warning] Image not found: {img_path}")
            continue
            
        try:
            img_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"\n[Warning] Failed to open image {img_path}: {e}")
            continue

        keyword_to_phrases = {}
        gt_boxes = []
        
        for ann in anns:
            phrase = ann['phrase']  
            attrs = ann['keywords'].get('attributes', [])
            target = ann['keywords'].get('target', '')
            combined_keyword = " ".join(attrs + [target]).strip()
            
            if combined_keyword not in keyword_to_phrases:
                keyword_to_phrases[combined_keyword] = set() 
            keyword_to_phrases[combined_keyword].add(phrase)
            
            x, y, w, h = ann['x'], ann['y'], ann['width'], ann['height']
            gt_boxes.append({
                'box': [x, y, x + w, y + h],
                'phrases': keyword_to_phrases[combined_keyword],
                'used': False  # <--- 新增：GT 使用标记
            })

        all_yolo_classes = list(keyword_to_phrases.keys())
        if not all_yolo_classes:
            continue
            
        yolo_model.set_classes(all_yolo_classes)
        results = yolo_model.predict(img_path, conf=conf_thresh, verbose=False)[0]
        
        # --- 核心修改：先收集 YOLO 预测，并按置信度从高到低排序 ---
        preds = []
        for p_box, p_cls, p_conf in zip(results.boxes.xyxy.cpu().numpy(), 
                                        results.boxes.cls.cpu().numpy(),
                                        results.boxes.conf.cpu().numpy()):
            preds.append({
                'box': p_box,
                'cls': int(p_cls),
                'conf': float(p_conf)
            })
            
        preds.sort(key=lambda x: x['conf'], reverse=True) # 置信度高的优先匹配 GT
        
        # 开始分配标签并提取 Logit
        for p in preds:
            detected_keyword = all_yolo_classes[p['cls']] 
            possible_original_phrases = keyword_to_phrases[detected_keyword]
            
            x1, y1, x2, y2 = map(int, p['box'])
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 寻找具有最大 IoU 的同类 GT
            ovmax = -1
            jmax = -1
            for j, gt in enumerate(gt_boxes):
                if not possible_original_phrases.isdisjoint(gt['phrases']):
                    iou = calculate_iou(p['box'], gt['box'])
                    if iou > ovmax:
                        ovmax = iou
                        jmax = j
            
            # --- 严谨的 Label 分配逻辑 ---
            label = 0 # 默认是假阳性 (负样本)
            if ovmax >= iou_thresh:
                if not gt_boxes[jmax]['used']:
                    label = 1 # 只有最大的且没被占用的才算真阳性
                    gt_boxes[jmax]['used'] = True
            
            crop_img = expand_crop(img_pil, [x1, y1, x2, y2], expansion_factor=1.5)
            target_logit, negative_logits = get_siglip_logits(
                crop_img, detected_keyword, siglip_processor, siglip_model, device
            )
            
            X_logits.append(target_logit)
            y_labels.append(label)
            
            for neg_logit in negative_logits:
                X_logits.append(neg_logit)
                y_labels.append(0)

    # ================= Stage 3: Logistic Regression =================
    print("\n" + "="*40)
    print("Fitting Platt Scaling parameters...")
    
    if len(X_logits) == 0:
        print("Error: No samples extracted! Check your image paths and YOLO predictions.")
        return
        
    X = np.array(X_logits).reshape(-1, 1)
    y = np.array(y_labels)
    
    pos_count = np.sum(y == 1)
    neg_count = np.sum(y == 0)
    print(f"Total Samples: {len(y)} (Positives: {pos_count}, Negatives: {neg_count})")
    
    if pos_count == 0 or neg_count == 0:
        print("Error: Need both positive and negative samples to fit. Try lowering conf_thresh.")
        return

    clf = LogisticRegression(solver='lbfgs', class_weight='balanced') 
    clf.fit(X, y)

    A = clf.coef_[0][0]
    B = clf.intercept_[0]

    print("-" * 40)
    print(f"🎉 Calibration Complete!")
    print(f"Optimal A (Scale) = {A:.4f}")
    print(f"Optimal B (Bias)  = {B:.4f}")
    print("="*40)

if __name__ == "__main__":
    JSON_FILE = "yolo_dataset/home_ovd_subset_5.json" 
    
    # 【注意】请确认这里的路径是你图片真正的存放位置！
    # 之前是 "yolo_dataset/indoors_subset/images"
    IMAGES_DIR = "yolo_dataset/filtered_images"                
    
    run_calibration(JSON_FILE, IMAGES_DIR)