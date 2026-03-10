import json
import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLOE

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
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def get_combined_label(ann):
    """
    Generate target + attribute combined label string for a given annotation.
    Example:
    Input: target="shirt", attributes=["blue", "white striped", "hanging"]
    Output: "blue white striped hanging shirt"
    """
    target = ann.get('keywords', {}).get('target', '')
    attributes = ann.get('keywords', {}).get('attributes', [])
    
    if attributes:
        attr_str = " ".join(attributes)
        label = f"{attr_str} {target}".strip()
    else:
        label = target.strip()
        
    return label

def inference_yolo_only(json_path, images_dir, conf_thresh=0.01, iou_thresh=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading YOLOE...")
    yolo_model = YOLOE("yoloe-26l-seg.pt").to(device)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    img_to_anns = {}
    for item in tqdm(data, desc='Loading JSON Doc'):
        img_id = str(item['image_id'])
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(item)
        
    # Global Label to ID
    global_label_to_id = {}
    for anns in img_to_anns.values():
        for ann in anns:
            label = get_combined_label(ann)
            if label not in global_label_to_id:
                global_label_to_id[label] = len(global_label_to_id)
                
    print(f"Total unique 'attribute+target' classes in dataset: {len(global_label_to_id)}")
    print(f"Validation Start! {len(img_to_anns)} Pics in total...")
    
    all_gts = []
    all_preds = []
    
    for img_id, anns in tqdm(img_to_anns.items(), desc="Evaluating"):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue

        local_labels = list(set([get_combined_label(ann) for ann in anns]))
        
        if len(local_labels) > 0:
            yolo_model.set_classes(local_labels)
        else:
            continue 
            
        for ann in anns:
            x, y, w, h = ann['x'], ann['y'], ann['width'], ann['height']
            gt_box = [x, y, x + w, y + h]
            label_str = get_combined_label(ann)
            gt_cls_id = global_label_to_id[label_str]
            
            all_gts.append({'img_id': img_id, 'cls': gt_cls_id, 'box': gt_box, 'used': False})
        
        # Predict with YOLOE
        results = yolo_model.predict(img_path, conf=conf_thresh, verbose=False)
        result = results[0]
        
        raw_pred_boxes = result.boxes.xyxy.cpu().numpy()
        raw_pred_classes = result.boxes.cls.cpu().numpy()
        raw_pred_confs = result.boxes.conf.cpu().numpy()

        # Global class ID mapping: local label index -> local label string -> global class ID
        for p_box, p_cls_local, p_conf in zip(raw_pred_boxes, raw_pred_classes, raw_pred_confs):
            label_str = local_labels[int(p_cls_local)]
            global_cls_id = global_label_to_id[label_str]
            
            all_preds.append({
                'img_id': img_id, 
                'cls': global_cls_id, 
                'box': p_box, 
                'conf': float(p_conf)
            })

    # ==========================================
    # Calculate Metrics: Precision, Recall, mAP50
    # ==========================================
    print("Computing metrics...")
    
    unique_classes = set([gt['cls'] for gt in all_gts])
    aps = []
    global_tp = 0
    total_gts = len(all_gts)
    total_preds = len(all_preds)

    for c in unique_classes:
        preds_c = [p for p in all_preds if p['cls'] == c]
        gts_c = [g for g in all_gts if g['cls'] == c]
        
        npos = len(gts_c)
        if npos == 0:
            continue
            
        preds_c.sort(key=lambda x: x['conf'], reverse=True)
        tp = np.zeros(len(preds_c))
        fp = np.zeros(len(preds_c))
        
        for g in gts_c:
            g['used'] = False
            
        for i, p in enumerate(preds_c):
            img_gts = [g for g in gts_c if g['img_id'] == p['img_id']]
            ovmax = -1
            jmax = -1
            
            for j, g in enumerate(img_gts):
                iou = calculate_iou(p['box'], g['box'])
                if iou > ovmax:
                    ovmax = iou
                    jmax = j
                    
            if ovmax >= iou_thresh:
                if not img_gts[jmax]['used']:
                    tp[i] = 1.0
                    img_gts[jmax]['used'] = True 
                else:
                    fp[i] = 1.0 
            else:
                fp[i] = 1.0 
                
        global_tp += np.sum(tp)
        
        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)
        rec = tp_cumsum / float(npos)
        prec = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
        
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i_list = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i_list + 1] - mrec[i_list]) * mpre[i_list + 1])
        aps.append(ap)

    map50 = np.mean(aps) if len(aps) > 0 else 0.0
    precision = global_tp / total_preds if total_preds > 0 else 0.0
    recall = global_tp / total_gts if total_gts > 0 else 0.0

    print("\n" + "="*40)
    print("Validation with YOLOE ONLY Succeeded!")
    print(f"Ground Truths (Total Targets): {total_gts}")
    print(f"Total Predictions made: {total_preds}")
    print(f"True Positives (Matched): {int(global_tp)}")
    print("-" * 40)
    print(f"Precision@{iou_thresh}: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall@{iou_thresh}:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"mAP@{iou_thresh}:       {map50:.4f} ({map50*100:.2f}%)")
    print("="*40)

if __name__ == "__main__":
    JSON_FILE = "yolo_dataset/indoors_subset/filtered_indoors_LM_vg.json" 
    IMAGES_DIR = "yolo_dataset/indoors_subset/images"                
    
    inference_yolo_only(JSON_FILE, IMAGES_DIR)