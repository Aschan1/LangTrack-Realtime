import json
import os
import torch
import torch.nn.functional as F
import dspy
import numpy as np
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLOE
from transformers import AutoProcessor, AutoModel
from LLMs.get_prompt import PromptProcessor

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

def siglip_verify(
    crop_img,
    phrase,
    processor,
    model,
    device,
    p_thresh=0.35,   
    margin=0.05      
):
    texts = [phrase, "a blank background", "a photo of a random object"]
    inputs = processor(
        text=texts,
        images=crop_img,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image[0]
    probs = F.softmax(logits, dim=0) 

    p_phrase = float(probs[0].item())
    p_negmax = float(torch.max(probs[1:]).item())

    is_top1 = (torch.argmax(probs).item() == 0)
    pass_thresh = (p_phrase >= p_thresh)
    pass_margin = ((p_phrase - p_negmax) >= margin)

    return (is_top1 and pass_thresh and pass_margin), p_phrase


def inference(json_path, images_dir, conf_thresh=0.01, iou_thresh=0.5, validation=False, pass_thresh=0.4, margin=0.1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading YOLOE...")
    yolo_model = YOLOE("yoloe-26l-seg.pt").to(device)
    yolo_model.to(device)

    print("Loading SigLIP...")
    #Replace with your actual SigLIP model path
    siglip_processor = AutoProcessor.from_pretrained("/home/chen/workplace/LangTrack-Realtime/ultralytics/SigLIP/siglip2-base-patch16-224")
    siglip_model = AutoModel.from_pretrained("/home/chen/workplace/LangTrack-Realtime/ultralytics/SigLIP/siglip2-base-patch16-224").to(device)
    siglip_model.eval()

    prompt_processor = None
    if not validation:
        print("Initializing LLM Prompt Processor...")
        prompt_processor = dspy.Predict(PromptProcessor)
        try:
            prompt_processor.load('./LLMs/optimized_prompt.json')
            print("Loaded optimized_prompt.json successfully!")
        except Exception as e:
            print(f"Warning: Did not load optimized prompt. Error: {e}")
    else:
        print("Validation mode ON: LLM is disabled.")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    img_to_anns = {}
    for item in tqdm(data, desc='Loading JSON Doc'):
        img_id = str(item['image_id'])
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(item)
        
    print(f"Validation Start! {len(img_to_anns)} Pics in total...")
    
    all_gts = []
    all_preds = []
    
    for img_id, anns in tqdm(img_to_anns.items(), desc="Evaluating"):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        
        try:
            img_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            continue

        # ==========================================
        # [Stage 1] LLM Processing & YOLO Prediction
        # ==========================================
        keyword_to_phrases = {}
        phrase_to_idx = {}

        for ann in anns:
            phrase = ann['phrase']  
            if phrase not in phrase_to_idx:
                phrase_to_idx[phrase] = len(phrase_to_idx) 
            
            keywords = ann['keywords'] 
            
            if keywords['target'] not in keyword_to_phrases:
                keyword_to_phrases[keywords['target']] = set() 

            keyword_to_phrases[keywords['target']].add(phrase)

        all_yolo_classes = list(keyword_to_phrases.keys())
        
        if len(all_yolo_classes) > 0:
            yolo_model.set_classes(all_yolo_classes)
        else:
            continue 
        
        results = yolo_model.predict(img_path, conf=conf_thresh, verbose=False)
        result = results[0]
        
        raw_pred_boxes = result.boxes.xyxy.cpu().numpy()
        raw_pred_classes = result.boxes.cls.cpu().numpy()

        # ==========================================
        # [Stage 2] SigLIP Filtering & Disambiguation
        # ==========================================
        filtered_pred_boxes = []
        filtered_pred_classes = [] 
        filtered_pred_confs = [] # store the confidence scores for evaluation

        for p_box, p_cls in zip(raw_pred_boxes, raw_pred_classes):
            detected_keyword = all_yolo_classes[int(p_cls)]
            possible_original_phrases = keyword_to_phrases[detected_keyword]
            
            x1, y1, x2, y2 = map(int, p_box)
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop_img = img_pil.crop((x1, y1, x2, y2))
            
            for orig_phrase in possible_original_phrases:
                is_valid, conf_score = siglip_verify(crop_img, orig_phrase, siglip_processor, siglip_model, device, p_thresh=pass_thresh, margin=margin)
                
                if is_valid:
                    filtered_pred_boxes.append(p_box)
                    filtered_pred_classes.append(phrase_to_idx[orig_phrase])
                    filtered_pred_confs.append(conf_score)

        # ==========================================
        # [Stage 3] Record for Global Evaluation
        # ==========================================
        for ann in anns:
            x, y, w, h = ann['x'], ann['y'], ann['width'], ann['height']
            gt_box = [x, y, x + w, y + h]
            gt_cls_id = phrase_to_idx[ann['phrase']]
            all_gts.append({'img_id': img_id, 'cls': gt_cls_id, 'box': gt_box, 'used': False})
            
        for p_box, p_cls, p_conf in zip(filtered_pred_boxes, filtered_pred_classes, filtered_pred_confs):
            all_preds.append({'img_id': img_id, 'cls': int(p_cls), 'box': p_box, 'conf': p_conf})

    # ==========================================
    # [Stage 4] Calculate Metrics: Precision, Recall, mAP50
    # ==========================================
    print("Computing metrics...")
    
    unique_classes = set([gt['cls'] for gt in all_gts])
    aps = []
    global_tp = 0
    global_fp = 0
    total_gts = len(all_gts)
    total_preds = len(all_preds)

    # Finding matches for each class and calculating TP/FP
    for c in unique_classes:
        preds_c = [p for p in all_preds if p['cls'] == c]
        gts_c = [g for g in all_gts if g['cls'] == c]
        
        npos = len(gts_c)
        if npos == 0:
            continue
            
        preds_c.sort(key=lambda x: x['conf'], reverse=True)
        tp = np.zeros(len(preds_c))
        fp = np.zeros(len(preds_c))
        
        # reset GT flags for this class
        for g in gts_c:
            g['used'] = False
            
        for i, p in enumerate(preds_c):
            # Find GTs in the same image
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
        global_fp += np.sum(fp)
        
        # Calculate mAP.
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
    print("Validation with SigLIP Filtering Succeeded!")
    print(f"Ground Truths (Total Targets): {total_gts}")
    print(f"Total Predictions made: {total_preds}")
    print(f"True Positives (Matched): {int(global_tp)}")
    print("-" * 40)
    print(f"Precision@{iou_thresh}: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall@{iou_thresh}:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"mAP@{iou_thresh}:       {map50:.4f} ({map50*100:.2f}%)")
    print("="*40)

if __name__ == "__main__":
    IS_VALIDATION = True

    if not IS_VALIDATION:
        main_lm = dspy.LM(
            model="openai/qwen3.5", 
            api_base="http://127.0.0.1:8080/v1",
            api_key="unused",
            cache=False, 
        )
        dspy.configure(lm=main_lm)

    JSON_FILE = "yolo_dataset/indoors_subset/filtered_indoors_LM_vg.json" 
    IMAGES_DIR = "yolo_dataset/indoors_subset/images"                
    
    inference(JSON_FILE, IMAGES_DIR, validation=IS_VALIDATION)