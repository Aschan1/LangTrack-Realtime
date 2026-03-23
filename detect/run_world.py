import json
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import dspy
import numpy as np
from tqdm import tqdm
from PIL import Image

# Ensure imports resolve to installed `ultralytics` package, not this script directory shadow.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLOE
from transformers import AutoProcessor, AutoModel
from LLMs.get_prompt import PromptProcessor


def resolve_siglip_source() -> str:
    """Prefer local SigLIP files, otherwise fallback to model id."""
    local_dir = SCRIPT_DIR / "SigLIP" / "siglip2-base-patch16-224"
    required_files = ("config.json", "preprocessor_config.json")
    if local_dir.is_dir() and all((local_dir / name).exists() for name in required_files):
        return str(local_dir)
    return os.getenv("SIGLIP_MODEL_ID", "google/siglip2-base-patch16-224")
def expand_crop(img_pil, box, expansion_factor=1.5):
    width, height = img_pil.size
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    
    # 计算扩展后的中心点和新的宽高
    cx, cy = x1 + w / 2, y1 + h / 2
    new_w, new_h = w * expansion_factor, h * expansion_factor
    
    # 计算新的坐标并确保不越界
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
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def siglip_verify(
    crop_img,
    phrase,
    processor,
    model,
    device
):
    # 丰富负面提示词，吸收不同类型的背景干扰
    texts = [
        phrase, 
        "a blank background", 
        "a photo of a random object", 
        "noise and meaningless texture",
        "cropped irrelevant background"
    ]
    inputs = processor(
        text=texts,
        images=crop_img,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image
    
    # --- Platt Scaling Calibration ---
    A = 0.6256
    B = 3.3241
    
    scaled_logits = A * logits + B
    probs = torch.sigmoid(scaled_logits)

    p_phrase = float(probs[0, 0].item())
    
    return p_phrase


def inference(json_path, images_dir, conf_thresh=0.01, iou_thresh=0.5, validation=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading YOLOE...")
    yolo_weights = REPO_ROOT / "yoloe-26l-seg.pt"
    yolo_model = YOLOE(str(yolo_weights)).to(device)
    yolo_model.to(device)

    print("Loading SigLIP...")
    siglip_source = resolve_siglip_source()
    print(f"Using SigLIP source: {siglip_source}")
    siglip_processor = AutoProcessor.from_pretrained(siglip_source)
    siglip_model = AutoModel.from_pretrained(siglip_source).to(device)
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
            img_w, img_h = img_pil.size
            img_area = img_w * img_h  # 提前计算整图面积
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
            attrs = keywords.get('attributes', [])
            target = keywords.get('target', '')
            
            combined_keyword = " ".join(attrs + [target]).strip()
            
            if combined_keyword not in keyword_to_phrases:
                keyword_to_phrases[combined_keyword] = set() 

            keyword_to_phrases[combined_keyword].add(phrase)

        all_yolo_classes = list(keyword_to_phrases.keys())
        
        if len(all_yolo_classes) > 0:
            yolo_model.set_classes(all_yolo_classes)
        else:
            continue
        
        results = yolo_model.predict(img_path, conf=conf_thresh, verbose=False)
        result = results[0]
        
        raw_pred_boxes = result.boxes.xyxy.cpu().numpy()
        raw_pred_classes = result.boxes.cls.cpu().numpy()
        raw_pred_confs = result.boxes.conf.cpu().numpy()

        # ==========================================
        # [Stage 2] SigLIP Filtering & Disambiguation (Dynamic & Bypass)
        # ==========================================
        filtered_pred_boxes = []
        filtered_pred_classes = []
        filtered_pred_confs = []
        
        # 定义动态 Alpha 的边界值参数 (你可以按需调整)
        MIN_ALPHA = 0.1          # 极小框最少听 SigLIP 多少 (10%)
        MAX_ALPHA = 0.60          # 极大框最多听 SigLIP 多少 (60%)
        SATURATION_RATIO = 0.25   # 当框面积占整图 25% 以上时，Alpha 达到最大值

        for p_box, p_cls, p_det_conf in zip(raw_pred_boxes, raw_pred_classes, raw_pred_confs):
            detected_keyword = all_yolo_classes[int(p_cls)] 
            possible_original_phrases = keyword_to_phrases[detected_keyword]
            
            x1, y1, x2, y2 = map(int, p_box)
            if x2 <= x1 or y2 <= y1:
                continue
                
            if float(p_det_conf) >= 0.4:
                fused_score = float(p_det_conf)
            
            else:
                # 1. 计算面积比例
                box_area = (x2 - x1) * (y2 - y1)
                area_ratio = box_area / img_area
                
                # 2. 线性插值计算当前框的 Alpha
                # min(..., 1.0) 保证面积超过 SATURATION_RATIO 时封顶
                alpha = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * min(area_ratio / SATURATION_RATIO, 1.0)
                
                # 3. SigLIP 验证 (仅针对模糊框进行高开销的推理)
                crop_img = expand_crop(img_pil, [x1, y1, x2, y2], expansion_factor=1.5)
                p_vlm_conf = siglip_verify(crop_img, detected_keyword, siglip_processor, siglip_model, device)
                
                # 4. 几何平均分数融合
                p_det_conf_safe = max(float(p_det_conf), 1e-6)
                p_vlm_conf_safe = max(float(p_vlm_conf), 1e-6)
                
                fused_score = (p_det_conf_safe ** (1 - alpha)) * (p_vlm_conf_safe ** alpha)
            
            # 记录结果
            if fused_score >= conf_thresh:
                for orig_phrase in possible_original_phrases:
                    filtered_pred_boxes.append(p_box)
                    filtered_pred_classes.append(phrase_to_idx[orig_phrase])
                    filtered_pred_confs.append(fused_score)

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
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    print("\n" + "="*40)
    print("Validation with SigLIP Filtering Succeeded!")
    print(f"Ground Truths (Total Targets): {total_gts}")
    print(f"Total Predictions made: {total_preds}")
    print(f"True Positives (Matched): {int(global_tp)}")
    print("-" * 40)
    print(f"Precision@{iou_thresh}: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall@{iou_thresh}:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score@{iou_thresh}:   {f1_score:.4f} ({f1_score*100:.2f}%)")
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
