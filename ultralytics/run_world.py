import json
import os
import torch
import dspy
import ast
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
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

def siglip_verify(crop_img, phrase, processor, model, device):
    texts = [phrase, "blank background", "a picture of a random object"]
    inputs = processor(text=texts, images=crop_img, padding="max_length", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits_per_image = outputs.logits_per_image
    probs = torch.sigmoid(logits_per_image).squeeze() 
    
    max_idx = torch.argmax(probs).item()
    return max_idx == 0

def inference(json_path, images_dir, conf_thresh=0.01, iou_thresh=0.5, validation=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading YOLO-World...")
    yolo_model = YOLO("ultralytics/yolov8x-worldv2.pt")
    yolo_model.to(device)

    print("Loading SigLIP...")
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
        
    total_targets = 0
    matched_targets = 0
    
    print(f"Validation Start! {len(img_to_anns)} Pics in total...")
    
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

        #Some complex logic to build the mapping from keywords to original phrases and phrase to class index
        for ann in anns:
            phrase = ann['phrase']  
            if phrase not in phrase_to_idx:
                phrase_to_idx[phrase] = len(phrase_to_idx) 
            
            keywords = ann['keywords'] 
            
            if keywords['target'] not in keyword_to_phrases:
                keyword_to_phrases[keywords['target']] = set() # Use a set to prevent duplicates

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

        for p_box, p_cls in zip(raw_pred_boxes, raw_pred_classes):
            detected_keyword = all_yolo_classes[int(p_cls)]
            
            # Now returns a list/set of strings instead of integers
            possible_original_phrases = keyword_to_phrases[detected_keyword]
            
            x1, y1, x2, y2 = map(int, p_box)
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop_img = img_pil.crop((x1, y1, x2, y2))
            
            for orig_phrase in possible_original_phrases:
                # orig_phrase is safely a string now
                is_valid = siglip_verify(crop_img, orig_phrase, siglip_processor, siglip_model, device)
                
                if is_valid:
                    filtered_pred_boxes.append(p_box)
                    filtered_pred_classes.append(phrase_to_idx[orig_phrase])

        # ==========================================
        # [Stage 3] Evaluation
        # ==========================================
        for ann in anns:
            total_targets += 1
            x, y, w, h = ann['x'], ann['y'], ann['width'], ann['height']
            gt_box = [x, y, x + w, y + h]
            gt_cls_id = phrase_to_idx[ann['phrase']]
            
            is_matched = False
            for p_box, p_cls in zip(filtered_pred_boxes, filtered_pred_classes):
                if int(p_cls) == gt_cls_id:
                    iou = calculate_iou(gt_box, p_box)
                    if iou >= iou_thresh:
                        is_matched = True
                        break 
            
            if is_matched:
                matched_targets += 1
                
    recall = matched_targets / total_targets if total_targets > 0 else 0
    print("\n" + "="*40)
    print("Validation with SigLIP Filtering Succeeded!")
    print(f"Ground Truths: {total_targets}")
    print(f"Matched (IoU >= {iou_thresh}): {matched_targets}")
    print(f"Recall@{iou_thresh}: {recall:.4f} ({recall*100:.2f}%)")
    print("="*40)

if __name__ == "__main__":
    IS_VALIDATION = True

    # ONLY FOR INFERENCE, you will skip it if you run the validation.
    if not IS_VALIDATION:
        main_lm = dspy.LM(
            model="openai/qwen3.5", 
            api_base="http://127.0.0.1:8080/v1",
            api_key="unused",
            cache=False, 
        )
        dspy.configure(lm=main_lm)

    JSON_FILE = "yolo_dataset/home_ovd_keywords_json_full.json" 
    IMAGES_DIR = "yolo_dataset/filtered_images"                
    
    inference(JSON_FILE, IMAGES_DIR, validation=IS_VALIDATION)