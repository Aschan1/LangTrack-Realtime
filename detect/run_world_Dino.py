import json
import os
from platform import processor
from pyexpat import model
import torch
import dspy
import ast
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoModelForZeroShotObjectDetection

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

def inference(json_path, images_dir, conf_thresh=0.01, iou_thresh=0.5, validation=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading GroundingDINO Base...")

    #model_id = "IDEA-Research/grounding-dino-tiny"#https://huggingface.co/docs/transformers/en/model_doc/grounding-dino
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

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

    # Initialize counters for metrics
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    # Per-class data for mAP computation: {class_name: [(confidence, is_tp), ...]}
    per_class_preds = {}
    per_class_num_gt = {}

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

        prompt = " . ".join(all_yolo_classes)
        if len(prompt) == 0:
            continue #If no classes detected by YOLO, skip DINO and evaluation for this image

        inputs = processor(
            images=img_pil,
            text=prompt,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            text_threshold=0.25,#Thr best
            target_sizes=[img_pil.size[::-1]]
        )
        #w, h = img_pil.size



        pred = results[0]

        raw_pred_boxes = []
        raw_pred_classes = []

        for box, phrase in zip(pred["boxes"], pred["labels"]):

            x1, y1, x2, y2 = box.tolist()

            raw_pred_boxes.append([x1, y1, x2, y2])

            if phrase in phrase_to_idx:
                raw_pred_classes.append(phrase_to_idx[phrase])
        filtered_pred_boxes = raw_pred_boxes

            # Build GT list for this image
        gt_boxes = []
        gt_phrases = []
        for ann in anns:
            total_targets += 1
            gt_boxes.append([ann['x'], ann['y'], ann['x'] + ann['width'], ann['y'] + ann['height']])
            gt_phrase = ann['phrase'].lower().strip()
            gt_phrases.append(gt_phrase)
            per_class_num_gt[gt_phrase] = per_class_num_gt.get(gt_phrase, 0) + 1

        # Build prediction list sorted by confidence (descending) for proper mAP
        pred_entries = []
        for i, (p_box, p_label, p_score) in enumerate(zip(filtered_pred_boxes, pred["labels"], pred["scores"])):
            conf = p_score.item() if hasattr(p_score, 'item') else float(p_score)
            pred_entries.append((conf, p_box, p_label.lower().strip(), i))
        pred_entries.sort(key=lambda x: x[0], reverse=True)

        # Match predictions to GTs in confidence order (highest first)
        matched_gts = set()
        used_preds = set()

        for conf, p_box, p_label, p_idx in pred_entries:
            best_iou = 0.0
            best_gt_idx = -1
            matched_gt_phrase = None

            for g_idx, (gt_box, gt_phrase) in enumerate(zip(gt_boxes, gt_phrases)):
                if g_idx in matched_gts:
                    continue
                # Same substring matching as original
                if gt_phrase in p_label or p_label in gt_phrase:
                    iou = calculate_iou(gt_box, p_box)
                    if iou >= iou_thresh and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                        matched_gt_phrase = gt_phrase

            if best_gt_idx != -1:
                # TP: record under the GT's class name so it matches per_class_num_gt
                tp += 1
                matched_gts.add(best_gt_idx)
                used_preds.add(p_idx)
                matched_targets += 1
                per_class_preds.setdefault(matched_gt_phrase, []).append((conf, 1))
            else:
                # FP: record under the prediction's own label
                fp += 1
                per_class_preds.setdefault(p_label, []).append((conf, 0))

        # GTs not matched are false negatives
        fn += (len(gt_boxes) - len(matched_gts))

    # ==========================================
    # Compute mAP@50 (11-point interpolation)
    # ==========================================
    import numpy as np

    def compute_ap(predictions, num_gt):
        """Compute Average Precision for a single class using 11-point interpolation."""
        if num_gt == 0:
            return 0.0
        # Sort predictions by confidence (descending)
        predictions.sort(key=lambda x: x[0], reverse=True)

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for conf, is_tp in predictions:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
            recalls.append(tp_cumsum / num_gt)

        # 11-point interpolation
        ap = 0.0
        for r_threshold in np.arange(0.0, 1.1, 0.1):
            # Find max precision at recall >= r_threshold
            p_at_r = [p for p, r in zip(precisions, recalls) if r >= r_threshold]
            if p_at_r:
                ap += max(p_at_r)
        ap /= 11.0
        return ap

    all_classes = set(per_class_num_gt.keys()) | set(per_class_preds.keys())
    ap_per_class = {}
    for cls in all_classes:
        num_gt = per_class_num_gt.get(cls, 0)
        preds = per_class_preds.get(cls, [])
        ap_per_class[cls] = compute_ap(preds, num_gt)

    # Only average over classes that have GT instances
    classes_with_gt = [cls for cls in ap_per_class if per_class_num_gt.get(cls, 0) > 0]
    mAP50 = np.mean([ap_per_class[cls] for cls in classes_with_gt]) if classes_with_gt else 0.0

    # Calculate Final Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # Accuracy in detection is often defined as TP / (TP + FP + FN)
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    print("\n" + "="*40)
    print("Validation with GroundingDINO Base Succeeded!")
    print(f"Ground Truths: {total_targets}")
    print(f"Matched (IoU >= {iou_thresh}): {matched_targets}")
    print(f"Recall@{iou_thresh}: {recall:.4f} ({recall*100:.2f}%)")
    print(f"Precision@{iou_thresh}: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Accuracy@{iou_thresh}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score@{iou_thresh}: {f1:.4f} ({f1*100:.2f}%)")
    print(f"mAP@{iou_thresh}: {mAP50:.4f} ({mAP50*100:.2f}%)")
    print(f"  Classes evaluated: {len(classes_with_gt)}")
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

    JSON_FILE = "yolo_dataset/indoors_subset/filtered_indoors_LM_vg.json" 
    IMAGES_DIR = "yolo_dataset/indoors_subset/images"                
    
    inference(JSON_FILE, IMAGES_DIR, validation=IS_VALIDATION)