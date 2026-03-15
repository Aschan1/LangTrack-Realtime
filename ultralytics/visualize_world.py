import json
import random
import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ULTRALYTICS_ROOT = REPO_ROOT / "src" / "ultralytics"

# Prefer complete local source package and avoid namespace-shadow imports from this script directory.
while str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if SRC_ULTRALYTICS_ROOT.is_dir() and str(SRC_ULTRALYTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ULTRALYTICS_ROOT))
elif str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLOE


def resolve_siglip_source() -> str:
    """Prefer local SigLIP files, otherwise fallback to HF model id."""
    local_dir = SCRIPT_DIR / "SigLIP" / "siglip2-base-patch16-224"
    required_files = ("config.json", "preprocessor_config.json")
    if local_dir.is_dir() and all((local_dir / name).exists() for name in required_files):
        return str(local_dir)
    return os.getenv("SIGLIP_MODEL_ID", "google/siglip2-base-patch16-224")


def resolve_visualize_inputs():
    """Resolve dataset JSON and image directory with optional env overrides."""
    json_override = os.getenv("VISUALIZE_JSON", "").strip()
    images_override = os.getenv("VISUALIZE_IMAGES", "").strip()

    json_candidates = [
        Path(json_override).expanduser() if json_override else None,
        REPO_ROOT / "yolo_dataset" / "home_ovd_keywords_json_full.json",
        REPO_ROOT / "yolo_dataset" / "indoors_subset" / "filtered_indoors_LM_vg.json",
        REPO_ROOT / "yolo_dataset" / "indoors_subset" / "filtered_indoors_LM.json",
        REPO_ROOT / "yolo_dataset" / "indoors_subset" / "filtered_indoors.json",
    ]
    image_candidates = [
        Path(images_override).expanduser() if images_override else None,
        REPO_ROOT / "yolo_dataset" / "filtered_images",
        REPO_ROOT / "yolo_dataset" / "indoors_subset" / "images",
    ]

    json_file = next((p for p in json_candidates if p and p.is_file()), None)
    images_dir = next((p for p in image_candidates if p and p.is_dir()), None)

    if json_file is None:
        available = sorted(str(p) for p in (REPO_ROOT / "yolo_dataset").rglob("*.json"))
        raise FileNotFoundError(
            "Could not find dataset JSON. Set VISUALIZE_JSON=/abs/path/file.json. "
            f"Found JSON files: {available}"
        )
    if images_dir is None:
        raise FileNotFoundError(
            "Could not find images directory. Set VISUALIZE_IMAGES=/abs/path/to/images."
        )
    return str(json_file), str(images_dir)


def siglip_get_score(crop_img, phrase, processor, model, device):
    """
    Scoring function that directly returns the matching probability.
    """
    text = phrase
    
    inputs = processor(text=[text], images=crop_img, padding="max_length", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits_per_image = outputs.logits_per_image
    prob = torch.sigmoid(logits_per_image).item() 
    
    print(f"    [SigLIP] Match probability for '{phrase}': {prob:.4f}")
    return prob

def visualize_pipeline(img_path, original_phrase, keywords, yolo_model, siglip_processor, siglip_model, device, output_path, gt_box=None, conf_thresh=0.2):
    # Read image (OpenCV for drawing, PIL for cropping for SigLIP)
    cv_img = cv2.imread(img_path)
    if cv_img is None:
        print(f"  [Error] Cannot read image: {img_path}")
        return
        
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    
    # ==========================================
    # Draw Ground Truth first (Yellow)
    # ==========================================
    if gt_box is not None:
        gx, gy, gw, gh = gt_box
        gx1, gy1, gx2, gy2 = int(gx), int(gy), int(gx + gw), int(gy + gh)
        
        cv2.rectangle(cv_img, (gx1, gy1), (gx2, gy2), (0, 255, 255), 3)
        
        label_gt = "Ground Truth"
        (w, h), _ = cv2.getTextSize(label_gt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(cv_img, (gx1, max(0, gy1 - 25)), (gx1 + w, gy1), (0, 255, 255), -1)
        cv2.putText(cv_img, label_gt, (gx1, max(0, gy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 1. Feed all keywords to YOLO
    yolo_model.set_classes(keywords)
    results = yolo_model.predict(img_path, conf=conf_thresh, verbose=False)[0]
    
    raw_pred_boxes = results.boxes.xyxy.cpu().numpy()
    raw_pred_classes = results.boxes.cls.cpu().numpy()
    raw_pred_confs = results.boxes.conf.cpu().numpy()

    print(f"  -> YOLO found {len(raw_pred_boxes)} candidate boxes.")

    # ==========================================
    # 2. [First pass] Iterate over all candidate boxes, get scores and find the highest score
    # ==========================================
    box_scores = []
    best_score = -1.0
    best_idx = -1

    for idx, (p_box, p_cls, p_conf) in enumerate(zip(raw_pred_boxes, raw_pred_classes, raw_pred_confs)):
        x1, y1, x2, y2 = map(int, p_box)
        
        # Out-of-bounds protection
        if x2 <= x1 or y2 <= y1:
            box_scores.append(-1.0)
            continue
            
        crop_img = pil_img.crop((x1, y1, x2, y2))
        
        # Get absolute probability
        score = siglip_get_score(crop_img, original_phrase, siglip_processor, siglip_model, device)
        box_scores.append(score)
        
        # Record the highest score
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx != -1:
        print(f"  -> Highest score selected (Score: {best_score:.4f})")

    # ==========================================
    # 3. [Second pass] Draw boxes based on the highest score logic
    # ==========================================
    for idx, (p_box, p_cls, p_conf) in enumerate(zip(raw_pred_boxes, raw_pred_classes, raw_pred_confs)):
        score = box_scores[idx]
        if score < 0:
            continue
            
        x1, y1, x2, y2 = map(int, p_box)
        detected_keyword = keywords[int(p_cls)]
        
        # --- Core judging logic (Only pass the highest score) ---
        is_valid = (idx == best_idx)
        
        # Set colors and text
        if is_valid:
            color = (0, 255, 0) # Green
            status = f"PASS ({score:.2f})"
            thickness = 3 
        else:
            color = (0, 0, 255) # Red
            status = f"FAIL ({score:.2f})"
            thickness = 2

        # Draw box
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, thickness)
        
        # Add label
        label = f"{detected_keyword} {status}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(cv_img, (x1, max(0, y1 - 20)), (x1 + w, y1), color, -1)
        cv2.putText(cv_img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 4. Save visualization results
    cv2.imwrite(output_path, cv_img)
    print(f"  -> Image saved to: {output_path}")


if __name__ == "__main__":
    # Path configuration
    JSON_FILE, IMAGES_DIR = resolve_visualize_inputs()
    OUTPUT_DIR = "visualizations"
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    # ==========================================
    # Load models in advance
    # ==========================================
    print("Loading YOLO and SigLIP models, please wait...")
    yolo_model = YOLOE("yoloe-26l-seg.pt").to(device)
    siglip_source = resolve_siglip_source()
    print(f"Using SigLIP source: {siglip_source}")
    siglip_processor = AutoProcessor.from_pretrained(siglip_source)
    siglip_model = AutoModel.from_pretrained(siglip_source).to(device)
    siglip_model.eval()
    print("Models loaded successfully!")

    # ==========================================
    # Read JSON and randomly sample
    # ==========================================
    print(f"\nReading dataset: {JSON_FILE}")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_samples = min(20, len(data))
    random_samples = random.sample(data, num_samples)
    
    print(f"Successfully sampled {num_samples} items, starting visualization...\n")

    for i, item in enumerate(random_samples):
        print(f"[{i+1}/{num_samples}] Processing Region ID: {item.get('region_id')} | Phrase: {item.get('phrase')}")
        
        img_id = str(item['image_id'])
        img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
        
        original_phrase = item['phrase']
        keywords = item.get('keywords', None) 
        keywords = [keywords['target']] if keywords else [original_phrase]
        gt_box = [item['x'], item['y'], item['width'], item['height']]
        
        output_path = os.path.join(OUTPUT_DIR, f"vis_sample_{i+1}_img{img_id}.jpg")
        
        if not os.path.exists(img_path):
            print(f"  [Warning] Image file not found: {img_path}, skipping this sample.")
            continue
            
        visualize_pipeline(
            img_path=img_path,
            original_phrase=original_phrase,
            keywords=keywords,
            yolo_model=yolo_model,
            siglip_processor=siglip_processor,
            siglip_model=siglip_model,
            device=device,
            output_path=output_path,
            gt_box=gt_box,
            conf_thresh=0.02
        )
        print("-" * 50)
        
    print(f"\nAll done! Please check the '{OUTPUT_DIR}' folder for results.")
