"""
YOLOE + SigLIP filtering with offline text embedding cache.

Combines the offline embedding technique (pre-encode targets/attributes via
MobileCLIP, cache to .pt) with the SigLIP verification and score fusion
pipeline from run_world.py.

Offline phase:
  - Pre-encode all targets (from dataset + common indoor objects) and all
    attributes via MobileCLIP, saving raw embeddings (before reprta) to a
    .pt cache file.

Online phase:
  - At inference, look up cached target/attribute embeddings, compose them
    (average + L2-normalize), run the lightweight reprta head, and call
    set_classes() with pre-composed embeddings.
  - Apply SigLIP filtering with dynamic alpha fusion on ambiguous detections.

Usage:
  # Step 1 — build cache (one-time)
  python detect/run_world_offline.py --build_cache

  # Step 2 — run evaluation with offline embeddings + SigLIP
  python detect/run_world_offline.py
"""

import json
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

from ultralytics import YOLOE
from transformers import AutoProcessor, AutoModel

# Reuse helpers from existing modules
from run_world import calculate_iou, expand_crop, siglip_verify, resolve_siglip_source
from run_yolo_offline import (
    EXTRA_COMMON_TARGETS,
    build_embedding_cache,
    compose_embedding,
)


# ───────────────────────────────────────────────────────────────────────────
# Inference with offline embeddings + SigLIP filtering
# ───────────────────────────────────────────────────────────────────────────

def inference_offline_world(json_path, images_dir, cache_path, yolo_weights,
                            conf_thresh=0.01, iou_thresh=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load embedding cache
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=True)
    cache_embeddings = cache_data["embeddings"]
    embed_dim = cache_data["embed_dim"]
    print(f"Loaded embedding cache: {len(cache_embeddings)} words, dim={embed_dim}")

    # Load YOLOE
    print("Loading YOLOE ...")
    yolo_model = YOLOE(str(yolo_weights)).to(device)
    inner = yolo_model.model
    head = inner.model[-1]  # YOLOEDetect head (has reprta)

    # Load SigLIP
    print("Loading SigLIP ...")
    siglip_source = resolve_siglip_source()
    print(f"Using SigLIP source: {siglip_source}")
    siglip_processor = AutoProcessor.from_pretrained(siglip_source)
    siglip_model = AutoModel.from_pretrained(siglip_source).to(device)
    siglip_model.eval()

    # Load dataset
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_to_anns = {}
    for item in tqdm(data, desc="Loading JSON"):
        img_id = str(item["image_id"])
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(item)

    print(f"Evaluating {len(img_to_anns)} images ...")

    all_gts = []
    all_preds = []
    cache_hits = 0
    cache_misses = 0

    # Dynamic alpha parameters (same as run_world.py)
    MIN_ALPHA = 0.1
    MAX_ALPHA = 0.60
    SATURATION_RATIO = 0.25

    for img_id, anns in tqdm(img_to_anns.items(), desc="Inference (offline+SigLIP)"):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue

        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_w, img_h = img_pil.size
            img_area = img_w * img_h
        except Exception:
            continue

        # ── Stage 1: Build per-image class list & compose embeddings ──
        keyword_to_phrases = {}
        phrase_to_idx = {}

        for ann in anns:
            phrase = ann["phrase"]
            if phrase not in phrase_to_idx:
                phrase_to_idx[phrase] = len(phrase_to_idx)

            keywords = ann["keywords"]
            attrs = keywords.get("attributes", [])
            target = keywords.get("target", "")
            combined_keyword = " ".join(attrs + [target]).strip()

            if combined_keyword not in keyword_to_phrases:
                keyword_to_phrases[combined_keyword] = set()
            keyword_to_phrases[combined_keyword].add(phrase)

        all_yolo_classes = list(keyword_to_phrases.keys())
        if not all_yolo_classes:
            continue

        # Compose embeddings from cache for each keyword
        composed_raw = []
        valid_labels = []
        fallback_labels = []

        for label in all_yolo_classes:
            # Parse label back into target + attributes
            # Find an annotation that matches this combined keyword
            ann_example = None
            for ann in anns:
                kw = ann["keywords"]
                attrs = kw.get("attributes", [])
                target = kw.get("target", "")
                if " ".join(attrs + [target]).strip() == label:
                    ann_example = ann
                    break

            if ann_example is not None:
                target = ann_example["keywords"].get("target", "")
                attrs = ann_example["keywords"].get("attributes", [])
            else:
                # Fallback: treat entire label as target
                target = label
                attrs = []

            emb = compose_embedding(target, attrs, cache_embeddings, embed_dim)
            if emb is not None:
                composed_raw.append(emb)
                valid_labels.append(label)
                cache_hits += 1
            else:
                fallback_labels.append(label)
                cache_misses += 1

        if valid_labels:
            raw_tensor = torch.stack(composed_raw).unsqueeze(0).to(device)
            with torch.no_grad():
                final_emb = head.get_tpe(raw_tensor)
            yolo_model.set_classes(valid_labels, final_emb)
            active_labels = valid_labels
        elif fallback_labels:
            yolo_model.set_classes(fallback_labels)
            active_labels = fallback_labels
        else:
            continue

        # ── Predict ──
        results = yolo_model.predict(img_path, conf=conf_thresh, verbose=False)
        result = results[0]

        raw_pred_boxes = result.boxes.xyxy.cpu().numpy()
        raw_pred_classes = result.boxes.cls.cpu().numpy()
        raw_pred_confs = result.boxes.conf.cpu().numpy()

        # ── Stage 2: SigLIP filtering & disambiguation ──
        filtered_pred_boxes = []
        filtered_pred_classes = []
        filtered_pred_confs = []

        for p_box, p_cls, p_det_conf in zip(raw_pred_boxes, raw_pred_classes, raw_pred_confs):
            detected_keyword = active_labels[int(p_cls)]
            possible_original_phrases = keyword_to_phrases.get(detected_keyword, set())

            x1, y1, x2, y2 = map(int, p_box)
            if x2 <= x1 or y2 <= y1:
                continue

            if float(p_det_conf) >= 0.1:
                fused_score = float(p_det_conf)
            else:
                # Dynamic alpha based on box area
                box_area = (x2 - x1) * (y2 - y1)
                area_ratio = box_area / img_area
                alpha = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * min(area_ratio / SATURATION_RATIO, 1.0)

                # SigLIP verification
                crop_img = expand_crop(img_pil, [x1, y1, x2, y2], expansion_factor=1.5)
                p_vlm_conf = siglip_verify(crop_img, detected_keyword, siglip_processor, siglip_model, device)

                # Geometric mean score fusion
                p_det_conf_safe = max(float(p_det_conf), 1e-6)
                p_vlm_conf_safe = max(float(p_vlm_conf), 1e-6)
                fused_score = (p_det_conf_safe ** (1 - alpha)) * (p_vlm_conf_safe ** alpha)

            if fused_score >= conf_thresh:
                for orig_phrase in possible_original_phrases:
                    filtered_pred_boxes.append(p_box)
                    filtered_pred_classes.append(phrase_to_idx[orig_phrase])
                    filtered_pred_confs.append(fused_score)

        # ── Stage 3: Record for global evaluation ──
        for ann in anns:
            x, y, w, h = ann["x"], ann["y"], ann["width"], ann["height"]
            gt_box = [x, y, x + w, y + h]
            gt_cls_id = phrase_to_idx[ann["phrase"]]
            all_gts.append({"img_id": img_id, "cls": gt_cls_id, "box": gt_box, "used": False})

        for p_box, p_cls, p_conf in zip(filtered_pred_boxes, filtered_pred_classes, filtered_pred_confs):
            all_preds.append({"img_id": img_id, "cls": int(p_cls), "box": p_box, "conf": p_conf})

    # ── Stage 4: Metrics ──
    print(f"\nCache hits: {cache_hits}  |  Cache misses: {cache_misses}  "
          f"|  Hit rate: {cache_hits / max(cache_hits + cache_misses, 1) * 100:.1f}%")

    compute_metrics(all_gts, all_preds, iou_thresh)


# ───────────────────────────────────────────────────────────────────────────
# Metrics (VOC-style mAP, same as run_world.py)
# ───────────────────────────────────────────────────────────────────────────

def compute_metrics(all_gts, all_preds, iou_thresh=0.5):
    print("Computing metrics ...")

    unique_classes = set(gt["cls"] for gt in all_gts)
    aps = []
    global_tp = 0
    total_gts = len(all_gts)
    total_preds = len(all_preds)

    for c in unique_classes:
        preds_c = [p for p in all_preds if p["cls"] == c]
        gts_c = [g for g in all_gts if g["cls"] == c]

        npos = len(gts_c)
        if npos == 0:
            continue

        preds_c.sort(key=lambda x: x["conf"], reverse=True)
        tp = np.zeros(len(preds_c))
        fp = np.zeros(len(preds_c))

        for g in gts_c:
            g["used"] = False

        for i, p in enumerate(preds_c):
            img_gts = [g for g in gts_c if g["img_id"] == p["img_id"]]
            ovmax = -1
            jmax = -1

            for j, g in enumerate(img_gts):
                iou = calculate_iou(p["box"], g["box"])
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

            if ovmax >= iou_thresh:
                if not img_gts[jmax]["used"]:
                    tp[i] = 1.0
                    img_gts[jmax]["used"] = True
                else:
                    fp[i] = 1.0
            else:
                fp[i] = 1.0

        global_tp += np.sum(tp)

        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)
        rec = tp_cumsum / float(npos)
        prec = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)

        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i_list = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i_list + 1] - mrec[i_list]) * mpre[i_list + 1])
        aps.append(ap)

    map50 = np.mean(aps) if len(aps) > 0 else 0.0
    precision = global_tp / total_preds if total_preds > 0 else 0.0
    recall = global_tp / total_gts if total_gts > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 40)
    print("Validation with YOLOE (Offline Embeddings + SigLIP)")
    print(f"Ground Truths: {total_gts}")
    print(f"Total Predictions: {total_preds}")
    print(f"True Positives: {int(global_tp)}")
    print("-" * 40)
    print(f"Precision@{iou_thresh}: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall@{iou_thresh}:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-Score@{iou_thresh}:   {f1_score:.4f} ({f1_score * 100:.2f}%)")
    print(f"mAP@{iou_thresh}:       {map50:.4f} ({map50 * 100:.2f}%)")
    print("=" * 40)


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YOLOE + SigLIP with offline text embedding cache"
    )
    parser.add_argument("--json_path", type=str,
                        default=f"/home/chen/workplace/LangTrack-Realtime/yolo_dataset/indoors_subset/filtered_indoors_LM_vg.json")
    parser.add_argument("--images_dir", type=str,
                        default=f"/home/chen/workplace/LangTrack-Realtime/yolo_dataset/indoors_subset/images")
    parser.add_argument("--cache_path", type=str,
                        default="/home/chen/workplace/LangTrack-Realtime/detect/embedding_cache.pt",
                        help="Path to save/load the embedding cache")
    parser.add_argument("--yolo_weights", type=str,
                        default= "/home/chen/workplace/LangTrack-Realtime/yoloe-26l-seg.pt")
    parser.add_argument("--conf_thresh", type=float, default=0.01)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--build_cache", action="store_true",
                        help="Build the embedding cache and exit")
    args = parser.parse_args()

    if args.build_cache:
        build_embedding_cache(args.json_path, args.cache_path, args.yolo_weights)
    else:
        if not os.path.exists(args.cache_path):
            print(f"Cache not found at {args.cache_path}. Building cache first ...")
            build_embedding_cache(args.json_path, args.cache_path, args.yolo_weights)
        inference_offline_world(
            args.json_path, args.images_dir, args.cache_path, args.yolo_weights,
            args.conf_thresh, args.iou_thresh,
        )


if __name__ == "__main__":
    main()
