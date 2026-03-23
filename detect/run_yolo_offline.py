"""
YOLOE with offline text embedding cache.

Offline phase:
  - Pre-encode all targets (from dataset + common indoor objects) and all
    attributes via MobileCLIP, saving raw embeddings (before reprta) to a
    .pt cache file.

Online phase:
  - At inference, look up cached target/attribute embeddings, compose them
    (average + L2-normalize), run the lightweight reprta head, and call
    set_classes() with pre-composed embeddings.  The expensive text encoder
    is never invoked during inference.

Usage:
  # Step 1 — build cache (one-time)
  python detect/run_yolo_offline.py --build_cache

  # Step 2 — run evaluation with offline embeddings
  python detect/run_yolo_offline.py
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

# Reuse IoU and mAP helpers from run_world
from run_world import calculate_iou

# ───────────────────────────────────────────────────────────────────────────
# Common indoor objects to add to the offline cache beyond dataset targets
# ───────────────────────────────────────────────────────────────────────────

EXTRA_COMMON_TARGETS = [
    # Furniture
    "sofa", "couch", "armchair", "recliner", "ottoman", "stool", "bench",
    "desk", "table", "coffee table", "dining table", "end table", "side table",
    "nightstand", "dresser", "wardrobe", "bookshelf", "bookcase", "cabinet",
    "shelf", "drawer", "chest", "cupboard",
    # Kitchen
    "refrigerator", "fridge", "oven", "microwave", "toaster", "blender",
    "coffee maker", "dishwasher", "sink", "faucet", "stove", "kettle",
    "cutting board", "pan", "pot", "wok", "spatula", "ladle",
    "plate", "bowl", "cup", "mug", "glass", "bottle", "jar", "can",
    "fork", "knife", "spoon", "chopsticks",
    # Electronics
    "television", "tv", "monitor", "laptop", "computer", "keyboard",
    "mouse", "remote control", "speaker", "headphones", "phone",
    "cell phone", "tablet", "printer", "router", "camera",
    # Bathroom
    "toilet", "bathtub", "shower", "towel", "soap", "shampoo",
    "toothbrush", "toothpaste", "mirror", "hair dryer",
    # Bedroom / Living
    "bed", "pillow", "blanket", "mattress", "lamp", "chandelier",
    "curtain", "rug", "carpet", "clock", "vase", "painting", "picture frame",
    # People and animals
    "person", "man", "woman", "child", "baby", "cat", "dog", "bird",
    # Misc indoor
    "door", "window", "wall", "ceiling", "floor", "stairs",
    "fan", "air conditioner", "heater", "radiator", "fireplace",
    "trash can", "bag", "box", "basket", "plant", "flower",
    "umbrella", "shoe", "hat", "backpack", "suitcase",
    "book", "newspaper", "magazine", "pen", "pencil", "scissors",
    "candle", "toy", "doll", "teddy bear",
]


# ───────────────────────────────────────────────────────────────────────────
# Cache building
# ───────────────────────────────────────────────────────────────────────────

def build_embedding_cache(json_path, cache_path, yolo_weights):
    """Encode all targets and attributes individually; save raw embeddings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset to collect targets and attributes
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    targets = set()
    attributes = set()
    for item in data:
        kw = item.get("keywords", {})
        t = kw.get("target", "")
        if t:
            targets.add(t.strip())
        for a in kw.get("attributes", []):
            if a:
                attributes.add(a.strip())

    # Add common targets
    for t in EXTRA_COMMON_TARGETS:
        targets.add(t.strip())

    targets = sorted(targets)
    attributes = sorted(attributes)
    all_words = targets + attributes

    print(f"Targets: {len(targets)}  |  Attributes: {len(attributes)}  |  Total words: {len(all_words)}")

    # Load YOLOE and get raw embeddings (without reprta)
    model = YOLOE(str(yolo_weights)).to(device)
    inner = model.model  # YOLOEModel / YOLOESegModel

    print("Encoding all words with MobileCLIP (without reprta) ...")
    # Encode in one batch for efficiency
    raw_embeddings = inner.get_text_pe(all_words, without_reprta=True)
    # raw_embeddings shape: (1, len(all_words), D)
    raw_embeddings = raw_embeddings.squeeze(0).cpu()  # (len(all_words), D)

    embed_dim = raw_embeddings.shape[-1]
    print(f"Embedding dim: {embed_dim}")

    # Build lookup: word → embedding vector (1-D tensor of size D)
    cache = {}
    for i, word in enumerate(all_words):
        cache[word] = raw_embeddings[i]

    # Save
    torch.save({
        "embeddings": cache,
        "targets": targets,
        "attributes": attributes,
        "embed_dim": embed_dim,
    }, cache_path)

    print(f"Cache saved to {cache_path}")
    print(f"  {len(targets)} targets, {len(attributes)} attributes")


# ───────────────────────────────────────────────────────────────────────────
# Embedding composition
# ───────────────────────────────────────────────────────────────────────────

def compose_embedding(target, attrs, cache_embeddings, embed_dim):
    """Compose a class embedding from cached target + attribute embeddings.

    Strategy: average the target embedding with each attribute embedding,
    then L2-normalize.  If a word is not in the cache, skip it.
    """
    vecs = []

    # Target embedding (required)
    if target in cache_embeddings:
        vecs.append(cache_embeddings[target])
    else:
        # Fall back: return None so caller knows to encode online
        return None

    # Attribute embeddings
    for a in attrs:
        if a in cache_embeddings:
            vecs.append(cache_embeddings[a])

    # Average + L2 normalize
    composed = torch.stack(vecs).mean(dim=0)
    composed = F.normalize(composed.unsqueeze(0), dim=-1, p=2).squeeze(0)
    return composed


# ───────────────────────────────────────────────────────────────────────────
# Inference with offline embeddings
# ───────────────────────────────────────────────────────────────────────────

def get_combined_label(ann):
    """Build the class label string from keywords (same as run_yolo.py)."""
    target = ann.get("keywords", {}).get("target", "")
    attributes = ann.get("keywords", {}).get("attributes", [])
    if attributes:
        return " ".join(attributes) + " " + target
    return target.strip()


def inference_offline(json_path, images_dir, cache_path, yolo_weights,
                      conf_thresh=0.05, iou_thresh=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load cache
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=True)
    cache_embeddings = cache_data["embeddings"]
    embed_dim = cache_data["embed_dim"]
    print(f"Loaded embedding cache: {len(cache_embeddings)} words, dim={embed_dim}")

    # Load YOLOE
    print("Loading YOLOE ...")
    model = YOLOE(str(yolo_weights)).to(device)
    inner = model.model
    head = inner.model[-1]  # YOLOEDetect head (has reprta)

    # Load dataset
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_to_anns = {}
    for item in tqdm(data, desc="Loading JSON"):
        img_id = str(item["image_id"])
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(item)

    # Global label → id
    global_label_to_id = {}
    for anns in img_to_anns.values():
        for ann in anns:
            label = get_combined_label(ann)
            if label not in global_label_to_id:
                global_label_to_id[label] = len(global_label_to_id)

    print(f"Unique classes: {len(global_label_to_id)}")
    print(f"Evaluating {len(img_to_anns)} images ...")

    all_gts = []
    all_preds = []
    cache_hits = 0
    cache_misses = 0

    for img_id, anns in tqdm(img_to_anns.items(), desc="Inference (offline)"):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue

        # Build per-image class list
        local_labels = list(set(get_combined_label(ann) for ann in anns))
        if not local_labels:
            continue

        # ── Compose embeddings from cache ──
        composed_raw = []  # raw embeddings before reprta
        valid_labels = []
        fallback_labels = []

        for label in local_labels:
            # Parse label → target + attributes
            ann_example = None
            for ann in anns:
                if get_combined_label(ann) == label:
                    ann_example = ann
                    break

            target = ann_example.get("keywords", {}).get("target", "")
            attrs = ann_example.get("keywords", {}).get("attributes", [])

            emb = compose_embedding(target, attrs, cache_embeddings, embed_dim)
            if emb is not None:
                composed_raw.append(emb)
                valid_labels.append(label)
                cache_hits += 1
            else:
                fallback_labels.append(label)
                cache_misses += 1

        if valid_labels:
            # Stack → (1, N, D), run reprta on GPU, then set_classes
            raw_tensor = torch.stack(composed_raw).unsqueeze(0).to(device)  # (1, N, D)
            with torch.no_grad():
                final_emb = head.get_tpe(raw_tensor)  # reprta + L2 normalize
            model.set_classes(valid_labels, final_emb)
        elif fallback_labels:
            # All labels missed cache — fall back to online encoding
            model.set_classes(fallback_labels)
        else:
            continue

        # ── Predict ──
        results = model.predict(img_path, conf=conf_thresh, verbose=False)
        result = results[0]

        raw_pred_boxes = result.boxes.xyxy.cpu().numpy()
        raw_pred_classes = result.boxes.cls.cpu().numpy()
        raw_pred_confs = result.boxes.conf.cpu().numpy()

        active_labels = valid_labels if valid_labels else fallback_labels

        # Record GTs
        for ann in anns:
            x, y, w, h = ann["x"], ann["y"], ann["width"], ann["height"]
            gt_box = [x, y, x + w, y + h]
            label_str = get_combined_label(ann)
            gt_cls_id = global_label_to_id[label_str]
            all_gts.append({"img_id": img_id, "cls": gt_cls_id, "box": gt_box, "used": False})

        # Record predictions
        for p_box, p_cls_local, p_conf in zip(raw_pred_boxes, raw_pred_classes, raw_pred_confs):
            label_str = active_labels[int(p_cls_local)]
            global_cls_id = global_label_to_id.get(label_str)
            if global_cls_id is None:
                continue
            all_preds.append({
                "img_id": img_id,
                "cls": global_cls_id,
                "box": p_box,
                "conf": float(p_conf),
            })

    # ── Metrics ──
    print(f"\nCache hits: {cache_hits}  |  Cache misses: {cache_misses}  "
          f"|  Hit rate: {cache_hits / max(cache_hits + cache_misses, 1) * 100:.1f}%")

    compute_metrics(all_gts, all_preds, iou_thresh)


# ───────────────────────────────────────────────────────────────────────────
# Metrics (same VOC-style mAP as run_yolo.py)
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
    print("Validation with YOLOE (Offline Embeddings)")
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
    parser = argparse.ArgumentParser(description="YOLOE with offline text embedding cache")
    parser.add_argument("--json_path", type=str,
                        default="yolo_dataset/indoors_subset/filtered_indoors_LM_vg.json")
    parser.add_argument("--images_dir", type=str,
                        default="yolo_dataset/indoors_subset/images")
    parser.add_argument("--cache_path", type=str,
                        default="detect/embedding_cache.pt",
                        help="Path to save/load the embedding cache")
    parser.add_argument("--yolo_weights", type=str,
                        default=str(REPO_ROOT / "yoloe-26l-seg.pt"))
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
        inference_offline(
            args.json_path, args.images_dir, args.cache_path, args.yolo_weights,
            args.conf_thresh, args.iou_thresh,
        )


if __name__ == "__main__":
    main()
