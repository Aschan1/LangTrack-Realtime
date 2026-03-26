"""
Evaluate GroundingDINO Base with the shared detection metrics used elsewhere in the repo.

This runner keeps the original target-only prompting behavior: each image is
prompted with the unique `keywords.target` values present in that image. Metrics
are then computed over those target labels, including REC.
"""

from __future__ import annotations

import json
import os
import site
import sys
from collections import defaultdict


def _prefer_env_site_packages() -> None:
    # Avoid picking up user-level packages (for example ~/.local/...) ahead of the active conda env.
    user_site = site.getusersitepackages()
    if not user_site:
        return

    user_site = os.path.abspath(user_site)
    sys.path[:] = [path for path in sys.path if not path or os.path.abspath(path) != user_site]


_prefer_env_site_packages()

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

try:
    from detect.refexp_metrics import compute_rec_metrics
except ImportError:
    from refexp_metrics import compute_rec_metrics


def ordered_unique(items):
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def normalize_label(text: str) -> str:
    return " ".join(str(text).strip().lower().replace(".", " ").split())


def get_target_label(annotation: dict) -> str:
    target = normalize_label(annotation.get("keywords", {}).get("target", ""))
    if target:
        return target
    return normalize_label(annotation.get("phrase", ""))


def resolve_detection_label(decoded_label: str, local_labels: list[str]) -> str | None:
    normalized_label = normalize_label(decoded_label)
    if not normalized_label:
        return None

    normalized_to_original = {normalize_label(label): label for label in local_labels}
    if normalized_label in normalized_to_original:
        return normalized_to_original[normalized_label]

    partial_matches = [
        label
        for label in local_labels
        if normalized_label in normalize_label(label) or normalize_label(label) in normalized_label
    ]
    partial_matches = ordered_unique(partial_matches)
    if len(partial_matches) == 1:
        return partial_matches[0]
    return None


def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    if inter_area == 0:
        return 0.0

    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    denom = box1_area + box2_area - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / float(denom)


def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    img_to_anns = defaultdict(list)
    global_label_to_id = {}
    for item in tqdm(data, desc="Loading JSON Doc"):
        img_id = str(item["image_id"])
        img_to_anns[img_id].append(item)

        label = get_target_label(item)
        if label and label not in global_label_to_id:
            global_label_to_id[label] = len(global_label_to_id)

    return data, dict(img_to_anns), global_label_to_id


def collect_predictions_and_gts(
    json_path,
    images_dir,
    box_threshold=0.25,
    text_threshold=0.25,
    validation=True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading GroundingDINO Base...")
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    if validation:
        print("Validation mode ON: target-only prompting is enabled.")
    else:
        print("Validation mode OFF requested, but this runner currently evaluates target-only prompts.")

    _, img_to_anns, global_label_to_id = load_dataset(json_path)

    all_gts = []
    all_preds = []
    stats = {
        "images_evaluated": 0,
        "skipped_missing_images": 0,
        "skipped_invalid_queries": 0,
        "unmatched_detection_labels": 0,
    }

    print(f"Validation Start! {len(img_to_anns)} Pics in total...")

    for img_id, anns in tqdm(img_to_anns.items(), desc="Evaluating"):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            stats["skipped_missing_images"] += 1
            continue

        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception:
            stats["skipped_missing_images"] += 1
            continue

        local_labels = ordered_unique(get_target_label(ann) for ann in anns if get_target_label(ann))
        invalid_queries = len(anns) - sum(1 for ann in anns if get_target_label(ann))
        stats["skipped_invalid_queries"] += invalid_queries
        if not local_labels:
            continue

        stats["images_evaluated"] += 1

        for ann in anns:
            label_str = get_target_label(ann)
            if not label_str:
                continue

            x, y, w, h = ann["x"], ann["y"], ann["width"], ann["height"]
            all_gts.append(
                {
                    "img_id": img_id,
                    "cls": global_label_to_id[label_str],
                    "box": [x, y, x + w, y + h],
                    "used": False,
                    "query_key": label_str,
                }
            )

        inputs = processor(
            images=img_pil,
            text=" . ".join(local_labels),
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[img_pil.size[::-1]],
        )[0]

        boxes = result["boxes"].detach().cpu().numpy() if "boxes" in result else np.empty((0, 4))
        scores = result["scores"].detach().cpu().numpy() if "scores" in result else np.empty((0,))
        text_labels = result.get("text_labels")
        if text_labels is None:
            text_labels = result.get("labels")
        if text_labels is None:
            text_labels = [""] * len(scores)

        for box, score, decoded_label in zip(boxes, scores, text_labels, strict=False):
            resolved_label = resolve_detection_label(str(decoded_label), local_labels)
            if resolved_label is None:
                stats["unmatched_detection_labels"] += 1
                continue

            all_preds.append(
                {
                    "img_id": img_id,
                    "cls": global_label_to_id[resolved_label],
                    "box": box.tolist(),
                    "conf": float(score),
                    "query_key": resolved_label,
                }
            )

    return all_gts, all_preds, stats


def compute_metrics(all_gts, all_preds, iou_thresh=0.5):
    unique_classes = set(gt["cls"] for gt in all_gts)
    global_tp = 0
    total_gts = len(all_gts)
    total_preds = len(all_preds)

    for cls_id in unique_classes:
        preds_c = [pred for pred in all_preds if pred["cls"] == cls_id]
        gts_c = [gt for gt in all_gts if gt["cls"] == cls_id]

        if not gts_c:
            continue

        preds_c.sort(key=lambda pred: pred["conf"], reverse=True)

        for gt in gts_c:
            gt["used"] = False

        for pred in preds_c:
            img_gts = [gt for gt in gts_c if gt["img_id"] == pred["img_id"]]
            ovmax = -1.0
            best_gt = None

            for gt in img_gts:
                iou = calculate_iou(pred["box"], gt["box"])
                if iou > ovmax:
                    ovmax = iou
                    best_gt = gt

            if ovmax >= iou_thresh and best_gt is not None and not best_gt["used"]:
                global_tp += 1
                best_gt["used"] = True

    precision = global_tp / total_preds if total_preds > 0 else 0.0
    recall = global_tp / total_gts if total_gts > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fp = total_preds - global_tp
    fn = total_gts - global_tp
    accuracy = global_tp / (global_tp + fp + fn) if (global_tp + fp + fn) > 0 else 0.0
    rec_summary = compute_rec_metrics(all_gts, all_preds, iou_thresh, calculate_iou)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "true_positives": int(global_tp),
        "total_gts": int(total_gts),
        "total_preds": int(total_preds),
        **rec_summary,
    }


def inference(json_path, images_dir, conf_thresh=0.01, iou_thresh=0.5, validation=False):
    _ = conf_thresh  # GroundingDINO Base currently uses fixed post-process thresholds below.
    all_gts, all_preds, stats = collect_predictions_and_gts(
        json_path=json_path,
        images_dir=images_dir,
        box_threshold=0.25,
        text_threshold=0.25,
        validation=validation,
    )
    metrics = compute_metrics(all_gts, all_preds, iou_thresh=iou_thresh)

    print("\n" + "=" * 40)
    print("Validation with GroundingDINO Base Succeeded!")
    print(f"Images evaluated: {stats['images_evaluated']}")
    print(f"Skipped missing images: {stats['skipped_missing_images']}")
    print(f"Skipped invalid queries: {stats['skipped_invalid_queries']}")
    print(f"Unmatched detection labels: {stats['unmatched_detection_labels']}")
    print("-" * 40)
    print(f"Ground Truths: {metrics['total_gts']}")
    print(f"Matched (IoU >= {iou_thresh}): {metrics['true_positives']}")
    print(f"Recall@{iou_thresh}: {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"Precision@{iou_thresh}: {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"Accuracy@{iou_thresh}: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"F1-Score@{iou_thresh}: {metrics['f1']:.4f} ({metrics['f1'] * 100:.2f}%)")
    print(
        f"REC@{iou_thresh}: {metrics['rec']:.4f} ({metrics['rec'] * 100:.2f}%) "
        f"[{metrics['rec_matched_queries']}/{metrics['rec_total_queries']}]"
    )
    print("=" * 40)

    return metrics


if __name__ == "__main__":
    IS_VALIDATION = True
    JSON_FILE = "yolo_dataset/indoors_subset/filtered_indoors_LM_vg.json"
    IMAGES_DIR = "yolo_dataset/indoors_subset/images"

    inference(JSON_FILE, IMAGES_DIR, validation=IS_VALIDATION)
