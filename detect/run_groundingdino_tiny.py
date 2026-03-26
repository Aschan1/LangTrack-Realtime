"""
Evaluate GroundingDINO Tiny with the same detection metrics used by the YOLO runners.

This script batches each image's unique phrase queries into one prompt whenever
possible, and automatically splits oversized prompts into multiple chunks when
they would exceed GroundingDINO Tiny's text limit. It then collects all
predictions and ground truths into the shared metric pipeline and reports:
  - Precision@IoU
  - Recall@IoU
  - F1@IoU
  - mAP@IoU
  - REC@IoU
"""

from __future__ import annotations

import argparse
import json
import os
import site
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _prefer_env_site_packages() -> None:
    """Avoid user-level package overlays outranking the active environment."""
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

from refexp_metrics import compute_rec_metrics


DEFAULT_JSON = REPO_ROOT / "yolo_dataset" / "indoors_subset" / "filtered_indoors_LM_vg_nonnull_spatial_relations.json"
DEFAULT_IMAGES = REPO_ROOT / "yolo_dataset" / "indoors_subset" / "images"
DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def ordered_unique(items):
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def normalize_query_label(text: str) -> str:
    return " ".join(str(text).strip().lower().replace(".", " ").split())


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


def get_query_text(annotation: dict) -> str:
    phrase = " ".join(str(annotation.get("phrase", "")).strip().lower().split())
    if phrase:
        return phrase

    keywords = annotation.get("keywords", {})
    target = " ".join(str(keywords.get("target", "")).strip().lower().split())
    attributes = [" ".join(str(attr).strip().lower().split()) for attr in keywords.get("attributes", []) if attr]
    if target and attributes:
        return " ".join(attributes + [target])
    return target


def prompt_for_query(query: str) -> str:
    return query if query.endswith(".") else f"{query}."


def prompt_for_queries(queries: list[str]) -> str:
    return " . ".join(query.rstrip(".") for query in queries if query).strip() + " ."


def count_prompt_tokens(processor, queries: list[str]) -> int:
    prompt = prompt_for_queries(queries)
    encoded = processor.tokenizer(prompt, add_special_tokens=True, truncation=False)
    return len(encoded["input_ids"])


def chunk_queries_for_model(processor, queries: list[str], max_text_len: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []

    for query in queries:
        candidate = [*current, query]
        if current and count_prompt_tokens(processor, candidate) > max_text_len:
            chunks.append(current)
            current = [query]
            continue
        current = candidate

    if current:
        chunks.append(current)

    return chunks


def resolve_detection_query(decoded_label: str, local_queries: list[str]) -> str | None:
    normalized_label = normalize_query_label(decoded_label)
    if not normalized_label:
        return None

    normalized_queries = {normalize_query_label(query): query for query in local_queries}
    if normalized_label in normalized_queries:
        return normalized_queries[normalized_label]

    partial_matches = [
        query
        for query in local_queries
        if normalized_label in normalize_query_label(query) or normalize_query_label(query) in normalized_label
    ]
    partial_matches = ordered_unique(partial_matches)
    if len(partial_matches) == 1:
        return partial_matches[0]
    return None


def get_gt_box(annotation: dict) -> list[float]:
    x, y, w, h = annotation["x"], annotation["y"], annotation["width"], annotation["height"]
    return [x, y, x + w, y + h]


def load_dataset(json_path: Path) -> tuple[list[dict], dict[str, list[dict]], dict[str, int]]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    img_to_anns = defaultdict(list)
    global_query_to_id = {}
    for ann in data:
        img_id = str(ann["image_id"])
        img_to_anns[img_id].append(ann)
        query = get_query_text(ann)
        if query and query not in global_query_to_id:
            global_query_to_id[query] = len(global_query_to_id)

    return data, dict(img_to_anns), global_query_to_id


def collect_predictions_and_gts(
    json_path: Path,
    images_dir: Path,
    model_id: str,
    box_threshold: float,
    text_threshold: float,
    limit: int,
    device: str,
    verbose: bool,
):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    max_text_len = int(
        getattr(model.config, "max_text_len", None)
        or getattr(getattr(processor, "tokenizer", None), "model_max_length", 256)
    )

    _, img_to_anns, global_query_to_id = load_dataset(json_path)
    entries = sorted(img_to_anns.items(), key=lambda item: int(item[0]))
    if limit > 0:
        entries = entries[:limit]

    all_gts = []
    all_preds = []
    stats = {
        "images_evaluated": 0,
        "queries_evaluated": 0,
        "model_forward_passes": 0,
        "skipped_missing_images": 0,
        "skipped_invalid_queries": 0,
        "unmatched_detection_labels": 0,
        "prompt_chunks": 0,
    }

    print(f"Loading GroundingDINO Tiny from {model_id} on {device}...")
    print(f"Unique query classes: {len(global_query_to_id)}")
    print(f"Validation Start! {len(entries)} images in total...")

    for img_id, anns in tqdm(entries, desc="Inference with GroundingDINO tiny"):
        img_path = images_dir / f"{img_id}.jpg"
        if not img_path.is_file():
            stats["skipped_missing_images"] += 1
            if verbose:
                print(f"Skipping missing image: {img_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:  # noqa: PERF203
            stats["skipped_missing_images"] += 1
            if verbose:
                print(f"Skipping unreadable image {img_path}: {exc}")
            continue

        local_queries = ordered_unique(get_query_text(ann) for ann in anns if get_query_text(ann))
        invalid_queries = len(anns) - sum(1 for ann in anns if get_query_text(ann))
        stats["skipped_invalid_queries"] += invalid_queries
        if not local_queries:
            continue

        stats["images_evaluated"] += 1

        for ann in anns:
            query = get_query_text(ann)
            if not query:
                continue
            all_gts.append(
                {
                    "img_id": img_id,
                    "cls": global_query_to_id[query],
                    "box": get_gt_box(ann),
                    "used": False,
                    "query_key": query,
                }
            )

        query_chunks = chunk_queries_for_model(processor, local_queries, max_text_len=max_text_len)
        stats["prompt_chunks"] += len(query_chunks)

        for query_chunk in query_chunks:
            inputs = processor(
                images=image,
                text=prompt_for_queries(query_chunk),
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]

            boxes = result["boxes"].detach().cpu().numpy() if "boxes" in result else np.empty((0, 4))
            scores = result["scores"].detach().cpu().numpy() if "scores" in result else np.empty((0,))
            text_labels = result.get("text_labels")
            if text_labels is None:
                text_labels = result.get("labels")
            if text_labels is None:
                text_labels = [""] * len(scores)

            for box, score, decoded_label in zip(boxes, scores, text_labels, strict=False):
                resolved_query = resolve_detection_query(str(decoded_label), query_chunk)
                if resolved_query is None:
                    stats["unmatched_detection_labels"] += 1
                    continue
                all_preds.append(
                    {
                        "img_id": img_id,
                        "cls": global_query_to_id[resolved_query],
                        "box": box.tolist(),
                        "conf": float(score),
                        "query_key": resolved_query,
                    }
                )

        stats["queries_evaluated"] += len(local_queries)
        stats["model_forward_passes"] += len(query_chunks)

    return all_gts, all_preds, stats


def compute_metrics(all_gts, all_preds, iou_thresh=0.5):
    unique_classes = set(gt["cls"] for gt in all_gts)
    aps = []
    global_tp = 0
    total_gts = len(all_gts)
    total_preds = len(all_preds)

    for cls_id in unique_classes:
        preds_c = [pred for pred in all_preds if pred["cls"] == cls_id]
        gts_c = [gt for gt in all_gts if gt["cls"] == cls_id]

        npos = len(gts_c)
        if npos == 0:
            continue

        preds_c.sort(key=lambda pred: pred["conf"], reverse=True)
        tp = np.zeros(len(preds_c))
        fp = np.zeros(len(preds_c))

        for gt in gts_c:
            gt["used"] = False

        for i, pred in enumerate(preds_c):
            img_gts = [gt for gt in gts_c if gt["img_id"] == pred["img_id"]]
            ovmax = -1.0
            jmax = -1

            for j, gt in enumerate(img_gts):
                iou = calculate_iou(pred["box"], gt["box"])
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

    map50 = np.mean(aps) if aps else 0.0
    precision = global_tp / total_preds if total_preds > 0 else 0.0
    recall = global_tp / total_gts if total_gts > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    rec_summary = compute_rec_metrics(all_gts, all_preds, iou_thresh, calculate_iou)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
        "map50": float(map50),
        "true_positives": int(global_tp),
        "total_gts": int(total_gts),
        "total_preds": int(total_preds),
        **rec_summary,
    }


def maybe_save_json(save_path: Path, all_gts, all_preds, stats, metrics):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "stats": stats,
                "metrics": metrics,
                "ground_truths": all_gts,
                "predictions": all_preds,
            },
            handle,
            indent=2,
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate GroundingDINO Tiny with shared detection metrics.")
    parser.add_argument("--json", default=str(DEFAULT_JSON), help="Annotation JSON path.")
    parser.add_argument("--images", default=str(DEFAULT_IMAGES), help="Directory containing <image_id>.jpg files.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="GroundingDINO model id or local path.")
    parser.add_argument("--box-threshold", type=float, default=0.25, help="GroundingDINO box threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="GroundingDINO text threshold.")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for metrics.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of images.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--save-json", default="", help="Optional path to save predictions, GTs, and metrics as JSON.")
    parser.add_argument("--verbose", action="store_true", help="Print details for skipped items.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    json_path = Path(args.json).expanduser().resolve()
    images_dir = Path(args.images).expanduser().resolve()

    all_gts, all_preds, stats = collect_predictions_and_gts(
        json_path=json_path,
        images_dir=images_dir,
        model_id=args.model_id,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        limit=args.limit,
        device=args.device,
        verbose=args.verbose,
    )
    metrics = compute_metrics(all_gts, all_preds, iou_thresh=args.iou_thresh)

    if args.save_json:
        maybe_save_json(Path(args.save_json).expanduser().resolve(), all_gts, all_preds, stats, metrics)

    print("\n" + "=" * 48)
    print("Validation with GroundingDINO Tiny Succeeded!")
    print(f"Images evaluated:           {stats['images_evaluated']}")
    print(f"Unique query classes:       {len({gt['query_key'] for gt in all_gts})}")
    print(f"Queries evaluated:          {stats['queries_evaluated']}")
    print(f"Model forward passes:       {stats['model_forward_passes']}")
    print(f"Prompt chunks used:         {stats['prompt_chunks']}")
    print(f"Skipped missing images:     {stats['skipped_missing_images']}")
    print(f"Skipped invalid queries:    {stats['skipped_invalid_queries']}")
    print(f"Unmatched detection labels: {stats['unmatched_detection_labels']}")
    print("-" * 48)
    print(f"Ground Truths (targets):    {metrics['total_gts']}")
    print(f"Total Predictions made:     {metrics['total_preds']}")
    print(f"True Positives (matched):   {metrics['true_positives']}")
    print("-" * 48)
    print(f"Precision@{args.iou_thresh}:          {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"Recall@{args.iou_thresh}:             {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"F1-Score@{args.iou_thresh}:           {metrics['f1']:.4f} ({metrics['f1'] * 100:.2f}%)")
    print(f"mAP@{args.iou_thresh}:                {metrics['map50']:.4f} ({metrics['map50'] * 100:.2f}%)")
    print(
        f"REC@{args.iou_thresh}:                {metrics['rec']:.4f} ({metrics['rec'] * 100:.2f}%) "
        f"[{metrics['rec_matched_queries']}/{metrics['rec_total_queries']}]"
    )
    print("=" * 48)


if __name__ == "__main__":
    main()
