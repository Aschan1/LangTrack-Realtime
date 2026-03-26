"""
Evaluate YOLOE-26L-seg with the lightweight target-anchor relation rescoring block.

This runner is intentionally closer to the other run_{model}.py scripts than the
earlier per-annotation dump utility:
  - annotations are grouped by image
  - inference runs once per unique query for that image
  - predictions and GT boxes are collected into a shared metric pass
  - IoU-based precision / recall / F1 / mAP / REC are reported

Because the relation block is defined only for exactly two prompts
(`target_prompt`, `anchor_prompt`), this script evaluates one unique
(target, anchor, spatial_relation) query at a time for each image rather than
running one prompt set for all annotations in the image at once.

The current relation module only uses target + anchor prompts. The
`spatial_relation` string is still preserved in the evaluation query identity so
metrics remain aligned with the dataset rows even though the model does not yet
consume that relation text directly.
"""

import argparse
import inspect
import json
import os
import site
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Prefer the active environment's packages over user-site overlays, which can
# otherwise inject an incompatible torchvision build into YOLOE imports.
USER_SITE_PATHS = {
    Path(path).resolve()
    for path in {
        site.getusersitepackages(),
        *(site.getsitepackages() if hasattr(site, "getsitepackages") else []),
    }
    if isinstance(path, str) and ".local" in path
}
sys.path[:] = [
    path
    for path in sys.path
    if not any(Path(path).resolve() == user_site for user_site in USER_SITE_PATHS if path)
]

# Ensure imports resolve to the full local Ultralytics package.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ULTRALYTICS_PACKAGE_ROOT = REPO_ROOT / "src" / "ultralytics"
while str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
while str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(ULTRALYTICS_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(1, str(SCRIPT_DIR))

from ultralytics import YOLOE
from ultralytics.utils.target_anchor_relation_adapter import load_target_anchor_relation_adapter
from refexp_metrics import compute_rec_metrics
from run_yolo_offline import build_embedding_cache, compose_embedding

INDOORS_SUBSET_DIR = REPO_ROOT / "yolo_dataset" / "indoors_subset"
DEFAULT_JSON = INDOORS_SUBSET_DIR / "filtered_indoors_LM_vg_nonnull_spatial_relations.json"
DEFAULT_IMAGES = INDOORS_SUBSET_DIR / "images"
DEFAULT_WEIGHTS = REPO_ROOT / "yoloe-26l-seg.pt"
DEFAULT_CACHE_PATH = REPO_ROOT / "detect" / "embedding_cache.pt"


def supports_relation_inference(model):
    """Return True when the loaded model forward path accepts relation-specific kwargs."""
    try:
        signature = inspect.signature(model.model.predict)
    except (TypeError, ValueError, AttributeError):
        return False

    parameters = signature.parameters
    if "use_target_anchor_relation" in parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())


def ordered_unique(items):
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


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


def parse_keywords(ann):
    keywords = ann.get("keywords", {}) if isinstance(ann, dict) else {}
    target = (keywords.get("target", "") or "").strip()
    anchor = (keywords.get("anchor_object", "") or "").strip()
    relation = (keywords.get("spatial_relation", "") or "").strip()
    return target, anchor, relation


def get_query_key(ann):
    target, anchor, relation = parse_keywords(ann)
    return json.dumps(
        {"target": target, "anchor_object": anchor, "spatial_relation": relation},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def get_gt_box(ann):
    x, y, w, h = ann["x"], ann["y"], ann["width"], ann["height"]
    return [x, y, x + w, y + h]


def resolve_json_path(json_arg):
    candidate = Path(json_arg)
    if candidate.is_absolute() or candidate.exists():
        return candidate

    indoors_candidate = INDOORS_SUBSET_DIR / json_arg
    if indoors_candidate.exists():
        return indoors_candidate

    return candidate


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOE target-anchor relation inference on the indoors subset."
    )
    parser.add_argument(
        "--json",
        default=str(DEFAULT_JSON),
        help="Annotation JSON path. Defaults to the non-null spatial-relation indoors subset file.",
    )
    parser.add_argument(
        "--images",
        default=str(DEFAULT_IMAGES),
        help="Directory containing images named <image_id>.jpg.",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="YOLOE segmentation checkpoint to evaluate.",
    )
    parser.add_argument(
        "--cache-path",
        default=str(DEFAULT_CACHE_PATH),
        help="Offline embedding cache path used for target and anchor prompts.",
    )
    parser.add_argument(
        "--adapter-path",
        default="",
        help="Optional target-anchor relation adapter checkpoint to load before evaluation.",
    )
    parser.add_argument("--conf", type=float, default=0.05, help="YOLOE confidence threshold.")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for matching.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of images to evaluate.")
    parser.add_argument(
        "--relation-topk-target",
        type=int,
        default=20,
        help="Top-K target candidates used by the relation rescoring block.",
    )
    parser.add_argument(
        "--relation-topk-anchor",
        type=int,
        default=10,
        help="Top-M anchor candidates used by the relation rescoring block.",
    )
    parser.add_argument(
        "--relation-lambda",
        type=float,
        default=0.5,
        help="Lambda multiplier for adding aggregated relation support to target logits.",
    )
    parser.add_argument(
        "--relation-agg",
        choices=("logsumexp", "max"),
        default="logsumexp",
        help="Reducer used to aggregate anchor support per target candidate.",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional path to save predictions, ground truths, and summary metrics as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image progress details for skipped/malformed entries.",
    )
    return parser


def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_to_anns = defaultdict(list)
    for ann in data:
        img_to_anns[str(ann["image_id"])].append(ann)

    return data, dict(img_to_anns)


def make_global_label_map(img_to_anns):
    global_label_to_id = {}
    for anns in img_to_anns.values():
        for ann in anns:
            query_key = get_query_key(ann)
            if query_key not in global_label_to_id:
                global_label_to_id[query_key] = len(global_label_to_id)
    return global_label_to_id


def extract_target_predictions(result, img_id, target_index_to_queries):
    if result.boxes is None or len(result.boxes) == 0:
        return []

    raw_boxes = result.boxes.xyxy.detach().cpu().numpy()
    raw_classes = result.boxes.cls.detach().cpu().numpy()
    raw_confs = result.boxes.conf.detach().cpu().numpy()

    predictions = []
    for p_box, p_cls_local, p_conf in zip(raw_boxes, raw_classes, raw_confs):
        local_index = int(p_cls_local)
        query_infos = target_index_to_queries.get(local_index)
        if not query_infos:
            continue

        box = p_box.tolist() if hasattr(p_box, "tolist") else list(p_box)
        for query_info in query_infos:
            predictions.append(
                {
                    "img_id": img_id,
                    "cls": query_info["global_cls_id"],
                    "box": box,
                    "conf": float(p_conf),
                    "p_label": query_info["target_label"],
                    "query_key": query_info["query_key"],
                }
            )

    return predictions


def collect_predictions_and_gts(
    json_path,
    images_dir,
    weights,
    cache_path,
    adapter_path,
    conf_thresh,
    relation_topk_target,
    relation_topk_anchor,
    relation_lambda,
    relation_agg,
    limit=0,
    verbose=False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_path = Path(cache_path)

    if not cache_path.exists():
        print(f"Offline cache not found at {cache_path}. Building cache first...")
        build_embedding_cache(str(json_path), str(cache_path), str(weights))

    cache_data = torch.load(str(cache_path), map_location="cpu", weights_only=True)
    cache_embeddings = cache_data["embeddings"]
    embed_dim = cache_data["embed_dim"]
    print(f"Loaded offline embedding cache: {len(cache_embeddings)} words, dim={embed_dim}")

    print("Loading YOLOE target-anchor relation model...")
    model = YOLOE(str(weights)).to(device)
    if adapter_path:
        payload = load_target_anchor_relation_adapter(model, adapter_path, strict=False)
        print(f"Loaded relation adapter from: {adapter_path} (epoch={payload.get('epoch')})")
        if payload.get("missing_keys") or payload.get("unexpected_keys"):
            print(
                "Adapter load details: "
                f"missing={len(payload.get('missing_keys', []))}, "
                f"unexpected={len(payload.get('unexpected_keys', []))}"
            )
    head = model.model.model[-1]
    relation_inference_supported = supports_relation_inference(model)
    if not relation_inference_supported:
        print(
            "Warning: current local YOLOE model does not implement target-anchor relation inference kwargs. "
            "Falling back to plain target/anchor prompting without relation rescoring."
        )

    _, img_to_anns = load_dataset(json_path)
    if limit > 0:
        img_items = list(img_to_anns.items())[:limit]
        img_to_anns = dict(img_items)

    global_label_to_id = make_global_label_map(img_to_anns)
    print(f"Unique query classes: {len(global_label_to_id)}")
    print(f"Validation Start! {len(img_to_anns)} images in total...")

    all_gts = []
    all_preds = []
    skipped_missing_image = 0
    skipped_invalid_query = 0
    query_pairs_evaluated = 0
    model_forward_passes = 0
    offline_prompt_hits = 0
    offline_prompt_fallbacks = 0
    prompt_embedding_cache = {}

    for img_id, anns in tqdm(img_to_anns.items(), desc="Inference with YOLOE relation"):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            skipped_missing_image += 1
            if verbose:
                print(f"Skipping missing image: {img_path}")
            continue

        query_to_anns = defaultdict(list)
        for ann in anns:
            query_to_anns[get_query_key(ann)].append(ann)

        pair_to_query_infos = defaultdict(list)
        for query_key, query_anns in query_to_anns.items():
            target, anchor, relation = parse_keywords(query_anns[0])
            if not target or not anchor:
                skipped_invalid_query += len(query_anns)
                if verbose:
                    print(
                        f"Skipping malformed query for image {img_id}: "
                        f"target={target!r}, anchor={anchor!r}, relation={relation!r}"
                    )
                continue

            global_cls_id = global_label_to_id[query_key]
            for ann in query_anns:
                all_gts.append(
                    {
                        "img_id": img_id,
                        "cls": global_cls_id,
                        "box": get_gt_box(ann),
                        "used": False,
                        "query_key": query_key,
                    }
                )

            pair_to_query_infos[(target, anchor)].append(
                {
                    "query_key": query_key,
                    "global_cls_id": global_cls_id,
                    "target_label": target,
                }
            )

        if not pair_to_query_infos:
            continue

        prompts_to_cache = ordered_unique([prompt for pair in pair_to_query_infos for prompt in pair])
        missing_prompts = [prompt for prompt in prompts_to_cache if prompt not in prompt_embedding_cache]
        if missing_prompts:
            raw_embeddings = []
            resolved_prompts = []
            for prompt in missing_prompts:
                raw_embedding = compose_embedding(prompt, [], cache_embeddings, embed_dim)
                if raw_embedding is None:
                    offline_prompt_fallbacks += 1
                    continue
                raw_embeddings.append(raw_embedding)
                resolved_prompts.append(prompt)
                offline_prompt_hits += 1

            if raw_embeddings:
                raw_tensor = torch.stack(raw_embeddings).unsqueeze(0).to(device)
                with torch.no_grad():
                    final_embeddings = head.get_tpe(raw_tensor).squeeze(0).detach().cpu()
                for prompt, embedding in zip(resolved_prompts, final_embeddings):
                    prompt_embedding_cache[prompt] = embedding

        use_online_fallback = any(
            target not in prompt_embedding_cache or anchor not in prompt_embedding_cache
            for target, anchor in pair_to_query_infos
        )

        prompt_labels = []
        pair_indices = []
        target_index_to_queries = {}
        for target, anchor in pair_to_query_infos:
            target_cls_idx = len(prompt_labels)
            anchor_cls_idx = target_cls_idx + 1
            prompt_labels.extend([target, anchor])
            pair_indices.append([target_cls_idx, anchor_cls_idx])
            target_index_to_queries[target_cls_idx] = pair_to_query_infos[(target, anchor)]

        if use_online_fallback:
            model.set_classes(prompt_labels)
        else:
            prompt_embeddings = torch.stack([prompt_embedding_cache[prompt] for prompt in prompt_labels]).unsqueeze(0).to(device)
            model.set_classes(prompt_labels, prompt_embeddings)

        predict_kwargs = {
            "source": img_path,
            "conf": conf_thresh,
            "verbose": False,
        }
        if relation_inference_supported:
            predict_kwargs.update(
                {
                    "use_target_anchor_relation": True,
                    "target_anchor_pair_indices": pair_indices,
                    "relation_topk_target": relation_topk_target,
                    "relation_topk_anchor": relation_topk_anchor,
                    "relation_lambda": relation_lambda,
                    "relation_agg": relation_agg,
                }
            )

        results = model.predict(**predict_kwargs)
        model_forward_passes += 1
        query_pairs_evaluated += len(pair_to_query_infos)

        result = results[0]
        all_preds.extend(
            extract_target_predictions(
                result=result,
                img_id=img_id,
                target_index_to_queries=target_index_to_queries,
            )
        )

    stats = {
        "images_evaluated": len(img_to_anns),
        "query_pairs_evaluated": query_pairs_evaluated,
        "model_forward_passes": model_forward_passes,
        "unique_query_classes": len(global_label_to_id),
        "skipped_missing_image": skipped_missing_image,
        "skipped_invalid_query": skipped_invalid_query,
        "offline_prompt_hits": offline_prompt_hits,
        "offline_prompt_fallbacks": offline_prompt_fallbacks,
        "unique_cached_prompts": len(prompt_embedding_cache),
    }
    return all_gts, all_preds, stats


def compute_metrics(all_gts, all_preds, iou_thresh):
    unique_classes = {gt["cls"] for gt in all_gts}
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

        for i, pred in enumerate(preds_c):
            img_gts = [g for g in gts_c if g["img_id"] == pred["img_id"]]
            ovmax = -1.0
            jmax = -1

            for j, gt in enumerate(img_gts):
                iou = calculate_iou(pred["box"], gt["box"])
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

            if ovmax >= iou_thresh and jmax >= 0:
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
        aps.append(float(ap))

    map50 = float(np.mean(aps)) if aps else 0.0
    precision = float(global_tp / total_preds) if total_preds > 0 else 0.0
    recall = float(global_tp / total_gts) if total_gts > 0 else 0.0
    f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    rec_summary = compute_rec_metrics(all_gts, all_preds, iou_thresh, calculate_iou)

    return {
        "ground_truths": total_gts,
        "predictions": total_preds,
        "true_positives": int(global_tp),
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "map": map50,
        **rec_summary,
    }


def maybe_save_json(save_path, summary, stats, all_gts, all_preds):
    if not save_path:
        return

    payload = {
        "summary": summary,
        "stats": stats,
        "ground_truths": all_gts,
        "predictions": all_preds,
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation JSON to: {save_path}")


def main():
    args = build_arg_parser().parse_args()

    json_path = resolve_json_path(args.json)
    images_dir = Path(args.images)
    weights = Path(args.weights)

    print(f"Using annotation JSON: {json_path}")
    print(f"Using images directory: {images_dir}")
    print(f"Using weights: {weights}")
    print(f"Using offline cache: {args.cache_path}")
    if args.adapter_path:
        print(f"Using adapter checkpoint: {args.adapter_path}")

    all_gts, all_preds, stats = collect_predictions_and_gts(
        json_path=json_path,
        images_dir=images_dir,
        weights=weights,
        cache_path=args.cache_path,
        adapter_path=args.adapter_path,
        conf_thresh=args.conf,
        relation_topk_target=args.relation_topk_target,
        relation_topk_anchor=args.relation_topk_anchor,
        relation_lambda=args.relation_lambda,
        relation_agg=args.relation_agg,
        limit=args.limit,
        verbose=args.verbose,
    )
    summary = compute_metrics(all_gts, all_preds, args.iou_thresh)

    print("\n" + "=" * 48)
    print("Validation with YOLOE Target-Anchor Relation Succeeded!")
    print(f"Images evaluated:           {stats['images_evaluated']}")
    print(f"Unique query classes:       {stats['unique_query_classes']}")
    print(f"Query pairs evaluated:      {stats['query_pairs_evaluated']}")
    print(f"Model forward passes:       {stats['model_forward_passes']}")
    print(f"Skipped missing images:     {stats['skipped_missing_image']}")
    print(f"Skipped invalid queries:    {stats['skipped_invalid_query']}")
    print(f"Offline prompt hits:        {stats['offline_prompt_hits']}")
    print(f"Offline prompt fallbacks:   {stats['offline_prompt_fallbacks']}")
    print(f"Unique cached prompts:      {stats['unique_cached_prompts']}")
    print("-" * 48)
    print(f"Ground Truths (targets):    {summary['ground_truths']}")
    print(f"Total Predictions made:     {summary['predictions']}")
    print(f"True Positives (matched):   {summary['true_positives']}")
    print("-" * 48)
    print(f"Precision@{args.iou_thresh}:          {summary['precision']:.4f} ({summary['precision'] * 100:.2f}%)")
    print(f"Recall@{args.iou_thresh}:             {summary['recall']:.4f} ({summary['recall'] * 100:.2f}%)")
    print(f"F1-Score@{args.iou_thresh}:           {summary['f1']:.4f} ({summary['f1'] * 100:.2f}%)")
    print(f"mAP@{args.iou_thresh}:                {summary['map']:.4f} ({summary['map'] * 100:.2f}%)")
    print(
        f"REC@{args.iou_thresh}:                {summary['rec']:.4f} ({summary['rec'] * 100:.2f}%) "
        f"[{summary['rec_matched_queries']}/{summary['rec_total_queries']}]"
    )
    print("=" * 48)

    maybe_save_json(args.save_json, summary, stats, all_gts, all_preds)


if __name__ == "__main__":
    main()
