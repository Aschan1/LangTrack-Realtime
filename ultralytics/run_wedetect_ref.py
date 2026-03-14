import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Ensure imports resolve to installed `ultralytics` package, not local namespace-shadow paths.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for shadow_path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    while shadow_path in sys.path:
        sys.path.remove(shadow_path)

from transformers import AutoProcessor
from ultralytics import YOLOE


def resolve_wedetect_repo_hint():
    env_hint = os.getenv("WEDETECT_REPO", "").strip()
    if env_hint:
        return Path(env_hint).expanduser().resolve()
    default_hint = Path.home() / "projects" / "WeDetect"
    if default_hint.is_dir():
        return default_hint.resolve()
    return None


def ensure_wedetect_repo_on_path():
    repo_path = resolve_wedetect_repo_hint()
    if repo_path and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    return repo_path


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


def get_combined_label(ann):
    target = ann.get("keywords", {}).get("target", "")
    attributes = ann.get("keywords", {}).get("attributes", [])
    if attributes:
        return f"{' '.join(attributes)} {target}".strip()
    return target.strip()


def get_target_label(ann):
    return (ann.get("keywords", {}).get("target", "") or "").strip()


def try_import_wedetect_ref_model_cls():
    """Prefer official WeDetect repo class; fallback to transformers>=4.57.1 export if available."""
    ensure_wedetect_repo_on_path()

    try:
        from wedetect_ref.models.qwen3vl_referring import Qwen3VLGroundingForConditionalGeneration

        source = "wedetect_ref.models.qwen3vl_referring"
        return Qwen3VLGroundingForConditionalGeneration, source
    except Exception:
        pass

    try:
        from transformers import Qwen3VLGroundingForConditionalGeneration

        source = "transformers"
        return Qwen3VLGroundingForConditionalGeneration, source
    except Exception as exc:
        raise RuntimeError(
            "Could not import Qwen3VLGroundingForConditionalGeneration. "
            "Install transformers>=4.57.1 or set WEDETECT_REPO=/path/to/WeDetect (or add it to PYTHONPATH)."
        ) from exc


def try_import_process_vision_info():
    """Use official vision preprocessing when available; fallback to simple image-only behavior."""
    ensure_wedetect_repo_on_path()

    try:
        from wedetect_ref.models.vision_process import process_vision_info

        return process_vision_info, "wedetect_ref.models.vision_process"
    except Exception:
        pass

    try:
        from qwen_vl_utils import process_vision_info

        return process_vision_info, "qwen_vl_utils"
    except Exception:
        pass

    def simple_process_vision_info(conversations, image_patch_size=16):
        del image_patch_size
        if isinstance(conversations, dict):
            conversations = [conversations]
        image_inputs = []
        for message in conversations:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for ele in content:
                if not isinstance(ele, dict):
                    continue
                if ele.get("type") == "image" and "image" in ele:
                    image_inputs.append(ele["image"])
                elif "image" in ele:
                    image_inputs.append(ele["image"])
        return image_inputs if image_inputs else None, None

    return simple_process_vision_info, "simple_fallback"


def load_wedetect_ref_model(model_id, device):
    model_cls, model_cls_source = try_import_wedetect_ref_model_cls()
    process_vision_info, process_source = try_import_process_vision_info()
    repo_hint = resolve_wedetect_repo_hint()

    print("Loading WeDetectRef model...")
    print(f"WeDetectRef checkpoint: {model_id}")
    print(f"WeDetect repo hint: {repo_hint if repo_hint else 'not set'}")
    print(f"Model class source: {model_cls_source}")
    print(f"Vision preprocess source: {process_source}")

    processor = AutoProcessor.from_pretrained(model_id)

    model_kwargs = {"torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32}
    if device == "cuda":
        model_kwargs["attn_implementation"] = os.getenv("WEDETECT_ATTN_IMPL", "flash_attention_2")

    try:
        model = model_cls.from_pretrained(model_id, **model_kwargs)
    except Exception as exc:
        if "attn_implementation" in model_kwargs:
            print(f"Warning: retrying model load without attn_implementation due to: {exc}")
            model_kwargs.pop("attn_implementation", None)
            model = model_cls.from_pretrained(model_id, **model_kwargs)
        else:
            raise

    model = model.to(device).eval()

    object_token_index = processor.tokenizer.convert_tokens_to_ids("<object>")
    if hasattr(model, "model"):
        setattr(model.model, "object_token_id", object_token_index)

    return processor, model, process_vision_info, object_token_index


def load_yoloe_proposal_model(weights_path, device):
    print("Loading proposal model (YOLOE)...")
    yolo_model = YOLOE(str(weights_path)).to(device)
    text_model_variant = os.getenv("YOLOE_TEXT_MODEL", "clip:ViT-B/32")
    yolo_model.model.text_model = text_model_variant
    print(f"YOLOE text model for proposals: {text_model_variant}")
    return yolo_model


def build_queries_for_image(anns, query_mode):
    seen = set()
    queries = []
    for ann in anns:
        q = get_combined_label(ann) if query_mode == "combined" else get_target_label(ann)
        if q and q not in seen:
            seen.add(q)
            queries.append(q)
    return queries


def build_ranked_queries_for_image(anns, query_mode, max_queries=None):
    """Build de-duplicated queries, optionally capped by within-image frequency."""
    counts = {}
    first_idx = {}

    for i, ann in enumerate(anns):
        q = get_combined_label(ann) if query_mode == "combined" else get_target_label(ann)
        if not q:
            continue
        counts[q] = counts.get(q, 0) + 1
        if q not in first_idx:
            first_idx[q] = i

    if not counts:
        return []

    queries = sorted(counts.keys(), key=lambda q: (-counts[q], first_idx[q]))
    if max_queries is not None and max_queries > 0:
        queries = queries[:max_queries]
    return queries


def score_query_with_wedetect_ref(
    processor,
    model,
    process_vision_info,
    object_token_index,
    image_pil,
    proposals,
    query_text,
    device,
):
    num_proposals = len(proposals)
    if num_proposals == 0:
        return torch.empty(0)

    proposal_str = "<object>" * num_proposals

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": copy.deepcopy(image_pil)},
                {"type": "text", "text": f'Please detect the "{query_text}" in the image'},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": proposal_str}],
        },
    ]

    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    texts = [processor.apply_chat_template(messages, tokenize=False)]

    model_inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
        do_resize=False,
    ).to(device)

    proposals_tensor = [torch.tensor(proposals, device=device, dtype=model.dtype)]
    ori_shapes = [image_pil.size]

    with torch.inference_mode():
        pred = model(
            **model_inputs,
            bboxes=copy.deepcopy(proposals_tensor),
            ori_shapes=ori_shapes,
            bboxes_id=object_token_index,
            image_inputs=image_inputs,
        )

    proposal_positions = model_inputs["input_ids"] == object_token_index
    pred_scores = pred.logits.sigmoid()[proposal_positions].view(-1)
    return pred_scores


def inference_wedetect_ref(
    json_path,
    images_dir,
    wedetect_ref_checkpoint,
    iou_thresh=0.5,
    score_thre=-1.0,
    query_mode="combined",
    proposal_conf=0.01,
    proposal_max_det=300,
    max_queries_per_image=0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    processor, model, process_vision_info, object_token_index = load_wedetect_ref_model(
        model_id=wedetect_ref_checkpoint,
        device=device,
    )

    yoloe_weights = REPO_ROOT / "yoloe-26l-seg.pt"
    yolo_model = load_yoloe_proposal_model(yoloe_weights, device)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_to_anns = {}
    for item in tqdm(data, desc="Loading JSON Doc"):
        img_to_anns.setdefault(str(item["image_id"]), []).append(item)

    global_label_to_id = {}
    for anns in img_to_anns.values():
        for ann in anns:
            label = get_combined_label(ann) if query_mode == "combined" else get_target_label(ann)
            if label and label not in global_label_to_id:
                global_label_to_id[label] = len(global_label_to_id)

    print(f"Total unique classes in dataset ({query_mode}): {len(global_label_to_id)}")
    print(f"Validation Start! {len(img_to_anns)} Pics in total...")

    all_gts = []
    all_preds = []
    images_with_no_props = 0

    for img_id, anns in tqdm(img_to_anns.items(), desc="Evaluating"):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue

        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        query_labels = build_ranked_queries_for_image(
            anns,
            query_mode=query_mode,
            max_queries=max_queries_per_image,
        )
        proposal_labels = build_queries_for_image(anns, query_mode="target")

        if not query_labels or not proposal_labels:
            continue

        for ann in anns:
            label = get_combined_label(ann) if query_mode == "combined" else get_target_label(ann)
            if not label:
                continue
            x, y, w, h = ann["x"], ann["y"], ann["width"], ann["height"]
            gt_box = [x, y, x + w, y + h]
            all_gts.append({"img_id": img_id, "cls": global_label_to_id[label], "box": gt_box, "used": False})

        # Proposal generation (YOLOE): WeDetectRef scores provided proposals.
        try:
            yolo_model.set_classes(proposal_labels)
            yolo_result = yolo_model.predict(
                img_path,
                conf=proposal_conf,
                iou=0.7,
                max_det=proposal_max_det,
                verbose=False,
            )[0]
            proposal_boxes = yolo_result.boxes.xyxy.detach().cpu().numpy().tolist()
        except Exception:
            proposal_boxes = []

        if not proposal_boxes:
            images_with_no_props += 1
            continue

        for query_label in query_labels:
            try:
                pred_scores = score_query_with_wedetect_ref(
                    processor=processor,
                    model=model,
                    process_vision_info=process_vision_info,
                    object_token_index=object_token_index,
                    image_pil=img_pil,
                    proposals=proposal_boxes,
                    query_text=query_label,
                    device=device,
                )
            except Exception:
                continue

            if pred_scores.numel() == 0:
                continue

            if score_thre < 0:
                top_val, top_idx = torch.topk(pred_scores.view(-1), 1, dim=0)
                keep_indices = top_idx.detach().cpu().tolist()
                keep_scores = top_val.detach().float().cpu().tolist()
            else:
                keep_mask = pred_scores > score_thre
                keep_indices = torch.nonzero(keep_mask, as_tuple=False).view(-1).detach().cpu().tolist()
                keep_scores = pred_scores[keep_mask].detach().float().cpu().tolist()

            for idx, score in zip(keep_indices, keep_scores):
                if idx >= len(proposal_boxes):
                    continue
                all_preds.append(
                    {
                        "img_id": img_id,
                        "cls": global_label_to_id[query_label],
                        "box": proposal_boxes[idx],
                        "conf": float(score),
                    }
                )

    print("Computing metrics...")

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

    map50 = np.mean(aps) if aps else 0.0
    precision = global_tp / total_preds if total_preds > 0 else 0.0
    recall = global_tp / total_gts if total_gts > 0 else 0.0
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    print("\n" + "=" * 40)
    print("Validation with WeDetectRef-2B Succeeded!")
    print(f"Ground Truths (Total Targets): {total_gts}")
    print(f"Total Predictions made: {total_preds}")
    print(f"Images with no proposals: {images_with_no_props}")
    print(f"True Positives (Matched): {int(global_tp)}")
    print("-" * 40)
    print(f"Precision@{iou_thresh}: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall@{iou_thresh}:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-Score@{iou_thresh}:   {f1_score:.4f} ({f1_score * 100:.2f}%)")
    print(f"mAP@{iou_thresh}:       {map50:.4f} ({map50 * 100:.2f}%)")
    print("=" * 40)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate WeDetectRef-2B on indoor subset JSON/images.")
    presets = {
        "fast": {"query_mode": "target", "proposal_conf": 0.1, "proposal_max_det": 50, "max_queries_per_image": 8},
        "balanced": {
            "query_mode": "target",
            "proposal_conf": 0.05,
            "proposal_max_det": 80,
            "max_queries_per_image": 12,
        },
        "quality": {
            "query_mode": "combined",
            "proposal_conf": 0.01,
            "proposal_max_det": 300,
            "max_queries_per_image": 0,
        },
    }
    parser.add_argument(
        "--json",
        default="yolo_dataset/indoors_subset/filtered_indoors_LM_vg.json",
        help="Path to annotation JSON",
    )
    parser.add_argument(
        "--images",
        default="yolo_dataset/indoors_subset/images",
        help="Directory containing images named <image_id>.jpg",
    )
    parser.add_argument(
        "--wedetect-ref-checkpoint",
        default=os.getenv("WEDETECT_REF_CHECKPOINT", "fushh7/WeDetect-Ref-2B"),
        help="HF model id or local path for WeDetectRef model",
    )
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for metric matching")
    parser.add_argument(
        "--score-thre",
        type=float,
        default=-1.0,
        help="Score threshold for WeDetectRef proposal scores. If <0, use top1 per query (official behavior).",
    )
    parser.add_argument(
        "--preset",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Speed/quality preset for query/proposal settings.",
    )
    parser.add_argument(
        "--query-mode",
        choices=["combined", "target"],
        default=None,
        help="Class construction mode for evaluation labels/queries. Overrides --preset when set.",
    )
    parser.add_argument(
        "--proposal-conf",
        type=float,
        default=None,
        help="YOLOE confidence threshold used to generate region proposals (higher is faster). Overrides --preset.",
    )
    parser.add_argument(
        "--proposal-max-det",
        type=int,
        default=None,
        help="Maximum number of YOLOE proposals per image (lower is faster/safer on 12GB). Overrides --preset.",
    )
    parser.add_argument(
        "--max-queries-per-image",
        type=int,
        default=None,
        help="Cap evaluated queries per image. 0 disables cap; lower values are much faster. Overrides --preset.",
    )
    args = parser.parse_args()

    preset = presets[args.preset]
    if args.query_mode is None:
        args.query_mode = preset["query_mode"]
    if args.proposal_conf is None:
        args.proposal_conf = preset["proposal_conf"]
    if args.proposal_max_det is None:
        args.proposal_max_det = preset["proposal_max_det"]
    if args.max_queries_per_image is None:
        args.max_queries_per_image = preset["max_queries_per_image"]

    return args


if __name__ == "__main__":
    args = parse_args()
    print(
        "Using preset/config: "
        f"preset={args.preset}, query_mode={args.query_mode}, proposal_conf={args.proposal_conf}, "
        f"proposal_max_det={args.proposal_max_det}, max_queries_per_image={args.max_queries_per_image}"
    )
    inference_wedetect_ref(
        json_path=args.json,
        images_dir=args.images,
        wedetect_ref_checkpoint=args.wedetect_ref_checkpoint,
        iou_thresh=args.iou_thresh,
        score_thre=args.score_thre,
        query_mode=args.query_mode,
        proposal_conf=args.proposal_conf,
        proposal_max_det=args.proposal_max_det,
        max_queries_per_image=args.max_queries_per_image,
    )
