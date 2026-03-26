import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor

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


def try_import_wedetect_ref_model_cls():
    ensure_wedetect_repo_on_path()

    try:
        from wedetect_ref.models.qwen3vl_referring import Qwen3VLGroundingForConditionalGeneration

        return Qwen3VLGroundingForConditionalGeneration, "wedetect_ref.models.qwen3vl_referring"
    except Exception:
        pass

    try:
        from transformers import Qwen3VLGroundingForConditionalGeneration

        return Qwen3VLGroundingForConditionalGeneration, "transformers"
    except Exception as exc:
        raise RuntimeError(
            "Could not import Qwen3VLGroundingForConditionalGeneration. "
            "Install transformers>=4.57.1 or set WEDETECT_REPO=/path/to/WeDetect."
        ) from exc


def try_import_process_vision_info():
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


def get_combined_label(ann):
    target = ann.get("keywords", {}).get("target", "")
    attributes = ann.get("keywords", {}).get("attributes", [])
    if attributes:
        return f"{' '.join(attributes)} {target}".strip()
    return target.strip()


def get_target_label(ann):
    return (ann.get("keywords", {}).get("target", "") or "").strip()


def get_structured_query_text(ann):
    keywords = ann.get("keywords", {}) if isinstance(ann, dict) else {}
    target = (keywords.get("target", "") or "").strip()
    attributes = keywords.get("attributes", []) or []
    if not isinstance(attributes, list):
        attributes = [str(attributes)]
    attributes = [str(a).strip() for a in attributes if str(a).strip()]
    anchor_object = (keywords.get("anchor_object", "") or "").strip()
    spatial_relation = (keywords.get("spatial_relation", "") or "").strip()
    payload = {
        "target": target,
        "attributes": attributes,
        "anchor_object": anchor_object,
        "spatial_relation": spatial_relation,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def get_query_label(ann, query_mode):
    if query_mode == "combined":
        return get_combined_label(ann)
    if query_mode == "target":
        return get_target_label(ann)
    if query_mode == "structured":
        return get_structured_query_text(ann)
    raise ValueError(f"Unsupported query_mode: {query_mode}")


def build_queries_for_image(anns, query_mode):
    seen = set()
    queries = []
    for ann in anns:
        q = get_query_label(ann, query_mode)
        if q and q not in seen:
            seen.add(q)
            queries.append(q)
    return queries


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
    print("Loading YOLOE proposal model...")
    yolo_model = YOLOE(str(weights_path)).to(device)
    text_model_variant = getattr(yolo_model.model, "text_model", "unknown")
    print(f"YOLOE text model for proposals: {text_model_variant}")
    return yolo_model


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
                {"type": "text", "text": query_text},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": proposal_str}]},
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
    return pred.logits.sigmoid()[proposal_positions].view(-1)


def resolve_visualize_inputs(json_arg, images_arg):
    json_candidates = [
        Path(json_arg).expanduser() if json_arg else None,
        Path(os.getenv("VISUALIZE_JSON", "")).expanduser() if os.getenv("VISUALIZE_JSON") else None,
        REPO_ROOT / "yolo_dataset" / "home_ovd_keywords_json_full.json",
        REPO_ROOT / "yolo_dataset" / "indoors_subset" / "filtered_indoors_LM_vg.json",
        REPO_ROOT / "yolo_dataset" / "indoors_subset" / "filtered_indoors_LM.json",
        REPO_ROOT / "yolo_dataset" / "indoors_subset" / "filtered_indoors.json",
    ]
    image_candidates = [
        Path(images_arg).expanduser() if images_arg else None,
        Path(os.getenv("VISUALIZE_IMAGES", "")).expanduser() if os.getenv("VISUALIZE_IMAGES") else None,
        REPO_ROOT / "yolo_dataset" / "filtered_images",
        REPO_ROOT / "yolo_dataset" / "indoors_subset" / "images",
    ]

    json_file = next((p for p in json_candidates if p and p.is_file()), None)
    images_dir = next((p for p in image_candidates if p and p.is_dir()), None)

    if json_file is None:
        available = sorted(str(p) for p in (REPO_ROOT / "yolo_dataset").rglob("*.json"))
        raise FileNotFoundError(
            "Could not find dataset JSON. Use --json or VISUALIZE_JSON. "
            f"Available JSON files: {available}"
        )
    if images_dir is None:
        raise FileNotFoundError("Could not find images directory. Use --images or VISUALIZE_IMAGES.")

    return str(json_file), str(images_dir)


def draw_visualization(cv_img, gt_box, predictions, query_label):
    gx, gy, gw, gh = gt_box
    gx1, gy1, gx2, gy2 = int(gx), int(gy), int(gx + gw), int(gy + gh)
    cv2.rectangle(cv_img, (gx1, gy1), (gx2, gy2), (0, 255, 255), 3)
    cv2.putText(cv_img, "GT", (gx1, max(0, gy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if not predictions:
        return cv_img

    best_idx = max(range(len(predictions)), key=lambda i: predictions[i][1])
    short_query = query_label if len(query_label) <= 40 else query_label[:37] + "..."

    for i, (box, score) in enumerate(predictions):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if i == best_idx else (0, 0, 255)
        thickness = 3 if i == best_idx else 2
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, thickness)
        label = f"{short_query} {score:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(cv_img, (x1, max(0, y1 - 20)), (x1 + w, y1), color, -1)
        cv2.putText(cv_img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return cv_img


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize WeDetectRef predictions on sampled annotations.")
    parser.add_argument("--json", default=None, help="Path to dataset JSON.")
    parser.add_argument("--images", default=None, help="Directory with <image_id>.jpg files.")
    parser.add_argument("--output-dir", default="visualizations_wedetect_ref", help="Output directory.")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of sampled annotations to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--query-mode", choices=["combined", "target", "structured"], default="combined")
    parser.add_argument(
        "--wedetect-ref-checkpoint",
        default=os.getenv("WEDETECT_REF_CHECKPOINT", "fushh7/WeDetect-Ref-2B"),
        help="HF model id or local path for WeDetectRef model.",
    )
    parser.add_argument("--proposal-conf", type=float, default=0.05, help="YOLOE proposal confidence threshold.")
    parser.add_argument("--proposal-max-det", type=int, default=80, help="Max proposals per image from YOLOE.")
    parser.add_argument("--score-thre", type=float, default=-1.0, help="If >=0, keep proposals above this score.")
    parser.add_argument("--topk-per-query", type=int, default=5, help="Used when score_thre < 0.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    json_file, images_dir = resolve_visualize_inputs(args.json, args.images)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Reading dataset: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("Dataset is empty. Nothing to visualize.")
        return

    img_to_anns = {}
    for item in data:
        img_to_anns.setdefault(str(item["image_id"]), []).append(item)

    processor, model, process_vision_info, object_token_index = load_wedetect_ref_model(
        model_id=args.wedetect_ref_checkpoint,
        device=device,
    )
    yolo_model = load_yoloe_proposal_model(REPO_ROOT / "yoloe-26l-seg.pt", device)

    sample_count = min(args.num_samples, len(data))
    samples = random.sample(data, sample_count)
    print(f"Sampled {sample_count} annotations for visualization.")

    saved = 0
    for i, ann in enumerate(samples, 1):
        img_id = str(ann["image_id"])
        img_path = Path(images_dir) / f"{img_id}.jpg"
        if not img_path.exists():
            print(f"[{i}/{sample_count}] Missing image: {img_path}")
            continue

        anns_for_image = img_to_anns.get(img_id, [])
        # Match the direct YOLOE baseline by proposing with attribute+target labels.
        proposal_labels = build_queries_for_image(anns_for_image, query_mode="combined")
        query_label = get_query_label(ann, args.query_mode)
        if not proposal_labels or not query_label:
            print(f"[{i}/{sample_count}] Skipping due to empty labels for image {img_id}.")
            continue

        try:
            image_pil = Image.open(img_path).convert("RGB")
            cv_img = cv2.imread(str(img_path))
            if cv_img is None:
                print(f"[{i}/{sample_count}] Failed to read image via OpenCV: {img_path}")
                continue

            yolo_model.set_classes(proposal_labels)
            yolo_result = yolo_model.predict(
                str(img_path),
                conf=args.proposal_conf,
                iou=0.7,
                max_det=args.proposal_max_det,
                verbose=False,
            )[0]
            proposal_boxes = yolo_result.boxes.xyxy.detach().cpu().numpy().tolist()
            if not proposal_boxes:
                print(f"[{i}/{sample_count}] No proposals for image {img_id}.")
                continue

            pred_scores = score_query_with_wedetect_ref(
                processor=processor,
                model=model,
                process_vision_info=process_vision_info,
                object_token_index=object_token_index,
                image_pil=image_pil,
                proposals=proposal_boxes,
                query_text=query_label,
                device=device,
            )
            if pred_scores.numel() == 0:
                print(f"[{i}/{sample_count}] Empty WeDetectRef scores for image {img_id}.")
                continue

            if args.score_thre < 0:
                k = min(max(int(args.topk_per_query), 1), int(pred_scores.numel()))
                top_val, top_idx = torch.topk(pred_scores.view(-1), k, dim=0)
                keep_indices = top_idx.detach().cpu().tolist()
                keep_scores = top_val.detach().float().cpu().tolist()
            else:
                keep_mask = pred_scores > args.score_thre
                keep_indices = torch.nonzero(keep_mask, as_tuple=False).view(-1).detach().cpu().tolist()
                keep_scores = pred_scores[keep_mask].detach().float().cpu().tolist()

            predictions = []
            for idx, score in zip(keep_indices, keep_scores):
                if idx < len(proposal_boxes):
                    predictions.append((proposal_boxes[idx], float(score)))

            gt_box = [ann["x"], ann["y"], ann["width"], ann["height"]]
            vis_img = draw_visualization(cv_img, gt_box, predictions, query_label)

            region_id = ann.get("region_id", f"idx{i}")
            output_path = output_dir / f"wedetect_vis_{i}_img{img_id}_region{region_id}.jpg"
            cv2.imwrite(str(output_path), vis_img)
            print(f"[{i}/{sample_count}] Saved: {output_path}")
            saved += 1
        except Exception as exc:
            print(f"[{i}/{sample_count}] Error processing image {img_id}: {exc}")

    print(f"Done. Saved {saved}/{sample_count} visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
