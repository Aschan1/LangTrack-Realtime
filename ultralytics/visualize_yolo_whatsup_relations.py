import argparse
import importlib.util
import json
import os
import random
import re
import site
import sys
import textwrap
from pathlib import Path

import cv2
import torch

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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ULTRALYTICS_PACKAGE_ROOT = REPO_ROOT / "src" / "ultralytics"

while str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
while str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(ULTRALYTICS_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PACKAGE_ROOT))

from ultralytics import YOLOE


def load_run_yolo_module():
    module_path = REPO_ROOT / "ultralytics" / "run_yolo.py"
    spec = importlib.util.spec_from_file_location("run_yolo_local", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load run_yolo module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUN_YOLO = load_run_yolo_module()


def sanitize_filename(text, max_length=80):
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")
    if not cleaned:
        cleaned = "sample"
    return cleaned[:max_length]


def resolve_inputs(json_arg, images_arg):
    json_path = Path(RUN_YOLO.resolve_json_path(json_arg)).expanduser()
    if not json_path.is_file():
        raise FileNotFoundError(f"Could not find dataset JSON: {json_path}")

    images_dir = (
        Path(images_arg).expanduser()
        if images_arg
        else REPO_ROOT / "yolo_dataset" / "indoors_subset" / "images"
    )
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Could not find images directory: {images_dir}")

    return json_path, images_dir


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


def xywh_to_xyxy(ann):
    x, y, w, h = ann["x"], ann["y"], ann["width"], ann["height"]
    return [x, y, x + w, y + h]


def build_local_labels(anns, query_mode, detect_anchor_objects):
    local_eval_labels = RUN_YOLO.ordered_unique(
        RUN_YOLO.get_query_label(ann, query_mode) for ann in anns
    )
    local_labels = list(local_eval_labels)
    if detect_anchor_objects:
        local_labels = RUN_YOLO.ordered_unique(
            list(local_eval_labels) + [RUN_YOLO.get_anchor_label(ann) for ann in anns]
        )
    return local_labels


def collect_label_predictions(result, local_labels, query_label, gt_boxes=None, *, forced=False):
    predictions = []
    gt_boxes = list(gt_boxes or [])

    pred_boxes = result.boxes.xyxy.detach().cpu().numpy().tolist()
    pred_classes = result.boxes.cls.detach().cpu().numpy().tolist()
    pred_confs = result.boxes.conf.detach().cpu().numpy().tolist()

    for box, cls_idx, conf in zip(pred_boxes, pred_classes, pred_confs):
        cls_idx = int(cls_idx)
        if cls_idx < 0 or cls_idx >= len(local_labels):
            continue

        pred_label = local_labels[cls_idx]
        if pred_label != query_label:
            continue

        best_iou = None
        if gt_boxes:
            best_iou = max(calculate_iou(box, gt_box) for gt_box in gt_boxes)

        predictions.append(
            {
                "box": box,
                "conf": float(conf),
                "iou": best_iou,
                "forced": forced,
                "label": pred_label,
            }
        )

    return predictions


def get_forced_best_prediction(yolo_model, img_path, query_label, gt_boxes, nms_iou):
    yolo_model.set_classes([query_label])
    result = yolo_model.predict(
        str(img_path),
        conf=0.0,
        iou=nms_iou,
        max_det=10,
        verbose=False,
    )[0]
    forced_predictions = collect_label_predictions(
        result,
        [query_label],
        query_label,
        gt_boxes,
        forced=True,
    )
    if not forced_predictions:
        return []
    return [max(forced_predictions, key=lambda item: item["conf"])]


def get_anchor_gt_boxes(anns_for_image, anchor_label):
    if not anchor_label:
        return []

    return [
        xywh_to_xyxy(ann)
        for ann in anns_for_image
        if RUN_YOLO.get_target_label(ann) == anchor_label
    ]


def draw_header(cv_img, ann, query_label, anchor_label):
    keywords = ann.get("keywords", {})
    relation = (keywords.get("spatial_relation", "") or "").strip()
    phrase = (ann.get("phrase", "") or "").strip()

    lines = []
    lines.extend(textwrap.wrap(f"Phrase: {phrase}", width=80) or ["Phrase:"])
    lines.extend(
        textwrap.wrap(
            f"Target: {query_label} | Relation: {relation or '-'} | Anchor: {anchor_label or '-'}",
            width=80,
        )
    )

    max_width = 0
    line_height = 22
    for line in lines:
        (width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        max_width = max(max_width, width)

    top = 12
    left = 12
    bottom = top + 12 + line_height * len(lines)
    right = left + max_width + 16

    cv2.rectangle(cv_img, (left, top), (right, bottom), (24, 24, 24), -1)

    for index, line in enumerate(lines, 1):
        y = top + index * line_height
        cv2.putText(cv_img, line, (left + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_gt_boxes(cv_img, gt_boxes, color, label_prefix):
    for index, box in enumerate(gt_boxes, 1):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
        label = label_prefix if len(gt_boxes) == 1 else f"{label_prefix} {index}"
        cv2.putText(
            cv_img,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def draw_predictions(cv_img, predictions, label_prefix, base_color, match_color, match_iou, has_gt, max_draw):
    predictions = sorted(predictions, key=lambda item: item["conf"], reverse=True)[:max_draw]
    best_idx = -1
    best_iou = -1.0

    if has_gt:
        for index, prediction in enumerate(predictions):
            iou = prediction["iou"] if prediction["iou"] is not None else -1.0
            if iou > best_iou:
                best_iou = iou
                best_idx = index

    for index, prediction in enumerate(predictions):
        box = prediction["box"]
        conf = prediction["conf"]
        iou = prediction["iou"]
        forced = prediction["forced"]

        is_match = has_gt and index == best_idx and iou is not None and iou >= match_iou
        color = match_color if is_match else base_color
        thickness = 3 if is_match else 2

        status = "MATCH" if is_match else ("MISS" if has_gt else "DET")
        if forced:
            status = f"FORCED {status}"

        text = f"{label_prefix} {status} c={conf:.2f}"
        if iou is not None:
            text += f" iou={iou:.2f}"

        x1, y1, x2, y2 = [int(v) for v in box]
        (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        top = max(0, y1 - 18)
        cv2.rectangle(cv_img, (x1, top), (x1 + width + 6, y1), color, -1)
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            cv_img,
            text,
            (x1 + 3, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Visualize indoor WhatSup-style YOLO predictions with both target and anchor detections."
    )
    parser.add_argument(
        "--json",
        default="whatsup",
        help=(
            "Annotation JSON preset, file name, or path. "
            "Default is the WhatSup-filtered indoors_subset JSON."
        ),
    )
    parser.add_argument("--images", default=None, help="Directory with <image_id>.jpg files.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "visualizations_yolo_whatsup_relations"),
        help="Output directory.",
    )
    parser.add_argument("--num-samples", type=int, default=20, help="Number of sampled annotations to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--weights",
        default=str(REPO_ROOT / "yoloe-26l-seg.pt"),
        help="Path to YOLOE weights.",
    )
    parser.add_argument(
        "--relation-head",
        default=str(REPO_ROOT / "outputs" / "relation_head_indoors_6500.pt"),
        help="Path to the learned spatial-adapter / relation-head checkpoint.",
    )
    parser.add_argument("--conf", type=float, default=0.01, help="YOLOE confidence threshold.")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="YOLOE NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold used to mark the best prediction as a match when GT boxes are available.",
    )
    parser.add_argument(
        "--query-mode",
        choices=RUN_YOLO.QUERY_MODES,
        default="combined",
        help="How to build target queries, matching ultralytics/run_yolo.py.",
    )
    parser.add_argument(
        "--force-prediction",
        action="store_true",
        help="Force one best-effort prediction for missing target or anchor labels.",
    )
    parser.add_argument(
        "--max-draw-per-label",
        type=int,
        default=8,
        help="Maximum number of target or anchor predictions to draw.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    random.seed(args.seed)

    json_path, images_dir = resolve_inputs(args.json, args.images)
    detect_anchor_objects = RUN_YOLO.should_detect_anchor_objects(json_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    relation_rows = [ann for ann in data if RUN_YOLO.get_anchor_label(ann)]
    if not relation_rows:
        raise ValueError(f"No annotations with anchor_object found in {json_path}")

    img_to_anns = {}
    for item in data:
        img_to_anns.setdefault(str(item["image_id"]), []).append(item)

    sample_count = min(args.num_samples, len(relation_rows))
    samples = random.sample(relation_rows, sample_count)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Reading dataset: {json_path}")
    print(f"Using images: {images_dir}")
    if detect_anchor_objects:
        print("WhatSup JSON detected: visualizing both target and anchor detections.")
    print("Loading YOLOE model...")
    yolo_model = YOLOE(str(args.weights)).to(device)

    saved = 0
    for index, ann in enumerate(samples, 1):
        img_id = str(ann["image_id"])
        img_path = images_dir / f"{img_id}.jpg"
        if not img_path.exists():
            print(f"[{index}/{sample_count}] Missing image: {img_path}")
            continue

        anns_for_image = img_to_anns.get(img_id, [])
        local_labels = build_local_labels(anns_for_image, args.query_mode, detect_anchor_objects)
        target_label = RUN_YOLO.get_query_label(ann, args.query_mode)
        anchor_label = RUN_YOLO.get_anchor_label(ann)
        if not local_labels or not target_label or not anchor_label:
            print(f"[{index}/{sample_count}] Skipping image {img_id} due to empty target/anchor labels.")
            continue

        cv_img = cv2.imread(str(img_path))
        if cv_img is None:
            print(f"[{index}/{sample_count}] Failed to read image via OpenCV: {img_path}")
            continue

        target_gt_boxes = [xywh_to_xyxy(ann)]
        anchor_gt_boxes = get_anchor_gt_boxes(anns_for_image, anchor_label)

        try:
            yolo_model.set_classes(local_labels)
            result = yolo_model.predict(
                str(img_path),
                conf=args.conf,
                iou=args.nms_iou,
                max_det=args.max_det,
                verbose=False,
            )[0]

            target_predictions = collect_label_predictions(
                result,
                local_labels,
                target_label,
                target_gt_boxes,
            )
            anchor_predictions = collect_label_predictions(
                result,
                local_labels,
                anchor_label,
                anchor_gt_boxes,
            )

            if args.force_prediction and not target_predictions:
                target_predictions = get_forced_best_prediction(
                    yolo_model,
                    img_path,
                    target_label,
                    target_gt_boxes,
                    args.nms_iou,
                )
            if args.force_prediction and not anchor_predictions:
                anchor_predictions = get_forced_best_prediction(
                    yolo_model,
                    img_path,
                    anchor_label,
                    anchor_gt_boxes,
                    args.nms_iou,
                )

            draw_header(cv_img, ann, target_label, anchor_label)
            draw_gt_boxes(cv_img, target_gt_boxes, (0, 255, 255), "TARGET GT")
            if anchor_gt_boxes:
                draw_gt_boxes(cv_img, anchor_gt_boxes, (255, 255, 0), "ANCHOR GT")

            draw_predictions(
                cv_img,
                target_predictions,
                label_prefix="TARGET",
                base_color=(0, 0, 255),
                match_color=(0, 200, 0),
                match_iou=args.match_iou,
                has_gt=True,
                max_draw=args.max_draw_per_label,
            )
            draw_predictions(
                cv_img,
                anchor_predictions,
                label_prefix="ANCHOR",
                base_color=(255, 0, 180),
                match_color=(255, 180, 0),
                match_iou=args.match_iou,
                has_gt=bool(anchor_gt_boxes),
                max_draw=args.max_draw_per_label,
            )

            region_id = ann.get("region_id", f"idx{index}")
            phrase_slug = sanitize_filename(ann.get("phrase", ""))
            output_path = output_dir / f"whatsup_yolo_vis_{index}_img{img_id}_region{region_id}_{phrase_slug}.jpg"
            cv2.imwrite(str(output_path), cv_img)
            print(
                f"[{index}/{sample_count}] Saved: {output_path} | "
                f"target_preds={len(target_predictions)} anchor_preds={len(anchor_predictions)} "
                f"anchor_gt_candidates={len(anchor_gt_boxes)}"
            )
            saved += 1
        except Exception as exc:
            print(f"[{index}/{sample_count}] Error processing image {img_id}: {exc}")

    print(f"Done. Saved {saved}/{sample_count} visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
