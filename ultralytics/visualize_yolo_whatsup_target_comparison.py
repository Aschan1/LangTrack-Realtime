import argparse
import importlib.util
import json
import random
import re
import site
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
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
from ultralytics.utils.relation_head import (
    load_relation_head_checkpoint,
    score_options_with_relation_head,
)


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
PAIRWISE_RELATIONS = (
    ("left_of", "to the left of"),
    ("right_of", "to the right of"),
    ("above", "above"),
    ("below", "below"),
    ("on", "on"),
    ("under", "under"),
    ("front", "in front of"),
    ("behind", "behind"),
)


@dataclass(frozen=True)
class Detection:
    det_id: int
    label: str
    conf: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class RelationOption:
    target: str
    relation: str
    anchor: str

    @property
    def is_pairwise(self):
        return True


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


def get_anchor_gt_boxes(anns_for_image, anchor_label):
    if not anchor_label:
        return []

    return [
        xywh_to_xyxy(ann)
        for ann in anns_for_image
        if RUN_YOLO.get_target_label(ann) == anchor_label
    ]


def collect_detections_by_label(result, local_labels):
    detections_by_label = defaultdict(list)
    pred_boxes = result.boxes.xyxy.detach().cpu().numpy().tolist()
    pred_classes = result.boxes.cls.detach().cpu().numpy().tolist()
    pred_confs = result.boxes.conf.detach().cpu().numpy().tolist()

    for det_id, (box, cls_idx, conf) in enumerate(zip(pred_boxes, pred_classes, pred_confs)):
        cls_idx = int(cls_idx)
        if cls_idx < 0 or cls_idx >= len(local_labels):
            continue

        label = local_labels[cls_idx]
        detections_by_label[label].append(
            Detection(
                det_id=det_id,
                label=label,
                conf=float(conf),
                box=tuple(float(value) for value in box),
            )
        )

    return detections_by_label


def collect_label_predictions(detections_by_label, query_label, gt_boxes=None, *, forced=False):
    predictions = []
    gt_boxes = list(gt_boxes or [])
    for detection in detections_by_label.get(query_label, []):
        box = list(detection.box)

        best_iou = None
        if gt_boxes:
            best_iou = max(calculate_iou(box, gt_box) for gt_box in gt_boxes)

        predictions.append(
            {
                "box": box,
                "conf": float(detection.conf),
                "iou": best_iou,
                "forced": forced,
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
        collect_detections_by_label(result, [query_label]),
        query_label,
        gt_boxes,
        forced=True,
    )
    if not forced_predictions:
        return []
    return [max(forced_predictions, key=lambda item: item["conf"])]


def predict_relation_with_adapter(detections_by_label, target_label, anchor_label, relation_head, image_width, image_height):
    if relation_head is None:
        return "adapter unavailable"
    if not detections_by_label.get(target_label) or not detections_by_label.get(anchor_label):
        return "n/a (missing target or anchor)"

    options = [
        RelationOption(target=target_label, relation=relation_code, anchor=anchor_label)
        for relation_code, _ in PAIRWISE_RELATIONS
    ]
    scores = score_options_with_relation_head(
        options,
        detections_by_label,
        image_width,
        image_height,
        relation_head,
    )
    best_index = max(range(len(scores)), key=scores.__getitem__)
    return f"{PAIRWISE_RELATIONS[best_index][1]} ({scores[best_index]:.3f})"


def run_single_query_set(
    yolo_model,
    img_path,
    labels,
    target_label,
    anchor_label,
    target_gt_boxes,
    anchor_gt_boxes,
    relation_head,
    image_width,
    image_height,
    args,
):
    yolo_model.set_classes(labels)
    result = yolo_model.predict(
        str(img_path),
        conf=args.conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        verbose=False,
    )[0]
    detections_by_label = collect_detections_by_label(result, labels)

    target_predictions = collect_label_predictions(
        detections_by_label,
        target_label,
        target_gt_boxes,
    )
    anchor_predictions = collect_label_predictions(
        detections_by_label,
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
    if args.force_prediction and anchor_label in labels and not anchor_predictions:
        anchor_predictions = get_forced_best_prediction(
            yolo_model,
            img_path,
            anchor_label,
            anchor_gt_boxes,
            args.nms_iou,
        )

    predicted_relation = (
        predict_relation_with_adapter(
            detections_by_label,
            target_label,
            anchor_label,
            relation_head,
            image_width,
            image_height,
        )
        if anchor_label in labels
        else "n/a (target-only)"
    )

    return target_predictions, anchor_predictions, predicted_relation


def draw_block_text(cv_img, lines, *, top, left, font_scale=0.55, thickness=2):
    max_width = 0
    line_height = 22
    for line in lines:
        (width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        max_width = max(max_width, width)

    bottom = top + 12 + line_height * len(lines)
    right = left + max_width + 16
    cv2.rectangle(cv_img, (left, top), (right, bottom), (24, 24, 24), -1)

    for index, line in enumerate(lines, 1):
        y = top + index * line_height
        cv2.putText(
            cv_img,
            line,
            (left + 8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )


def draw_header(cv_img, ann, target_label, anchor_label, mode_title):
    keywords = ann.get("keywords", {})
    relation = (keywords.get("spatial_relation", "") or "").strip()
    phrase = (ann.get("phrase", "") or "").strip()

    lines = [mode_title]
    lines.extend(textwrap.wrap(f"Phrase: {phrase}", width=55) or ["Phrase:"])
    lines.extend(
        textwrap.wrap(
            f"Target: {target_label} | Relation: {relation or '-'} | Anchor: {anchor_label or '-'}",
            width=55,
        )
    )
    draw_block_text(cv_img, lines, top=12, left=12)


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


def draw_panel_summary(cv_img, labels, target_predictions, anchor_predictions, predicted_relation):
    lines = [
        f"Queries: {', '.join(labels)}",
        f"Target preds: {len(target_predictions)}",
        f"Anchor preds: {len(anchor_predictions)}",
        f"Pred relation: {predicted_relation}",
    ]
    draw_block_text(cv_img, lines, top=110, left=12, font_scale=0.5, thickness=1)


def add_panel_divider(left_panel, right_panel):
    height = max(left_panel.shape[0], right_panel.shape[0])
    divider = 255 * (left_panel[:height, :1].copy() * 0 + 1)
    divider[:] = (40, 40, 40)
    return cv2.hconcat([left_panel, divider, right_panel])


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize indoor WhatSup-style YOLO predictions side-by-side: "
            "target+anchor queries versus target-only queries."
        )
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
        default=str(REPO_ROOT / "visualizations_yolo_whatsup_target_comparison"),
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
        help="Force one best-effort prediction for missing queried labels.",
    )
    parser.add_argument(
        "--max-draw-per-label",
        type=int,
        default=8,
        help="Maximum number of target or anchor predictions to draw per panel.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    random.seed(args.seed)

    json_path, images_dir = resolve_inputs(args.json, args.images)
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
    print("Loading YOLOE model...")
    yolo_model = YOLOE(str(args.weights)).to(device)
    relation_head_path = Path(args.relation_head)
    relation_head = None
    if relation_head_path.is_file():
        relation_head = load_relation_head_checkpoint(relation_head_path, device=device)
        print(f"Using spatial adapter: {relation_head_path}")
    else:
        print(f"Spatial adapter checkpoint not found, relation prediction disabled: {relation_head_path}")

    saved = 0
    for index, ann in enumerate(samples, 1):
        img_id = str(ann["image_id"])
        img_path = images_dir / f"{img_id}.jpg"
        if not img_path.exists():
            print(f"[{index}/{sample_count}] Missing image: {img_path}")
            continue

        anns_for_image = img_to_anns.get(img_id, [])
        target_label = RUN_YOLO.get_query_label(ann, args.query_mode)
        anchor_label = RUN_YOLO.get_anchor_label(ann)
        if not target_label or not anchor_label:
            print(f"[{index}/{sample_count}] Skipping image {img_id} due to empty target/anchor labels.")
            continue

        base_img = cv2.imread(str(img_path))
        if base_img is None:
            print(f"[{index}/{sample_count}] Failed to read image via OpenCV: {img_path}")
            continue

        target_gt_boxes = [xywh_to_xyxy(ann)]
        anchor_gt_boxes = get_anchor_gt_boxes(anns_for_image, anchor_label)
        image_height, image_width = base_img.shape[:2]

        try:
            labels_both = RUN_YOLO.ordered_unique([target_label, anchor_label])
            labels_target_only = [target_label]

            target_preds_both, anchor_preds_both, predicted_relation_both = run_single_query_set(
                yolo_model,
                img_path,
                labels_both,
                target_label,
                anchor_label,
                target_gt_boxes,
                anchor_gt_boxes,
                relation_head,
                image_width,
                image_height,
                args,
            )
            target_preds_only, anchor_preds_only, predicted_relation_only = run_single_query_set(
                yolo_model,
                img_path,
                labels_target_only,
                target_label,
                anchor_label,
                target_gt_boxes,
                anchor_gt_boxes,
                relation_head,
                image_width,
                image_height,
                args,
            )

            left_img = base_img.copy()
            right_img = base_img.copy()

            for panel_img, panel_title in (
                (left_img, "Mode: target + anchor"),
                (right_img, "Mode: target only"),
            ):
                draw_header(panel_img, ann, target_label, anchor_label, panel_title)
                draw_gt_boxes(panel_img, target_gt_boxes, (0, 255, 255), "TARGET GT")
                if anchor_gt_boxes:
                    draw_gt_boxes(panel_img, anchor_gt_boxes, (255, 255, 0), "ANCHOR GT")

            draw_predictions(
                left_img,
                target_preds_both,
                label_prefix="TARGET",
                base_color=(0, 0, 255),
                match_color=(0, 200, 0),
                match_iou=args.match_iou,
                has_gt=True,
                max_draw=args.max_draw_per_label,
            )
            draw_predictions(
                left_img,
                anchor_preds_both,
                label_prefix="ANCHOR",
                base_color=(255, 0, 180),
                match_color=(255, 180, 0),
                match_iou=args.match_iou,
                has_gt=bool(anchor_gt_boxes),
                max_draw=args.max_draw_per_label,
            )
            draw_panel_summary(
                left_img,
                labels_both,
                target_preds_both,
                anchor_preds_both,
                predicted_relation_both,
            )

            draw_predictions(
                right_img,
                target_preds_only,
                label_prefix="TARGET",
                base_color=(0, 0, 255),
                match_color=(0, 200, 0),
                match_iou=args.match_iou,
                has_gt=True,
                max_draw=args.max_draw_per_label,
            )
            draw_panel_summary(
                right_img,
                labels_target_only,
                target_preds_only,
                anchor_preds_only,
                predicted_relation_only,
            )

            combined_img = add_panel_divider(left_img, right_img)

            region_id = ann.get("region_id", f"idx{index}")
            phrase_slug = sanitize_filename(ann.get("phrase", ""))
            output_path = (
                output_dir
                / f"whatsup_compare_{index}_img{img_id}_region{region_id}_{phrase_slug}.jpg"
            )
            cv2.imwrite(str(output_path), combined_img)
            print(
                f"[{index}/{sample_count}] Saved: {output_path} | "
                f"both(target={len(target_preds_both)}, anchor={len(anchor_preds_both)}) "
                f"target_only(target={len(target_preds_only)})"
            )
            saved += 1
        except Exception as exc:
            print(f"[{index}/{sample_count}] Error processing image {img_id}: {exc}")

    print(f"Done. Saved {saved}/{sample_count} visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
