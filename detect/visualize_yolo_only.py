import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ULTRALYTICS_ROOT = REPO_ROOT / "src" / "ultralytics"

# Prefer the full local source package and avoid namespace shadowing from this script directory.
while str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if SRC_ULTRALYTICS_ROOT.is_dir() and str(SRC_ULTRALYTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ULTRALYTICS_ROOT))
elif str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLOE


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


def get_combined_label(ann):
    target = ann.get("keywords", {}).get("target", "")
    attributes = ann.get("keywords", {}).get("attributes", [])

    if attributes:
        return f"{' '.join(attributes)} {target}".strip()
    return target.strip()


def build_local_labels(anns):
    seen = set()
    labels = []
    for ann in anns:
        label = get_combined_label(ann)
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def draw_query_header(cv_img, query_label):
    short_query = query_label if len(query_label) <= 70 else query_label[:67] + "..."
    label = f"Query: {short_query}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(cv_img, (10, 10), (20 + w, 20 + h), (20, 20, 20), -1)
    cv2.putText(cv_img, label, (15, 15 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return cv_img


def draw_visualization(cv_img, gt_box, predictions, query_label, match_iou):
    draw_query_header(cv_img, query_label)

    if not predictions:
        gx, gy, gw, gh = gt_box
        gx1, gy1, gx2, gy2 = int(gx), int(gy), int(gx + gw), int(gy + gh)
        cv2.rectangle(cv_img, (gx1, gy1), (gx2, gy2), (0, 255, 255), 3)
        cv2.putText(cv_img, "GT", (gx1, max(24, gy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        return cv_img

    best_idx = -1
    best_iou = -1.0
    for i, (_, _, iou) in enumerate(predictions):
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    for i, (box, conf, iou) in enumerate(predictions):
        x1, y1, x2, y2 = map(int, box)
        is_match = i == best_idx and iou >= match_iou
        color = (0, 255, 0) if is_match else (0, 0, 255)
        thickness = 3 if is_match else 2

        status = "MATCH" if is_match else "MISS"
        label = f"{status} c={conf:.2f} iou={iou:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - 20)
        cv2.rectangle(cv_img, (x1, top), (x1 + w, y1), color, -1)
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(cv_img, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    gx, gy, gw, gh = gt_box
    gx1, gy1, gx2, gy2 = int(gx), int(gy), int(gx + gw), int(gy + gh)
    cv2.rectangle(cv_img, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
    cv2.putText(cv_img, "GT", (gx1, max(24, gy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    return cv_img


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize plain YOLOE predictions using the run_yolo setup.")
    parser.add_argument("--json", default=None, help="Path to dataset JSON.")
    parser.add_argument("--images", default=None, help="Directory with <image_id>.jpg files.")
    parser.add_argument("--output-dir", default="visualizations_yolo_only", help="Output directory.")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of sampled annotations to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--weights",
        default=str(REPO_ROOT / "yoloe-26l-seg.pt"),
        help="Path to YOLOE weights.",
    )
    parser.add_argument("--conf", type=float, default=0.01, help="YOLOE confidence threshold.")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="YOLOE NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold used to mark the best same-class prediction as a match.",
    )
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

    print("Loading YOLOE model...")
    yolo_model = YOLOE(str(args.weights)).to(device)

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
        local_labels = build_local_labels(anns_for_image)
        query_label = get_combined_label(ann)
        if not local_labels or not query_label:
            print(f"[{i}/{sample_count}] Skipping due to empty labels for image {img_id}.")
            continue

        try:
            cv_img = cv2.imread(str(img_path))
            if cv_img is None:
                print(f"[{i}/{sample_count}] Failed to read image via OpenCV: {img_path}")
                continue

            yolo_model.set_classes(local_labels)
            result = yolo_model.predict(
                str(img_path),
                conf=args.conf,
                iou=args.nms_iou,
                max_det=args.max_det,
                verbose=False,
            )[0]

            predictions = []
            gt_xyxy = [ann["x"], ann["y"], ann["x"] + ann["width"], ann["y"] + ann["height"]]
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
                predictions.append((box, float(conf), calculate_iou(box, gt_xyxy)))

            predictions.sort(key=lambda item: item[1], reverse=True)

            gt_box = [ann["x"], ann["y"], ann["width"], ann["height"]]
            vis_img = draw_visualization(cv_img, gt_box, predictions, query_label, args.match_iou)

            region_id = ann.get("region_id", f"idx{i}")
            output_path = output_dir / f"yolo_vis_{i}_img{img_id}_region{region_id}.jpg"
            cv2.imwrite(str(output_path), vis_img)
            print(f"[{i}/{sample_count}] Saved: {output_path} | preds_for_query={len(predictions)}")
            saved += 1
        except Exception as exc:
            print(f"[{i}/{sample_count}] Error processing image {img_id}: {exc}")

    print(f"Done. Saved {saved}/{sample_count} visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
