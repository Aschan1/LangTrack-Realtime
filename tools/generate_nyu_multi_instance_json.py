#!/usr/bin/env python3
import argparse
import base64
import io
import json
import math
import os
import site
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageColor, ImageDraw, ImageFont
from scipy import ndimage
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")


MIN_COMPONENT_AREA = 500
MAX_TARGET_INSTANCES = 6
RELATION_ORDER = [
    "in front of",
    "under",
    "to the left of",
    "to the right of",
    "above",
    "below",
]
EXCLUDED_CLASS_NAMES = {"wall", "ceiling", "unknown"}
PROMPT_VERSION = "2026-03-25-v6"
DEFAULT_MODEL = (
    os.environ.get("OPENROUTER_MODEL")
    or os.environ.get("NYU_RELATION_MODEL")
    or os.environ.get("OPENAI_MODEL")
    or "gpt-4.1-mini"
)
DEFAULT_MAX_RETRIES = 3
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_PROVIDER = os.environ.get("OPENROUTER_PROVIDER", "").strip()
OPENROUTER_APP_URL = os.environ.get("OPENROUTER_APP_URL", "https://github.com/openai/codex")
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "LangTrack-Realtime")

NYU_DIR = REPO_ROOT / "nyu_dataset"
IMAGES_DIR = NYU_DIR / "images"
LABELS_DIR = NYU_DIR / "labels"
CLASS_NAMES_PATH = NYU_DIR / "class_names.json"
MAT_PATH = REPO_ROOT / "nyu_depth_v2_labeled.mat"
OUTPUT_PATH = NYU_DIR / "filtered_nyu_LM_vg_multi_instance.json"
OUTPUT_PATH_RGB = NYU_DIR / "filtered_nyu_LM_vg_multi_instance_rgb_only.json"
CACHE_DIR = NYU_DIR / "chatgpt_relation_cache"
CACHE_DIR_RGB = NYU_DIR / "chatgpt_relation_cache_rgb_only"
DEFAULT_YOLOE_WEIGHTS = REPO_ROOT / "yoloe-26l-seg.pt"

ANNOTATION_COLORS = (
    "#f94144",
    "#f3722c",
    "#f8961e",
    "#f9c74f",
    "#90be6d",
    "#43aa8b",
    "#577590",
    "#277da1",
)

YOLOE = None
RGB_DETECTOR = None
RGB_DETECTOR_WEIGHTS = None
RGB_DETECTOR_CLASSES = None


@dataclass
class Component:
    component_id: int | None
    class_id: int
    class_name: str
    x: int
    y: int
    width: int
    height: int
    area: int
    depth_median: float
    mask_centroid_x: float
    mask_centroid_y: float
    mask: np.ndarray | None = field(repr=False, compare=False, default=None)

    def prompt_dict(self) -> dict:
        if self.component_id is None:
            raise ValueError("Component IDs must be assigned before prompt serialization.")

        depth_value = None
        if math.isfinite(self.depth_median):
            depth_value = round(self.depth_median, 4)

        return {
            "component_id": self.component_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "locator_bbox": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "area_pixels": self.area,
            "depth_median_m": depth_value,
        }


@dataclass(frozen=True)
class TargetGroup:
    class_id: int
    class_name: str
    target_components: tuple[Component, ...]

    def prompt_dict(self) -> dict:
        return {
            "target_class_id": self.class_id,
            "target_class_name": self.class_name,
            "target_component_ids": [component.component_id for component in self.target_components],
            "disallowed_anchor_class_id": self.class_id,
        }


@dataclass(frozen=True)
class ClientConfig:
    client: OpenAI
    backend: str
    provider_slug: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate NYU multi-instance phrase annotations with at most one repeated target group per image, "
            "using a shared anchor that yields distinct relations across that group."
        )
    )
    parser.add_argument(
        "--image-id",
        help="Only process one NYU image id. Accepts values like 0, 0000, or 0000.png.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Process at most this many images after filtering. Useful for smoke tests.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model name to use for relation selection. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--component-source",
        choices=("rgb", "semantic"),
        default="rgb",
        help=(
            "How to extract candidate repeated objects before choosing at most one repeated target group per image: "
            "RGB detections only, or semantic connected components."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Where to write the generated JSON. Defaults to the full dataset path, or a safe per-image/per-subset path.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory used to cache one ChatGPT response per target group.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_YOLOE_WEIGHTS,
        help=f"YOLOE weights used when --component-source=rgb. Default: {DEFAULT_YOLOE_WEIGHTS}",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.05,
        help="YOLOE confidence threshold used for RGB-only component extraction.",
    )
    parser.add_argument(
        "--det-iou",
        type=float,
        default=0.7,
        help="YOLOE NMS IoU threshold used for RGB-only component extraction.",
    )
    parser.add_argument(
        "--det-max-det",
        type=int,
        default=300,
        help="Maximum detections returned per image in RGB-only mode.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached ChatGPT responses and fetch fresh ones.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Do not make API calls. Fail if a needed image response is not already cached.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="Optional pause between API calls to reduce rate-limit pressure.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retries for an API call. Default: {DEFAULT_MAX_RETRIES}",
    )
    return parser.parse_args()


def normalize_image_id(raw_value: str) -> int:
    normalized = raw_value.strip()
    if normalized.endswith(".png"):
        normalized = Path(normalized).stem
    return int(normalized)


def resolve_output_path(args: argparse.Namespace) -> Path:
    default_output = OUTPUT_PATH_RGB if args.component_source == "rgb" else OUTPUT_PATH
    if args.output_path is not None:
        return args.output_path
    if args.image_id is not None:
        image_id = normalize_image_id(args.image_id)
        suffix = "rgb_only" if args.component_source == "rgb" else "semantic"
        return NYU_DIR / f"filtered_nyu_LM_vg_multi_instance_{suffix}_{image_id:04d}.json"
    if args.max_images is not None:
        suffix = "rgb_only" if args.component_source == "rgb" else "semantic"
        return NYU_DIR / f"filtered_nyu_LM_vg_multi_instance_{suffix}_first_{args.max_images}.json"
    return default_output


def resolve_cache_dir(args: argparse.Namespace) -> Path:
    if args.cache_dir is not None:
        return args.cache_dir
    return CACHE_DIR_RGB if args.component_source == "rgb" else CACHE_DIR


def load_class_names() -> dict[int, str]:
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {int(key): str(value) for key, value in payload.items()}


def load_label_map(image_id: int) -> np.ndarray:
    label_path = LABELS_DIR / f"{image_id:04d}.png"
    return np.array(Image.open(label_path), dtype=np.int32)


def load_rgb_image(image_id: int) -> Image.Image:
    image_path = IMAGES_DIR / f"{image_id:04d}.png"
    return Image.open(image_path).convert("RGB")


def class_name_to_id_map() -> dict[str, int]:
    lookup: dict[str, int] = {}
    for class_id, class_name in load_class_names().items():
        normalized = class_name.strip()
        if not normalized or normalized in {"unlabeled"}:
            continue
        lookup.setdefault(normalized, class_id)
    return lookup


def ensure_yoloe_import():
    global YOLOE
    if YOLOE is not None:
        return YOLOE

    user_site_paths = {
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
        if not any(Path(path).resolve() == user_site for user_site in user_site_paths if path)
    ]

    script_dir = REPO_ROOT / "detect"
    ultralytics_package_root = REPO_ROOT / "src" / "ultralytics"
    while str(REPO_ROOT) in sys.path:
        sys.path.remove(str(REPO_ROOT))
    while str(script_dir) in sys.path:
        sys.path.remove(str(script_dir))
    if str(ultralytics_package_root) not in sys.path:
        sys.path.insert(0, str(ultralytics_package_root))
    if str(script_dir) not in sys.path:
        sys.path.insert(1, str(script_dir))

    from ultralytics import YOLOE as _YOLOE

    YOLOE = _YOLOE
    return YOLOE


def ensure_rgb_detector(weights: Path, detector_classes: list[str]):
    global RGB_DETECTOR, RGB_DETECTOR_WEIGHTS, RGB_DETECTOR_CLASSES
    yoloe_cls = ensure_yoloe_import()

    weights = weights.expanduser().resolve()
    if RGB_DETECTOR is not None and RGB_DETECTOR_WEIGHTS == weights and RGB_DETECTOR_CLASSES == tuple(detector_classes):
        return RGB_DETECTOR

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    model = yoloe_cls(str(weights)).to(device)
    model.set_classes(detector_classes)
    RGB_DETECTOR = model
    RGB_DETECTOR_WEIGHTS = weights
    RGB_DETECTOR_CLASSES = tuple(detector_classes)
    return RGB_DETECTOR


def extract_components(label_map: np.ndarray, depth_map: np.ndarray, class_id: int, class_name: str) -> list[Component]:
    labeled_components, count = ndimage.label(label_map == class_id)
    if count == 0:
        return []

    components: list[Component] = []
    for component_id, slices in enumerate(ndimage.find_objects(labeled_components), start=1):
        if slices is None:
            continue

        component_mask = labeled_components[slices] == component_id
        area = int(component_mask.sum())
        if area < MIN_COMPONENT_AREA:
            continue

        y_slice, x_slice = slices
        x = int(x_slice.start)
        y = int(y_slice.start)
        width = int(x_slice.stop - x_slice.start)
        height = int(y_slice.stop - y_slice.start)

        component_depths = depth_map[slices][component_mask]
        finite_depths = component_depths[np.isfinite(component_depths)]
        depth_median = float(np.median(finite_depths)) if finite_depths.size else math.inf
        mask_y, mask_x = np.nonzero(component_mask)
        if mask_x.size:
            mask_centroid_x = float(x + mask_x.mean())
            mask_centroid_y = float(y + mask_y.mean())
        else:
            mask_centroid_x = float(x + (width - 1) / 2.0)
            mask_centroid_y = float(y + (height - 1) / 2.0)

        components.append(
            Component(
                component_id=None,
                class_id=class_id,
                class_name=class_name,
                x=x,
                y=y,
                width=width,
                height=height,
                area=area,
                depth_median=depth_median,
                mask_centroid_x=mask_centroid_x,
                mask_centroid_y=mask_centroid_y,
                mask=component_mask.copy(),
            )
        )

    components.sort(key=lambda comp: (comp.x, comp.y, comp.width, comp.height))
    return components


def extract_components_from_rgb(
    image: Image.Image,
    *,
    class_name_to_id: dict[str, int],
    weights: Path,
    conf_thresh: float,
    iou_thresh: float,
    max_det: int,
) -> dict[int, list[Component]]:
    detector_classes = [
        class_name
        for class_name in sorted(class_name_to_id)
        if class_name not in EXCLUDED_CLASS_NAMES and class_name != "unlabeled"
    ]
    model = ensure_rgb_detector(weights, detector_classes)
    result = model.predict(
        source=np.array(image),
        conf=conf_thresh,
        iou=iou_thresh,
        max_det=max_det,
        verbose=False,
    )[0]

    components_by_class: dict[int, list[Component]] = defaultdict(list)
    if result.boxes is None or len(result.boxes) == 0:
        return {}

    pred_boxes = result.boxes.xyxy.detach().cpu().numpy()
    pred_classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    pred_confs = result.boxes.conf.detach().cpu().numpy()
    for box, class_index, conf in zip(pred_boxes, pred_classes, pred_confs, strict=False):
        if class_index < 0 or class_index >= len(detector_classes):
            continue
        class_name = detector_classes[class_index]
        class_id = class_name_to_id.get(class_name)
        if class_id is None:
            continue

        x1, y1, x2, y2 = [float(value) for value in box]
        x = max(0, int(round(x1)))
        y = max(0, int(round(y1)))
        width = max(1, int(round(x2 - x1)))
        height = max(1, int(round(y2 - y1)))
        area = int(width * height)
        if area < MIN_COMPONENT_AREA:
            continue

        components_by_class[class_id].append(
            Component(
                component_id=None,
                class_id=class_id,
                class_name=class_name,
                x=x,
                y=y,
                width=width,
                height=height,
                area=area,
                depth_median=math.inf,
                mask_centroid_x=float(x + width / 2.0),
                mask_centroid_y=float(y + height / 2.0),
                mask=None,
            )
        )

    for class_id in list(components_by_class):
        components_by_class[class_id].sort(key=lambda comp: (comp.x, comp.y, -comp.area))
    return dict(components_by_class)


def collect_rgb_image_ids(args: argparse.Namespace) -> list[int]:
    available_ids = sorted(int(path.stem) for path in IMAGES_DIR.glob("*.png"))
    if args.image_id is not None:
        image_id = normalize_image_id(args.image_id)
        if image_id not in set(available_ids):
            raise ValueError(f"image_id {image_id} is not available in {IMAGES_DIR}")
        return [image_id]
    if args.max_images is not None:
        available_ids = available_ids[: args.max_images]
    return available_ids


def assign_component_ids(components_by_class: dict[int, list[Component]]) -> list[Component]:
    all_components: list[Component] = []
    next_id = 1
    for class_id in sorted(components_by_class):
        for component in components_by_class[class_id]:
            component.component_id = next_id
            all_components.append(component)
            next_id += 1
    return all_components


def build_target_groups(components_by_class: dict[int, list[Component]]) -> tuple[dict[int, TargetGroup], int]:
    target_groups: dict[int, TargetGroup] = {}
    dropped_too_many = 0

    for class_id in sorted(components_by_class):
        target_components = components_by_class[class_id]
        if len(target_components) < 2:
            continue
        if len(target_components) > MAX_TARGET_INSTANCES:
            dropped_too_many += 1
            continue

        target_groups[class_id] = TargetGroup(
            class_id=class_id,
            class_name=target_components[0].class_name,
            target_components=tuple(target_components),
        )

    return target_groups, dropped_too_many


def component_center(component: Component) -> tuple[float, float]:
    return (
        float(component.x + component.width / 2.0),
        float(component.y + component.height / 2.0),
    )


def overlap_length(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def bbox_iou(first: Component, second: Component) -> float:
    inter_w = overlap_length(first.x, first.x + first.width, second.x, second.x + second.width)
    inter_h = overlap_length(first.y, first.y + first.height, second.y, second.y + second.height)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    union_area = float(first.area + second.area - inter_area)
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def normalized_center_distance(first: Component, second: Component, image_size: tuple[int, int]) -> float:
    image_width, image_height = image_size
    first_cx, first_cy = component_center(first)
    second_cx, second_cy = component_center(second)
    dx = (first_cx - second_cx) / max(float(image_width), 1.0)
    dy = (first_cy - second_cy) / max(float(image_height), 1.0)
    return math.hypot(dx, dy)


def pair_relation_scores(target: Component, anchor: Component, image_size: tuple[int, int]) -> dict[str, float]:
    image_width, image_height = image_size
    target_cx, target_cy = component_center(target)
    anchor_cx, anchor_cy = component_center(anchor)
    dx = (target_cx - anchor_cx) / max(float(image_width), 1.0)
    dy = (target_cy - anchor_cy) / max(float(image_height), 1.0)

    horizontal_overlap = overlap_length(target.x, target.x + target.width, anchor.x, anchor.x + anchor.width)
    vertical_overlap = overlap_length(target.y, target.y + target.height, anchor.y, anchor.y + anchor.height)
    horizontal_overlap /= max(1.0, float(min(target.width, anchor.width)))
    vertical_overlap /= max(1.0, float(min(target.height, anchor.height)))
    alignment_bonus = max(0.0, 1.0 - min(abs(dx), 1.0))
    support_bonus = max(0.0, 1.0 - min(abs(dy), 1.0))

    depth_score = 0.0
    if math.isfinite(target.depth_median) and math.isfinite(anchor.depth_median):
        depth_delta = anchor.depth_median - target.depth_median
        depth_scale = max(abs(anchor.depth_median), abs(target.depth_median), 1e-6)
        depth_score = max(0.0, depth_delta / depth_scale)

    return {
        "to the left of": max(0.0, -dx) + 0.25 * vertical_overlap,
        "to the right of": max(0.0, dx) + 0.25 * vertical_overlap,
        "above": max(0.0, -dy) + 0.25 * horizontal_overlap,
        "below": max(0.0, dy) + 0.25 * horizontal_overlap,
        "under": max(0.0, dy) + 0.40 * horizontal_overlap + 0.15 * alignment_bonus,
        "in front of": depth_score + 0.10 * support_bonus,
    }


def score_anchor_for_target_group(
    target_group: TargetGroup,
    anchor: Component,
    image_size: tuple[int, int],
) -> float:
    best_relations: list[str] = []
    best_scores: list[float] = []
    margins: list[float] = []
    target_distances: list[float] = []

    for target in target_group.target_components:
        relation_scores = pair_relation_scores(target, anchor, image_size)
        ranked_relations = sorted(relation_scores.items(), key=lambda item: item[1], reverse=True)
        best_relation, best_score = ranked_relations[0]
        second_score = ranked_relations[1][1] if len(ranked_relations) > 1 else 0.0
        best_relations.append(best_relation)
        best_scores.append(best_score)
        margins.append(best_score - second_score)
        target_distances.append(normalized_center_distance(target, anchor, image_size))

    distinct_relation_count = len(set(best_relations))
    distinct_ratio = distinct_relation_count / max(len(best_relations), 1)
    mean_best_score = sum(best_scores) / max(len(best_scores), 1)
    min_best_score = min(best_scores, default=0.0)
    mean_margin = sum(margins) / max(len(margins), 1)
    mean_distance = sum(target_distances) / max(len(target_distances), 1)

    return (
        3.0 * distinct_ratio
        + 1.8 * mean_best_score
        + 0.9 * min_best_score
        + 0.7 * mean_margin
        - 1.2 * mean_distance
    )


def score_target_group_cleanliness(
    target_group: TargetGroup,
    all_components: list[Component],
    image_size: tuple[int, int],
) -> float:
    image_width, image_height = image_size
    image_area = max(1.0, float(image_width * image_height))
    targets = list(target_group.target_components)

    pair_count = 0
    overlap_penalty = 0.0
    separation_bonus = 0.0
    for index, first in enumerate(targets):
        for second in targets[index + 1 :]:
            pair_count += 1
            overlap_penalty += bbox_iou(first, second)
            separation_bonus += normalized_center_distance(first, second, image_size)
    if pair_count > 0:
        overlap_penalty /= pair_count
        separation_bonus /= pair_count

    area_fractions = [component.area / image_area for component in targets]
    tiny_penalty = sum(1.0 for area_fraction in area_fractions if area_fraction < 0.004) / max(len(area_fractions), 1)
    huge_penalty = sum(1.0 for area_fraction in area_fractions if area_fraction > 0.20) / max(len(area_fractions), 1)

    anchor_candidates = [
        component
        for component in all_components
        if component.class_id != target_group.class_id and component.class_name not in EXCLUDED_CLASS_NAMES
    ]
    best_anchor_score = 0.0
    for anchor in anchor_candidates:
        best_anchor_score = max(best_anchor_score, score_anchor_for_target_group(target_group, anchor, image_size))

    return (
        2.4 * best_anchor_score
        + 1.2 * separation_bonus
        - 2.0 * overlap_penalty
        - 1.3 * tiny_penalty
        - 1.6 * huge_penalty
        + 0.15 * len(targets)
    )


def sorted_target_groups(
    target_groups: dict[int, TargetGroup],
    all_components: list[Component],
    image_size: tuple[int, int],
) -> list[TargetGroup]:
    scored_groups = [
        (
            score_target_group_cleanliness(target_group, all_components, image_size),
            target_group,
        )
        for target_group in target_groups.values()
    ]
    scored_groups.sort(
        key=lambda item: (
            -item[0],
            -len(item[1].target_components),
            item[1].class_name,
            item[1].class_id,
        )
    )
    return [target_group for _, target_group in scored_groups]


def render_component_map(image: Image.Image, components: list[Component]) -> Image.Image:
    annotated = image.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    for component in components:
        if component.component_id is None:
            raise ValueError("Component IDs must be assigned before rendering.")

        color = ANNOTATION_COLORS[(component.component_id - 1) % len(ANNOTATION_COLORS)]
        fill_rgba = ImageColor.getrgb(color) + (96,)
        outline_rgba = ImageColor.getrgb(color) + (220,)

        if component.mask is not None:
            local_mask = component.mask.astype(np.uint8) * 255
            mask_image = Image.fromarray(local_mask, mode="L")
            fill_patch = Image.new("RGBA", (component.width, component.height), fill_rgba)
            overlay.paste(fill_patch, (component.x, component.y), mask_image)

            edge_mask = component.mask & ~ndimage.binary_erosion(component.mask, structure=np.ones((3, 3), dtype=bool))
            edge_image = Image.fromarray(edge_mask.astype(np.uint8) * 255, mode="L")
            outline_patch = Image.new("RGBA", (component.width, component.height), outline_rgba)
            overlay.paste(outline_patch, (component.x, component.y), edge_image)
        else:
            draw.rectangle(
                (
                    component.x,
                    component.y,
                    component.x + component.width,
                    component.y + component.height,
                ),
                fill=fill_rgba,
                outline=outline_rgba,
                width=3,
            )

        label = str(component.component_id)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_left = int(round(component.mask_centroid_x - (text_width + 6) / 2.0))
        text_top = int(round(component.mask_centroid_y - (text_height + 4) / 2.0))
        text_left = max(0, min(annotated.width - text_width - 6, text_left))
        text_top = max(0, min(annotated.height - text_height - 4, text_top))
        text_right = min(annotated.width - 1, text_left + text_width + 6)
        text_bottom = min(annotated.height - 1, text_top + text_height + 4)

        draw.rectangle((text_left, text_top, text_right, text_bottom), fill=outline_rgba)
        draw.text((text_left + 3, text_top + 2), label, fill="black", font=font)

    annotated = Image.alpha_composite(annotated, overlay)
    return annotated.convert("RGB")


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def compute_focus_crop_box(target_group: TargetGroup, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x1 = min(component.x for component in target_group.target_components)
    y1 = min(component.y for component in target_group.target_components)
    x2 = max(component.x + component.width for component in target_group.target_components)
    y2 = max(component.y + component.height for component in target_group.target_components)

    group_width = max(1, x2 - x1)
    group_height = max(1, y2 - y1)
    margin = int(max(48, 0.35 * max(group_width, group_height)))

    crop_x1 = max(0, x1 - margin)
    crop_y1 = max(0, y1 - margin)
    crop_x2 = min(image_width, x2 + margin)
    crop_y2 = min(image_height, y2 + margin)

    min_crop_size = 192
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    if crop_width < min_crop_size:
        missing = min_crop_size - crop_width
        grow_left = missing // 2
        grow_right = missing - grow_left
        crop_x1 = max(0, crop_x1 - grow_left)
        crop_x2 = min(image_width, crop_x2 + grow_right)
        if crop_x2 - crop_x1 < min_crop_size:
            if crop_x1 == 0:
                crop_x2 = min(image_width, min_crop_size)
            else:
                crop_x1 = max(0, image_width - min_crop_size)

    if crop_height < min_crop_size:
        missing = min_crop_size - crop_height
        grow_top = missing // 2
        grow_bottom = missing - grow_top
        crop_y1 = max(0, crop_y1 - grow_top)
        crop_y2 = min(image_height, crop_y2 + grow_bottom)
        if crop_y2 - crop_y1 < min_crop_size:
            if crop_y1 == 0:
                crop_y2 = min(image_height, min_crop_size)
            else:
                crop_y1 = max(0, image_height - min_crop_size)

    return int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)


def crop_box_to_dict(crop_box: tuple[int, int, int, int]) -> dict[str, int]:
    x1, y1, x2, y2 = crop_box
    return {
        "x": int(x1),
        "y": int(y1),
        "width": int(x2 - x1),
        "height": int(y2 - y1),
    }


def crop_image(image: Image.Image, crop_box: tuple[int, int, int, int]) -> Image.Image:
    return image.crop(crop_box)


def build_prompt_payload(
    image_id: int,
    target_group: TargetGroup,
    all_components: list[Component],
    crop_box: tuple[int, int, int, int],
    component_source: str,
) -> dict:
    return {
        "image_id": image_id,
        "component_source": component_source,
        "allowed_relations": RELATION_ORDER,
        "excluded_class_names": sorted(EXCLUDED_CLASS_NAMES),
        "focus_target_group": target_group.prompt_dict(),
        "focus_crop_bbox": crop_box_to_dict(crop_box),
        "components": [component.prompt_dict() for component in all_components],
    }


def response_schema() -> dict:
    return {
        "name": "nyu_multi_instance_relations",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "groups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target_class_id": {"type": "integer"},
                            "anchor_component_id": {"type": "integer"},
                            "assignments": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "target_component_id": {"type": "integer"},
                                        "spatial_relation": {
                                            "type": "string",
                                            "enum": RELATION_ORDER,
                                        },
                                    },
                                    "required": ["target_component_id", "spatial_relation"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["target_class_id", "anchor_component_id", "assignments"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["groups"],
            "additionalProperties": False,
        },
    }


def build_messages(
    image_id: int,
    target_group: TargetGroup,
    prompt_payload: dict,
    original_image_url: str,
    annotated_image_url: str,
    cropped_original_image_url: str,
    cropped_annotated_image_url: str,
    component_source: str,
) -> list[dict]:
    payload_text = json.dumps(prompt_payload, indent=2, ensure_ascii=False)
    annotation_kind = "colored object masks and component_id labels drawn on the objects" if component_source == "semantic" else "colored bounding boxes and component_id labels"
    system_text = (
        "You generate spatial-relation dataset annotations for indoor images. "
        "You are solving one repeated target group at a time. "
        "Use the full-scene images, the zoomed crop images, and the structured component metadata together. "
        "Infer relations between the visible objects themselves, not between their metadata boxes. "
        "For each repeated target group, either choose exactly one shared anchor component and assign a distinct "
        "relation to every target component, or omit that target group entirely if you cannot do that confidently. "
        "Use only these relation labels: "
        + ", ".join(RELATION_ORDER)
        + ". Return JSON only."
    )
    user_text = (
        f"Image {image_id:04d} is shown four times for the focus target group '{target_group.class_name}'.\n"
        "The first image is the original full RGB frame.\n"
        f"The second image is the full frame with {annotation_kind}.\n"
        "The third image is a zoomed crop around the focus target group plus nearby context.\n"
        f"The fourth image is the same crop with {annotation_kind}.\n\n"
        "Requirements:\n"
        "- Every returned group must use exactly one anchor_component_id.\n"
        "- Every returned group must assign every target component exactly one relation.\n"
        "- All relations inside a group must be distinct.\n"
        "- Never choose an anchor from the same class as the target class.\n"
        "- Judge each relation from the actual objects and scene layout, not from comparing locator boxes.\n"
        "- Use the zoomed crop for fine local judgments and the full-scene view for global context.\n"
        "- Use the RGB images as the primary evidence for where each object is.\n"
        "- Treat locator_bbox as an ID lookup hint only; do not derive the relation from box centers, edges, or box overlap alone.\n"
        "- If the visible object shape/placement disagrees with what a box would suggest, trust the visible object.\n"
        "- Prefer visually meaningful anchors that make the full set of relations natural.\n"
        "- Use 'under' only when the target looks physically under or directly beneath the anchor.\n"
        "- If no good single anchor exists for a target group, omit that group.\n\n"
        "Structured metadata:\n"
        f"{payload_text}"
    )
    if component_source == "semantic":
        user_text = user_text.replace(
            "- Use the RGB images as the primary evidence for where each object is.\n",
            "- Use the RGB images and colored object masks as the primary evidence for where each object is.\n",
        ).replace(
            "- If no good single anchor exists for a target group, omit that group.\n\n",
            "- Use 'in front of' only when the target appears closer to the camera than the anchor; depth_median_m is a hint, not a rule.\n"
            "- If no good single anchor exists for a target group, omit that group.\n\n",
        )
    return [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": original_image_url}},
                {"type": "image_url", "image_url": {"url": annotated_image_url}},
                {"type": "image_url", "image_url": {"url": cropped_original_image_url}},
                {"type": "image_url", "image_url": {"url": cropped_annotated_image_url}},
            ],
        },
    ]


def cache_path_for_group(cache_dir: Path, image_id: int, target_class_id: int) -> Path:
    return cache_dir / f"{image_id:04d}_{target_class_id}.json"


def load_cached_response(cache_dir: Path, image_id: int, target_class_id: int, model: str) -> dict | None:
    cache_path = cache_path_for_group(cache_dir, image_id, target_class_id)
    if not cache_path.exists():
        return None

    with cache_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("prompt_version") != PROMPT_VERSION:
        return None
    if payload.get("model") != model:
        return None
    if int(payload.get("target_class_id", -1)) != target_class_id:
        return None
    if not isinstance(payload.get("response"), dict):
        return None
    return payload["response"]


def save_cached_response(
    cache_dir: Path,
    image_id: int,
    target_class_id: int,
    model: str,
    response_payload: dict,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path_for_group(cache_dir, image_id, target_class_id)
    cache_payload = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "image_id": image_id,
        "target_class_id": target_class_id,
        "response": response_payload,
    }
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(cache_payload, handle, indent=2, ensure_ascii=False)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_completion_text(response) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_chunks.append(str(item.get("text", "")))
            elif hasattr(item, "type") and getattr(item, "type") == "text":
                text_value = getattr(item, "text", "")
                if hasattr(text_value, "value"):
                    text_chunks.append(str(text_value.value))
                else:
                    text_chunks.append(str(text_value))
        return "".join(text_chunks)

    return str(content)


def should_fallback_to_json_object(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in ("json_schema", "response_format", "schema"))


def request_chatgpt_relations(
    client_config: ClientConfig,
    model: str,
    image_id: int,
    target_group: TargetGroup,
    prompt_payload: dict,
    original_image: Image.Image,
    annotated_image: Image.Image,
    cropped_original_image: Image.Image,
    cropped_annotated_image: Image.Image,
    max_retries: int,
    component_source: str,
) -> dict:
    client = client_config.client
    messages = build_messages(
        image_id=image_id,
        target_group=target_group,
        prompt_payload=prompt_payload,
        original_image_url=image_to_data_url(original_image),
        annotated_image_url=image_to_data_url(annotated_image),
        cropped_original_image_url=image_to_data_url(cropped_original_image),
        cropped_annotated_image_url=image_to_data_url(cropped_annotated_image),
        component_source=component_source,
    )

    request_kwargs = {
        "model": model,
        "messages": messages,
    }
    if not model.lower().startswith("gpt-5"):
        request_kwargs["temperature"] = 0
    if client_config.backend == "openrouter":
        extra_body = {}
        if client_config.provider_slug:
            extra_body["provider"] = {
                "order": [client_config.provider_slug],
                "allow_fallbacks": False,
            }
        if extra_body:
            request_kwargs["extra_body"] = extra_body

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            try:
                response = client.chat.completions.create(
                    **request_kwargs,
                    response_format={
                        "type": "json_schema",
                        "json_schema": response_schema(),
                    },
                )
            except Exception as exc:
                if not should_fallback_to_json_object(exc):
                    raise
                response = client.chat.completions.create(
                    **request_kwargs,
                    response_format={"type": "json_object"},
                )

            text = strip_code_fences(extract_completion_text(response))
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("ChatGPT response must parse to a JSON object.")
            return parsed
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(min(2 ** (attempt - 1), 8))

    if last_error is None:
        raise RuntimeError(f"ChatGPT request for image {image_id:04d} failed without an error.")
    raise RuntimeError(f"ChatGPT request for image {image_id:04d} failed: {last_error}") from last_error


def parse_group_decisions(
    response_payload: dict,
    target_groups: dict[int, TargetGroup],
    all_components: list[Component],
) -> dict[int, tuple[Component, list[str]]]:
    components_by_id = {component.component_id: component for component in all_components}
    decisions: dict[int, tuple[Component, list[str]]] = {}

    groups = response_payload.get("groups", [])
    if not isinstance(groups, list):
        return decisions

    for raw_group in groups:
        if not isinstance(raw_group, dict):
            continue

        try:
            target_class_id = int(raw_group["target_class_id"])
            anchor_component_id = int(raw_group["anchor_component_id"])
            assignments = raw_group["assignments"]
        except (KeyError, TypeError, ValueError):
            continue

        target_group = target_groups.get(target_class_id)
        if target_group is None or target_class_id in decisions:
            continue
        if not isinstance(assignments, list):
            continue

        anchor = components_by_id.get(anchor_component_id)
        if anchor is None:
            continue
        if anchor.class_id == target_class_id:
            continue
        if anchor.class_name in EXCLUDED_CLASS_NAMES:
            continue

        expected_target_ids = {component.component_id for component in target_group.target_components}
        relation_map: dict[int, str] = {}
        used_relations: set[str] = set()
        valid = True

        for assignment in assignments:
            if not isinstance(assignment, dict):
                valid = False
                break
            try:
                target_component_id = int(assignment["target_component_id"])
                spatial_relation = str(assignment["spatial_relation"])
            except (KeyError, TypeError, ValueError):
                valid = False
                break

            if target_component_id not in expected_target_ids:
                valid = False
                break
            if target_component_id in relation_map:
                valid = False
                break
            if spatial_relation not in RELATION_ORDER:
                valid = False
                break
            if spatial_relation in used_relations:
                valid = False
                break

            relation_map[target_component_id] = spatial_relation
            used_relations.add(spatial_relation)

        if not valid or set(relation_map) != expected_target_ids:
            continue

        ordered_relations = [relation_map[component.component_id] for component in target_group.target_components]
        decisions[target_class_id] = (anchor, ordered_relations)

    return decisions


def resolve_chatgpt_decision_for_group(
    image_id: int,
    target_group: TargetGroup,
    all_components: list[Component],
    original_image: Image.Image,
    annotated_image: Image.Image,
    client_config: ClientConfig | None,
    args: argparse.Namespace,
    component_source: str,
) -> tuple[tuple[Component, list[str]] | None, bool]:
    cached_payload = None
    if not args.force_refresh:
        cached_payload = load_cached_response(args.cache_dir, image_id, target_group.class_id, args.model)
        if cached_payload is not None:
            decisions = parse_group_decisions(
                cached_payload,
                {target_group.class_id: target_group},
                all_components,
            )
            return decisions.get(target_group.class_id), True

    if args.cache_only:
        raise RuntimeError(
            f"Cache miss for image {image_id:04d} target_class_id {target_group.class_id} while --cache-only is enabled."
        )

    if client_config is None:
        raise RuntimeError(
            "No API client is configured, and no matching cached response was found. "
            "Set OPENROUTER_API_KEY or OPENAI_API_KEY."
        )

    crop_box = compute_focus_crop_box(target_group, original_image.size)
    cropped_original_image = crop_image(original_image, crop_box)
    cropped_annotated_image = crop_image(annotated_image, crop_box)
    response_payload = request_chatgpt_relations(
        client_config=client_config,
        model=args.model,
        image_id=image_id,
        target_group=target_group,
        prompt_payload=build_prompt_payload(image_id, target_group, all_components, crop_box, component_source),
        original_image=original_image,
        annotated_image=annotated_image,
        cropped_original_image=cropped_original_image,
        cropped_annotated_image=cropped_annotated_image,
        max_retries=args.max_retries,
        component_source=component_source,
    )
    save_cached_response(args.cache_dir, image_id, target_group.class_id, args.model, response_payload)
    decisions = parse_group_decisions(
        response_payload,
        {target_group.class_id: target_group},
        all_components,
    )
    return decisions.get(target_group.class_id), False


def is_cache_only_cache_miss(exc: Exception, args: argparse.Namespace) -> bool:
    return args.cache_only and "while --cache-only is enabled" in str(exc)


def append_annotations_for_group(
    annotations: list[dict],
    *,
    image_id: int,
    target_group: TargetGroup,
    anchor: Component,
    relations: list[str],
) -> None:
    for ordinal, (component, relation) in enumerate(zip(target_group.target_components, relations), start=1):
        region_id = image_id * 1_000_000 + target_group.class_id * 1_000 + ordinal
        annotations.append(
            {
                "region_id": region_id,
                "width": component.width,
                "height": component.height,
                "image_id": image_id,
                "phrase": f"{component.class_name} {relation} {anchor.class_name}",
                "y": component.y,
                "x": component.x,
                "keywords": {
                    "target": component.class_name,
                    "attributes": [],
                    "anchor_object": anchor.class_name,
                    "spatial_relation": relation,
                },
            }
        )


def collect_image_ids(total_images: int, args: argparse.Namespace) -> list[int]:
    if args.image_id is not None:
        image_id = normalize_image_id(args.image_id)
        if image_id < 0 or image_id >= total_images:
            raise ValueError(f"image_id {image_id} is out of range 0..{total_images - 1}")
        return [image_id]

    image_ids = list(range(total_images))
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]
    return image_ids


def build_annotations(args: argparse.Namespace, client_config: ClientConfig | None) -> list[dict]:
    class_names = load_class_names()
    class_name_lookup = class_name_to_id_map()
    annotations: list[dict] = []

    kept_target_groups = 0
    dropped_too_many = 0
    dropped_no_anchor = 0
    dropped_excluded_class = 0
    cached_groups = 0
    api_groups = 0
    images_with_selected_group = 0
    images_without_valid_group = 0
    skipped_additional_groups = 0
    cache_miss_groups = 0

    if args.component_source == "semantic":
        with h5py.File(MAT_PATH, "r") as mat_file:
            depths = mat_file["depths"]
            image_ids = collect_image_ids(depths.shape[0], args)

            for image_id in tqdm(image_ids, desc="Generating NYU multi-instance phrases with ChatGPT"):
                label_map = load_label_map(image_id)
                depth_map = np.transpose(depths[image_id], (1, 0))
                original_image = load_rgb_image(image_id)

                components_by_class: dict[int, list[Component]] = {}

                for class_id in sorted(int(class_value) for class_value in np.unique(label_map) if int(class_value) != 0):
                    class_name = class_names.get(class_id, str(class_id))
                    if class_name in EXCLUDED_CLASS_NAMES:
                        dropped_excluded_class += 1
                        continue
                    components = extract_components(label_map, depth_map, class_id, class_name)
                    if components:
                        components_by_class[class_id] = components

                if not components_by_class:
                    continue

                all_components = assign_component_ids(components_by_class)
                target_groups, too_many_for_image = build_target_groups(components_by_class)
                dropped_too_many += too_many_for_image

                if not target_groups:
                    continue

                annotated_image = render_component_map(original_image, all_components)
                candidate_groups = sorted_target_groups(target_groups, all_components, original_image.size)
                selected_group = False

                for group_index, target_group in enumerate(candidate_groups):
                    try:
                        chosen, from_cache = resolve_chatgpt_decision_for_group(
                            image_id=image_id,
                            target_group=target_group,
                            all_components=all_components,
                            original_image=original_image,
                            annotated_image=annotated_image,
                            client_config=client_config,
                            args=args,
                            component_source=args.component_source,
                        )
                    except RuntimeError as exc:
                        if not is_cache_only_cache_miss(exc, args):
                            raise
                        cache_miss_groups += 1
                        continue
                    if from_cache:
                        cached_groups += 1
                    else:
                        api_groups += 1
                        if args.pause_seconds > 0:
                            time.sleep(args.pause_seconds)

                    if chosen is None:
                        dropped_no_anchor += 1
                        continue

                    anchor, relations = chosen
                    kept_target_groups += 1
                    images_with_selected_group += 1
                    skipped_additional_groups += len(candidate_groups) - group_index - 1
                    append_annotations_for_group(
                        annotations,
                        image_id=image_id,
                        target_group=target_group,
                        anchor=anchor,
                        relations=relations,
                    )
                    selected_group = True
                    break

                if not selected_group:
                    images_without_valid_group += 1
    else:
        image_ids = collect_rgb_image_ids(args)
        for image_id in tqdm(image_ids, desc="Generating NYU multi-instance phrases from RGB only"):
            original_image = load_rgb_image(image_id)
            components_by_class = extract_components_from_rgb(
                original_image,
                class_name_to_id=class_name_lookup,
                weights=args.weights.expanduser().resolve(),
                conf_thresh=args.det_conf,
                iou_thresh=args.det_iou,
                max_det=args.det_max_det,
            )
            if not components_by_class:
                continue

            all_components = assign_component_ids(components_by_class)
            target_groups, too_many_for_image = build_target_groups(components_by_class)
            dropped_too_many += too_many_for_image
            if not target_groups:
                continue

            annotated_image = render_component_map(original_image, all_components)
            candidate_groups = sorted_target_groups(target_groups, all_components, original_image.size)
            selected_group = False
            for group_index, target_group in enumerate(candidate_groups):
                try:
                    chosen, from_cache = resolve_chatgpt_decision_for_group(
                        image_id=image_id,
                        target_group=target_group,
                        all_components=all_components,
                        original_image=original_image,
                        annotated_image=annotated_image,
                        client_config=client_config,
                        args=args,
                        component_source=args.component_source,
                    )
                except RuntimeError as exc:
                    if not is_cache_only_cache_miss(exc, args):
                        raise
                    cache_miss_groups += 1
                    continue
                if from_cache:
                    cached_groups += 1
                else:
                    api_groups += 1
                    if args.pause_seconds > 0:
                        time.sleep(args.pause_seconds)

                if chosen is None:
                    dropped_no_anchor += 1
                    continue

                anchor, relations = chosen
                kept_target_groups += 1
                images_with_selected_group += 1
                skipped_additional_groups += len(candidate_groups) - group_index - 1
                append_annotations_for_group(
                    annotations,
                    image_id=image_id,
                    target_group=target_group,
                    anchor=anchor,
                    relations=relations,
                )
                selected_group = True
                break

            if not selected_group:
                images_without_valid_group += 1

    print(f"Kept repeated target groups: {kept_target_groups}")
    print(f"Dropped target groups with >{MAX_TARGET_INSTANCES} instances: {dropped_too_many}")
    print(f"Candidate target groups without a valid shared anchor: {dropped_no_anchor}")
    print(f"Skipped excluded classes ({sorted(EXCLUDED_CLASS_NAMES)}): {dropped_excluded_class}")
    print(f"Target groups served from cache: {cached_groups}")
    print(f"Target groups fetched from ChatGPT: {api_groups}")
    print(f"Target groups skipped because cache-only had no cached response: {cache_miss_groups}")
    print(f"Images with one selected repeated target group: {images_with_selected_group}")
    print(f"Images with repeated targets but no valid shared-anchor group: {images_without_valid_group}")
    print(f"Additional candidate groups skipped after selecting one per image: {skipped_additional_groups}")
    print(f"Generated annotations: {len(annotations)}")
    return annotations


def validate_annotations(annotations: list[dict]) -> None:
    if not isinstance(annotations, list):
        raise ValueError("Output must be a JSON list.")

    expected_keys = {"region_id", "width", "height", "image_id", "phrase", "y", "x", "keywords"}
    expected_keyword_keys = {"target", "attributes", "anchor_object", "spatial_relation"}
    allowed_relations = set(RELATION_ORDER)

    seen_region_ids: set[int] = set()
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    image_targets: dict[int, set[str]] = defaultdict(set)

    for entry in annotations:
        if set(entry.keys()) != expected_keys:
            raise ValueError(f"Unexpected keys for entry: {entry.keys()}")
        if set(entry["keywords"].keys()) != expected_keyword_keys:
            raise ValueError(f"Unexpected keyword keys for entry: {entry['keywords'].keys()}")
        if entry["width"] <= 0 or entry["height"] <= 0:
            raise ValueError(f"Non-positive box size for region {entry['region_id']}")
        if not (0 <= entry["x"] < 640 and 0 <= entry["y"] < 480):
            raise ValueError(f"Top-left corner out of bounds for region {entry['region_id']}")
        if entry["x"] + entry["width"] > 640 or entry["y"] + entry["height"] > 480:
            raise ValueError(f"Box exceeds image bounds for region {entry['region_id']}")
        if entry["keywords"]["spatial_relation"] not in allowed_relations:
            raise ValueError(f"Invalid relation for region {entry['region_id']}")
        if "inside" in entry["phrase"]:
            raise ValueError(f"'inside' should never appear in phrase for region {entry['region_id']}")
        if entry["keywords"]["target"] in EXCLUDED_CLASS_NAMES:
            raise ValueError(f"Excluded target class found in region {entry['region_id']}")
        if entry["keywords"]["anchor_object"] in EXCLUDED_CLASS_NAMES:
            raise ValueError(f"Excluded anchor class found in region {entry['region_id']}")
        if entry["region_id"] in seen_region_ids:
            raise ValueError(f"Duplicate region_id found: {entry['region_id']}")

        seen_region_ids.add(entry["region_id"])
        grouped[(entry["image_id"], entry["keywords"]["target"])].append(entry)
        image_targets[entry["image_id"]].add(entry["keywords"]["target"])

    for image_id, targets in image_targets.items():
        if len(targets) != 1:
            raise ValueError(f"Image {image_id} has multiple repeated target groups: {sorted(targets)}")

    for (image_id, target), group in grouped.items():
        if len(group) < 2:
            raise ValueError(f"Group {(image_id, target)} has fewer than 2 entries.")
        if len(group) > MAX_TARGET_INSTANCES:
            raise ValueError(f"Group {(image_id, target)} has more than {MAX_TARGET_INSTANCES} entries.")

        anchors = {entry["keywords"]["anchor_object"] for entry in group}
        if len(anchors) != 1:
            raise ValueError(f"Group {(image_id, target)} uses multiple anchors: {anchors}")

        relations = [entry["keywords"]["spatial_relation"] for entry in group]
        if len(relations) != len(set(relations)):
            raise ValueError(f"Group {(image_id, target)} reuses a relation: {relations}")


def build_client(args: argparse.Namespace) -> ClientConfig | None:
    if args.cache_only:
        return None

    if OPENROUTER_API_KEY:
        headers = {
            "HTTP-Referer": OPENROUTER_APP_URL,
            "X-Title": OPENROUTER_APP_NAME,
        }
        return ClientConfig(
            client=OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                default_headers=headers,
            ),
            backend="openrouter",
            provider_slug=OPENROUTER_PROVIDER or None,
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return ClientConfig(
            client=OpenAI(api_key=api_key),
            backend="openai",
        )
    return None


def validate_backend_configuration(client_config: ClientConfig | None, model: str, cache_only: bool = False) -> None:
    normalized_model = model.strip().lower()
    looks_like_openrouter_model = "/" in normalized_model and not normalized_model.startswith(("gpt-", "o1", "o3", "o4"))

    if cache_only:
        return

    if looks_like_openrouter_model and (client_config is None or client_config.backend != "openrouter"):
        raise SystemExit(
            "The selected model appears to be an OpenRouter-style model ID, but the script is not using the "
            "OpenRouter backend. Set OPENROUTER_API_KEY in .env so the generator uses OpenRouter, or switch "
            f"--model/NYU_RELATION_MODEL to an OpenAI model. Current model: {model}"
        )


def main() -> None:
    args = parse_args()
    args.output_path = resolve_output_path(args)
    args.cache_dir = resolve_cache_dir(args)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    client_config = build_client(args)
    validate_backend_configuration(client_config, args.model, cache_only=args.cache_only)
    if client_config is not None:
        print(
            f"Using {client_config.backend} backend"
            + (
                f" with provider {client_config.provider_slug}"
                if client_config.provider_slug
                else ""
            )
            + f" and model {args.model}"
        )
    else:
        print("No API backend configured; only cached responses will be available.")

    annotations = build_annotations(args=args, client_config=client_config)
    validate_annotations(annotations)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(annotations, handle, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
