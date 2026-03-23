from __future__ import annotations

import argparse
import inspect
import io
import json
import math
import site
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Prefer the active environment's packages over user-site overlays.
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

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DATA_ROOT = REPO_ROOT / "whatsup_vlms_data"
ALL_DATASETS = (
    "controlled_images",
    "controlled_clevr",
    "coco_qa_one_obj",
    "coco_qa_two_obj",
    "vg_qa_one_obj",
    "vg_qa_two_obj",
)
PROMPT_MODES = ("object", "full_caption")
RELATION_FROM_FILENAME = {
    "left_of": "left_of",
    "right_of": "right_of",
    "on": "on",
    "under": "under",
    "in-front_of": "front",
    "behind": "behind",
}
RELATION_ALIASES = {
    "to the left of": "left_of",
    "to the right of": "right_of",
    "to the top of": "above",
    "to the bottom of": "below",
    "to the front of": "front",
    "to the behind of": "behind",
    "in front of": "front",
    "behind": "behind",
    "above": "above",
    "below": "below",
    "on": "on",
    "under": "under",
    "left": "left",
    "right": "right",
    "top": "top",
    "bottom": "bottom",
    "front": "front",
}


@dataclass(frozen=True)
class ParsedCaption:
    text: str
    target: str
    relation: str
    anchor: Optional[str] = None

    @property
    def is_pairwise(self) -> bool:
        return self.anchor is not None


@dataclass(frozen=True)
class BenchmarkSample:
    dataset_name: str
    sample_id: str
    image_key: str
    image_hint: Optional[str]
    correct_index: int
    options: tuple[ParsedCaption, ...]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    json_path: Path
    image_source: str
    pairwise_ground_truth_first: bool = True


@dataclass(frozen=True)
class Detection:
    det_id: int
    label: str
    conf: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class PromptSpec:
    text: str
    object_labels: tuple[str, ...]


DATASET_SPECS = {
    "controlled_images": DatasetSpec(
        name="controlled_images",
        json_path=DATA_ROOT / "controlled_images_dataset.json",
        image_source="controlled_images",
    ),
    "controlled_clevr": DatasetSpec(
        name="controlled_clevr",
        json_path=DATA_ROOT / "controlled_clevr_dataset.json",
        image_source="controlled_clevr",
    ),
    "coco_qa_one_obj": DatasetSpec(
        name="coco_qa_one_obj",
        json_path=DATA_ROOT / "coco_qa_one_obj.json",
        image_source="coco",
    ),
    "coco_qa_two_obj": DatasetSpec(
        name="coco_qa_two_obj",
        json_path=DATA_ROOT / "coco_qa_two_obj.json",
        image_source="coco",
    ),
    "vg_qa_one_obj": DatasetSpec(
        name="vg_qa_one_obj",
        json_path=DATA_ROOT / "vg_qa_one_obj.json",
        image_source="vg",
    ),
    "vg_qa_two_obj": DatasetSpec(
        name="vg_qa_two_obj",
        json_path=DATA_ROOT / "vg_qa_two_obj.json",
        image_source="vg",
    ),
}


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def strip_leading_article(text: str) -> str:
    tokens = normalize_text(text).split()
    if tokens and tokens[0] in {"a", "an", "the"}:
        tokens = tokens[1:]
    return " ".join(tokens)


def canonicalize_relation(raw_relation: str) -> str:
    relation = normalize_text(raw_relation)
    if relation not in RELATION_ALIASES:
        raise ValueError(f"Unsupported relation text: {raw_relation!r}")
    return RELATION_ALIASES[relation]


def parse_caption(text: str) -> ParsedCaption:
    text = " ".join(text.strip().split())

    single_patterns = (
        r"^A photo of (?:a|an) (?P<target>.+?) on the (?P<relation>left|right|top|bottom|front|behind)$",
    )
    pair_patterns = (
        r"^A photo of (?:a|an) (?P<target>.+?) (?P<relation>to the left of|to the right of|to the top of|to the bottom of|to the front of|to the behind of|above|below|on|under|in front of|behind) (?:a|an) (?P<anchor>.+)$",
        r"^A (?P<target>.+?) (?P<relation>on|under|to the left of|to the right of|in front of|behind) (?:a|an) (?P<anchor>.+)$",
    )

    import re

    for pattern in single_patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return ParsedCaption(
                text=text,
                target=normalize_text(match.group("target")),
                relation=canonicalize_relation(match.group("relation")),
                anchor=None,
            )

    for pattern in pair_patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return ParsedCaption(
                text=text,
                target=normalize_text(match.group("target")),
                relation=canonicalize_relation(match.group("relation")),
                anchor=normalize_text(match.group("anchor")),
            )

    raise ValueError(f"Could not parse caption: {text!r}")


class ArchiveReader:
    def __init__(self, archive_path: Path):
        self.archive_path = archive_path
        self._handle = None
        self._members = None
        self._kind = "zip" if archive_path.suffix == ".zip" else "tar"

    def _ensure_open(self) -> None:
        if self._handle is not None:
            return
        if self._kind == "zip":
            self._handle = zipfile.ZipFile(self.archive_path)
            self._members = set(self._handle.namelist())
        else:
            self._handle = tarfile.open(self.archive_path, "r:*")
            self._members = {member.name for member in self._handle.getmembers()}

    def has_member(self, name: str) -> bool:
        self._ensure_open()
        return name in self._members

    def read_bytes(self, name: str) -> bytes:
        self._ensure_open()
        if self._kind == "zip":
            return self._handle.read(name)
        extracted = self._handle.extractfile(name)
        if extracted is None:
            raise FileNotFoundError(f"Could not extract {name} from {self.archive_path}")
        data = extracted.read()
        extracted.close()
        return data


class ImageResolver:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.archive_readers = {
            "controlled_images": ArchiveReader(DATA_ROOT / "controlled_images.tar.gz"),
            "controlled_clevr": ArchiveReader(DATA_ROOT / "controlled_clevr.tar.gz"),
            "coco": ArchiveReader(DATA_ROOT / "val2017.zip"),
            "vg": ArchiveReader(DATA_ROOT / "vg_images.tar.gz"),
        }

    def _candidate_paths(self, source: str, image_key: str, image_hint: Optional[str]) -> list[Path]:
        candidates: list[Path] = []
        if image_hint:
            hint_path = Path(image_hint)
            candidates.extend(
                [
                    self.repo_root / hint_path,
                    self.repo_root / hint_path.name,
                    self.repo_root / hint_path.parts[-2] / hint_path.name if len(hint_path.parts) >= 2 else self.repo_root / hint_path.name,
                    DATA_ROOT / hint_path.parts[-2] / hint_path.name if len(hint_path.parts) >= 2 else DATA_ROOT / hint_path.name,
                    DATA_ROOT / hint_path.name,
                    DATA_ROOT / hint_path,
                ]
            )

        if source == "coco":
            filename = f"{int(image_key):012d}.jpg"
            candidates.extend(
                [
                    self.repo_root / "val2017" / filename,
                    self.repo_root / "data" / "val2017" / filename,
                    DATA_ROOT / "val2017" / filename,
                ]
            )
        elif source == "vg":
            filename = f"{image_key}.jpg"
            candidates.extend(
                [
                    self.repo_root / "vg_images" / filename,
                    self.repo_root / "data" / "vg_images" / filename,
                    DATA_ROOT / "vg_images" / filename,
                ]
            )

        unique_candidates: list[Path] = []
        seen = set()
        for candidate in candidates:
            candidate = candidate.resolve() if candidate.is_absolute() else candidate
            key = str(candidate)
            if key not in seen:
                unique_candidates.append(candidate)
                seen.add(key)
        return unique_candidates

    def _archive_member(self, source: str, image_key: str, image_hint: Optional[str]) -> str:
        if source in {"controlled_images", "controlled_clevr"}:
            if not image_hint:
                raise ValueError(f"{source} samples require image_hint")
            folder = "controlled_images" if source == "controlled_images" else "controlled_clevr"
            return f"{folder}/{Path(image_hint).name}"
        if source == "coco":
            return f"val2017/{int(image_key):012d}.jpg"
        if source == "vg":
            return f"vg_images/{image_key}.jpg"
        raise ValueError(f"Unsupported image source: {source}")

    def load(self, source: str, image_key: str, image_hint: Optional[str]) -> tuple[Image.Image, str]:
        for candidate in self._candidate_paths(source, image_key, image_hint):
            if candidate.exists():
                with Image.open(candidate) as image:
                    return image.convert("RGB"), str(candidate)

        member = self._archive_member(source, image_key, image_hint)
        archive = self.archive_readers[source]
        if not archive.has_member(member):
            raise FileNotFoundError(f"Missing image {member} in {archive.archive_path}")

        data = archive.read_bytes(member)
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB"), f"{archive.archive_path}!/{member}"


def infer_controlled_correct_index(image_hint: str, options: tuple[ParsedCaption, ...]) -> int:
    stem = Path(image_hint).stem
    for token, relation in RELATION_FROM_FILENAME.items():
        if f"_{token}_" in stem or stem.endswith(f"_{token}"):
            for index, option in enumerate(options):
                if option.relation == relation:
                    return index
    return 0


def load_samples(spec: DatasetSpec, max_samples: Optional[int]) -> list[BenchmarkSample]:
    with open(spec.json_path, "r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    samples: list[BenchmarkSample] = []
    if spec.name.startswith("controlled_"):
        for index, item in enumerate(raw_data):
            options = tuple(parse_caption(text) for text in item["caption_options"])
            correct_index = infer_controlled_correct_index(item["image_path"], options)
            samples.append(
                BenchmarkSample(
                    dataset_name=spec.name,
                    sample_id=f"{spec.name}:{index}",
                    image_key=Path(item["image_path"]).stem,
                    image_hint=item["image_path"],
                    correct_index=correct_index,
                    options=options,
                )
            )
            if max_samples and len(samples) >= max_samples:
                break
        return samples

    for index, row in enumerate(raw_data):
        image_key = str(row[0])
        options = tuple(parse_caption(text) for text in row[1:])
        correct_index = 0 if spec.pairwise_ground_truth_first else len(options) - 1
        samples.append(
            BenchmarkSample(
                dataset_name=spec.name,
                sample_id=f"{spec.name}:{index}",
                image_key=image_key,
                image_hint=None,
                correct_index=correct_index,
                options=options,
            )
        )
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def sigmoid(value: float, scale: float = 12.0) -> float:
    return 1.0 / (1.0 + math.exp(-scale * value))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def overlap_ratio_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    overlap = max(0.0, min(a2, b2) - max(a1, b1))
    denom = max(1e-6, min(a2 - a1, b2 - b1))
    return clamp01(overlap / denom)


def score_single_relation(
    box: tuple[float, float, float, float],
    relation: str,
    image_width: float,
    image_height: float,
) -> float:
    center_x, center_y = box_center(box)
    x_fraction = center_x / max(image_width, 1.0)
    y_fraction = center_y / max(image_height, 1.0)

    if relation == "left":
        return sigmoid(0.5 - x_fraction)
    if relation == "right":
        return sigmoid(x_fraction - 0.5)
    if relation == "top":
        return sigmoid(0.5 - y_fraction)
    if relation == "bottom":
        return sigmoid(y_fraction - 0.5)
    if relation == "front":
        return sigmoid(y_fraction - 0.5)
    if relation == "behind":
        return sigmoid(0.5 - y_fraction)
    raise ValueError(f"Unsupported single-object relation: {relation}")


def score_pair_relation(
    target_box: tuple[float, float, float, float],
    anchor_box: tuple[float, float, float, float],
    relation: str,
    image_width: float,
    image_height: float,
) -> float:
    target_cx, target_cy = box_center(target_box)
    anchor_cx, anchor_cy = box_center(anchor_box)

    delta_x = (anchor_cx - target_cx) / max(image_width, 1.0)
    delta_y = (anchor_cy - target_cy) / max(image_height, 1.0)
    overlap_x = overlap_ratio_1d(target_box[0], target_box[2], anchor_box[0], anchor_box[2])

    if relation == "left_of":
        return sigmoid(delta_x)
    if relation == "right_of":
        return sigmoid(-delta_x)
    if relation == "above":
        return sigmoid(delta_y)
    if relation == "below":
        return sigmoid(-delta_y)
    if relation == "on":
        vertical_order = sigmoid(delta_y)
        contact = math.exp(-abs(target_box[3] - anchor_box[1]) / max(0.12 * image_height, 1.0))
        return clamp01(0.4 * vertical_order + 0.4 * overlap_x + 0.2 * contact)
    if relation == "under":
        vertical_order = sigmoid(-delta_y)
        contact = math.exp(-abs(target_box[1] - anchor_box[3]) / max(0.12 * image_height, 1.0))
        return clamp01(0.4 * vertical_order + 0.4 * overlap_x + 0.2 * contact)
    if relation in {"front", "behind"}:
        area_delta = math.log((box_area(target_box) + 1.0) / (box_area(anchor_box) + 1.0))
        area_delta = max(-1.0, min(1.0, area_delta / 2.0))
        depth_score = 0.75 * (-delta_y) + 0.25 * area_delta
        return sigmoid(depth_score) if relation == "front" else sigmoid(-depth_score)

    raise ValueError(f"Unsupported pairwise relation: {relation}")


def score_option(
    option: ParsedCaption,
    detections_by_label: dict[str, list[Detection]],
    image_width: int,
    image_height: int,
) -> float:
    target_detections = detections_by_label.get(option.target, [])
    if not target_detections:
        return 0.0

    if not option.is_pairwise:
        return max(
            detection.conf * score_single_relation(detection.box, option.relation, image_width, image_height)
            for detection in target_detections
        )

    anchor_detections = detections_by_label.get(option.anchor or "", [])
    if not anchor_detections:
        return 0.0

    best_score = 0.0
    for target_det in target_detections:
        for anchor_det in anchor_detections:
            if target_det.label == anchor_det.label and target_det.det_id == anchor_det.det_id:
                continue
            relation_score = score_pair_relation(
                target_det.box,
                anchor_det.box,
                option.relation,
                image_width,
                image_height,
            )
            detection_score = math.sqrt(max(target_det.conf, 1e-8) * max(anchor_det.conf, 1e-8))
            best_score = max(best_score, detection_score * relation_score)
    return best_score


def choose_prediction(option_scores: list[float]) -> Optional[int]:
    if not option_scores:
        return None
    best_score = max(option_scores)
    if best_score <= 0.0:
        return None
    best_indices = [index for index, score in enumerate(option_scores) if math.isclose(score, best_score, rel_tol=1e-8, abs_tol=1e-8)]
    if len(best_indices) != 1:
        return None
    return best_indices[0]


def post_process_grounding_outputs(
    processor: AutoProcessor,
    outputs,
    input_ids,
    target_sizes: list[tuple[int, int]],
    box_threshold: float,
    text_threshold: float,
):
    kwargs = {"text_threshold": text_threshold, "target_sizes": target_sizes}
    signature = inspect.signature(processor.post_process_grounded_object_detection).parameters
    if "box_threshold" in signature:
        kwargs["box_threshold"] = box_threshold
    else:
        kwargs["threshold"] = box_threshold
    return processor.post_process_grounded_object_detection(outputs, input_ids, **kwargs)


def build_grounding_prompt(label: str) -> str:
    return f"{label.strip().rstrip('.')}."


def text_label_matches_object(text_label: str, object_label: str) -> bool:
    normalized_text_label = strip_leading_article(text_label).strip(" .")
    normalized_object_label = strip_leading_article(object_label).strip(" .")
    if not normalized_text_label or not normalized_object_label:
        return False
    return (
        normalized_text_label == normalized_object_label
        or normalized_text_label in normalized_object_label
        or normalized_object_label in normalized_text_label
    )


def build_prompt_specs(image_samples: list[BenchmarkSample], prompt_mode: str) -> list[PromptSpec]:
    if prompt_mode == "object":
        object_labels = sorted(
            {
                caption_label
                for sample in image_samples
                for option in sample.options
                for caption_label in ([option.target] + ([option.anchor] if option.anchor else []))
            }
        )
        return [PromptSpec(text=label, object_labels=(label,)) for label in object_labels]

    if prompt_mode == "full_caption":
        prompt_to_objects: dict[str, set[str]] = defaultdict(set)
        for sample in image_samples:
            for option in sample.options:
                prompt_to_objects[option.text].add(option.target)
                if option.anchor:
                    prompt_to_objects[option.text].add(option.anchor)
        return [
            PromptSpec(text=text, object_labels=tuple(sorted(object_labels)))
            for text, object_labels in sorted(prompt_to_objects.items())
        ]

    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def resolve_detection_labels(prompt_spec: PromptSpec, decoded_text_label: str, prompt_mode: str) -> tuple[str, ...]:
    if prompt_mode == "object":
        return prompt_spec.object_labels

    matched_labels = tuple(
        object_label
        for object_label in prompt_spec.object_labels
        if text_label_matches_object(decoded_text_label, object_label)
    )
    if matched_labels:
        return matched_labels

    # For one-object captions, fall back to the only object even if the decoder returns a noisy span.
    if len(prompt_spec.object_labels) == 1:
        return prompt_spec.object_labels

    return ()


def run_groundingdino(
    processor: AutoProcessor,
    model: AutoModelForZeroShotObjectDetection,
    image: Image.Image,
    prompt_specs: list[PromptSpec],
    prompt_mode: str,
    device: str,
    box_threshold: float,
    text_threshold: float,
    max_det: int,
) -> dict[str, list[Detection]]:
    detections_by_label: dict[str, list[Detection]] = defaultdict(list)
    if not prompt_specs:
        return detections_by_label

    prompts = [build_grounding_prompt(prompt_spec.text) for prompt_spec in prompt_specs]
    inputs = processor(
        images=[image] * len(prompts),
        text=prompts,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = [image.size[::-1]] * len(prompts)
    results = post_process_grounding_outputs(
        processor=processor,
        outputs=outputs,
        input_ids=inputs.input_ids,
        target_sizes=target_sizes,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    det_id = 0
    for prompt_spec, result in zip(prompt_specs, results):
        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        text_labels = result.get("text_labels")
        if text_labels is None:
            text_labels = result.get("labels")
        if text_labels is None:
            text_labels = [""] * len(scores)
        keep = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:max_det]
        for index in keep:
            decoded_text_label = str(text_labels[index]) if index < len(text_labels) else ""
            resolved_labels = resolve_detection_labels(prompt_spec, decoded_text_label, prompt_mode)
            for label in resolved_labels:
                detections_by_label[label].append(
                    Detection(
                        det_id=det_id,
                        label=label,
                        conf=float(scores[index]),
                        box=tuple(float(value) for value in boxes[index]),
                    )
                )
            det_id += 1

    return detections_by_label


def empty_summary(dataset_name: str, unique_images: int) -> dict:
    return {
        "dataset": dataset_name,
        "unique_images": unique_images,
        "evaluated_samples": 0,
        "correct": 0,
        "abstained": 0,
        "missing_images": 0,
        "target_detected": 0,
        "anchor_detected": 0,
        "pair_available": 0,
        "per_relation": defaultdict(Counter),
    }


def update_relation_stats(summary: dict, relation: str, is_correct: bool, predicted_index: Optional[int], target_found: bool, anchor_found: bool, pair_found: bool) -> None:
    relation_stats = summary["per_relation"][relation]
    relation_stats["samples"] += 1
    relation_stats["correct"] += int(is_correct)
    relation_stats["abstained"] += int(predicted_index is None)
    relation_stats["target_detected"] += int(target_found)
    relation_stats["anchor_detected"] += int(anchor_found)
    relation_stats["pair_available"] += int(pair_found)


def finalize_summary(summary: dict) -> dict:
    total = summary["evaluated_samples"]
    summary["accuracy"] = summary["correct"] / total if total else 0.0
    summary["abstain_rate"] = summary["abstained"] / total if total else 0.0
    summary["target_detect_rate"] = summary["target_detected"] / total if total else 0.0
    summary["anchor_detect_rate"] = summary["anchor_detected"] / total if total else 0.0
    summary["pair_available_rate"] = summary["pair_available"] / total if total else 0.0

    per_relation = {}
    for relation, stats in summary["per_relation"].items():
        samples = stats["samples"]
        per_relation[relation] = {
            "samples": samples,
            "accuracy": stats["correct"] / samples if samples else 0.0,
            "abstain_rate": stats["abstained"] / samples if samples else 0.0,
            "target_detect_rate": stats["target_detected"] / samples if samples else 0.0,
            "anchor_detect_rate": stats["anchor_detected"] / samples if samples else 0.0,
            "pair_available_rate": stats["pair_available"] / samples if samples else 0.0,
        }
    summary["per_relation"] = dict(sorted(per_relation.items()))
    return summary


def evaluate_dataset(
    spec: DatasetSpec,
    samples: list[BenchmarkSample],
    processor: AutoProcessor,
    model: AutoModelForZeroShotObjectDetection,
    resolver: ImageResolver,
    prompt_mode: str,
    device: str,
    box_threshold: float,
    text_threshold: float,
    max_det: int,
) -> tuple[dict, list[dict]]:
    grouped_samples: dict[str, list[BenchmarkSample]] = defaultdict(list)
    for sample in samples:
        grouped_samples[sample.image_key].append(sample)

    summary = empty_summary(spec.name, unique_images=len(grouped_samples))
    detailed_rows: list[dict] = []

    progress = tqdm(grouped_samples.items(), desc=f"Evaluating {spec.name}", unit="image")
    for image_key, image_samples in progress:
        image_hint = image_samples[0].image_hint
        try:
            image, resolved_from = resolver.load(spec.image_source, image_key, image_hint)
        except FileNotFoundError as exc:
            summary["missing_images"] += 1
            print(f"Warning: {exc}")
            continue

        image_width, image_height = image.size
        prompt_specs = build_prompt_specs(image_samples, prompt_mode)
        detections_by_label = run_groundingdino(
            processor=processor,
            model=model,
            image=image,
            prompt_specs=prompt_specs,
            prompt_mode=prompt_mode,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_det=max_det,
        )

        for sample in image_samples:
            scores = [score_option(option, detections_by_label, image_width, image_height) for option in sample.options]
            predicted_index = choose_prediction(scores)
            correct_option = sample.options[sample.correct_index]

            target_found = bool(detections_by_label.get(correct_option.target))
            anchor_found = True
            if correct_option.anchor is not None:
                anchor_found = bool(detections_by_label.get(correct_option.anchor))
            pair_found = target_found and anchor_found

            is_correct = predicted_index == sample.correct_index

            summary["evaluated_samples"] += 1
            summary["correct"] += int(is_correct)
            summary["abstained"] += int(predicted_index is None)
            summary["target_detected"] += int(target_found)
            summary["anchor_detected"] += int(anchor_found)
            summary["pair_available"] += int(pair_found)
            update_relation_stats(
                summary,
                relation=correct_option.relation,
                is_correct=is_correct,
                predicted_index=predicted_index,
                target_found=target_found,
                anchor_found=anchor_found,
                pair_found=pair_found,
            )

            detailed_rows.append(
                {
                    "dataset": spec.name,
                    "sample_id": sample.sample_id,
                    "image_key": sample.image_key,
                    "image_ref": resolved_from,
                    "correct_index": sample.correct_index,
                    "predicted_index": predicted_index,
                    "is_correct": is_correct,
                    "relation": correct_option.relation,
                    "target_detected": target_found,
                    "anchor_detected": anchor_found,
                    "pair_available": pair_found,
                    "options": [option.text for option in sample.options],
                    "option_scores": scores,
                }
            )

    return finalize_summary(summary), detailed_rows


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 72)
    print(f"Dataset: {summary['dataset']}")
    print(f"Unique images:      {summary['unique_images']}")
    print(f"Evaluated samples:  {summary['evaluated_samples']}")
    print(f"Missing images:     {summary['missing_images']}")
    print(f"Accuracy:           {summary['accuracy']:.4f} ({summary['accuracy'] * 100:.2f}%)")
    print(f"Abstain rate:       {summary['abstain_rate']:.4f} ({summary['abstain_rate'] * 100:.2f}%)")
    print(f"Target detect rate: {summary['target_detect_rate']:.4f} ({summary['target_detect_rate'] * 100:.2f}%)")
    if any(stats["anchor_detect_rate"] < 1.0 for stats in summary["per_relation"].values()):
        print(f"Anchor detect rate: {summary['anchor_detect_rate']:.4f} ({summary['anchor_detect_rate'] * 100:.2f}%)")
    print(f"Pair available:     {summary['pair_available_rate']:.4f} ({summary['pair_available_rate'] * 100:.2f}%)")
    print("-" * 72)
    print("Per relation:")
    for relation, stats in summary["per_relation"].items():
        print(
            f"  {relation:>8} | n={stats['samples']:4d} | "
            f"acc={stats['accuracy']:.3f} | "
            f"abstain={stats['abstain_rate']:.3f} | "
            f"pair={stats['pair_available_rate']:.3f}"
        )
    print("=" * 72)


def aggregate_summaries(summaries: list[dict]) -> dict:
    total_samples = sum(summary["evaluated_samples"] for summary in summaries)
    total_correct = sum(summary["correct"] for summary in summaries)
    total_abstained = sum(summary["abstained"] for summary in summaries)
    total_missing = sum(summary["missing_images"] for summary in summaries)
    total_target_detected = sum(summary["target_detected"] for summary in summaries)
    total_anchor_detected = sum(summary["anchor_detected"] for summary in summaries)
    total_pair_available = sum(summary["pair_available"] for summary in summaries)
    total_unique_images = sum(summary["unique_images"] for summary in summaries)

    return {
        "dataset": "overall",
        "unique_images": total_unique_images,
        "evaluated_samples": total_samples,
        "correct": total_correct,
        "abstained": total_abstained,
        "missing_images": total_missing,
        "target_detected": total_target_detected,
        "anchor_detected": total_anchor_detected,
        "pair_available": total_pair_available,
        "accuracy": total_correct / total_samples if total_samples else 0.0,
        "abstain_rate": total_abstained / total_samples if total_samples else 0.0,
        "target_detect_rate": total_target_detected / total_samples if total_samples else 0.0,
        "anchor_detect_rate": total_anchor_detected / total_samples if total_samples else 0.0,
        "pair_available_rate": total_pair_available / total_samples if total_samples else 0.0,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GroundingDINO on the spatial-relation benchmark files in whatsup_vlms_data. "
            "For the pairwise QA JSON files, the script assumes the first caption is the ground-truth option."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", *ALL_DATASETS],
        help="Datasets to evaluate. Use 'all' to run the full benchmark.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="GroundingDINO model id or local checkpoint path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25, help="GroundingDINO box threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="GroundingDINO text threshold.")
    parser.add_argument("--max-det", type=int, default=50, help="Maximum detections per image.")
    parser.add_argument(
        "--prompt-mode",
        choices=PROMPT_MODES,
        default="object",
        help=(
            "How to build GroundingDINO prompts per image: "
            "'object' uses extracted target/anchor labels, "
            "and 'full_caption' uses the full caption text from the benchmark file."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional per-dataset cap to speed up smoke tests.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path for a detailed JSON report.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    selected_datasets = list(ALL_DATASETS) if "all" in args.datasets else args.datasets

    print(f"Loading GroundingDINO from {args.model_id} on {args.device}...")
    print(f"Using prompt mode: {args.prompt_mode}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(args.device)
    model.eval()
    resolver = ImageResolver(REPO_ROOT)

    all_summaries: list[dict] = []
    all_rows: list[dict] = []

    for dataset_name in selected_datasets:
        spec = DATASET_SPECS[dataset_name]
        samples = load_samples(spec, max_samples=args.max_samples)
        print(f"\nLoaded {len(samples)} samples from {spec.json_path}")
        summary, detailed_rows = evaluate_dataset(
            spec=spec,
            samples=samples,
            processor=processor,
            model=model,
            resolver=resolver,
            prompt_mode=args.prompt_mode,
            device=args.device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            max_det=args.max_det,
        )
        print_summary(summary)
        all_summaries.append(summary)
        all_rows.extend(detailed_rows)

    if len(all_summaries) > 1:
        overall = aggregate_summaries(all_summaries)
        print("\n" + "#" * 72)
        print("Overall")
        print(f"Datasets:           {', '.join(selected_datasets)}")
        print(f"Unique images:      {overall['unique_images']}")
        print(f"Evaluated samples:  {overall['evaluated_samples']}")
        print(f"Missing images:     {overall['missing_images']}")
        print(f"Accuracy:           {overall['accuracy']:.4f} ({overall['accuracy'] * 100:.2f}%)")
        print(f"Abstain rate:       {overall['abstain_rate']:.4f} ({overall['abstain_rate'] * 100:.2f}%)")
        print(f"Target detect rate: {overall['target_detect_rate']:.4f} ({overall['target_detect_rate'] * 100:.2f}%)")
        print(f"Pair available:     {overall['pair_available_rate']:.4f} ({overall['pair_available_rate'] * 100:.2f}%)")
        print("#" * 72)

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "datasets": all_summaries,
                    "samples": all_rows,
                },
                handle,
                indent=2,
            )
        print(f"\nSaved report to {args.report_json}")


if __name__ == "__main__":
    main()
