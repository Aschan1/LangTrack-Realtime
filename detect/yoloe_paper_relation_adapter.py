#!/usr/bin/env python3
"""
Train and evaluate a Hu et al. CVPR 2018 style object-relation adapter on top of YOLOE.

This script is intentionally separate from the lightweight `reprta` prompt adapter trainer.
It keeps YOLOE frozen, harvests target/anchor proposals with offline prompt embeddings, and
learns a standalone relation module that follows the paper's core object-relation design:

  - proposal appearance features
  - pairwise geometric relation embeddings
  - multi-relation attention
  - residual feature update

Practical note:
YOLOE's current local offline path does not expose detector-side RoI pooled proposal features,
so this script uses cropped image patches plus prompt embeddings and proposal metadata as the
proposal appearance feature source. The relation block itself is paper-style; only the source of
proposal appearance features differs from a region-based detector.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import site
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
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

from refexp_metrics import compute_rec_metrics

YOLOE = None
build_embedding_cache = None
compose_embedding = None


INDOORS_SUBSET_DIR = REPO_ROOT / "yolo_dataset" / "indoors_subset"
NYU_DATASET_DIR = REPO_ROOT / "nyu_dataset"
DEFAULT_JSON = INDOORS_SUBSET_DIR / "filtered_indoors_LM_vg_nonnull_spatial_relations.json"
DEFAULT_IMAGES = INDOORS_SUBSET_DIR / "images"
DEFAULT_NYU_RGB_JSON = NYU_DATASET_DIR / "filtered_nyu_LM_vg_multi_instance_rgb_only.json"
DEFAULT_NYU_IMAGES = NYU_DATASET_DIR / "images"
DEFAULT_WEIGHTS = REPO_ROOT / "yoloe-26l-seg.pt"
DEFAULT_CACHE_PATH = REPO_ROOT / "detect" / "embedding_cache.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "paper_relation_adapter"
DEFAULT_PROPOSAL_CACHE = DEFAULT_OUTPUT_DIR / "proposal_cache.pt"
DEFAULT_NYU_OUTPUT_DIR = REPO_ROOT / "outputs" / "paper_relation_adapter_nyu_rgb_only"
DEFAULT_NYU_PROPOSAL_CACHE = DEFAULT_NYU_OUTPUT_DIR / "proposal_cache.pt"


def ensure_runtime_imports() -> tuple[Any, Any, Any]:
    global YOLOE, build_embedding_cache, compose_embedding
    if YOLOE is None:
        from ultralytics import YOLOE as _YOLOE

        YOLOE = _YOLOE
    if build_embedding_cache is None or compose_embedding is None:
        from run_yolo_offline import build_embedding_cache as _build_embedding_cache, compose_embedding as _compose_embedding

        build_embedding_cache = _build_embedding_cache
        compose_embedding = _compose_embedding
    return YOLOE, build_embedding_cache, compose_embedding


@dataclass(frozen=True)
class QueryEpisode:
    image_id: str
    image_path: str
    image_width: int
    image_height: int
    query_key: str
    target: str
    anchor: str
    relation: str
    global_cls_id: int
    target_gt_boxes: tuple[tuple[float, float, float, float], ...]
    anchor_gt_boxes: tuple[tuple[float, float, float, float], ...]
    target_candidates: tuple[dict[str, Any], ...]
    anchor_candidates: tuple[dict[str, Any], ...]


def ordered_unique(items):
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_json_path(json_arg: str | Path) -> Path:
    candidate = Path(json_arg).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()

    indoors_candidate = INDOORS_SUBSET_DIR / candidate
    if indoors_candidate.exists():
        return indoors_candidate.resolve()

    return candidate.resolve()


def resolve_path(path_arg: str | Path) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def resolve_image_path(images_dir: Path, image_id: str | int) -> Path | None:
    image_id = str(image_id)
    candidate_stems = []
    for stem in (image_id, image_id.zfill(4), image_id.zfill(5)):
        if stem not in candidate_stems:
            candidate_stems.append(stem)
    for stem in candidate_stems:
        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = images_dir / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate.resolve()
    return None


def parse_keywords(ann: dict) -> tuple[str, str, str]:
    keywords = ann.get("keywords", {}) if isinstance(ann, dict) else {}
    target = (keywords.get("target", "") or "").strip()
    anchor = (keywords.get("anchor_object", "") or "").strip()
    relation = (keywords.get("spatial_relation", "") or "").strip()
    return target, anchor, relation


def get_query_key(ann: dict) -> str:
    target, anchor, relation = parse_keywords(ann)
    return json.dumps(
        {"target": target, "anchor_object": anchor, "spatial_relation": relation},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def get_gt_box(ann: dict) -> tuple[float, float, float, float]:
    x, y, w, h = ann["x"], ann["y"], ann["width"], ann["height"]
    return float(x), float(y), float(x + w), float(y + h)


def calculate_iou(box1, box2) -> float:
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0.0, x2_inter - x1_inter) * max(0.0, y2_inter - y1_inter)
    if inter_area == 0:
        return 0.0

    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    denom = box1_area + box2_area - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / float(denom)


def box_center(box) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def dedupe_boxes(boxes: list[tuple[float, float, float, float]]) -> tuple[tuple[float, float, float, float], ...]:
    unique = []
    seen = set()
    for box in boxes:
        key = tuple(round(float(value), 4) for value in box)
        if key in seen:
            continue
        seen.add(key)
        unique.append(tuple(float(value) for value in box))
    unique.sort()
    return tuple(unique)


def load_dataset(json_path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    with open(json_path, "r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    img_to_anns: dict[str, list[dict]] = defaultdict(list)
    for ann in annotations:
        img_to_anns[str(ann["image_id"])].append(ann)
    return annotations, dict(img_to_anns)


def make_global_label_map(img_to_anns: dict[str, list[dict]]) -> dict[str, int]:
    global_label_to_id = {}
    for anns in img_to_anns.values():
        for ann in anns:
            query_key = get_query_key(ann)
            if query_key not in global_label_to_id:
                global_label_to_id[query_key] = len(global_label_to_id)
    return global_label_to_id


def collect_detections_by_label(result, local_labels: list[str]) -> dict[str, list[dict]]:
    detections_by_label: dict[str, list[dict]] = defaultdict(list)
    if result.boxes is None or len(result.boxes) == 0:
        return detections_by_label

    pred_boxes = result.boxes.xyxy.detach().cpu().numpy().tolist()
    pred_classes = result.boxes.cls.detach().cpu().numpy().tolist()
    pred_confs = result.boxes.conf.detach().cpu().numpy().tolist()

    for det_id, (box, cls_idx, conf) in enumerate(zip(pred_boxes, pred_classes, pred_confs)):
        cls_idx = int(cls_idx)
        if cls_idx < 0 or cls_idx >= len(local_labels):
            continue
        label = local_labels[cls_idx]
        detections_by_label[label].append(
            {
                "det_id": det_id,
                "label": label,
                "conf": float(conf),
                "box": tuple(float(value) for value in box),
            }
        )

    for label in detections_by_label:
        detections_by_label[label].sort(key=lambda item: item["conf"], reverse=True)
    return detections_by_label


def ensure_prompt_embeddings(
    model,
    prompts: list[str],
    *,
    cache_embeddings: dict[str, torch.Tensor],
    embed_dim: int,
    prompt_embedding_cache: dict[str, torch.Tensor],
    stats: dict[str, int],
) -> None:
    _, _, compose_embedding_fn = ensure_runtime_imports()
    prompts = [prompt for prompt in prompts if prompt and prompt not in prompt_embedding_cache]
    if not prompts:
        return

    head = model.model.model[-1]
    device = next(model.model.parameters()).device

    raw_embeddings = []
    resolved_prompts = []
    fallback_prompts = []
    for prompt in prompts:
        raw_embedding = compose_embedding_fn(prompt, [], cache_embeddings, embed_dim)
        if raw_embedding is None:
            fallback_prompts.append(prompt)
            continue
        raw_embeddings.append(raw_embedding)
        resolved_prompts.append(prompt)
        stats["offline_prompt_hits"] += 1

    if raw_embeddings:
        raw_tensor = torch.stack(raw_embeddings).unsqueeze(0).to(device)
        with torch.no_grad():
            final_embeddings = head.get_tpe(raw_tensor).squeeze(0).detach().cpu()
        for prompt, embedding in zip(resolved_prompts, final_embeddings, strict=False):
            prompt_embedding_cache[prompt] = embedding.float().contiguous()

    if fallback_prompts:
        with torch.no_grad():
            final_embeddings = model.model.get_text_pe(
                fallback_prompts,
                cache_clip_model=True,
            ).squeeze(0).detach().cpu()
        for prompt, embedding in zip(fallback_prompts, final_embeddings, strict=False):
            prompt_embedding_cache[prompt] = embedding.float().contiguous()
        stats["offline_prompt_fallbacks"] += len(fallback_prompts)


def build_proposal_cache(
    json_path: Path,
    images_dir: Path,
    weights: Path,
    cache_path: Path,
    proposal_cache_path: Path,
    *,
    conf_thresh: float,
    topk_target: int,
    topk_anchor: int,
    limit: int,
    verbose: bool,
    force_rebuild: bool,
) -> dict[str, Any]:
    YOLOE_cls, build_embedding_cache_fn, _ = ensure_runtime_imports()
    proposal_cache_path = proposal_cache_path.expanduser().resolve()
    if proposal_cache_path.is_file() and not force_rebuild:
        return torch.load(str(proposal_cache_path), map_location="cpu", weights_only=False)

    if not cache_path.exists():
        print(f"Offline cache not found at {cache_path}. Building cache first...")
        build_embedding_cache_fn(str(json_path), str(cache_path), str(weights))

    cache_payload = torch.load(str(cache_path), map_location="cpu", weights_only=True)
    cache_embeddings = cache_payload["embeddings"]
    embed_dim = int(cache_payload["embed_dim"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLOE_cls(str(weights)).to(device)
    model.model.eval()

    _, img_to_anns = load_dataset(json_path)
    if limit > 0:
        img_to_anns = dict(list(img_to_anns.items())[:limit])
    global_label_to_id = make_global_label_map(img_to_anns)

    prompt_embedding_cache: dict[str, torch.Tensor] = {}
    stats = {
        "images_total": len(img_to_anns),
        "images_processed": 0,
        "images_missing": 0,
        "query_total": 0,
        "query_kept": 0,
        "query_missing_target": 0,
        "query_missing_anchor": 0,
        "offline_prompt_hits": 0,
        "offline_prompt_fallbacks": 0,
        "model_forward_passes": 0,
    }
    episodes: list[dict[str, Any]] = []

    progress = tqdm(img_to_anns.items(), desc="Building proposal cache", unit="image")
    for image_id, anns in progress:
        image_path = resolve_image_path(images_dir, image_id)
        if image_path is None:
            stats["images_missing"] += 1
            if verbose:
                print(f"Skipping missing image for image_id={image_id!r} in {images_dir}")
            continue

        query_to_anns: dict[str, list[dict]] = defaultdict(list)
        for ann in anns:
            query_to_anns[get_query_key(ann)].append(ann)

        labels = ordered_unique(
            [prompt for query_anns in query_to_anns.values() for prompt in parse_keywords(query_anns[0])[:2] if prompt]
        )
        if not labels:
            continue

        ensure_prompt_embeddings(
            model,
            labels,
            cache_embeddings=cache_embeddings,
            embed_dim=embed_dim,
            prompt_embedding_cache=prompt_embedding_cache,
            stats=stats,
        )

        prompt_embeddings = torch.stack([prompt_embedding_cache[label] for label in labels]).unsqueeze(0).to(device)
        model.set_classes(labels, prompt_embeddings)
        result = model.predict(
            source=str(image_path),
            conf=conf_thresh,
            verbose=False,
        )[0]
        stats["model_forward_passes"] += 1
        stats["images_processed"] += 1

        detections_by_label = collect_detections_by_label(result, labels)
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        for query_key, query_anns in query_to_anns.items():
            target, anchor, relation = parse_keywords(query_anns[0])
            if not target or not anchor:
                continue

            stats["query_total"] += 1
            target_dets = tuple(detections_by_label.get(target, [])[:topk_target])
            anchor_dets = tuple(detections_by_label.get(anchor, [])[:topk_anchor])
            stats["query_missing_target"] += int(not target_dets)
            stats["query_missing_anchor"] += int(not anchor_dets)

            target_gt_boxes = dedupe_boxes([get_gt_box(ann) for ann in query_anns])
            anchor_gt_boxes = dedupe_boxes([get_gt_box(ann) for ann in anns if parse_keywords(ann)[0] == anchor])
            episode = QueryEpisode(
                image_id=str(image_id),
                image_path=str(image_path),
                image_width=int(image_width),
                image_height=int(image_height),
                query_key=query_key,
                target=target,
                anchor=anchor,
                relation=relation,
                global_cls_id=int(global_label_to_id[query_key]),
                target_gt_boxes=target_gt_boxes,
                anchor_gt_boxes=anchor_gt_boxes,
                target_candidates=target_dets,
                anchor_candidates=anchor_dets,
            )
            episodes.append(
                {
                    "image_id": episode.image_id,
                    "image_path": episode.image_path,
                    "image_width": episode.image_width,
                    "image_height": episode.image_height,
                    "query_key": episode.query_key,
                    "target": episode.target,
                    "anchor": episode.anchor,
                    "relation": episode.relation,
                    "global_cls_id": episode.global_cls_id,
                    "target_gt_boxes": episode.target_gt_boxes,
                    "anchor_gt_boxes": episode.anchor_gt_boxes,
                    "target_candidates": episode.target_candidates,
                    "anchor_candidates": episode.anchor_candidates,
                }
            )
            stats["query_kept"] += 1

    payload = {
        "episodes": episodes,
        "prompt_embeddings": prompt_embedding_cache,
        "global_label_to_id": global_label_to_id,
        "stats": stats,
        "json_path": str(json_path),
        "images_dir": str(images_dir),
        "weights": str(weights),
        "conf_thresh": float(conf_thresh),
        "topk_target": int(topk_target),
        "topk_anchor": int(topk_anchor),
    }
    proposal_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, proposal_cache_path)
    print(f"Saved proposal cache to: {proposal_cache_path}")
    return payload


def split_episodes_by_image(
    episodes: list[dict[str, Any]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    image_ids = sorted({episode["image_id"] for episode in episodes})
    if not image_ids or val_fraction <= 0.0:
        return episodes, []
    if len(image_ids) < 2:
        return episodes, []

    rng = random.Random(seed)
    shuffled = image_ids[:]
    rng.shuffle(shuffled)
    val_count = min(max(int(round(len(shuffled) * val_fraction)), 1), len(shuffled) - 1)
    val_ids = set(shuffled[:val_count])
    train = [episode for episode in episodes if episode["image_id"] not in val_ids]
    val = [episode for episode in episodes if episode["image_id"] in val_ids]
    return train, val


def crop_box_tensor(image: Image.Image, box, crop_size: int) -> torch.Tensor:
    width, height = image.size
    x1 = int(max(0, min(width, math.floor(box[0]))))
    y1 = int(max(0, min(height, math.floor(box[1]))))
    x2 = int(max(0, min(width, math.ceil(box[2]))))
    y2 = int(max(0, min(height, math.ceil(box[3]))))
    if x2 <= x1 or y2 <= y1:
        patch = Image.new("RGB", (crop_size, crop_size), color=(0, 0, 0))
    else:
        patch = image.crop((x1, y1, x2, y2)).convert("RGB").resize((crop_size, crop_size), Image.BILINEAR)
    patch_np = np.asarray(patch, dtype=np.float32) / 255.0
    return torch.from_numpy(patch_np).permute(2, 0, 1).contiguous()


class ProposalEpisodeDataset(Dataset):
    def __init__(
        self,
        episodes: list[dict[str, Any]],
        *,
        prompt_embeddings: dict[str, torch.Tensor],
        crop_size: int,
        positive_iou: float,
        require_positive: bool,
        require_anchor: bool,
    ) -> None:
        self.prompt_embeddings = prompt_embeddings
        self.crop_size = int(crop_size)
        self.positive_iou = float(positive_iou)

        filtered = []
        skip_stats = Counter()
        for episode in episodes:
            target_candidates = list(episode["target_candidates"])
            anchor_candidates = list(episode["anchor_candidates"])
            if not target_candidates:
                skip_stats["missing_target_candidates"] += 1
                continue
            if require_anchor and not anchor_candidates:
                skip_stats["missing_anchor_candidates"] += 1
                continue

            target_positive = [
                max((calculate_iou(candidate["box"], gt_box) for gt_box in episode["target_gt_boxes"]), default=0.0) >= self.positive_iou
                for candidate in target_candidates
            ]
            if require_positive and not any(target_positive):
                skip_stats["missing_positive_target"] += 1
                continue

            filtered.append(episode)

        self.episodes = filtered
        self.skip_stats = dict(skip_stats)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self.episodes[index]
        proposals = []
        with Image.open(episode["image_path"]) as image:
            image = image.convert("RGB")
            for role, label, candidates in (
                ("target", episode["target"], episode["target_candidates"]),
                ("anchor", episode["anchor"], episode["anchor_candidates"]),
            ):
                prompt_embedding = self.prompt_embeddings[label].float().contiguous()
                for candidate in candidates:
                    box = tuple(float(value) for value in candidate["box"])
                    proposals.append(
                        {
                            "role": role,
                            "box": box,
                            "conf": float(candidate["conf"]),
                            "crop": crop_box_tensor(image, box, self.crop_size),
                            "text_feat": prompt_embedding,
                            "target_positive": float(
                                max(
                                    (calculate_iou(box, gt_box) for gt_box in episode["target_gt_boxes"]),
                                    default=0.0,
                                )
                                >= self.positive_iou
                            )
                            if role == "target"
                            else -1.0,
                            "anchor_positive": float(
                                max(
                                    (calculate_iou(box, gt_box) for gt_box in episode["anchor_gt_boxes"]),
                                    default=0.0,
                                )
                                >= self.positive_iou
                            )
                            if role == "anchor"
                            else -1.0,
                        }
                    )

        return {
            "image_id": episode["image_id"],
            "query_key": episode["query_key"],
            "target": episode["target"],
            "anchor": episode["anchor"],
            "relation": episode["relation"],
            "global_cls_id": int(episode["global_cls_id"]),
            "image_wh": torch.tensor([float(episode["image_width"]), float(episode["image_height"])], dtype=torch.float32),
            "target_gt_boxes": [tuple(map(float, box)) for box in episode["target_gt_boxes"]],
            "anchor_gt_boxes": [tuple(map(float, box)) for box in episode["anchor_gt_boxes"]],
            "proposals": proposals,
        }


def collate_proposal_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_props = max(len(sample["proposals"]) for sample in batch)
    text_dim = batch[0]["proposals"][0]["text_feat"].numel()
    crop_shape = batch[0]["proposals"][0]["crop"].shape

    crops = torch.zeros(len(batch), max_props, *crop_shape, dtype=torch.float32)
    boxes = torch.zeros(len(batch), max_props, 4, dtype=torch.float32)
    confs = torch.zeros(len(batch), max_props, dtype=torch.float32)
    text_feats = torch.zeros(len(batch), max_props, text_dim, dtype=torch.float32)
    role_ids = torch.zeros(len(batch), max_props, dtype=torch.long)
    target_labels = torch.full((len(batch), max_props), -1.0, dtype=torch.float32)
    anchor_labels = torch.full((len(batch), max_props), -1.0, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_props, dtype=torch.bool)

    meta = []
    for batch_index, sample in enumerate(batch):
        meta.append(
            {
                "image_id": sample["image_id"],
                "query_key": sample["query_key"],
                "target": sample["target"],
                "anchor": sample["anchor"],
                "relation": sample["relation"],
                "global_cls_id": sample["global_cls_id"],
                "target_gt_boxes": sample["target_gt_boxes"],
                "anchor_gt_boxes": sample["anchor_gt_boxes"],
            }
        )
        for proposal_index, proposal in enumerate(sample["proposals"]):
            crops[batch_index, proposal_index] = proposal["crop"]
            boxes[batch_index, proposal_index] = torch.tensor(proposal["box"], dtype=torch.float32)
            confs[batch_index, proposal_index] = float(proposal["conf"])
            text_feats[batch_index, proposal_index] = proposal["text_feat"]
            role_ids[batch_index, proposal_index] = 0 if proposal["role"] == "target" else 1
            target_labels[batch_index, proposal_index] = float(proposal["target_positive"])
            anchor_labels[batch_index, proposal_index] = float(proposal["anchor_positive"])
            mask[batch_index, proposal_index] = True

    return {
        "crops": crops,
        "boxes": boxes,
        "confs": confs,
        "text_feats": text_feats,
        "role_ids": role_ids,
        "target_labels": target_labels,
        "anchor_labels": anchor_labels,
        "mask": mask,
        "image_wh": torch.stack([sample["image_wh"] for sample in batch], dim=0),
        "meta": meta,
    }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


class ConvCropEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).flatten(1)
        return self.proj(x)


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    dim = max(int(dim), 2)
    half_dim = max(dim // 2, 1)
    device = values.device
    if half_dim == 1:
        freq = torch.ones(1, device=device, dtype=values.dtype)
    else:
        freq = torch.exp(
            torch.arange(half_dim, device=device, dtype=values.dtype)
            * (-math.log(10000.0) / float(max(half_dim - 1, 1)))
        )
    scaled = values.unsqueeze(-1) * freq
    embedding = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding[..., :dim]


def pairwise_geometry_embedding(boxes: torch.Tensor, geom_dim: int) -> torch.Tensor:
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-4)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-4)
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0

    delta_x = torch.log(torch.clamp(torch.abs(centers_x[:, None] - centers_x[None, :]) / widths[None, :], min=1e-4))
    delta_y = torch.log(torch.clamp(torch.abs(centers_y[:, None] - centers_y[None, :]) / heights[None, :], min=1e-4))
    delta_w = torch.log(torch.clamp(widths[:, None] / widths[None, :], min=1e-4))
    delta_h = torch.log(torch.clamp(heights[:, None] / heights[None, :], min=1e-4))
    raw = torch.stack([delta_x, delta_y, delta_w, delta_h], dim=-1)

    per_feature_dim = max(int(math.ceil(geom_dim / 4.0)), 2)
    embedded = torch.cat(
        [sinusoidal_embedding(raw[..., feature_index], per_feature_dim) for feature_index in range(raw.shape[-1])],
        dim=-1,
    )
    if embedded.shape[-1] < geom_dim:
        embedded = F.pad(embedded, (0, geom_dim - embedded.shape[-1]))
    return embedded[..., :geom_dim]


class ObjectRelationBlock(nn.Module):
    def __init__(self, feature_dim: int, *, num_relations: int, key_dim: int, geom_dim: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_relations = int(num_relations)
        self.key_dim = int(key_dim)
        self.geom_dim = int(geom_dim)

        base_value_dim = feature_dim // num_relations
        remainder = feature_dim % num_relations
        self.query_proj = nn.ModuleList(nn.Linear(feature_dim, key_dim) for _ in range(num_relations))
        self.key_proj = nn.ModuleList(nn.Linear(feature_dim, key_dim) for _ in range(num_relations))
        self.value_proj = nn.ModuleList(
            nn.Linear(feature_dim, base_value_dim + (1 if head_index < remainder else 0))
            for head_index in range(num_relations)
        )
        self.geom_proj = nn.ModuleList(nn.Linear(geom_dim, 1) for _ in range(num_relations))

    def forward(self, features: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        if features.shape[0] <= 1:
            return features

        geom_embedding = pairwise_geometry_embedding(boxes, self.geom_dim)
        relation_outputs = []
        scale = math.sqrt(float(self.key_dim))
        for q_proj, k_proj, v_proj, g_proj in zip(
            self.query_proj,
            self.key_proj,
            self.value_proj,
            self.geom_proj,
            strict=False,
        ):
            queries = q_proj(features)
            keys = k_proj(features)
            values = v_proj(features)
            appearance_logits = (queries @ keys.transpose(0, 1)) / scale
            geometry_weight = F.relu(g_proj(geom_embedding).squeeze(-1))
            attention_logits = appearance_logits + torch.log(geometry_weight + 1e-6)
            attention = torch.softmax(attention_logits, dim=1)
            relation_outputs.append(attention @ values)

        relation_feature = torch.cat(relation_outputs, dim=-1)
        return features + relation_feature


class PaperRelationAdapter(nn.Module):
    def __init__(
        self,
        *,
        text_dim: int,
        feature_dim: int,
        num_relations: int,
        key_dim: int,
        geom_dim: int,
        num_blocks: int,
        crop_feature_dim: int,
    ) -> None:
        super().__init__()
        self.crop_encoder = ConvCropEncoder(crop_feature_dim)
        self.text_proj = nn.Linear(text_dim, feature_dim // 2)
        self.meta_proj = nn.Sequential(
            nn.Linear(11, feature_dim // 4),
            nn.SiLU(),
            nn.Linear(feature_dim // 4, feature_dim // 4),
            nn.SiLU(),
        )
        self.appearance_proj = nn.Sequential(
            nn.Linear(crop_feature_dim + feature_dim // 2 + feature_dim // 4, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.blocks = nn.ModuleList(
            ObjectRelationBlock(
                feature_dim,
                num_relations=num_relations,
                key_dim=key_dim,
                geom_dim=geom_dim,
            )
            for _ in range(num_blocks)
        )
        self.target_classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.SiLU(),
            nn.Linear(feature_dim // 2, 1),
        )

    def encode_proposals(
        self,
        crops: torch.Tensor,
        text_feats: torch.Tensor,
        boxes: torch.Tensor,
        confs: torch.Tensor,
        role_ids: torch.Tensor,
        image_wh: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_props = mask.shape
        flat_mask = mask.reshape(-1)
        flat_crops = crops.reshape(batch_size * max_props, *crops.shape[2:])
        crop_features = torch.zeros(
            batch_size * max_props,
            self.crop_encoder.proj.out_features,
            device=crops.device,
            dtype=crops.dtype,
        )
        if flat_mask.any():
            crop_features[flat_mask] = self.crop_encoder(flat_crops[flat_mask])
        crop_features = crop_features.view(batch_size, max_props, -1)

        widths = image_wh[:, 0:1].clamp(min=1.0)
        heights = image_wh[:, 1:2].clamp(min=1.0)
        x1 = boxes[..., 0] / widths
        y1 = boxes[..., 1] / heights
        x2 = boxes[..., 2] / widths
        y2 = boxes[..., 3] / heights
        cx = ((boxes[..., 0] + boxes[..., 2]) / 2.0) / widths
        cy = ((boxes[..., 1] + boxes[..., 3]) / 2.0) / heights
        bw = (boxes[..., 2] - boxes[..., 0]).clamp(min=1.0) / widths
        bh = (boxes[..., 3] - boxes[..., 1]).clamp(min=1.0) / heights
        role_one_hot = F.one_hot(role_ids.clamp(min=0), num_classes=2).float()
        meta = torch.cat(
            [
                x1.unsqueeze(-1),
                y1.unsqueeze(-1),
                x2.unsqueeze(-1),
                y2.unsqueeze(-1),
                cx.unsqueeze(-1),
                cy.unsqueeze(-1),
                bw.unsqueeze(-1),
                bh.unsqueeze(-1),
                confs.unsqueeze(-1),
                role_one_hot,
            ],
            dim=-1,
        )
        encoded = torch.cat(
            [
                crop_features,
                self.text_proj(text_feats),
                self.meta_proj(meta),
            ],
            dim=-1,
        )
        return self.appearance_proj(encoded)

    def forward(
        self,
        *,
        crops: torch.Tensor,
        text_feats: torch.Tensor,
        boxes: torch.Tensor,
        confs: torch.Tensor,
        role_ids: torch.Tensor,
        image_wh: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encode_proposals(crops, text_feats, boxes, confs, role_ids, image_wh, mask)
        outputs = []
        for sample_features, sample_boxes, sample_mask in zip(encoded, boxes, mask, strict=False):
            valid_features = sample_features[sample_mask]
            valid_boxes = sample_boxes[sample_mask]
            for block in self.blocks:
                valid_features = block(valid_features, valid_boxes)
            logits = self.target_classifier(valid_features).squeeze(-1)
            padded_logits = torch.full((sample_mask.shape[0],), -1e9, device=logits.device, dtype=logits.dtype)
            padded_logits[sample_mask] = logits
            outputs.append(padded_logits)
        return torch.stack(outputs, dim=0)


def compute_loss(
    logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    relation_score_weight: float,
    pos_weight: float,
) -> torch.Tensor:
    target_mask = batch["mask"] & (batch["role_ids"] == 0) & (batch["target_labels"] >= 0.0)
    if not target_mask.any():
        return logits.sum() * 0.0

    bce = F.binary_cross_entropy_with_logits(
        logits[target_mask],
        batch["target_labels"][target_mask],
        pos_weight=torch.tensor(float(pos_weight), device=logits.device),
    )

    if relation_score_weight <= 0.0:
        return bce

    fused_scores = torch.sigmoid(logits[target_mask]) * batch["confs"][target_mask]
    positive_scores = fused_scores[batch["target_labels"][target_mask] > 0.5]
    negative_scores = fused_scores[batch["target_labels"][target_mask] <= 0.5]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return bce

    margin_loss = F.relu(negative_scores.max() - positive_scores.max() + 0.1)
    return bce + float(relation_score_weight) * margin_loss


def summarize_query_predictions(
    *,
    episode_meta: dict[str, Any],
    target_candidates: list[dict[str, Any]],
    target_scores: list[float],
    iou_thresh: float,
) -> dict[str, Any]:
    if not target_candidates:
        return {
            "predicted_index": None,
            "is_correct": False,
            "target_found": False,
            "anchor_found": bool(episode_meta["anchor_gt_boxes"]),
            "pair_found": False,
            "best_iou": 0.0,
        }

    predicted_index = max(range(len(target_scores)), key=target_scores.__getitem__)
    best_candidate = target_candidates[predicted_index]
    best_iou = max(
        (calculate_iou(best_candidate["box"], gt_box) for gt_box in episode_meta["target_gt_boxes"]),
        default=0.0,
    )
    return {
        "predicted_index": predicted_index,
        "is_correct": bool(best_iou >= iou_thresh),
        "target_found": True,
        "anchor_found": bool(episode_meta["anchor_gt_boxes"]),
        "pair_found": True,
        "best_iou": float(best_iou),
    }


def empty_relation_summary(name: str, unique_images: int) -> dict[str, Any]:
    return {
        "dataset": name,
        "unique_images": int(unique_images),
        "evaluated_samples": 0,
        "correct": 0,
        "abstained": 0,
        "target_detected": 0,
        "anchor_detected": 0,
        "pair_available": 0,
        "per_relation": defaultdict(Counter),
    }


def update_relation_stats(
    summary: dict[str, Any],
    *,
    relation: str,
    is_correct: bool,
    predicted_index: int | None,
    target_found: bool,
    anchor_found: bool,
    pair_found: bool,
) -> None:
    relation_stats = summary["per_relation"][relation]
    relation_stats["samples"] += 1
    relation_stats["correct"] += int(is_correct)
    relation_stats["abstained"] += int(predicted_index is None)
    relation_stats["target_detected"] += int(target_found)
    relation_stats["anchor_detected"] += int(anchor_found)
    relation_stats["pair_available"] += int(pair_found)


def finalize_relation_summary(summary: dict[str, Any]) -> dict[str, Any]:
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
            "samples": int(samples),
            "accuracy": stats["correct"] / samples if samples else 0.0,
            "abstain_rate": stats["abstained"] / samples if samples else 0.0,
            "target_detect_rate": stats["target_detected"] / samples if samples else 0.0,
            "anchor_detect_rate": stats["anchor_detected"] / samples if samples else 0.0,
            "pair_available_rate": stats["pair_available"] / samples if samples else 0.0,
        }
    summary["per_relation"] = dict(sorted(per_relation.items()))
    return summary


def compute_detection_metrics(all_gts: list[dict[str, Any]], all_preds: list[dict[str, Any]], iou_thresh: float) -> dict[str, Any]:
    unique_classes = {gt["cls"] for gt in all_gts}
    aps = []
    global_tp = 0
    global_fp = 0
    total_gts = len(all_gts)
    total_preds = len(all_preds)

    for cls_id in unique_classes:
        preds_c = [pred for pred in all_preds if pred["cls"] == cls_id]
        gts_c = [gt.copy() for gt in all_gts if gt["cls"] == cls_id]
        npos = len(gts_c)
        if npos == 0:
            continue

        preds_c.sort(key=lambda item: item["conf"], reverse=True)
        tp = np.zeros(len(preds_c))
        fp = np.zeros(len(preds_c))

        for gt in gts_c:
            gt["used"] = False

        for pred_index, pred in enumerate(preds_c):
            img_gts = [gt for gt in gts_c if gt["img_id"] == pred["img_id"]]
            ovmax = -1.0
            matched_index = -1
            for gt_index, gt in enumerate(img_gts):
                iou = calculate_iou(pred["box"], gt["box"])
                if iou > ovmax:
                    ovmax = iou
                    matched_index = gt_index

            if ovmax >= iou_thresh and matched_index >= 0:
                if not img_gts[matched_index]["used"]:
                    tp[pred_index] = 1.0
                    img_gts[matched_index]["used"] = True
                else:
                    fp[pred_index] = 1.0
            else:
                fp[pred_index] = 1.0

        global_tp += int(tp.sum())
        global_fp += int(fp.sum())

        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)
        recall = tp_cumsum / float(npos)
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for index in range(mpre.size - 1, 0, -1):
            mpre[index - 1] = np.maximum(mpre[index - 1], mpre[index])
        i_list = np.where(mrec[1:] != mrec[:-1])[0]
        aps.append(float(np.sum((mrec[i_list + 1] - mrec[i_list]) * mpre[i_list + 1])))

    fn = max(total_gts - global_tp, 0)
    precision = float(global_tp / total_preds) if total_preds > 0 else 0.0
    recall = float(global_tp / total_gts) if total_gts > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = float(global_tp / (global_tp + global_fp + fn)) if (global_tp + global_fp + fn) > 0 else 0.0
    map50 = float(np.mean(aps)) if aps else 0.0
    rec_summary = compute_rec_metrics(all_gts, all_preds, iou_thresh, calculate_iou)

    return {
        "ground_truths": int(total_gts),
        "predictions": int(total_preds),
        "true_positives": int(global_tp),
        "false_positives": int(global_fp),
        "false_negatives": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "map": map50,
        **rec_summary,
    }


def evaluate_model(
    model: PaperRelationAdapter,
    loader: DataLoader,
    *,
    device: torch.device,
    iou_thresh: float,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    all_gts: list[dict[str, Any]] = []
    all_preds: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    image_ids = set()
    summary = empty_relation_summary("indoors_subset", unique_images=0)

    progress = tqdm(loader, desc="Evaluating relation adapter", leave=False)
    with torch.no_grad():
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            logits = model(
                crops=batch["crops"],
                text_feats=batch["text_feats"],
                boxes=batch["boxes"],
                confs=batch["confs"],
                role_ids=batch["role_ids"],
                image_wh=batch["image_wh"],
                mask=batch["mask"],
            )
            fused_scores = torch.sigmoid(logits) * batch["confs"]

            for batch_index, meta in enumerate(batch["meta"]):
                image_ids.add(meta["image_id"])
                target_mask = batch["mask"][batch_index] & (batch["role_ids"][batch_index] == 0)
                anchor_mask = batch["mask"][batch_index] & (batch["role_ids"][batch_index] == 1)
                target_candidates = []
                target_scores = []

                for gt_box in meta["target_gt_boxes"]:
                    all_gts.append(
                        {
                            "img_id": meta["image_id"],
                            "cls": meta["global_cls_id"],
                            "box": list(gt_box),
                            "query_key": meta["query_key"],
                        }
                    )

                target_indices = torch.where(target_mask)[0].tolist()
                for local_rank, proposal_index in enumerate(target_indices):
                    box = batch["boxes"][batch_index, proposal_index].detach().cpu().tolist()
                    conf = float(fused_scores[batch_index, proposal_index].detach().cpu())
                    target_candidates.append({"box": box, "conf": conf, "rank": local_rank})
                    target_scores.append(conf)
                    all_preds.append(
                        {
                            "img_id": meta["image_id"],
                            "cls": meta["global_cls_id"],
                            "box": box,
                            "conf": conf,
                            "query_key": meta["query_key"],
                            "p_label": meta["target"],
                        }
                    )

                query_summary = summarize_query_predictions(
                    episode_meta=meta,
                    target_candidates=target_candidates,
                    target_scores=target_scores,
                    iou_thresh=iou_thresh,
                )
                target_found = bool(target_indices)
                anchor_found = bool(torch.where(anchor_mask)[0].numel())
                pair_found = target_found and anchor_found

                summary["evaluated_samples"] += 1
                summary["correct"] += int(query_summary["is_correct"])
                summary["abstained"] += int(query_summary["predicted_index"] is None)
                summary["target_detected"] += int(target_found)
                summary["anchor_detected"] += int(anchor_found)
                summary["pair_available"] += int(pair_found)
                update_relation_stats(
                    summary,
                    relation=meta["relation"],
                    is_correct=bool(query_summary["is_correct"]),
                    predicted_index=query_summary["predicted_index"],
                    target_found=target_found,
                    anchor_found=anchor_found,
                    pair_found=pair_found,
                )

                detail_rows.append(
                    {
                        "image_id": meta["image_id"],
                        "query_key": meta["query_key"],
                        "relation": meta["relation"],
                        "target": meta["target"],
                        "anchor": meta["anchor"],
                        "predicted_index": query_summary["predicted_index"],
                        "is_correct": bool(query_summary["is_correct"]),
                        "best_iou": float(query_summary["best_iou"]),
                        "target_detected": target_found,
                        "anchor_detected": anchor_found,
                        "pair_available": pair_found,
                        "candidate_scores": target_scores,
                    }
                )

    summary["unique_images"] = len(image_ids)
    summary = finalize_relation_summary(summary)
    detection_metrics = compute_detection_metrics(all_gts, all_preds, iou_thresh)
    return summary, detection_metrics, all_gts, all_preds, detail_rows


def run_epoch(
    model: PaperRelationAdapter,
    loader: DataLoader,
    *,
    optimizer: AdamW | None,
    device: torch.device,
    relation_score_weight: float,
    pos_weight: float,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_queries = 0

    progress = tqdm(loader, desc="train" if is_training else "val", leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(
            crops=batch["crops"],
            text_feats=batch["text_feats"],
            boxes=batch["boxes"],
            confs=batch["confs"],
            role_ids=batch["role_ids"],
            image_wh=batch["image_wh"],
            mask=batch["mask"],
        )
        loss = compute_loss(
            logits,
            batch,
            relation_score_weight=relation_score_weight,
            pos_weight=pos_weight,
        )

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        fused_scores = (torch.sigmoid(logits) * batch["confs"]).detach().cpu()
        for batch_index, meta in enumerate(batch["meta"]):
            target_mask = (batch["mask"][batch_index] & (batch["role_ids"][batch_index] == 0)).detach().cpu()
            target_indices = torch.where(target_mask)[0].tolist()
            if not target_indices:
                total_queries += 1
                continue
            best_index = max(target_indices, key=lambda proposal_index: float(fused_scores[batch_index, proposal_index]))
            pred_box = batch["boxes"][batch_index, best_index].detach().cpu().tolist()
            best_iou = max((calculate_iou(pred_box, gt_box) for gt_box in meta["target_gt_boxes"]), default=0.0)
            total_correct += int(best_iou >= 0.5)
            total_queries += 1

        average_loss = total_loss / max(len(loader), 1)
        accuracy = total_correct / total_queries if total_queries else 0.0
        progress.set_postfix(loss=f"{average_loss:.4f}", top1=f"{accuracy:.3f}")

    return total_loss / max(len(loader), 1), (total_correct / total_queries if total_queries else 0.0)


def build_model_from_config(text_dim: int, args: argparse.Namespace) -> PaperRelationAdapter:
    return PaperRelationAdapter(
        text_dim=text_dim,
        feature_dim=args.feature_dim,
        num_relations=args.num_relations,
        key_dim=args.key_dim,
        geom_dim=args.geom_dim,
        num_blocks=args.num_blocks,
        crop_feature_dim=args.crop_feature_dim,
    )


def save_checkpoint(
    path: Path,
    *,
    model: PaperRelationAdapter,
    args: argparse.Namespace,
    epoch: int,
    history: list[dict[str, Any]],
    text_dim: int,
    proposal_cache: Path,
    train_size: int,
    val_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "history": history,
            "args": vars(args),
            "text_dim": int(text_dim),
            "proposal_cache": str(proposal_cache),
            "train_samples": int(train_size),
            "val_samples": int(val_size),
        },
        path,
    )


def print_relation_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"Dataset:            {summary['dataset']}")
    print(f"Unique images:      {summary['unique_images']}")
    print(f"Evaluated samples:  {summary['evaluated_samples']}")
    print(f"Accuracy:           {summary['accuracy']:.4f} ({summary['accuracy'] * 100:.2f}%)")
    print(f"Abstain rate:       {summary['abstain_rate']:.4f} ({summary['abstain_rate'] * 100:.2f}%)")
    print(f"Target detect rate: {summary['target_detect_rate']:.4f} ({summary['target_detect_rate'] * 100:.2f}%)")
    print(f"Anchor detect rate: {summary['anchor_detect_rate']:.4f} ({summary['anchor_detect_rate'] * 100:.2f}%)")
    print(f"Pair available:     {summary['pair_available_rate']:.4f} ({summary['pair_available_rate'] * 100:.2f}%)")
    print("-" * 72)
    print("Per relation:")
    for relation, stats in summary["per_relation"].items():
        print(
            f"  {relation:>10} | n={stats['samples']:4d} | "
            f"acc={stats['accuracy']:.3f} | "
            f"pair={stats['pair_available_rate']:.3f}"
        )
    print("=" * 72)


def print_detection_summary(metrics: dict[str, Any], iou_thresh: float) -> None:
    print("-" * 72)
    print(f"Ground Truths (targets):    {metrics['ground_truths']}")
    print(f"Total Predictions made:     {metrics['predictions']}")
    print(f"True Positives (matched):   {metrics['true_positives']}")
    print(f"False Positives:            {metrics['false_positives']}")
    print(f"False Negatives:            {metrics['false_negatives']}")
    print("-" * 72)
    print(f"Precision@{iou_thresh}:          {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"Recall@{iou_thresh}:             {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"F1-Score@{iou_thresh}:           {metrics['f1']:.4f} ({metrics['f1'] * 100:.2f}%)")
    print(f"Accuracy@{iou_thresh}:           {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"mAP@{iou_thresh}:                {metrics['map']:.4f} ({metrics['map'] * 100:.2f}%)")
    print(
        f"REC@{iou_thresh}:                {metrics['rec']:.4f} ({metrics['rec'] * 100:.2f}%) "
        f"[{metrics['rec_matched_queries']}/{metrics['rec_total_queries']}]"
    )


def maybe_save_report(
    save_path: Path | None,
    *,
    proposal_stats: dict[str, Any],
    relation_summary: dict[str, Any],
    detection_metrics: dict[str, Any],
    all_gts: list[dict[str, Any]],
    all_preds: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
) -> None:
    if save_path is None:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "proposal_stats": proposal_stats,
                "relation_summary": relation_summary,
                "detection_metrics": detection_metrics,
                "ground_truths": all_gts,
                "predictions": all_preds,
                "details": detail_rows,
            },
            handle,
            indent=2,
        )
    print(f"Saved report to: {save_path}")


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", default=str(DEFAULT_JSON), help="Spatial-relation annotation JSON path.")
    parser.add_argument("--images", default=str(DEFAULT_IMAGES), help="Directory containing image files that match image_id values.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLOE checkpoint used to harvest proposals.")
    parser.add_argument("--cache-path", default=str(DEFAULT_CACHE_PATH), help="Offline text embedding cache path.")
    parser.add_argument(
        "--proposal-cache",
        default=str(DEFAULT_PROPOSAL_CACHE),
        help="Path to the YOLOE proposal cache used by this relation adapter.",
    )
    parser.add_argument("--conf", type=float, default=0.05, help="YOLOE confidence threshold used during proposal harvest.")
    parser.add_argument("--topk-target", type=int, default=20, help="Top-K target proposals kept per query.")
    parser.add_argument("--topk-anchor", type=int, default=10, help="Top-M anchor proposals kept per query.")
    parser.add_argument("--limit", type=int, default=0, help="Optional image cap for smoke tests.")
    parser.add_argument("--force-rebuild-cache", action="store_true", help="Rebuild the proposal cache even if it exists.")
    parser.add_argument("--verbose", action="store_true", help="Print extra cache-building diagnostics.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train or evaluate a paper-style object relation adapter on frozen YOLOE proposals "
            "using the configured dataset defaults."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train the paper-style relation adapter.")
    add_common_data_args(train)
    train.set_defaults(
        json=str(DEFAULT_NYU_RGB_JSON),
        images=str(DEFAULT_NYU_IMAGES),
        proposal_cache=str(DEFAULT_NYU_PROPOSAL_CACHE),
    )
    train.add_argument("--output-dir", default=str(DEFAULT_NYU_OUTPUT_DIR), help="Directory for checkpoints and training history.")
    train.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    train.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    train.add_argument("--workers", type=int, default=0, help="DataLoader workers.")
    train.add_argument("--device", default="", help="Torch device override.")
    train.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    train.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    train.add_argument("--val-fraction", type=float, default=0.1, help="Image-level validation fraction.")
    train.add_argument("--seed", type=int, default=42, help="Random seed.")
    train.add_argument("--crop-size", type=int, default=96, help="Proposal crop size.")
    train.add_argument("--positive-iou", type=float, default=0.5, help="IoU used to label positive target candidates.")
    train.add_argument("--pos-weight", type=float, default=4.0, help="Positive class weight for BCE.")
    train.add_argument(
        "--relation-score-weight",
        type=float,
        default=0.25,
        help="Extra margin loss weight encouraging positives to outrank negatives.",
    )
    train.add_argument("--feature-dim", type=int, default=512, help="Relation-module feature dimension.")
    train.add_argument("--crop-feature-dim", type=int, default=128, help="Crop encoder output dimension.")
    train.add_argument("--num-relations", type=int, default=8, help="Number of relation heads.")
    train.add_argument("--key-dim", type=int, default=64, help="Relation attention key/query dimension.")
    train.add_argument("--geom-dim", type=int, default=64, help="Pairwise geometry embedding dimension.")
    train.add_argument("--num-blocks", type=int, default=1, help="Number of stacked relation blocks.")
    train.add_argument(
        "--allow-train-without-anchor",
        action="store_true",
        help="Keep training samples even when no anchor candidates were detected.",
    )

    evaluate = subparsers.add_parser("eval", help="Evaluate a trained paper-style relation adapter.")
    add_common_data_args(evaluate)
    evaluate.set_defaults(
        json=str(INDOORS_SUBSET_DIR / "filtered_indoors_LM_vg.json"),
        images=str(DEFAULT_IMAGES),
        proposal_cache=str(DEFAULT_OUTPUT_DIR / "proposal_cache_eval_filtered_indoors_LM_vg.pt"),
    )
    evaluate.add_argument("--checkpoint", required=True, help="Path to a trained relation adapter checkpoint.")
    evaluate.add_argument("--device", default="", help="Torch device override.")
    evaluate.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    evaluate.add_argument("--workers", type=int, default=0, help="DataLoader workers.")
    evaluate.add_argument("--crop-size", type=int, default=96, help="Proposal crop size.")
    evaluate.add_argument("--positive-iou", type=float, default=0.5, help="IoU used to label positive target candidates.")
    evaluate.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold used for evaluation metrics.")
    evaluate.add_argument("--report-json", default="", help="Optional JSON report path.")

    return parser


def train_main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    json_path = resolve_json_path(args.json)
    images_dir = resolve_path(args.images)
    weights = resolve_path(args.weights)
    cache_path = resolve_path(args.cache_path)
    proposal_cache_path = resolve_path(args.proposal_cache)
    output_dir = resolve_path(args.output_dir)

    print(f"Using annotations:  {json_path}")
    print(f"Using images:       {images_dir}")
    print(f"Using YOLOE:        {weights}")
    print(f"Using offline cache:{cache_path}")
    print(f"Using proposals:    {proposal_cache_path}")
    print(f"Using output dir:   {output_dir}")
    print(f"Using device:       {device}")

    proposal_payload = build_proposal_cache(
        json_path=json_path,
        images_dir=images_dir,
        weights=weights,
        cache_path=cache_path,
        proposal_cache_path=proposal_cache_path,
        conf_thresh=args.conf,
        topk_target=args.topk_target,
        topk_anchor=args.topk_anchor,
        limit=args.limit,
        verbose=args.verbose,
        force_rebuild=args.force_rebuild_cache,
    )
    prompt_embeddings = proposal_payload["prompt_embeddings"]
    text_dim = int(next(iter(prompt_embeddings.values())).numel())
    train_episodes, val_episodes = split_episodes_by_image(
        proposal_payload["episodes"],
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    train_dataset = ProposalEpisodeDataset(
        train_episodes,
        prompt_embeddings=prompt_embeddings,
        crop_size=args.crop_size,
        positive_iou=args.positive_iou,
        require_positive=True,
        require_anchor=not args.allow_train_without_anchor,
    )
    val_dataset = ProposalEpisodeDataset(
        val_episodes,
        prompt_embeddings=prompt_embeddings,
        crop_size=args.crop_size,
        positive_iou=args.positive_iou,
        require_positive=True,
        require_anchor=False,
    )

    if not len(train_dataset):
        raise RuntimeError(
            "Training split is empty after filtering. Try lowering --positive-iou, increasing --topk-target, "
            "or enabling --allow-train-without-anchor."
        )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_proposal_batch,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs) if len(val_dataset) else None

    model = build_model_from_config(text_dim, args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    print()
    print("Dataset summary")
    print(f"  proposal cache episodes:   {len(proposal_payload['episodes'])}")
    print(f"  train episodes kept:       {len(train_dataset)}")
    print(f"  val episodes kept:         {len(val_dataset)}")
    print(f"  prompt embedding dim:      {text_dim}")
    print(f"  train skip stats:          {train_dataset.skip_stats}")
    print(f"  val skip stats:            {val_dataset.skip_stats}")
    print(f"  proposal cache stats:      {proposal_payload['stats']}")
    print()

    history: list[dict[str, Any]] = []
    best_metric = -float("inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_top1 = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            relation_score_weight=args.relation_score_weight,
            pos_weight=args.pos_weight,
        )
        if val_loader is not None:
            val_loss, val_top1 = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                relation_score_weight=args.relation_score_weight,
                pos_weight=args.pos_weight,
            )
        else:
            val_loss = math.nan
            val_top1 = math.nan

        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_top1={train_top1:.4f} | "
            f"val_loss={val_loss:.4f} val_top1={val_top1:.4f}"
        )

        score = val_top1 if not math.isnan(val_top1) else train_top1
        if score > best_metric:
            best_metric = score
            save_checkpoint(
                best_path,
                model=model,
                args=args,
                epoch=epoch,
                history=history,
                text_dim=text_dim,
                proposal_cache=proposal_cache_path,
                train_size=len(train_dataset),
                val_size=len(val_dataset),
            )

        save_checkpoint(
            last_path,
            model=model,
            args=args,
            epoch=epoch,
            history=history,
            text_dim=text_dim,
            proposal_cache=proposal_cache_path,
            train_size=len(train_dataset),
            val_size=len(val_dataset),
        )

    history_path = output_dir / "train_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "history": history,
                "proposal_stats": proposal_payload["stats"],
                "train_skip_stats": train_dataset.skip_stats,
                "val_skip_stats": val_dataset.skip_stats,
            },
            handle,
            indent=2,
        )
    print(f"\nSaved best checkpoint to: {best_path}")
    print(f"Saved last checkpoint to: {last_path}")
    print(f"Saved history to:         {history_path}")


def eval_main(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    json_path = resolve_json_path(args.json)
    images_dir = resolve_path(args.images)
    weights = resolve_path(args.weights)
    cache_path = resolve_path(args.cache_path)
    proposal_cache_path = resolve_path(args.proposal_cache)
    checkpoint_path = resolve_path(args.checkpoint)
    report_json = resolve_path(args.report_json) if args.report_json else None

    print(f"Using annotations:  {json_path}")
    print(f"Using images:       {images_dir}")
    print(f"Using YOLOE:        {weights}")
    print(f"Using proposals:    {proposal_cache_path}")
    print(f"Using checkpoint:   {checkpoint_path}")
    print(f"Using device:       {device}")

    proposal_payload = build_proposal_cache(
        json_path=json_path,
        images_dir=images_dir,
        weights=weights,
        cache_path=cache_path,
        proposal_cache_path=proposal_cache_path,
        conf_thresh=args.conf,
        topk_target=args.topk_target,
        topk_anchor=args.topk_anchor,
        limit=args.limit,
        verbose=args.verbose,
        force_rebuild=args.force_rebuild_cache,
    )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    prompt_embeddings = proposal_payload["prompt_embeddings"]
    text_dim = int(checkpoint.get("text_dim") or next(iter(prompt_embeddings.values())).numel())

    config_source = argparse.Namespace(**checkpoint["args"])
    model = build_model_from_config(text_dim, config_source).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    dataset = ProposalEpisodeDataset(
        proposal_payload["episodes"],
        prompt_embeddings=prompt_embeddings,
        crop_size=args.crop_size,
        positive_iou=args.positive_iou,
        require_positive=False,
        require_anchor=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=collate_proposal_batch,
    )

    relation_summary, detection_metrics, all_gts, all_preds, detail_rows = evaluate_model(
        model,
        loader,
        device=device,
        iou_thresh=args.iou_thresh,
    )

    print_relation_summary(relation_summary)
    print_detection_summary(detection_metrics, args.iou_thresh)
    maybe_save_report(
        report_json,
        proposal_stats=proposal_payload["stats"],
        relation_summary=relation_summary,
        detection_metrics=detection_metrics,
        all_gts=all_gts,
        all_preds=all_preds,
        detail_rows=detail_rows,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "train":
        train_main(args)
    elif args.command == "eval":
        eval_main(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
