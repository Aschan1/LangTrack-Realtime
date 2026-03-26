#!/usr/bin/env python3
"""
Shared utilities for a Dynamic Head style relation adapter on top of offline YOLOE proposals.

This module reuses the generic YOLOE proposal-cache flow from the earlier paper-style adapter,
but replaces the proposal encoder with a DyHead-inspired head based on:
  - scale-aware attention across pyramid levels
  - spatial-aware aggregation with deformable-conv fallback
  - task-aware dynamic ReLU

Practical note:
The original paper applies DyHead to detector feature pyramids. In this repo YOLOE does not
expose a clean offline head plugin path with intermediate FPN tensors, so this adapter builds a
small per-proposal feature pyramid from proposal crops and then applies DyHead blocks there.
The training/eval flow and metrics still operate on frozen YOLOE proposals from the default
indoors spatial-relation dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from yoloe_paper_relation_adapter import (
        DEFAULT_CACHE_PATH,
        DEFAULT_IMAGES,
        DEFAULT_JSON,
        DEFAULT_WEIGHTS,
        ProposalEpisodeDataset,
        build_proposal_cache,
        choose_device,
        collate_proposal_batch,
        compute_detection_metrics,
        ensure_runtime_imports,
        finalize_relation_summary,
        move_batch_to_device,
        print_detection_summary,
        print_relation_summary,
        resolve_json_path,
        resolve_path,
        set_seed,
        summarize_query_predictions,
        update_relation_stats,
        empty_relation_summary,
    )
except ModuleNotFoundError:
    from detect.yoloe_paper_relation_adapter import (
        DEFAULT_CACHE_PATH,
        DEFAULT_IMAGES,
        DEFAULT_JSON,
        DEFAULT_WEIGHTS,
        ProposalEpisodeDataset,
        build_proposal_cache,
        choose_device,
        collate_proposal_batch,
        compute_detection_metrics,
        ensure_runtime_imports,
        finalize_relation_summary,
        move_batch_to_device,
        print_detection_summary,
        print_relation_summary,
        resolve_json_path,
        resolve_path,
        set_seed,
        summarize_query_predictions,
        update_relation_stats,
        empty_relation_summary,
    )


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "dyhead_relation_adapter"
DEFAULT_PROPOSAL_CACHE = DEFAULT_OUTPUT_DIR / "proposal_cache.pt"
DEFAULT_RELATION_CACHE = DEFAULT_OUTPUT_DIR / "relation_text_cache.pt"


def build_relation_vocab(episodes: list[dict[str, Any]]) -> dict[str, int]:
    return {relation: index for index, relation in enumerate(sorted({episode["relation"] for episode in episodes}))}


def format_relation_prompt(relation: str, prompt_style: str) -> str:
    relation = str(relation).strip()
    prompt_style = str(prompt_style).strip().lower()
    if prompt_style == "raw":
        return relation
    if prompt_style == "template":
        return f"object {relation} object"
    if prompt_style == "framed":
        return f"a target object {relation} an anchor object"
    raise ValueError(f"Unsupported relation prompt style: {prompt_style}")


def build_relation_text_cache(
    episodes: list[dict[str, Any]],
    *,
    weights: Path,
    relation_cache_path: Path,
    force_rebuild: bool,
    prompt_style: str,
) -> dict[str, torch.Tensor]:
    relation_cache_path = relation_cache_path.expanduser().resolve()
    relations = sorted({str(episode["relation"]).strip() for episode in episodes if str(episode["relation"]).strip()})
    prompt_map = {relation: format_relation_prompt(relation, prompt_style) for relation in relations}

    if relation_cache_path.is_file() and not force_rebuild:
        payload = torch.load(str(relation_cache_path), map_location="cpu", weights_only=False)
        cached_embeddings = payload.get("embeddings", {})
        cached_style = str(payload.get("prompt_style", "framed"))
        if cached_style == prompt_style and all(relation in cached_embeddings for relation in relations):
            return {
                relation: cached_embeddings[relation].float().contiguous()
                for relation in relations
            }

    YOLOE_cls, _, _ = ensure_runtime_imports()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLOE_cls(str(weights)).to(device)
    model.model.eval()

    prompts = [prompt_map[relation] for relation in relations]
    with torch.no_grad():
        encoded = model.model.get_text_pe(prompts, cache_clip_model=True).squeeze(0).detach().cpu()

    relation_embeddings = {
        relation: embedding.float().contiguous()
        for relation, embedding in zip(relations, encoded, strict=False)
    }
    relation_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": relation_embeddings,
            "prompt_style": prompt_style,
            "prompt_map": prompt_map,
            "weights": str(weights),
        },
        relation_cache_path,
    )
    return relation_embeddings


def build_dyhead_collate_fn(relation_to_id: dict[str, int], relation_text_embeddings: dict[str, torch.Tensor]):
    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated = collate_proposal_batch(batch)
        collated["relation_ids"] = torch.tensor(
            [relation_to_id[sample["relation"]] for sample in batch],
            dtype=torch.long,
        )
        collated["relation_text_feats"] = torch.stack(
            [relation_text_embeddings[sample["relation"]].float().contiguous() for sample in batch],
            dim=0,
        )
        return collated

    return _collate


def pair_geometry_features(
    target_boxes: torch.Tensor,
    anchor_boxes: torch.Tensor,
    image_wh: torch.Tensor,
    target_confs: torch.Tensor,
    anchor_confs: torch.Tensor,
    feature_dim: int,
) -> torch.Tensor:
    target_w = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=1.0)
    target_h = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=1.0)
    anchor_w = (anchor_boxes[:, 2] - anchor_boxes[:, 0]).clamp(min=1.0)
    anchor_h = (anchor_boxes[:, 3] - anchor_boxes[:, 1]).clamp(min=1.0)

    target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2.0
    target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2.0
    anchor_cx = (anchor_boxes[:, 0] + anchor_boxes[:, 2]) / 2.0
    anchor_cy = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2.0

    width = image_wh[0].clamp(min=1.0)
    height = image_wh[1].clamp(min=1.0)

    delta_x = (anchor_cx - target_cx) / width
    delta_y = (anchor_cy - target_cy) / height
    delta_w = torch.log(anchor_w / target_w)
    delta_h = torch.log(anchor_h / target_h)

    inter_x1 = torch.maximum(target_boxes[:, 0], anchor_boxes[:, 0])
    inter_y1 = torch.maximum(target_boxes[:, 1], anchor_boxes[:, 1])
    inter_x2 = torch.minimum(target_boxes[:, 2], anchor_boxes[:, 2])
    inter_y2 = torch.minimum(target_boxes[:, 3], anchor_boxes[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter = inter_w * inter_h
    target_area = target_w * target_h
    anchor_area = anchor_w * anchor_h
    union = (target_area + anchor_area - inter).clamp(min=1e-6)
    iou = inter / union

    overlap_x = inter_w / torch.minimum(target_w, anchor_w).clamp(min=1e-6)
    overlap_y = inter_h / torch.minimum(target_h, anchor_h).clamp(min=1e-6)
    abs_delta_x = delta_x.abs()
    abs_delta_y = delta_y.abs()
    center_distance = torch.sqrt(abs_delta_x.square() + abs_delta_y.square() + 1e-6)
    target_inside_anchor = inter / target_area.clamp(min=1e-6)
    anchor_inside_target = inter / anchor_area.clamp(min=1e-6)
    anchor_is_right = (anchor_cx >= target_cx).float()
    anchor_is_below = (anchor_cy >= target_cy).float()

    features = torch.stack(
        [
            delta_x,
            delta_y,
            delta_w,
            delta_h,
            iou,
            overlap_x,
            overlap_y,
            target_confs,
            anchor_confs,
            abs_delta_x,
            abs_delta_y,
            center_distance,
            target_inside_anchor,
            anchor_inside_target,
            anchor_is_right,
            anchor_is_below,
        ],
        dim=-1,
    )
    if features.shape[-1] < feature_dim:
        features = F.pad(features, (0, feature_dim - features.shape[-1]))
    return features[..., :feature_dim]


def proposal_metadata_features(
    boxes: torch.Tensor,
    confs: torch.Tensor,
    role_ids: torch.Tensor,
    image_wh: torch.Tensor,
) -> torch.Tensor:
    widths = image_wh[:, 0].clamp(min=1.0)
    heights = image_wh[:, 1].clamp(min=1.0)
    x1 = boxes[:, 0] / widths
    y1 = boxes[:, 1] / heights
    x2 = boxes[:, 2] / widths
    y2 = boxes[:, 3] / heights
    bw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0) / widths
    bh = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0) / heights
    cx = ((boxes[:, 0] + boxes[:, 2]) / 2.0) / widths
    cy = ((boxes[:, 1] + boxes[:, 3]) / 2.0) / heights
    area = bw * bh
    aspect = torch.log(torch.clamp(bw / bh.clamp(min=1e-6), min=1e-4))
    role_one_hot = F.one_hot(role_ids.clamp(min=0), num_classes=2).float()
    return torch.cat(
        [
            x1.unsqueeze(-1),
            y1.unsqueeze(-1),
            x2.unsqueeze(-1),
            y2.unsqueeze(-1),
            cx.unsqueeze(-1),
            cy.unsqueeze(-1),
            bw.unsqueeze(-1),
            bh.unsqueeze(-1),
            area.unsqueeze(-1),
            aspect.unsqueeze(-1),
            confs.unsqueeze(-1),
            role_one_hot,
        ],
        dim=-1,
    )


class SpatialAggregationConv(nn.Module):
    def __init__(self, channels: int, *, stride: int = 1, use_deform: bool = True) -> None:
        super().__init__()
        self.use_deform = False
        self.offset_conv = None
        self.mask_conv = None
        self.conv = None

        if use_deform:
            try:
                from torchvision.ops import DeformConv2d

                self.offset_conv = nn.Conv2d(channels, 18, kernel_size=3, stride=stride, padding=1)
                self.mask_conv = nn.Conv2d(channels, 9, kernel_size=3, stride=stride, padding=1)
                self.conv = DeformConv2d(channels, channels, kernel_size=3, stride=stride, padding=1)
                self.use_deform = True
            except Exception:
                self.use_deform = False

        if not self.use_deform:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1)

        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_deform:
            offset = self.offset_conv(x)
            mask = self.mask_conv(x).sigmoid()
            x = self.conv(x, offset, mask)
        else:
            x = self.conv(x)
        return self.act(self.norm(x))


class ScaleAwareAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.hardsigmoid(self.proj(self.pool(x)))


class DyReLU(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels * 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = self.fc2(F.relu(self.fc1(self.pool(x)), inplace=False))
        a1, b1, a2, b2 = coeffs.chunk(4, dim=1)
        a1 = 2.0 * torch.sigmoid(a1)
        a2 = 2.0 * torch.sigmoid(a2) - 1.0
        b1 = 2.0 * torch.sigmoid(b1) - 1.0
        b2 = 2.0 * torch.sigmoid(b2) - 1.0
        return torch.maximum(a1 * x + b1, a2 * x + b2)


class DyHeadBlock(nn.Module):
    def __init__(self, channels: int, *, use_deform: bool = True) -> None:
        super().__init__()
        self.mid_conv = SpatialAggregationConv(channels, stride=1, use_deform=use_deform)
        self.high_conv = SpatialAggregationConv(channels, stride=1, use_deform=use_deform)
        self.low_conv = SpatialAggregationConv(channels, stride=2, use_deform=use_deform)
        self.scale_attn = ScaleAwareAttention(channels)
        self.task_attn = DyReLU(channels)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = []
        total_levels = len(features)
        for level in range(total_levels):
            current = self.mid_conv(features[level])
            current_attn = self.scale_attn(current)
            fused = current * current_attn
            norm = current_attn

            if level > 0:
                low = self.low_conv(features[level - 1])
                if low.shape[-2:] != current.shape[-2:]:
                    low = F.interpolate(low, size=current.shape[-2:], mode="bilinear", align_corners=False)
                low_attn = self.scale_attn(low)
                fused = fused + low * low_attn
                norm = norm + low_attn

            if level < total_levels - 1:
                high = self.high_conv(features[level + 1])
                if high.shape[-2:] != current.shape[-2:]:
                    high = F.interpolate(high, size=current.shape[-2:], mode="bilinear", align_corners=False)
                high_attn = self.scale_attn(high)
                fused = fused + high * high_attn
                norm = norm + high_attn

            outputs.append(self.task_attn(fused / norm.clamp(min=1e-6)))
        return outputs


class ProposalPyramidEncoder(nn.Module):
    def __init__(self, channels: int, *, num_levels: int) -> None:
        super().__init__()
        assert num_levels >= 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, channels // 2),
            nn.SiLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )
        self.levels = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, channels),
                nn.SiLU(),
            )
            for _ in range(num_levels - 1)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        current = self.stem(x)
        outputs.append(current)
        for layer in self.levels:
            current = layer(current)
            outputs.append(current)
        return outputs


class DyHeadRelationAdapter(nn.Module):
    def __init__(
        self,
        *,
        text_dim: int,
        relation_text_dim: int,
        relation_vocab_size: int,
        channels: int,
        num_levels: int,
        num_blocks: int,
        text_proj_dim: int,
        relation_dim: int,
        pair_geo_dim: int,
        pair_lambda: float,
        use_deform: bool,
        support_agg: str,
        use_meta_context: bool,
        use_relation_geo_bias: bool,
        use_relation_text: bool,
        use_learned_relation_residual: bool,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_levels = int(num_levels)
        self.text_proj_dim = int(text_proj_dim)
        self.relation_dim = int(relation_dim)
        self.pair_geo_dim = int(pair_geo_dim)
        self.pair_lambda = float(pair_lambda)
        self.support_agg = str(support_agg)
        self.use_meta_context = bool(use_meta_context)
        self.use_relation_geo_bias = bool(use_relation_geo_bias)
        self.use_relation_text = bool(use_relation_text)
        self.use_learned_relation_residual = bool(use_learned_relation_residual)

        self.encoder = ProposalPyramidEncoder(channels, num_levels=num_levels)
        self.blocks = nn.ModuleList(DyHeadBlock(channels, use_deform=use_deform) for _ in range(num_blocks))
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_proj_dim),
            nn.LayerNorm(text_proj_dim),
            nn.SiLU(),
        )
        if self.use_meta_context:
            self.meta_proj = nn.Sequential(
                nn.Linear(13, channels),
                nn.LayerNorm(channels),
                nn.SiLU(),
                nn.Linear(channels, channels),
            )
            self.context_proj = nn.Sequential(
                nn.Linear(text_proj_dim + channels, channels * 2),
                nn.LayerNorm(channels * 2),
                nn.SiLU(),
                nn.Linear(channels * 2, num_levels * channels * 2),
            )
        else:
            self.meta_proj = None
            self.context_proj = None
        if self.use_relation_text:
            self.relation_text_proj = nn.Sequential(
                nn.Linear(relation_text_dim, relation_dim),
                nn.LayerNorm(relation_dim),
                nn.SiLU(),
            )
        else:
            self.relation_text_proj = None
        self.relation_embedding = (
            nn.Embedding(relation_vocab_size, relation_dim) if self.use_learned_relation_residual else None
        )
        self.base_head = nn.Sequential(
            nn.LayerNorm(channels + text_proj_dim + relation_dim),
            nn.Linear(channels + text_proj_dim + relation_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, 1),
        )
        self.pair_head = nn.Sequential(
            nn.LayerNorm(channels * 2 + text_proj_dim * 2 + relation_dim + pair_geo_dim),
            nn.Linear(channels * 2 + text_proj_dim * 2 + relation_dim + pair_geo_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, 1),
        )
        if self.use_relation_geo_bias:
            self.geo_bias_head = nn.Sequential(
                nn.Linear(relation_dim + pair_geo_dim, relation_dim),
                nn.LayerNorm(relation_dim),
                nn.SiLU(),
                nn.Linear(relation_dim, 1),
            )
        else:
            self.geo_bias_head = None

    def encode_valid_proposals(
        self,
        crops: torch.Tensor,
        text_features: torch.Tensor,
        meta_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        levels = self.encoder(crops)
        if self.use_meta_context:
            assert meta_features is not None
            meta_context = self.meta_proj(meta_features)
            context = self.context_proj(torch.cat([text_features, meta_context], dim=-1))
            context = context.view(crops.shape[0], self.num_levels, 2, self.channels)
            for level_index, level in enumerate(levels):
                scale = 1.0 + 0.25 * torch.tanh(context[:, level_index, 0]).unsqueeze(-1).unsqueeze(-1)
                bias = context[:, level_index, 1].unsqueeze(-1).unsqueeze(-1)
                levels[level_index] = level * scale + bias
        for block in self.blocks:
            levels = block(levels)
        pooled = torch.stack([feature.mean(dim=(2, 3)) for feature in levels], dim=0).mean(dim=0)
        return pooled

    def forward(
        self,
        *,
        crops: torch.Tensor,
        text_feats: torch.Tensor,
        boxes: torch.Tensor,
        confs: torch.Tensor,
        role_ids: torch.Tensor,
        image_wh: torch.Tensor,
        relation_ids: torch.Tensor,
        relation_text_feats: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_props = mask.shape
        flat_mask = mask.reshape(-1)
        flat_crops = crops.reshape(batch_size * max_props, *crops.shape[2:])
        text_features = self.text_proj(text_feats)
        flat_text_features = text_features.reshape(batch_size * max_props, -1)
        flat_boxes = boxes.reshape(batch_size * max_props, 4)
        flat_confs = confs.reshape(batch_size * max_props)
        flat_role_ids = role_ids.reshape(batch_size * max_props)
        flat_image_wh = image_wh[:, None, :].expand(batch_size, max_props, 2).reshape(batch_size * max_props, 2)
        proposal_features = torch.zeros(
            batch_size * max_props,
            self.channels,
            device=crops.device,
            dtype=crops.dtype,
        )
        if flat_mask.any():
            meta_features = None
            if self.use_meta_context:
                meta_features = proposal_metadata_features(
                    flat_boxes[flat_mask],
                    flat_confs[flat_mask],
                    flat_role_ids[flat_mask],
                    flat_image_wh[flat_mask],
                )
            proposal_features[flat_mask] = self.encode_valid_proposals(
                flat_crops[flat_mask],
                flat_text_features[flat_mask],
                meta_features,
            )
        proposal_features = proposal_features.view(batch_size, max_props, self.channels)
        relation_features = None
        if self.relation_text_proj is not None:
            assert relation_text_feats is not None
            relation_features = self.relation_text_proj(relation_text_feats)
        if self.relation_embedding is not None:
            learned_relation = self.relation_embedding(relation_ids)
            relation_features = learned_relation if relation_features is None else relation_features + learned_relation
        if relation_features is None:
            raise RuntimeError("DyHeadRelationAdapter needs at least one relation feature source.")

        logits = torch.full((batch_size, max_props), -1e9, device=crops.device, dtype=crops.dtype)
        for batch_index in range(batch_size):
            sample_mask = mask[batch_index]
            sample_target_mask = sample_mask & (role_ids[batch_index] == 0)
            sample_anchor_mask = sample_mask & (role_ids[batch_index] == 1)
            relation_feature = relation_features[batch_index]

            if not sample_target_mask.any():
                continue

            target_feat = proposal_features[batch_index, sample_target_mask]
            target_text = text_features[batch_index, sample_target_mask]
            target_boxes = boxes[batch_index, sample_target_mask]
            target_confs = confs[batch_index, sample_target_mask]
            target_relation = relation_feature.unsqueeze(0).expand(target_feat.shape[0], -1)
            base_logits = self.base_head(torch.cat([target_feat, target_text, target_relation], dim=-1)).squeeze(-1)

            if sample_anchor_mask.any():
                anchor_feat = proposal_features[batch_index, sample_anchor_mask]
                anchor_text = text_features[batch_index, sample_anchor_mask]
                anchor_boxes = boxes[batch_index, sample_anchor_mask]
                anchor_confs = confs[batch_index, sample_anchor_mask]

                target_count = target_feat.shape[0]
                anchor_count = anchor_feat.shape[0]
                expanded_target_feat = target_feat.unsqueeze(1).expand(target_count, anchor_count, -1)
                expanded_anchor_feat = anchor_feat.unsqueeze(0).expand(target_count, anchor_count, -1)
                expanded_target_text = target_text.unsqueeze(1).expand(target_count, anchor_count, -1)
                expanded_anchor_text = anchor_text.unsqueeze(0).expand(target_count, anchor_count, -1)
                expanded_relation = relation_feature.view(1, 1, -1).expand(target_count, anchor_count, -1)
                geo = pair_geometry_features(
                    target_boxes.unsqueeze(1).expand(target_count, anchor_count, 4).reshape(-1, 4),
                    anchor_boxes.unsqueeze(0).expand(target_count, anchor_count, 4).reshape(-1, 4),
                    image_wh[batch_index],
                    target_confs.unsqueeze(1).expand(target_count, anchor_count).reshape(-1),
                    anchor_confs.unsqueeze(0).expand(target_count, anchor_count).reshape(-1),
                    self.pair_geo_dim,
                ).view(target_count, anchor_count, -1)

                pair_inputs = torch.cat(
                    [
                        expanded_target_feat,
                        expanded_anchor_feat,
                        expanded_target_text,
                        expanded_anchor_text,
                        expanded_relation,
                        geo,
                    ],
                    dim=-1,
                )
                pair_logits = self.pair_head(pair_inputs).squeeze(-1)
                if self.use_relation_geo_bias:
                    pair_logits = pair_logits + self.geo_bias_head(
                        torch.cat([expanded_relation, geo], dim=-1)
                    ).squeeze(-1)
                if self.support_agg == "max":
                    support = pair_logits.max(dim=1).values
                else:
                    support = torch.logsumexp(pair_logits, dim=1) - math.log(float(anchor_count))
                final_logits = base_logits + self.pair_lambda * support
            else:
                final_logits = base_logits

            target_indices = torch.where(sample_target_mask)[0]
            logits[batch_index, target_indices] = final_logits

        return logits


def compute_loss(
    logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    pos_weight: float,
    relation_margin_weight: float,
    listwise_weight: float,
    hard_negative_k: int,
    hard_negative_margin: float,
) -> torch.Tensor:
    target_mask = batch["mask"] & (batch["role_ids"] == 0) & (batch["target_labels"] >= 0.0)
    if not target_mask.any():
        return logits.sum() * 0.0

    bce = F.binary_cross_entropy_with_logits(
        logits[target_mask],
        batch["target_labels"][target_mask],
        pos_weight=torch.tensor(float(pos_weight), device=logits.device),
    )

    listwise_terms = []
    hard_negative_terms = []
    for batch_index in range(logits.shape[0]):
        sample_mask = batch["mask"][batch_index] & (batch["role_ids"][batch_index] == 0) & (batch["target_labels"][batch_index] >= 0.0)
        if not sample_mask.any():
            continue

        sample_logits = logits[batch_index, sample_mask]
        sample_labels = batch["target_labels"][batch_index, sample_mask]
        positive_mask = sample_labels > 0.5
        if not positive_mask.any():
            continue

        if listwise_weight > 0.0:
            listwise_terms.append(torch.logsumexp(sample_logits, dim=0) - torch.logsumexp(sample_logits[positive_mask], dim=0))

        if relation_margin_weight > 0.0:
            negative_logits = sample_logits[~positive_mask]
            if negative_logits.numel():
                positive_best = sample_logits[positive_mask].max()
                if hard_negative_k > 0:
                    k = min(int(hard_negative_k), int(negative_logits.numel()))
                    negative_logits = negative_logits.topk(k).values
                hard_negative_terms.append(F.relu(negative_logits - positive_best + float(hard_negative_margin)).mean())

    loss = bce
    if listwise_terms:
        loss = loss + float(listwise_weight) * torch.stack(listwise_terms).mean()
    if hard_negative_terms:
        loss = loss + float(relation_margin_weight) * torch.stack(hard_negative_terms).mean()
    return loss


def run_epoch(
    model: DyHeadRelationAdapter,
    loader: DataLoader,
    *,
    optimizer: AdamW | None,
    device: torch.device,
    pos_weight: float,
    relation_margin_weight: float,
    listwise_weight: float,
    hard_negative_k: int,
    hard_negative_margin: float,
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
            relation_ids=batch["relation_ids"],
            relation_text_feats=batch["relation_text_feats"],
            mask=batch["mask"],
        )
        loss = compute_loss(
            logits,
            batch,
            pos_weight=pos_weight,
            relation_margin_weight=relation_margin_weight,
            listwise_weight=listwise_weight,
            hard_negative_k=hard_negative_k,
            hard_negative_margin=hard_negative_margin,
        )

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        fused_scores = (torch.sigmoid(logits) * batch["confs"]).detach().cpu()
        for batch_index, meta in enumerate(batch["meta"]):
            target_mask = (batch["mask"][batch_index] & (batch["role_ids"][batch_index] == 0)).detach().cpu()
            target_indices = torch.where(target_mask)[0].tolist()
            total_queries += 1
            if not target_indices:
                continue
            best_index = max(target_indices, key=lambda proposal_index: float(fused_scores[batch_index, proposal_index]))
            pred_box = batch["boxes"][batch_index, best_index].detach().cpu().tolist()
            best_iou = max(
                (
                    _box_iou(pred_box, gt_box)
                    for gt_box in meta["target_gt_boxes"]
                ),
                default=0.0,
            )
            total_correct += int(best_iou >= 0.5)

        progress.set_postfix(
            loss=f"{(total_loss / max(len(loader), 1)):.4f}",
            top1=f"{(total_correct / total_queries if total_queries else 0.0):.3f}",
        )

    return total_loss / max(len(loader), 1), (total_correct / total_queries if total_queries else 0.0)


def _box_iou(box1, box2) -> float:
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0.0, x2_inter - x1_inter) * max(0.0, y2_inter - y1_inter)
    if inter_area == 0.0:
        return 0.0
    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = box1_area + box2_area - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def evaluate_model(
    model: DyHeadRelationAdapter,
    loader: DataLoader,
    *,
    device: torch.device,
    iou_thresh: float,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    all_gts: list[dict[str, Any]] = []
    all_preds: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    summary = empty_relation_summary("indoors_subset", unique_images=0)
    image_ids = set()

    progress = tqdm(loader, desc="Evaluating DyHead adapter", leave=False)
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
                relation_ids=batch["relation_ids"],
                relation_text_feats=batch["relation_text_feats"],
                mask=batch["mask"],
            )
            fused_scores = torch.sigmoid(logits) * batch["confs"]

            for batch_index, meta in enumerate(batch["meta"]):
                image_ids.add(meta["image_id"])
                target_mask = batch["mask"][batch_index] & (batch["role_ids"][batch_index] == 0)
                anchor_mask = batch["mask"][batch_index] & (batch["role_ids"][batch_index] == 1)
                target_indices = torch.where(target_mask)[0].tolist()
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


def save_checkpoint(
    path: Path,
    *,
    model: DyHeadRelationAdapter,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    epoch: int,
    text_dim: int,
    relation_text_dim: int,
    relation_to_id: dict[str, int],
    proposal_cache: Path,
    relation_cache: Path,
    train_size: int,
    val_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "history": history,
            "epoch": int(epoch),
            "text_dim": int(text_dim),
            "relation_text_dim": int(relation_text_dim),
            "relation_to_id": relation_to_id,
            "proposal_cache": str(proposal_cache),
            "relation_cache": str(relation_cache),
            "train_samples": int(train_size),
            "val_samples": int(val_size),
        },
        path,
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


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", default=str(DEFAULT_JSON), help="Spatial-relation annotation JSON path.")
    parser.add_argument("--images", default=str(DEFAULT_IMAGES), help="Directory containing image files that match image_id values.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLOE checkpoint used to harvest proposals.")
    parser.add_argument("--cache-path", default=str(DEFAULT_CACHE_PATH), help="Offline text embedding cache path.")
    parser.add_argument(
        "--proposal-cache",
        default=str(DEFAULT_PROPOSAL_CACHE),
        help="Path to the offline YOLOE proposal cache for the DyHead adapter.",
    )
    parser.add_argument(
        "--relation-cache",
        default=str(DEFAULT_RELATION_CACHE),
        help="Path to the dedicated offline relation-text embedding cache for the DyHead adapter.",
    )
    parser.add_argument("--conf", type=float, default=0.05, help="YOLOE confidence threshold used during proposal harvest.")
    parser.add_argument("--topk-target", type=int, default=20, help="Top-K target proposals kept per query.")
    parser.add_argument("--topk-anchor", type=int, default=10, help="Top-M anchor proposals kept per query.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on images for smoke tests.")
    parser.add_argument("--force-rebuild-cache", action="store_true", help="Rebuild the proposal cache even if it exists.")
    parser.add_argument("--force-rebuild-relation-cache", action="store_true", help="Rebuild the relation-text cache even if it exists.")
    parser.add_argument(
        "--relation-prompt-style",
        choices=("template", "framed", "raw"),
        default="template",
        help="How to phrase spatial relations before encoding them into the dedicated relation cache.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra proposal-cache diagnostics.")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--channels", type=int, default=128, help="DyHead internal channel width.")
    parser.add_argument("--num-levels", type=int, default=3, help="Number of crop-pyramid levels.")
    parser.add_argument("--num-blocks", type=int, default=1, help="Number of stacked DyHead blocks.")
    parser.add_argument("--text-proj-dim", type=int, default=128, help="Prompt embedding projection width.")
    parser.add_argument("--relation-dim", type=int, default=64, help="Relation embedding width.")
    parser.add_argument("--pair-geo-dim", type=int, default=16, help="Pair geometry feature width.")
    parser.add_argument("--pair-lambda", type=float, default=1.0, help="Weight applied to aggregated anchor support.")
    parser.add_argument("--support-agg", choices=("logsumexp", "max"), default="max", help="How to aggregate pair support over anchors.")
    parser.add_argument("--enable-deform", dest="use_deform", action="store_true", help="Enable deformable convolutions in spatial attention.")
    parser.add_argument("--disable-deform", dest="use_deform", action="store_false", help="Use standard convolutions instead of deformable ones in spatial attention.")
    parser.add_argument("--disable-meta-context", dest="use_meta_context", action="store_false", help="Disable early box-metadata conditioning before DyHead blocks.")
    parser.add_argument("--disable-relation-geo-bias", dest="use_relation_geo_bias", action="store_false", help="Disable the relation-conditioned geometry bias added to pair logits.")
    parser.add_argument("--disable-relation-text", dest="use_relation_text", action="store_false", help="Disable the dedicated relation-text cache features and fall back to learned relation IDs only.")
    parser.add_argument("--disable-learned-relation-residual", dest="use_learned_relation_residual", action="store_false", help="Disable the learned relation-ID residual and use cached relation text only.")
    parser.set_defaults(use_deform=False, use_meta_context=True, use_relation_geo_bias=True)
    parser.set_defaults(use_relation_text=True, use_learned_relation_residual=True)


def build_model_from_args(
    text_dim: int,
    relation_text_dim: int,
    relation_vocab_size: int,
    args: argparse.Namespace,
) -> DyHeadRelationAdapter:
    use_deform = bool(getattr(args, "use_deform", not bool(getattr(args, "disable_deform", False))))
    use_meta_context = bool(getattr(args, "use_meta_context", False))
    use_relation_geo_bias = bool(getattr(args, "use_relation_geo_bias", False))
    return DyHeadRelationAdapter(
        text_dim=text_dim,
        relation_text_dim=relation_text_dim,
        relation_vocab_size=relation_vocab_size,
        channels=int(getattr(args, "channels", 128)),
        num_levels=int(getattr(args, "num_levels", 3)),
        num_blocks=int(getattr(args, "num_blocks", 1)),
        text_proj_dim=int(getattr(args, "text_proj_dim", 128)),
        relation_dim=int(getattr(args, "relation_dim", 64)),
        pair_geo_dim=int(getattr(args, "pair_geo_dim", 16)),
        pair_lambda=float(getattr(args, "pair_lambda", 1.0)),
        use_deform=use_deform,
        support_agg=str(getattr(args, "support_agg", "max")),
        use_meta_context=use_meta_context,
        use_relation_geo_bias=use_relation_geo_bias,
        use_relation_text=bool(getattr(args, "use_relation_text", True)),
        use_learned_relation_residual=bool(getattr(args, "use_learned_relation_residual", True)),
    )


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PROPOSAL_CACHE",
    "DEFAULT_RELATION_CACHE",
    "ProposalEpisodeDataset",
    "add_common_data_args",
    "add_model_args",
    "build_dyhead_collate_fn",
    "build_model_from_args",
    "build_proposal_cache",
    "build_relation_text_cache",
    "build_relation_vocab",
    "choose_device",
    "compute_detection_metrics",
    "evaluate_model",
    "format_relation_prompt",
    "maybe_save_report",
    "move_batch_to_device",
    "print_detection_summary",
    "print_relation_summary",
    "resolve_json_path",
    "resolve_path",
    "run_epoch",
    "save_checkpoint",
    "set_seed",
]
