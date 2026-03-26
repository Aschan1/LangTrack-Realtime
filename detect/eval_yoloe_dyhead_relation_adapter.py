#!/usr/bin/env python3
"""
Evaluate a Dynamic Head style relation adapter on top of offline YOLOE proposals.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from yoloe_dyhead_relation_adapter_common import (
    ProposalEpisodeDataset,
    add_common_data_args,
    add_model_args,
    build_dyhead_collate_fn,
    build_model_from_args,
    build_proposal_cache,
    build_relation_text_cache,
    choose_device,
    evaluate_model,
    maybe_save_report,
    print_detection_summary,
    print_relation_summary,
    resolve_json_path,
    resolve_path,
)

try:
    from yoloe_paper_relation_adapter import DEFAULT_NYU_IMAGES, DEFAULT_NYU_RGB_JSON
except ModuleNotFoundError:
    from detect.yoloe_paper_relation_adapter import DEFAULT_NYU_IMAGES, DEFAULT_NYU_RGB_JSON


DEFAULT_NYU_DYHEAD_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "dyhead_relation_adapter_nyu_rgb_only"
DEFAULT_NYU_DYHEAD_PROPOSAL_CACHE = DEFAULT_NYU_DYHEAD_OUTPUT_DIR / "proposal_cache.pt"
DEFAULT_NYU_DYHEAD_RELATION_CACHE = DEFAULT_NYU_DYHEAD_OUTPUT_DIR / "relation_text_cache.pt"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a DyHead-style spatial relation adapter on filtered_indoors_LM_vg.json by default."
    )
    add_common_data_args(parser)
    add_model_args(parser)
    parser.set_defaults(
        json=str(Path(__file__).resolve().parent.parent / "yolo_dataset" / "indoors_subset" / "filtered_indoors_LM_vg.json"),
        images=str(Path(__file__).resolve().parent.parent / "yolo_dataset" / "indoors_subset" / "images"),
        proposal_cache=str(Path(__file__).resolve().parent.parent / "outputs" / "dyhead_relation_adapter" / "proposal_cache_eval_filtered_indoors_LM_vg.pt"),
        relation_cache=str(Path(__file__).resolve().parent.parent / "outputs" / "dyhead_relation_adapter" / "relation_text_cache_eval_filtered_indoors_LM_vg.pt"),
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a trained DyHead relation adapter checkpoint.")
    parser.add_argument("--device", default="", help="Torch device override.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--crop-size", type=int, default=96, help="Proposal crop size.")
    parser.add_argument("--positive-iou", type=float, default=0.5, help="Positive IoU threshold used in dataset labels.")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold used in evaluation metrics.")
    parser.add_argument("--report-json", default="", help="Optional path to save a detailed JSON report.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = choose_device(args.device)
    json_path = resolve_json_path(args.json)
    images_dir = resolve_path(args.images)
    weights = resolve_path(args.weights)
    cache_path = resolve_path(args.cache_path)
    proposal_cache_path = resolve_path(args.proposal_cache)
    relation_cache_path = resolve_path(args.relation_cache)
    checkpoint_path = resolve_path(args.checkpoint)
    report_json = resolve_path(args.report_json) if args.report_json else None

    print(f"Using annotations:   {json_path}")
    print(f"Using images:        {images_dir}")
    print(f"Using YOLOE:         {weights}")
    print(f"Using offline cache: {cache_path}")
    print(f"Using proposals:     {proposal_cache_path}")
    print(f"Using relation cache:{relation_cache_path}")
    print(f"Using checkpoint:    {checkpoint_path}")
    print(f"Using device:        {device}")

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
    relation_to_id = checkpoint["relation_to_id"]
    prompt_embeddings = proposal_payload["prompt_embeddings"]
    text_dim = int(checkpoint.get("text_dim") or next(iter(prompt_embeddings.values())).numel())
    all_episodes = proposal_payload["episodes"]
    supported_relations = set(relation_to_id)
    filtered_episodes = [episode for episode in all_episodes if episode["relation"] in supported_relations]
    skipped_relation_counts = Counter(
        episode["relation"] for episode in all_episodes if episode["relation"] not in supported_relations
    )
    if skipped_relation_counts:
        skipped_total = sum(skipped_relation_counts.values())
        print(
            "Skipping episodes with relations unsupported by this checkpoint: "
            f"{skipped_total} episode(s) across {len(skipped_relation_counts)} relation(s)."
        )
        print(
            "Unsupported relation counts: "
            + ", ".join(f"{relation}={count}" for relation, count in sorted(skipped_relation_counts.items()))
        )
    model_args = argparse.Namespace(**checkpoint["args"])
    state_dict = checkpoint["model_state_dict"]
    has_relation_text_proj = any(key.startswith("relation_text_proj.") for key in state_dict)
    if not has_relation_text_proj:
        model_args.use_relation_text = False
    relation_prompt_style = str(getattr(model_args, "relation_prompt_style", getattr(args, "relation_prompt_style", "framed")))
    relation_text_embeddings = build_relation_text_cache(
        filtered_episodes,
        weights=weights,
        relation_cache_path=relation_cache_path,
        force_rebuild=args.force_rebuild_relation_cache,
        prompt_style=relation_prompt_style,
    )
    relation_text_dim = int(
        checkpoint.get("relation_text_dim") or next(iter(relation_text_embeddings.values())).numel()
    )

    model = build_model_from_args(text_dim, relation_text_dim, len(relation_to_id), model_args).to(device)
    model.load_state_dict(state_dict, strict=True)

    dataset = ProposalEpisodeDataset(
        filtered_episodes,
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
        collate_fn=build_dyhead_collate_fn(relation_to_id, relation_text_embeddings),
    )

    relation_summary, detection_metrics, all_gts, all_preds, detail_rows = evaluate_model(
        model,
        loader,
        device=device,
        iou_thresh=args.iou_thresh,
    )

    print_relation_summary(relation_summary)
    print_detection_summary(detection_metrics, args.iou_thresh)
    proposal_stats = dict(proposal_payload["stats"])
    proposal_stats["query_skipped_unsupported_relation"] = int(sum(skipped_relation_counts.values()))
    proposal_stats["unsupported_relations"] = dict(sorted(skipped_relation_counts.items()))
    maybe_save_report(
        report_json,
        proposal_stats=proposal_stats,
        relation_summary=relation_summary,
        detection_metrics=detection_metrics,
        all_gts=all_gts,
        all_preds=all_preds,
        detail_rows=detail_rows,
    )
    if report_json is not None:
        print(f"Saved report to: {report_json}")


if __name__ == "__main__":
    main()
