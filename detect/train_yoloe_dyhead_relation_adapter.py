#!/usr/bin/env python3
"""
Train a Dynamic Head style relation adapter on top of offline YOLOE proposals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from yoloe_dyhead_relation_adapter_common import (
    ProposalEpisodeDataset,
    add_common_data_args,
    add_model_args,
    build_dyhead_collate_fn,
    build_model_from_args,
    build_proposal_cache,
    build_relation_text_cache,
    build_relation_vocab,
    choose_device,
    resolve_json_path,
    resolve_path,
    run_epoch,
    save_checkpoint,
    set_seed,
)

try:
    from yoloe_paper_relation_adapter import split_episodes_by_image
except ModuleNotFoundError:
    from detect.yoloe_paper_relation_adapter import split_episodes_by_image

try:
    from yoloe_paper_relation_adapter import DEFAULT_NYU_IMAGES, DEFAULT_NYU_RGB_JSON
except ModuleNotFoundError:
    from detect.yoloe_paper_relation_adapter import DEFAULT_NYU_IMAGES, DEFAULT_NYU_RGB_JSON


DEFAULT_NYU_DYHEAD_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "dyhead_relation_adapter_nyu_rgb_only"
DEFAULT_NYU_DYHEAD_PROPOSAL_CACHE = DEFAULT_NYU_DYHEAD_OUTPUT_DIR / "proposal_cache.pt"
DEFAULT_NYU_DYHEAD_RELATION_CACHE = DEFAULT_NYU_DYHEAD_OUTPUT_DIR / "relation_text_cache.pt"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a DyHead-style spatial relation adapter on the NYU RGB-only relation dataset by default."
    )
    add_common_data_args(parser)
    add_model_args(parser)
    parser.set_defaults(
        json=str(DEFAULT_NYU_RGB_JSON),
        images=str(DEFAULT_NYU_IMAGES),
        proposal_cache=str(DEFAULT_NYU_DYHEAD_PROPOSAL_CACHE),
        relation_cache=str(DEFAULT_NYU_DYHEAD_RELATION_CACHE),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_NYU_DYHEAD_OUTPUT_DIR), help="Directory for checkpoints and history.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--device", default="", help="Torch device override.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--crop-size", type=int, default=96, help="Proposal crop size.")
    parser.add_argument("--positive-iou", type=float, default=0.5, help="Positive IoU threshold used in dataset labels.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation split fraction by image.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--pos-weight", type=float, default=3.0, help="Positive weight for BCE loss.")
    parser.add_argument("--relation-margin-weight", type=float, default=0.2, help="Weight for the hard-negative margin loss.")
    parser.add_argument("--listwise-weight", type=float, default=0.15, help="Weight for the per-query listwise ranking loss.")
    parser.add_argument("--hard-negative-k", type=int, default=3, help="Number of hardest negatives to mine per query.")
    parser.add_argument("--hard-negative-margin", type=float, default=0.2, help="Margin used for hard-negative ranking.")
    parser.add_argument("--allow-train-without-anchor", action="store_true", help="Keep training episodes that have target proposals but no anchor proposals.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    json_path = resolve_json_path(args.json)
    images_dir = resolve_path(args.images)
    weights = resolve_path(args.weights)
    cache_path = resolve_path(args.cache_path)
    proposal_cache_path = resolve_path(args.proposal_cache)
    relation_cache_path = resolve_path(args.relation_cache)
    output_dir = resolve_path(args.output_dir)

    print(f"Using annotations:   {json_path}")
    print(f"Using images:        {images_dir}")
    print(f"Using YOLOE:         {weights}")
    print(f"Using offline cache: {cache_path}")
    print(f"Using proposals:     {proposal_cache_path}")
    print(f"Using relation cache:{relation_cache_path}")
    print(f"Using output dir:    {output_dir}")
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
    prompt_embeddings = proposal_payload["prompt_embeddings"]
    relation_text_embeddings = build_relation_text_cache(
        proposal_payload["episodes"],
        weights=weights,
        relation_cache_path=relation_cache_path,
        force_rebuild=args.force_rebuild_relation_cache,
        prompt_style=args.relation_prompt_style,
    )
    text_dim = int(next(iter(prompt_embeddings.values())).numel())
    relation_text_dim = int(next(iter(relation_text_embeddings.values())).numel())
    relation_to_id = build_relation_vocab(proposal_payload["episodes"])

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

    collate_fn = build_dyhead_collate_fn(relation_to_id, relation_text_embeddings)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs) if len(val_dataset) else None

    model = build_model_from_args(text_dim, relation_text_dim, len(relation_to_id), args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    print()
    print("Dataset summary")
    print(f"  proposal cache episodes:   {len(proposal_payload['episodes'])}")
    print(f"  train episodes kept:       {len(train_dataset)}")
    print(f"  val episodes kept:         {len(val_dataset)}")
    print(f"  prompt embedding dim:      {text_dim}")
    print(f"  relation text dim:         {relation_text_dim}")
    print(f"  train skip stats:          {train_dataset.skip_stats}")
    print(f"  val skip stats:            {val_dataset.skip_stats}")
    print(f"  proposal cache stats:      {proposal_payload['stats']}")
    print()

    history = []
    best_metric = -float("inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_top1 = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            pos_weight=args.pos_weight,
            relation_margin_weight=args.relation_margin_weight,
            listwise_weight=args.listwise_weight,
            hard_negative_k=args.hard_negative_k,
            hard_negative_margin=args.hard_negative_margin,
        )
        if val_loader is not None:
            val_loss, val_top1 = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                pos_weight=args.pos_weight,
                relation_margin_weight=args.relation_margin_weight,
                listwise_weight=args.listwise_weight,
                hard_negative_k=args.hard_negative_k,
                hard_negative_margin=args.hard_negative_margin,
            )
        else:
            val_loss = float("nan")
            val_top1 = float("nan")

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

        score = val_top1 if val_loader is not None else train_top1
        if score > best_metric:
            best_metric = score
            save_checkpoint(
                best_path,
                model=model,
                args=args,
                history=history,
                epoch=epoch,
                text_dim=text_dim,
                relation_text_dim=relation_text_dim,
                relation_to_id=relation_to_id,
                proposal_cache=proposal_cache_path,
                relation_cache=relation_cache_path,
                train_size=len(train_dataset),
                val_size=len(val_dataset),
            )

        save_checkpoint(
            last_path,
            model=model,
            args=args,
            history=history,
            epoch=epoch,
            text_dim=text_dim,
            relation_text_dim=relation_text_dim,
            relation_to_id=relation_to_id,
            proposal_cache=proposal_cache_path,
            relation_cache=relation_cache_path,
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
                "relation_prompt_style": args.relation_prompt_style,
                "relation_cache": str(relation_cache_path),
            },
            handle,
            indent=2,
        )
    print(f"\nSaved best checkpoint to: {best_path}")
    print(f"Saved last checkpoint to: {last_path}")
    print(f"Saved history to:         {history_path}")


if __name__ == "__main__":
    main()
