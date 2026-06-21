"""Real spatial backend: the trained DyHead relation adapter.

This wires the existing, trained adapter
(``detect/yoloe_dyhead_relation_adapter_common.py``) into the
:class:`stage2.spatial_backend.SpatialScorer` protocol, so the streaming
``score(prefix_text, tracks)`` API runs the production model unchanged.

The scoring math mirrors ``detect/demo_dyhead.py`` exactly:

    fused = sigmoid(adapter_logits) * detector_conf

All heavy imports (torch, ultralytics, the adapter module) are deferred to
construction / call time so importing :mod:`stage2` stays cheap and dependency-free.

Each ``Track`` fed to this backend must carry a ``crop`` (a 3xHxW float tensor or
HxWx3 numpy array in [0, 1]); use :func:`attach_crops` to populate them from a frame.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from stage2.query import Query
from stage2.spatial_backend import _clamp01
from stage2.tracks import Track

_DETECT_DIR = Path(__file__).resolve().parent.parent  # detect/
_REPO_ROOT = _DETECT_DIR.parent


def _ensure_paths() -> None:
    for p in (str(_REPO_ROOT), str(_DETECT_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def attach_crops(tracks: list[Track], frame_bgr, crop_size: int = 96) -> list[Track]:
    """Populate ``track.crop`` from a BGR frame (OpenCV order), in place.

    Mirrors ``demo_dyhead.crop_box_tensor`` but keeps this module self-contained.
    """
    import cv2  # local import
    import numpy as np
    import torch
    from PIL import Image

    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    width, height = image.size
    for t in tracks:
        x1 = int(max(0, min(width, math.floor(t.box[0]))))
        y1 = int(max(0, min(height, math.floor(t.box[1]))))
        x2 = int(max(0, min(width, math.ceil(t.box[2]))))
        y2 = int(max(0, min(height, math.ceil(t.box[3]))))
        if x2 <= x1 or y2 <= y1:
            patch = Image.new("RGB", (crop_size, crop_size), color=(0, 0, 0))
        else:
            patch = image.crop((x1, y1, x2, y2)).convert("RGB").resize((crop_size, crop_size), Image.BILINEAR)
        patch_np = np.asarray(patch, dtype=np.float32) / 255.0
        t.crop = torch.from_numpy(patch_np).permute(2, 0, 1).contiguous()
    return tracks


class DyHeadSpatialScorer:
    """Score tracks with the trained DyHead relation adapter.

    Construct directly with a loaded model + embedder, or via
    :meth:`from_checkpoint` which loads YOLOE (used only as the text encoder) and
    the adapter, just like the demo.
    """

    def __init__(
        self,
        *,
        model: Any,
        relation_to_id: dict[str, int],
        text_embedder: Callable[[list[str]], Any],
        relation_prompt_style: str = "template",
        crop_size: int = 96,
        device: Optional[str] = None,
        topk_target: int = 20,
        topk_anchor: int = 10,
    ) -> None:
        import torch  # local import

        self._torch = torch
        self.model = model
        self.relation_to_id = dict(relation_to_id)
        self.text_embedder = text_embedder
        self.relation_prompt_style = relation_prompt_style
        self.crop_size = crop_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.topk_target = topk_target
        self.topk_anchor = topk_anchor
        self._label_cache: dict[str, Any] = {}
        self._relation_cache: dict[str, Any] = {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        weights_path: str | Path,
        *,
        device: Optional[str] = None,
        crop_size: int = 96,
    ) -> "DyHeadSpatialScorer":
        _ensure_paths()
        import argparse

        import torch
        from ultralytics import YOLOE

        from yoloe_dyhead_relation_adapter_common import build_model_from_args, resolve_path

        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        yolo_model = YOLOE(str(resolve_path(str(weights_path)))).to(dev)
        yolo_model.model.eval()

        checkpoint = torch.load(str(resolve_path(str(checkpoint_path))), map_location="cpu", weights_only=False)
        relation_to_id = checkpoint["relation_to_id"]
        model_args = argparse.Namespace(**checkpoint["args"])
        state_dict = checkpoint["model_state_dict"]
        if not any(k.startswith("relation_text_proj.") for k in state_dict):
            model_args.use_relation_text = False
        relation_text_dim = (
            int(checkpoint["relation_text_dim"])
            if checkpoint.get("relation_text_dim") is not None
            else int(checkpoint["text_dim"])
        )
        relation_prompt_style = str(getattr(model_args, "relation_prompt_style", "template"))
        model = build_model_from_args(
            int(checkpoint["text_dim"]), relation_text_dim, len(relation_to_id), model_args
        ).to(dev)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        def embedder(prompts: list[str]):
            with torch.no_grad():
                return yolo_model.model.get_text_pe(prompts, cache_clip_model=True)

        return cls(
            model=model,
            relation_to_id=relation_to_id,
            text_embedder=embedder,
            relation_prompt_style=relation_prompt_style,
            crop_size=crop_size,
            device=str(dev),
        )

    def _label_feat(self, label: str):
        if label not in self._label_cache:
            emb = self.text_embedder([label]).squeeze(0)[0].detach().cpu().float().contiguous()
            self._label_cache[label] = emb
        return self._label_cache[label]

    def _relation_feat(self, relation: str):
        _ensure_paths()
        from yoloe_dyhead_relation_adapter_common import format_relation_prompt

        if relation not in self._relation_cache:
            prompt = format_relation_prompt(relation, self.relation_prompt_style)
            emb = self.text_embedder([prompt]).squeeze(0)[0].detach().cpu().float().contiguous()
            self._relation_cache[relation] = emb
        return self._relation_cache[relation]

    def score_tracks(self, query: Query, tracks: list[Track]) -> dict[int, float]:
        torch = self._torch
        targets = [t for t in tracks if query.target and t.label == query.target][: self.topk_target]
        if query.is_empty or not targets:
            return {t.track_id: 0.0 for t in tracks}

        # No supported relation -> plain target detection (mirrors the demo).
        if not query.has_relation or query.relation not in self.relation_to_id:
            return {t.track_id: _clamp01(t.conf) for t in targets}

        anchors = [t for t in tracks if t.label == query.anchor][: self.topk_anchor]
        if not anchors:
            return {t.track_id: _clamp01(t.conf) for t in targets}

        proposals: list[tuple[Track, int]] = []  # (track, role_id)
        target_positions: list[int] = []
        for t in targets:
            target_positions.append(len(proposals))
            proposals.append((t, 0))
        for a in anchors:
            proposals.append((a, 1))

        if any(p[0].crop is None for p in proposals):
            raise ValueError(
                "DyHeadSpatialScorer requires track.crop on every track; call attach_crops(tracks, frame)."
            )

        n = len(proposals)
        crop0 = proposals[0][0].crop
        crops = torch.zeros(1, n, *crop0.shape, dtype=torch.float32)
        boxes = torch.zeros(1, n, 4, dtype=torch.float32)
        confs = torch.zeros(1, n, dtype=torch.float32)
        text_feats = torch.zeros(1, n, self._label_feat(query.target).numel(), dtype=torch.float32)
        role_ids = torch.zeros(1, n, dtype=torch.long)
        mask = torch.ones(1, n, dtype=torch.bool)

        for i, (trk, role) in enumerate(proposals):
            crops[0, i] = trk.crop if torch.is_tensor(trk.crop) else torch.as_tensor(trk.crop)
            boxes[0, i] = torch.tensor(trk.box, dtype=torch.float32)
            confs[0, i] = float(trk.conf)
            label = query.target if role == 0 else query.anchor
            text_feats[0, i] = self._label_feat(label)
            role_ids[0, i] = role

        w, h = targets[0].image_wh
        with torch.no_grad():
            logits = self.model(
                crops=crops.to(self.device),
                text_feats=text_feats.to(self.device),
                boxes=boxes.to(self.device),
                confs=confs.to(self.device),
                role_ids=role_ids.to(self.device),
                image_wh=torch.tensor([[float(w), float(h)]], dtype=torch.float32).to(self.device),
                relation_ids=torch.tensor([self.relation_to_id[query.relation]], dtype=torch.long).to(self.device),
                relation_text_feats=self._relation_feat(query.relation).unsqueeze(0).to(self.device),
                mask=mask.to(self.device),
            )
            fused = (torch.sigmoid(logits) * confs.to(self.device)).squeeze(0).detach().cpu()

        scores: dict[int, float] = {}
        for pos, (trk, _role) in enumerate(proposals):
            if pos in target_positions:
                scores[trk.track_id] = _clamp01(float(fused[pos]))
        return scores
