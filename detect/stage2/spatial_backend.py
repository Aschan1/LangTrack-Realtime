"""Instantaneous spatial scoring backends.

A backend answers one question, with no memory of time:

    "Given this (possibly partial) query, how well does each track match, right now?"

    score_tracks(query, tracks) -> {track_id: raw_score in [0, 1]}

The temporal layer (EMA, hysteresis) lives in :class:`stage2.scorer.TrackScorer`
and wraps any backend. Two backends ship here:

* :class:`GeometricSpatialScorer` - dependency-free. Scores the target/anchor
  relation from box geometry alone. It is the *mock* that unblocks Person C and the
  tests (Day-0 contract), and is simultaneously a legitimate "no learned adapter"
  geometric baseline for the paper.

The trained DyHead adapter lives in :mod:`stage2.dyhead_backend` (kept separate so
this module imports without torch).
"""

from __future__ import annotations

import math
from typing import Iterable, Protocol, runtime_checkable

from stage2.query import Query
from stage2.tracks import Track


@runtime_checkable
class SpatialScorer(Protocol):
    def score_tracks(self, query: Query, tracks: list[Track]) -> dict[int, float]:
        """Instantaneous score in [0, 1] per track id for the candidate targets."""
        ...


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# Geometric predicate per canonical relation: given normalized signed offsets
# (dx, dy) of the *target* center relative to the *anchor* center (image-normalized,
# x right-positive, y down-positive), return a margin whose sigmoid is the score.
# Larger margin => more consistent with the relation.
def _relation_margin(relation: str, dx: float, dy: float) -> float:
    # scale controls how sharply the score saturates with displacement.
    scale = 8.0
    if relation == "to the left of":
        return scale * (-dx)  # target left of anchor => dx < 0
    if relation == "to the right of":
        return scale * dx
    if relation == "above":
        return scale * (-dy)  # target above anchor => dy < 0 (y grows downward)
    if relation in ("below", "under"):
        return scale * dy
    if relation in ("in front of", "behind"):
        # No depth in 2D; "in front of" weakly prefers lower-in-image (closer).
        sign = 1.0 if relation == "in front of" else -1.0
        return scale * 0.5 * sign * dy
    return 0.0


class GeometricSpatialScorer:
    """Score targets by detector confidence and (when known) relation geometry.

    * Empty query -> every track scores 0 ("no idea yet").
    * Target only -> score is the detector confidence of the matching-label tracks.
    * Target + relation + anchor -> confidence blended with how well each target sits
      in the requested relation to its best-matching anchor.
    """

    def __init__(self, *, conf_weight: float = 0.5) -> None:
        if not 0.0 <= conf_weight <= 1.0:
            raise ValueError("conf_weight must be in [0, 1]")
        self.conf_weight = conf_weight

    def _candidates(self, tracks: Iterable[Track], label: str) -> list[Track]:
        return [t for t in tracks if label and t.label == label]

    def score_tracks(self, query: Query, tracks: list[Track]) -> dict[int, float]:
        targets = self._candidates(tracks, query.target)
        if query.is_empty or not targets:
            return {t.track_id: 0.0 for t in tracks}

        if not query.has_relation:
            # Plain target detection (mirrors the demo's behaviour with no relation).
            return {t.track_id: _clamp01(t.conf) for t in targets}

        anchors = self._candidates(tracks, query.anchor)
        if not anchors:
            return {t.track_id: _clamp01(t.conf) for t in targets}

        scores: dict[int, float] = {}
        for tgt in targets:
            tcx, tcy = tgt.center
            best_rel = 0.0
            for anc in anchors:
                acx, acy = anc.center
                w, h = tgt.image_wh
                dx = (tcx - acx) / max(w, 1.0)
                dy = (tcy - acy) / max(h, 1.0)
                margin = _relation_margin(query.relation, dx, dy)
                # Anchor confidence gates how much we trust this anchor.
                rel = _sigmoid(margin) * _clamp01(anc.conf)
                best_rel = max(best_rel, rel)
            score = self.conf_weight * _clamp01(tgt.conf) + (1.0 - self.conf_weight) * best_rel
            scores[tgt.track_id] = _clamp01(score)
        return scores


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)
