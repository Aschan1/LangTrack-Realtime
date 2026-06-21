"""Stage 2 - Scoring (Person B's deliverable).

This package implements the LangTrack scoring layer behind the frozen Day-0 contract:

    score(prefix_text, tracks) -> {track_id: score}

The contract is realised by ``TrackScorer.score``. Internally it:

1. Parses the (possibly incomplete) transcription *prefix* into a structured
   :class:`~stage2.query.Query` ``{target, anchor, relation}`` - fields stay empty
   while the prefix is incomplete.
2. Asks a pluggable :class:`~stage2.spatial_backend.SpatialScorer` for an *instant*
   per-track score for that query.
3. Keeps a per-track running average (EMA) of the score over time - the anti-jitter
   fix - and applies the stability fixes (selection hysteresis, box smoothing).

Two scoring modes are supported and form the paper's core ablation (Stage 4,
"Flip count"):

* ``"frame"`` - no temporal memory (instantaneous scores), and
* ``"track"`` - EMA over time.

The spatial backend is pluggable so that:

* :class:`~stage2.spatial_backend.GeometricSpatialScorer` (a.k.a. the *mock*) unblocks
  Person C and the unit tests with zero heavy dependencies, while
* :class:`~stage2.dyhead_backend.DyHeadSpatialScorer` wires the real trained DyHead
  relation adapter (``detect/yoloe_dyhead_relation_adapter_common.py``) into the very
  same API.
"""

from stage2.query import Query, HeuristicPrefixParser, QueryParser
from stage2.tracks import Track
from stage2.spatial_backend import SpatialScorer, GeometricSpatialScorer
from stage2.smoothing import OneEuroBoxSmoother, EMABoxSmoother, BoxSmoother
from stage2.scorer import TrackScorer, ScoreResult
from stage2.tracker_config import build_bytetrack_args

__all__ = [
    "Query",
    "QueryParser",
    "HeuristicPrefixParser",
    "Track",
    "SpatialScorer",
    "GeometricSpatialScorer",
    "BoxSmoother",
    "OneEuroBoxSmoother",
    "EMABoxSmoother",
    "TrackScorer",
    "ScoreResult",
    "build_bytetrack_args",
]
