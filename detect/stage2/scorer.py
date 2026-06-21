"""Stage 2 - the stateful scorer behind the frozen API.

    score(prefix_text, tracks) -> {track_id: score}

``TrackScorer`` is the object Person C drives in the replay loop (Stage 3). It is
stateful *within one utterance* and must be :meth:`reset` between samples.

Pipeline per call (matches the plan, Stage 2):

1. Parse the prefix into a structured :class:`~stage2.query.Query`.
2. Ask the spatial backend for an instant per-track score.
3. ``track`` mode only: fold each track's instant score into its running average
   ``S_new = alpha * s_now + (1 - alpha) * S_old`` (EMA, the anti-jitter fix).
   ``frame`` mode skips the EMA entirely (instantaneous scores) - the other arm of
   the paper's core ablation.
4. Apply selection hysteresis (two thresholds) so the top guess does not blink, and
   box smoothing so the acted-on box does not shake.

The dict returned is exactly the contract. Richer per-step state (the hysteretic
selection, the smoothed box) is available via :meth:`score_full` / the attributes,
without breaking the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from stage2.query import HeuristicPrefixParser, Query, QueryParser
from stage2.smoothing import BoxSmoother, EMABoxSmoother
from stage2.spatial_backend import GeometricSpatialScorer, SpatialScorer
from stage2.tracks import Box, Track


@dataclass
class ScoreResult:
    """Full per-step output (the contract dict is :attr:`scores`)."""

    scores: dict[int, float]  # {track_id: score}  <- the API contract
    query: Query
    top_track_id: Optional[int]  # argmax of scores (instantaneous best guess)
    selected_track_id: Optional[int]  # hysteresis-stabilised guess (what a robot acts on)
    selected_box: Optional[Box]  # smoothed box of the selected track
    t: Optional[float]


class TrackScorer:
    """Stateful scorer implementing ``score(prefix_text, tracks)``.

    Parameters
    ----------
    spatial:
        Instantaneous backend. Defaults to the dependency-free
        :class:`GeometricSpatialScorer` so the object is usable out of the box.
    mode:
        ``"track"`` (EMA over time) or ``"frame"`` (no memory). This switch is the
        paper's core ablation.
    alpha:
        EMA weight on the current frame, ``S_new = alpha*s_now + (1-alpha)*S_old``.
        The plan's default is 0.6; swept over {0.4, 0.6, 0.8}.
    select_high / select_low:
        Selection hysteresis thresholds. A track must reach ``select_high`` to *become*
        the selection; it stays selected until it falls below ``select_low``. Pulling
        them apart stops the chosen object from blinking in and out.
    box_smoother_factory:
        Called with no args to build a per-track :class:`BoxSmoother`. ``None`` disables
        box smoothing.
    """

    def __init__(
        self,
        spatial: Optional[SpatialScorer] = None,
        *,
        parser: Optional[QueryParser] = None,
        mode: str = "track",
        alpha: float = 0.6,
        select_high: float = 0.5,
        select_low: float = 0.3,
        box_smoother_factory: Optional[Callable[[], BoxSmoother]] = EMABoxSmoother,
    ) -> None:
        if mode not in ("track", "frame"):
            raise ValueError("mode must be 'track' or 'frame'")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= select_low <= select_high <= 1.0:
            raise ValueError("require 0 <= select_low <= select_high <= 1")

        self.spatial = spatial or GeometricSpatialScorer()
        self.parser = parser or HeuristicPrefixParser()
        self.mode = mode
        self.alpha = alpha
        self.select_high = select_high
        self.select_low = select_low
        self._box_smoother_factory = box_smoother_factory

        self._ema: dict[int, float] = {}
        self._selected: Optional[int] = None
        self._smoothers: dict[int, BoxSmoother] = {}
        self._frame_index = 0

    # -- lifecycle -------------------------------------------------------------
    def reset(self) -> None:
        """Clear all per-utterance state. Call between samples in replay."""
        self._ema.clear()
        self._selected = None
        self._smoothers.clear()
        self._frame_index = 0

    # -- the contract ----------------------------------------------------------
    def score(self, prefix_text: str, tracks: list[Track], t: Optional[float] = None) -> dict[int, float]:
        """The frozen Day-0 API: prefix + tracks -> per-track score."""
        return self.score_full(prefix_text, tracks, t).scores

    # -- the contract, plus stabilised selection & smoothed box ----------------
    def score_full(self, prefix_text: str, tracks: list[Track], t: Optional[float] = None) -> ScoreResult:
        if t is None:
            t = float(self._frame_index)
        self._frame_index += 1

        query = self.parser.parse(prefix_text)
        instant = self.spatial.score_tracks(query, tracks)

        if self.mode == "frame":
            scores = {tid: float(s) for tid, s in instant.items()}
        else:
            scores = self._apply_ema(instant)

        top = max(scores, key=lambda k: scores[k]) if scores else None
        selected = self._apply_hysteresis(scores)
        selected_box = self._smooth_selected_box(selected, tracks, t)

        return ScoreResult(
            scores=scores,
            query=query,
            top_track_id=top,
            selected_track_id=selected,
            selected_box=selected_box,
            t=t,
        )

    # -- internals -------------------------------------------------------------
    def _apply_ema(self, instant: dict[int, float]) -> dict[int, float]:
        out: dict[int, float] = {}
        for tid, s_now in instant.items():
            s_old = self._ema.get(tid)
            s_new = float(s_now) if s_old is None else self.alpha * float(s_now) + (1.0 - self.alpha) * s_old
            self._ema[tid] = s_new
            out[tid] = s_new
        # Forget tracks that are no longer present so they cannot win later.
        for tid in list(self._ema):
            if tid not in instant:
                del self._ema[tid]
        return out

    def _apply_hysteresis(self, scores: dict[int, float]) -> Optional[int]:
        if not scores:
            self._selected = None
            return None

        best = max(scores, key=lambda k: scores[k])
        best_score = scores[best]

        if self._selected is None or self._selected not in scores:
            # Nothing held: only commit if we clear the high bar.
            self._selected = best if best_score >= self.select_high else None
            return self._selected

        held_score = scores[self._selected]
        if held_score < self.select_low:
            # Held guess has decayed: hand over to the best, but only if it is confident.
            self._selected = best if best_score >= self.select_high else None
        elif best != self._selected and best_score >= self.select_high:
            # A different track is now confidently better: switch.
            self._selected = best
        # else: keep the current selection (the sticky region between the thresholds).
        return self._selected

    def _smooth_selected_box(self, selected: Optional[int], tracks: list[Track], t: float) -> Optional[Box]:
        if selected is None or self._box_smoother_factory is None:
            return None
        track = next((trk for trk in tracks if trk.track_id == selected), None)
        if track is None:
            return None
        smoother = self._smoothers.get(selected)
        if smoother is None:
            smoother = self._box_smoother_factory()
            self._smoothers[selected] = smoother
        return smoother.smooth(track.box, t)
