"""Frame-level vs track-level comparison + EMA alpha sweep (plan D5-6).

This is Person B's core-ablation harness. It drives a :class:`TrackScorer` over a
set of *samples* (each a replayed prefix/track stream) and reports the **flip
count** - how often the top guess changes during an utterance. The frame-vs-track
flip-count contrast "IS our main ablation".

A *sample* is a dict::

    {
        "id": "home_0042",
        "gt_track": 7,                     # optional, enables correctness of final pick
        "steps": [(t, prefix_text, [Track, ...]), ...],
    }

Metric definitions for the paper (anytime accuracy, time-to-lock, commit curve) are
owned by Person C's evaluator; this module deliberately computes only the stability
metric Person B is responsible for, from the same trajectories.

Run ``python -m stage2.ablation`` for a self-contained synthetic demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from stage2.scorer import ScoreResult, TrackScorer
from stage2.spatial_backend import GeometricSpatialScorer, SpatialScorer
from stage2.tracks import Track

Step = tuple[float, str, list[Track]]


def flip_count(ids: Sequence[Optional[int]]) -> int:
    """Number of times the (non-empty) top/selected guess changes."""
    seq = [i for i in ids if i is not None]
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def run_sample(scorer: TrackScorer, steps: Sequence[Step]) -> list[ScoreResult]:
    scorer.reset()
    return [scorer.score_full(prefix, tracks, t) for (t, prefix, tracks) in steps]


@dataclass
class AblationRow:
    mode: str
    alpha: Optional[float]
    mean_top_flips: float
    mean_selected_flips: float
    final_accuracy: Optional[float]  # None if no sample carries gt_track
    n_samples: int

    def __str__(self) -> str:
        alpha = "  -  " if self.alpha is None else f"{self.alpha:.2f}"
        acc = "  n/a " if self.final_accuracy is None else f"{self.final_accuracy:6.3f}"
        return (
            f"{self.mode:5s} a={alpha}  "
            f"top_flips={self.mean_top_flips:5.2f}  "
            f"sel_flips={self.mean_selected_flips:5.2f}  "
            f"final_acc={acc}  (n={self.n_samples})"
        )


def _final_accuracy(results: list[ScoreResult], gt_track: Optional[int]) -> Optional[bool]:
    if gt_track is None or not results:
        return None
    last = results[-1]
    return last.top_track_id == gt_track


def evaluate_config(
    samples: Sequence[dict],
    *,
    mode: str,
    alpha: float,
    spatial_factory: Callable[[], SpatialScorer],
    scorer_kwargs: Optional[dict] = None,
) -> AblationRow:
    top_flips: list[int] = []
    sel_flips: list[int] = []
    correct: list[bool] = []
    for sample in samples:
        scorer = TrackScorer(spatial_factory(), mode=mode, alpha=alpha, **(scorer_kwargs or {}))
        results = run_sample(scorer, sample["steps"])
        top_flips.append(flip_count([r.top_track_id for r in results]))
        sel_flips.append(flip_count([r.selected_track_id for r in results]))
        acc = _final_accuracy(results, sample.get("gt_track"))
        if acc is not None:
            correct.append(acc)
    n = len(samples)
    return AblationRow(
        mode=mode,
        alpha=alpha if mode == "track" else None,
        mean_top_flips=sum(top_flips) / max(n, 1),
        mean_selected_flips=sum(sel_flips) / max(n, 1),
        final_accuracy=(sum(correct) / len(correct)) if correct else None,
        n_samples=n,
    )


def compare_frame_vs_track(
    samples: Sequence[dict],
    *,
    alphas: Sequence[float] = (0.4, 0.6, 0.8),
    spatial_factory: Callable[[], SpatialScorer] = GeometricSpatialScorer,
    scorer_kwargs: Optional[dict] = None,
) -> list[AblationRow]:
    """Run the frame baseline + each track-mode alpha; return one row each."""
    rows = [
        evaluate_config(
            samples, mode="frame", alpha=1.0, spatial_factory=spatial_factory, scorer_kwargs=scorer_kwargs
        )
    ]
    for alpha in alphas:
        rows.append(
            evaluate_config(
                samples, mode="track", alpha=alpha, spatial_factory=spatial_factory, scorer_kwargs=scorer_kwargs
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Self-contained synthetic demonstration (no heavy deps).
# --------------------------------------------------------------------------- #
def _synthetic_samples(n: int = 40, seed: int = 0) -> list[dict]:
    """home_0042-style scenes: two cups (left=GT, right=distractor) + a kettle.

    Per-frame detector-confidence jitter makes the *instantaneous* argmax flip
    between the two near-tied cups; the EMA is meant to absorb that.
    """
    import random

    rng = random.Random(seed)
    w, h = 640.0, 480.0
    prefixes = [
        (0.6, "the"),
        (0.9, "the cup"),
        (1.2, "the cup left"),
        (1.8, "the cup left of"),
        (2.3, "the cup left of the kettle"),
    ]
    samples: list[dict] = []
    for s in range(n):
        steps: list[Step] = []
        for (t, prefix) in prefixes:
            # Two cups close together (a near tie); kettle to the right.
            jitter = lambda: rng.uniform(-0.18, 0.18)
            cup_left = Track(7, "cup", (180, 320, 240, 400), conf=0.62 + jitter(), image_wh=(w, h))
            cup_right = Track(12, "cup", (300, 320, 360, 400), conf=0.60 + jitter(), image_wh=(w, h))
            kettle = Track(3, "kettle", (380, 300, 460, 420), conf=0.80, image_wh=(w, h))
            steps.append((t, prefix, [cup_left, cup_right, kettle]))
        samples.append({"id": f"home_{s:04d}", "gt_track": 7, "steps": steps})
    return samples


def _main() -> None:
    samples = _synthetic_samples()
    print("Frame-level vs track-level (EMA) + alpha sweep")
    print("Synthetic home_0042-style scenes: two near-tied cups, GT = left cup (track 7)")
    print("Lower flip counts = steadier guess (a robot that doesn't swing its head).\n")
    rows = compare_frame_vs_track(samples)
    for row in rows:
        print("  " + str(row))
    print(
        "\nExpected pattern: 'frame' shows the most top-guess flips; track-mode flips\n"
        "drop as alpha decreases (more smoothing). This is the paper's Flip-count ablation."
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    _main()
