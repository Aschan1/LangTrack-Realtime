"""Unit tests for the Stage-2 scoring layer (Person B).

Runnable two ways:
    pytest detect/stage2/tests/test_stage2.py
    python  detect/stage2/tests/test_stage2.py     # built-in runner, no pytest needed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # detect/ on path

from stage2.ablation import compare_frame_vs_track, flip_count, _synthetic_samples
from stage2.query import HeuristicPrefixParser, Query
from stage2.scorer import TrackScorer
from stage2.smoothing import EMABoxSmoother, OneEuroBoxSmoother
from stage2.spatial_backend import GeometricSpatialScorer
from stage2.tracks import Track

W, H = 640.0, 480.0


def _cup_scene(conf_left=0.6, conf_right=0.6):
    return [
        Track(7, "cup", (180, 320, 240, 400), conf=conf_left, image_wh=(W, H)),
        Track(12, "cup", (300, 320, 360, 400), conf=conf_right, image_wh=(W, H)),
        Track(3, "kettle", (380, 300, 460, 420), conf=0.8, image_wh=(W, H)),
    ]


# --- query parsing --------------------------------------------------------- #
def test_prefix_parser_incremental():
    p = HeuristicPrefixParser()
    assert p.parse("the") == Query()
    assert p.parse("the cup") == Query(target="cup")
    q = p.parse("the cup left of")
    assert q.target == "cup" and q.relation == "to the left of" and q.anchor == ""
    assert not q.has_relation  # anchor not heard yet
    q = p.parse("the cup left of the kettle")
    assert q.target == "cup" and q.relation == "to the left of" and q.anchor == "kettle"
    assert q.has_relation


def test_prefix_parser_attributes_and_aliases():
    q = HeuristicPrefixParser().parse("the blue cup to the right of the kettle")
    assert q.target == "cup" and q.attributes == ("blue",)
    assert q.relation == "to the right of" and q.anchor == "kettle"
    assert q.target_prompt == "blue cup"


def test_empty_query_scores_zero():
    scorer = TrackScorer(GeometricSpatialScorer(), mode="frame")
    scores = scorer.score("the", _cup_scene())
    assert set(scores) == {7, 12, 3}
    assert all(v == 0.0 for v in scores.values())


# --- geometric backend ----------------------------------------------------- #
def test_geometry_breaks_the_tie():
    # Equal confidence; only geometry ("left of kettle") should separate the cups.
    scores = GeometricSpatialScorer().score_tracks(
        Query(target="cup", anchor="kettle", relation="to the left of"), _cup_scene()
    )
    assert scores[7] > scores[12]  # left cup wins
    # Anchor itself is not a target candidate.
    assert 3 not in scores


def test_target_only_uses_confidence():
    scores = GeometricSpatialScorer().score_tracks(Query(target="cup"), _cup_scene(0.7, 0.3))
    assert scores[7] == 0.7 and scores[12] == 0.3


# --- EMA semantics --------------------------------------------------------- #
def test_ema_math():
    scorer = TrackScorer(GeometricSpatialScorer(), mode="track", alpha=0.6, box_smoother_factory=None)
    # target-only => score == conf, so we can predict the EMA exactly.
    q = "the cup"
    s1 = scorer.score(q, _cup_scene(0.5, 0.0))[7]
    assert abs(s1 - 0.5) < 1e-9  # first frame seeds the EMA
    s2 = scorer.score(q, _cup_scene(1.0, 0.0))[7]
    assert abs(s2 - (0.6 * 1.0 + 0.4 * 0.5)) < 1e-9  # 0.8


def test_frame_mode_has_no_memory():
    scorer = TrackScorer(GeometricSpatialScorer(), mode="frame", box_smoother_factory=None)
    scorer.score("the cup", _cup_scene(0.5, 0.0))
    s2 = scorer.score("the cup", _cup_scene(1.0, 0.0))[7]
    assert s2 == 1.0  # purely instantaneous


def test_ema_forgets_absent_tracks():
    scorer = TrackScorer(GeometricSpatialScorer(), mode="track", alpha=0.6, box_smoother_factory=None)
    scorer.score("the cup", _cup_scene())
    # track 7 disappears
    scene = [t for t in _cup_scene() if t.track_id != 7]
    scores = scorer.score("the cup", scene)
    assert 7 not in scores


# --- EMA reduces flips (the core ablation result) -------------------------- #
def test_track_mode_reduces_flips_vs_frame():
    rows = compare_frame_vs_track(_synthetic_samples(n=60, seed=1), alphas=(0.4, 0.6, 0.8))
    frame = next(r for r in rows if r.mode == "frame")
    track_a04 = next(r for r in rows if r.mode == "track" and r.alpha == 0.4)
    # Smoothing must not increase instability, and should help on these noisy ties.
    assert track_a04.mean_top_flips <= frame.mean_top_flips
    assert track_a04.mean_top_flips < frame.mean_top_flips + 1e-9
    # More smoothing (lower alpha) should not flip more than less smoothing.
    track_a08 = next(r for r in rows if r.mode == "track" and r.alpha == 0.8)
    assert track_a04.mean_top_flips <= track_a08.mean_top_flips + 1e-9


def test_flip_count_helper():
    assert flip_count([7, 7, 7]) == 0
    assert flip_count([7, 12, 7]) == 2
    assert flip_count([None, 7, None, 7]) == 0  # Nones ignored
    assert flip_count([7, 12]) == 1


# --- hysteresis ------------------------------------------------------------ #
def test_hysteresis_prevents_blinking():
    # A score that dithers around a single threshold would blink; with a low/high
    # gap the selection should latch once and hold.
    scorer = TrackScorer(
        GeometricSpatialScorer(),
        mode="frame",
        select_high=0.6,
        select_low=0.3,
        box_smoother_factory=None,
    )
    confs = [0.7, 0.45, 0.55, 0.4, 0.5, 0.65]  # all above low(0.3) after first commit
    selected = []
    for c in confs:
        res = scorer.score_full("the cup", _cup_scene(c, 0.0))
        selected.append(res.selected_track_id)
    assert selected[0] == 7  # committed at high
    assert all(s == 7 for s in selected)  # never blinks off (stays above low)


def test_hysteresis_handover_only_when_confident():
    scorer = TrackScorer(
        GeometricSpatialScorer(),
        mode="frame",
        select_high=0.6,
        select_low=0.3,
        box_smoother_factory=None,
    )
    # Commit to 7, then make 12 marginally better but below the high bar: keep 7.
    scorer.score_full("the cup", _cup_scene(0.7, 0.0))
    res = scorer.score_full("the cup", _cup_scene(0.2, 0.55))  # 7 drops below low, 12 below high
    assert res.selected_track_id is None or res.selected_track_id == 12
    # Now 12 clears the high bar -> hand over.
    res = scorer.score_full("the cup", _cup_scene(0.0, 0.7))
    assert res.selected_track_id == 12


# --- box smoothing --------------------------------------------------------- #
def test_ema_box_smoother_converges_and_dampens():
    sm = EMABoxSmoother(alpha=0.5)
    assert sm.smooth((0, 0, 10, 10)) == (0, 0, 10, 10)  # first = passthrough
    out = sm.smooth((10, 10, 20, 20))
    assert out == (5.0, 5.0, 15.0, 15.0)  # halfway


def test_one_euro_smoother_runs_and_tracks_steady_state():
    sm = OneEuroBoxSmoother(freq=30.0, min_cutoff=1.0, beta=0.05)
    box = (100.0, 100.0, 150.0, 160.0)
    out = None
    for i in range(50):
        out = sm.smooth(box, t=i / 30.0)
    # After many identical observations it should sit on the box.
    assert all(abs(a - b) < 1.0 for a, b in zip(out, box))


def test_box_smoothing_reduces_jitter():
    import random

    rng = random.Random(0)
    sm = OneEuroBoxSmoother(freq=30.0, min_cutoff=0.5, beta=0.0)
    base = (100.0, 100.0, 150.0, 160.0)
    raw_jitter = 0.0
    smoothed_jitter = 0.0
    prev_raw = None
    prev_sm = None
    for i in range(100):
        noisy = tuple(v + rng.uniform(-5, 5) for v in base)
        out = sm.smooth(noisy, t=i / 30.0)
        if prev_raw is not None:
            raw_jitter += abs(noisy[0] - prev_raw)
            smoothed_jitter += abs(out[0] - prev_sm)
        prev_raw, prev_sm = noisy[0], out[0]
    assert smoothed_jitter < raw_jitter  # the whole point


# --- reset / lifecycle ----------------------------------------------------- #
def test_reset_clears_state():
    scorer = TrackScorer(GeometricSpatialScorer(), mode="track", alpha=0.6, box_smoother_factory=None)
    scorer.score("the cup", _cup_scene(0.5, 0.0))
    scorer.reset()
    s = scorer.score("the cup", _cup_scene(1.0, 0.0))[7]
    assert s == 1.0  # EMA reseeded from scratch after reset


# --- built-in runner ------------------------------------------------------- #
def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
