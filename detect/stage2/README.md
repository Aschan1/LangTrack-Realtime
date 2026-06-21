# Stage 2 — Scoring (Person B)

Person B's deliverable from the two-week sprint plan: the scoring layer behind the
frozen Day-0 contract

```
score(prefix_text, tracks) -> {track_id: score}
```

Person C drives this in the replay loop (Stage 3) and never looks inside.

## What's here

| File | Role |
|------|------|
| `tracks.py` | `Track` — the object that crosses the API boundary (id, label, box, conf). |
| `query.py` | `Query` + incremental prefix parsers. Fields stay empty while the prefix is incomplete. |
| `spatial_backend.py` | `SpatialScorer` protocol + `GeometricSpatialScorer` (dependency-free instant scorer / no-adapter geometric baseline / mock). |
| `dyhead_backend.py` | `DyHeadSpatialScorer` — the **real** trained DyHead relation adapter behind the same protocol. |
| `smoothing.py` | `EMABoxSmoother`, `OneEuroBoxSmoother` — box-coordinate smoothing. |
| `scorer.py` | `TrackScorer` — **the contract**. EMA, frame/track modes, selection hysteresis, box smoothing. |
| `tracker_config.py` | `build_bytetrack_args` — ByteTrack with the two thresholds pulled apart (appear/stay hysteresis). |
| `ablation.py` | Frame-vs-track comparison + EMA α-sweep (plan D5–6, the core ablation). |
| `tests/test_stage2.py` | Unit tests (run with pytest or directly). |

## The pipeline inside `score()`

1. **Parse** the prefix → `{target, anchor, relation}` (`query.py`). Incomplete →
   empty fields, e.g. `"the cup left of"` has a relation but no anchor yet.
2. **Instant score** every track for that query (`SpatialScorer` backend).
3. **EMA** (track mode): `S_new = α·s_now + (1−α)·S_old` — the anti-jitter fix.
   Frame mode skips this (instantaneous) — the other arm of the ablation.
4. **Hysteresis** on the selected track (two thresholds) so the top guess doesn't
   blink; **box smoothing** so the acted-on box doesn't shake.

`score()` returns the contract dict. `score_full()` additionally returns the
hysteresis-stabilised `selected_track_id` and the smoothed `selected_box`.

## Usage

### Offline replay (Person C / tests) — geometric backend, no heavy deps

```python
from stage2 import TrackScorer, Track, GeometricSpatialScorer

scorer = TrackScorer(GeometricSpatialScorer(), mode="track", alpha=0.6)
scorer.reset()  # once per sample
for (t, prefix, tracks) in sample_stream:        # tracks: list[Track]
    scores = scorer.score(prefix, tracks, t)     # {track_id: score}
```

### Live / real model — trained DyHead adapter

```python
from stage2 import TrackScorer
from stage2.dyhead_backend import DyHeadSpatialScorer, attach_crops

backend = DyHeadSpatialScorer.from_checkpoint(
    "outputs/dyhead_relation_adapter_nyu_rgb_only/best.pt",
    "yoloe-26l-seg.pt",
)
scorer = TrackScorer(backend, mode="track", alpha=0.6)

# each frame: tracks come from YOLOE + ByteTrack; attach crops for the adapter
attach_crops(tracks, frame_bgr, crop_size=96)
scores = scorer.score(prefix_text, tracks, t)
```

The DyHead backend mirrors `detect/demo_dyhead.py` exactly: `fused = sigmoid(logits)·conf`.

### ByteTrack hysteresis

```python
from stage2 import build_bytetrack_args
from ultralytics.trackers.byte_tracker import BYTETracker

tracker = BYTETracker(build_bytetrack_args(track_high_thresh=0.6, track_low_thresh=0.1), frame_rate=30)
```

## Running things

```bash
cd detect
python stage2/tests/test_stage2.py     # unit tests (16, no pytest needed)
python -m stage2.ablation              # frame-vs-track + α-sweep demo
```

The ablation demo on synthetic near-tie scenes shows the expected result: frame-level
has the most top-guess flips; track-mode flips drop as α decreases (more smoothing).

## Scope notes / hand-offs

- **Metric definitions** (anytime accuracy, time-to-lock, commit curve, the edge
  rules) are Person C's evaluator. This package computes only **flip count**, the
  stability metric Person B owns, from the same trajectories.
- The `Track` contract is the proposed Day-0 lock; confirm the exact fields with
  Persons A/C. The geometric backend lets everyone work against mock data immediately.
- `GeometricSpatialScorer` is both the mock *and* a legitimate "no learned adapter"
  geometric baseline for the paper.
