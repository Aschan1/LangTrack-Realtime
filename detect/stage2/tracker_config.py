"""ByteTrack configuration with the two thresholds pulled apart (hysteresis).

The plan's Stage-2 stability fix: ByteTrack already exposes ``track_high_thresh`` (a
detection must clear this to *start*/confirm a track) and ``track_low_thresh`` (a
weak detection above this can still keep an existing track alive). Setting them to
the same value makes objects near the boundary blink in and out; pulling them apart
gives appear/stay hysteresis.

``build_bytetrack_args`` returns an ``argparse.Namespace`` ready for
``ultralytics.trackers.byte_tracker.BYTETracker(args, frame_rate=...)``.
"""

from __future__ import annotations

import argparse


def build_bytetrack_args(
    *,
    track_high_thresh: float = 0.6,  # appear: pulled UP from the 0.5 default
    track_low_thresh: float = 0.1,  # stay: kept low
    new_track_thresh: float = 0.6,  # confirm a brand-new track at the high bar
    track_buffer: int = 30,
    match_thresh: float = 0.8,
    fuse_score: bool = False,
) -> argparse.Namespace:
    """Build BYTETracker args with appear/stay hysteresis.

    Defaults pull ``high`` (0.6) and ``low`` (0.1) apart. Set them equal to recover
    the no-hysteresis baseline for the ablation.
    """
    if not 0.0 <= track_low_thresh <= track_high_thresh <= 1.0:
        raise ValueError("require 0 <= track_low_thresh <= track_high_thresh <= 1")
    return argparse.Namespace(
        track_high_thresh=float(track_high_thresh),
        track_low_thresh=float(track_low_thresh),
        new_track_thresh=float(new_track_thresh),
        track_buffer=int(track_buffer),
        match_thresh=float(match_thresh),
        fuse_score=bool(fuse_score),
    )


def baseline_bytetrack_args() -> argparse.Namespace:
    """No-hysteresis baseline (high == low == new == 0.5) for the ablation."""
    return build_bytetrack_args(track_high_thresh=0.5, track_low_thresh=0.5, new_track_thresh=0.5)
