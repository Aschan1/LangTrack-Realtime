"""The ``Track`` object that crosses the Stage-2 API boundary.

A *track* is a persistent object identity across frames (YOLOE + ByteTrack). At a
given timestep it is, in practice, a single detection (box + confidence + class
label) carrying a stable ``track_id``. This is the unit Person C feeds to
``score(prefix_text, tracks)`` and the unit the spatial backend scores.

Only plain Python / numpy types are used here so the contract has no heavy
dependencies. The optional ``crop`` field carries a precomputed proposal crop
(an ``np.ndarray`` HxWx3 in [0, 1] or a torch tensor 3xHxW) for the real DyHead
backend; the geometric mock backend ignores it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

Box = tuple[float, float, float, float]  # xyxy in pixels


@dataclass
class Track:
    """One tracked object at one timestep."""

    track_id: int
    label: str  # the YOLOE class this track instance belongs to, e.g. "cup"
    box: Box  # xyxy in the current frame
    conf: float  # detector confidence this frame
    image_wh: tuple[float, float]  # (width, height) of the frame
    crop: Optional[Any] = None  # optional precomputed crop for the DyHead backend
    text_feat: Optional[Any] = None  # optional precomputed label embedding (DyHead)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.track_id = int(self.track_id)
        self.label = str(self.label).strip().lower()
        self.box = tuple(float(v) for v in self.box)  # type: ignore[assignment]
        if len(self.box) != 4:
            raise ValueError(f"box must be xyxy of length 4, got {self.box!r}")
        self.conf = float(self.conf)
        self.image_wh = (float(self.image_wh[0]), float(self.image_wh[1]))

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def tracks_from_bytetrack(rows: Sequence[Sequence[float]], labels: Sequence[str], image_wh: tuple[float, float]) -> list[Track]:
    """Adapt the ``BYTETracker.update`` output to ``Track`` objects.

    ``BYTETracker.update`` returns rows ``[x1, y1, x2, y2, track_id, conf, cls, det_idx]``.
    ``labels`` maps the integer class id to the YOLOE class string used in the query.
    """
    out: list[Track] = []
    for row in rows:
        x1, y1, x2, y2, track_id, conf, cls = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
        cls_idx = int(cls)
        label = labels[cls_idx] if 0 <= cls_idx < len(labels) else str(cls_idx)
        out.append(
            Track(
                track_id=int(track_id),
                label=label,
                box=(float(x1), float(y1), float(x2), float(y2)),
                conf=float(conf),
                image_wh=image_wh,
            )
        )
    return out
