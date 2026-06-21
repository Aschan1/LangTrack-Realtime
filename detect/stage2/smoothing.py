"""Box-coordinate smoothing so the acted-on box doesn't visually shake.

Two filters, both operating independently on each of the 4 xyxy coordinates:

* :class:`EMABoxSmoother` - exponential moving average, one constant ``alpha``.
* :class:`OneEuroBoxSmoother` - the 1-Euro filter (Casiez et al., CHI 2012): an
  adaptive low-pass filter whose cutoff rises with speed, so it is smooth when the
  box is still and responsive when it moves fast. Preferred for visible boxes.

Both satisfy the :class:`BoxSmoother` protocol and are dependency-free.
"""

from __future__ import annotations

import math
from typing import Optional, Protocol, runtime_checkable

Box = tuple[float, float, float, float]


@runtime_checkable
class BoxSmoother(Protocol):
    def smooth(self, box: Box, t: Optional[float] = None) -> Box: ...

    def reset(self) -> None: ...


class EMABoxSmoother:
    """``b_new = alpha * measurement + (1 - alpha) * b_old`` per coordinate."""

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: Optional[Box] = None

    def reset(self) -> None:
        self._state = None

    def smooth(self, box: Box, t: Optional[float] = None) -> Box:
        if self._state is None:
            self._state = tuple(float(v) for v in box)  # type: ignore[assignment]
            return self._state
        a = self.alpha
        self._state = tuple(a * float(m) + (1.0 - a) * s for m, s in zip(box, self._state))  # type: ignore[assignment]
        return self._state


class _LowPass:
    def __init__(self) -> None:
        self.y: Optional[float] = None

    def __call__(self, x: float, alpha: float) -> float:
        if self.y is None:
            self.y = x
        else:
            self.y = alpha * x + (1.0 - alpha) * self.y
        return self.y


class _OneEuro:
    """Scalar 1-Euro filter."""

    def __init__(self, freq: float, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t: Optional[float] = None
        self._last_x: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t = None
        self._last_x = None

    def __call__(self, x: float, t: Optional[float]) -> float:
        if t is not None and self._last_t is not None and t > self._last_t:
            self.freq = 1.0 / (t - self._last_t)
        if t is not None:
            self._last_t = t

        dx = 0.0 if self._last_x is None else (x - self._last_x) * self.freq
        self._last_x = x
        edx = self._dx(dx, self._alpha(self.d_cutoff, 1.0 / self.freq))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x(x, self._alpha(cutoff, 1.0 / self.freq))


class OneEuroBoxSmoother:
    """1-Euro filter applied independently to x1, y1, x2, y2.

    ``min_cutoff`` lower => smoother but laggier; ``beta`` higher => more responsive
    to fast motion. Defaults are reasonable for ~30 fps tracking boxes.
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ) -> None:
        self._filters = [_OneEuro(freq, min_cutoff, beta, d_cutoff) for _ in range(4)]

    def reset(self) -> None:
        for f in self._filters:
            f.reset()

    def smooth(self, box: Box, t: Optional[float] = None) -> Box:
        return tuple(f(float(v), t) for f, v in zip(self._filters, box))  # type: ignore[return-value]
