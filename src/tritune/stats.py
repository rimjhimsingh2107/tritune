from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


def geomean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None and v > 0 and math.isfinite(v)]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


@dataclass
class Verdict:
    better: bool
    gain: float  # fractional p50 improvement, positive = faster
    threshold: float
    reason: str


def compare_latency(
    new_p50: float,
    new_p95: float | None,
    old_p50: float,
    old_p95: float | None,
    min_gain: float = 0.01,
) -> Verdict:
    """Is `new` faster than `old` by more than the measurement noise?

    Noise is estimated from the p50->p95 spread of both measurements. It's a
    heuristic, not a hypothesis test, but it stops the optimizer from chasing
    1% wins that are just jitter.
    """

    def spread(p50: float, p95: float | None) -> float:
        return 0.0 if not p95 or p50 <= 0 else max(0.0, (p95 - p50) / p50)

    noise = 0.5 * max(spread(new_p50, new_p95), spread(old_p50, old_p95))
    threshold = max(min_gain, noise)
    gain = (old_p50 - new_p50) / old_p50
    if gain > threshold:
        return Verdict(
            True, gain, threshold, f"{gain:.1%} faster (> {threshold:.1%} noise threshold)"
        )
    if gain < -threshold:
        return Verdict(False, gain, threshold, f"regression: {-gain:.1%} slower")
    return Verdict(False, gain, threshold, f"within noise ({gain:+.1%}, threshold {threshold:.1%})")
