"""Shared-scale normalization for candidate improvements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class EpsilonScaleResult:
    """Normalized candidate improvements and unchanged admission tolerance."""

    delta_r: float
    delta_n: Tuple[float, ...]
    epsilon: float


@dataclass(frozen=True)
class EpsilonScalingConfig:
    """Configuration for optional shared-scale normalization."""

    enabled: bool = False
    payoff_scale: float = 82.7272727
    motive_scale: float = 235.1594902

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.payoff_scale)
            or not np.isfinite(self.motive_scale)
            or self.payoff_scale <= 0
            or self.motive_scale <= 0
        ):
            raise ValueError("Payoff and motive scales must be finite and positive")


def apply_epsilon_scale(
    *,
    delta_r: float,
    delta_n: Sequence[float],
    epsilon: float,
    payoff_scale: float,
    motive_scale: float,
) -> EpsilonScaleResult:
    """Normalize payoff and motive improvements while preserving epsilon.

    A single shared scale is applied to all motive objectives so their
    relative magnitudes are preserved.
    """
    values = np.asarray(delta_n, dtype=float).reshape(-1)

    if not np.isfinite(delta_r):
        raise ValueError("delta_r must be finite")
    if not np.all(np.isfinite(values)):
        raise ValueError("delta_n values must be finite")
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    if (
        not np.isfinite(payoff_scale)
        or not np.isfinite(motive_scale)
        or payoff_scale <= 0.0
        or motive_scale <= 0.0
    ):
        raise ValueError("Scaling values must be finite and positive")

    scaled_delta_r = float(delta_r) / float(payoff_scale)
    scaled_delta_n = values / float(motive_scale)

    if (
        not np.isfinite(scaled_delta_r)
        or not np.all(np.isfinite(scaled_delta_n))
    ):
        raise ValueError("Epsilon scaling produced non-finite values")

    return EpsilonScaleResult(
        delta_r=scaled_delta_r,
        delta_n=tuple(float(value) for value in scaled_delta_n),
        epsilon=float(epsilon),
    )
