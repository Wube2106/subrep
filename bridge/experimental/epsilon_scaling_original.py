"""Experimental epsilon-scale transformations for bridge-level PDS inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class EpsilonScaleResult:
    delta_r: float
    delta_n: Tuple[float, ...]
    epsilon: float


def apply_epsilon_scale(
    *,
    method: str,
    delta_r: float,
    delta_n: Sequence[float],
    epsilon: float,
    payoff_scale: Optional[float] = None,
    motive_scale: Optional[float] = None,
    objective_scales: Optional[Sequence[float]] = None,
    baseline_return: Optional[float] = None,
) -> EpsilonScaleResult:
    """Transform candidate improvements and epsilon using approach A–D or F."""
    method = method.strip().upper().replace("-", "_")
    values = np.asarray(delta_n, dtype=float).reshape(-1)

    if not np.isfinite(delta_r):
        raise ValueError("delta_r must be finite")
    if not np.all(np.isfinite(values)):
        raise ValueError("delta_n values must be finite")
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")

    scaled_delta_r = float(delta_r)
    scaled_delta_n = values.copy()
    effective_epsilon = float(epsilon)

    if method == "A":
        if payoff_scale is None or motive_scale is None:
            raise ValueError("Approach A requires payoff_scale and motive_scale")
        if (
            not np.isfinite(payoff_scale)
            or not np.isfinite(motive_scale)
            or payoff_scale <= 0
            or motive_scale <= 0
        ):
            raise ValueError("Scaling values must be finite and positive")
        scaled_delta_r /= payoff_scale
        scaled_delta_n /= motive_scale

    elif method == "B":
        if payoff_scale is None or objective_scales is None:
            raise ValueError("Approach B requires payoff_scale and objective_scales")
        scales = np.asarray(objective_scales, dtype=float).reshape(-1)
        if (
            not np.isfinite(payoff_scale)
            or payoff_scale <= 0
            or scales.shape != values.shape
        ):
            raise ValueError("Invalid payoff or objective scales")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("Objective scales must be finite and positive")
        scaled_delta_r /= payoff_scale
        scaled_delta_n /= scales

    elif method == "C_PAYOFF":
        if (
            payoff_scale is None
            or not np.isfinite(payoff_scale)
            or payoff_scale <= 0
        ):
            raise ValueError("C_PAYOFF requires a finite positive payoff_scale")
        effective_epsilon *= payoff_scale

    elif method == "C_MOTIVE":
        if (
            motive_scale is None
            or not np.isfinite(motive_scale)
            or motive_scale <= 0
        ):
            raise ValueError("C_MOTIVE requires a finite positive motive_scale")
        effective_epsilon *= motive_scale

    elif method == "D":
        if baseline_return is None or not np.isfinite(baseline_return):
            raise ValueError("Approach D requires a finite baseline_return")
        effective_epsilon *= abs(float(baseline_return))

    elif method == "F":
        pass

    else:
        raise ValueError("Unsupported per-candidate epsilon scaling method: {!r}".format(method))

    if (
        not np.isfinite(scaled_delta_r)
        or not np.all(np.isfinite(scaled_delta_n))
        or not np.isfinite(effective_epsilon)
    ):
        raise ValueError("Epsilon scaling produced non-finite values")

    return EpsilonScaleResult(
        delta_r=scaled_delta_r,
        delta_n=tuple(float(value) for value in scaled_delta_n),
        epsilon=effective_epsilon,
    )


@dataclass(frozen=True)
class EpsilonScalingConfig:
    """Optional bridge-level configuration for epsilon scaling."""

    enabled: bool = False
    method: str = "F"
    payoff_scale: float = 82.7272727
    motive_scale: float = 235.1594902
    objective_scales: Tuple[float, ...] = (357.073526, 113.245455)

    def __post_init__(self):
        if (
            not np.isfinite(self.payoff_scale)
            or not np.isfinite(self.motive_scale)
            or self.payoff_scale <= 0
            or self.motive_scale <= 0
        ):
            raise ValueError("Payoff and motive scales must be finite and positive")
        if any(
            not np.isfinite(scale) or scale <= 0
            for scale in self.objective_scales
        ):
            raise ValueError("Objective scales must be finite and positive")


def rank_based_epsilon_admission(
    *,
    delta_r_values: Sequence[float],
    delta_n_values: Sequence[Sequence[float]],
    epsilon: float,
) -> Tuple[Tuple[float, ...], Tuple[bool, ...]]:
    """Apply E1's per-seed rank-based admission rule to a candidate batch.

    Candidates are ranked by ascending raw score:
        delta_r + min(delta_n)

    Tied scores receive their average ascending rank. The normalized rank
    margin is compared with -epsilon, matching the E1 experiment.
    """
    if len(delta_r_values) != len(delta_n_values):
        raise ValueError("Payoff and motive candidate counts must match")

    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")

    if not delta_r_values:
        return (), ()

    scores = []
    for delta_r, delta_n in zip(delta_r_values, delta_n_values):
        motives = np.asarray(delta_n, dtype=float).reshape(-1)
        if motives.size == 0 or not np.all(np.isfinite(motives)):
            raise ValueError("Each candidate must have finite objective values")
        if not np.isfinite(delta_r):
            raise ValueError("Candidate payoff values must be finite")
        scores.append(float(delta_r) + float(np.min(motives)))

    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)

    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1

        average_rank = ((i + 1) + j) / 2.0
        for position in range(i, j):
            original_index = ordered[position][0]
            ranks[original_index] = average_rank

        i = j

    candidate_count = len(scores)
    if candidate_count == 1:
        margins = (0.0,)
    else:
        margins = tuple(
            (rank - candidate_count) / (candidate_count - 1)
            for rank in ranks
        )

    admitted = tuple(margin >= -epsilon for margin in margins)
    return margins, admitted
