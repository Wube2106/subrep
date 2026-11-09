"""Rank-based admission rules for experimental candidate evaluation."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

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
