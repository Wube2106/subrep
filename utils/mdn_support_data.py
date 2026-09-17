"""Leakage-resistant data splitting for context-conditioned support targets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from utils.weight_set_store import WeightSetStore


@dataclass(frozen=True)
class SupportStoreSplit:
    """Context-disjoint support stores and a reproducibility manifest."""

    train: WeightSetStore
    validation: WeightSetStore
    test: WeightSetStore
    manifest: dict[str, Any]


def split_weight_set_store(
    store: WeightSetStore,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> SupportStoreSplit:
    """Split all vertices as one group per context.

    Contexts, rather than individual vertices, are shuffled. This prevents
    repeated observations of one context from crossing partition boundaries.
    At least three contexts are required so every partition is non-empty.
    """
    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction], dtype=np.float64
    )
    if not np.all(np.isfinite(fractions)) or np.any(fractions <= 0.0):
        raise ValueError("train, validation, and test fractions must be finite and positive")
    if not np.isclose(float(np.sum(fractions)), 1.0, atol=1e-9, rtol=0.0):
        raise ValueError(
            f"train, validation, and test fractions must sum to 1, got {float(np.sum(fractions)):.6f}"
        )

    entries = store.get_all_context_vertices()
    if len(entries) < 3:
        raise ValueError("At least three distinct contexts are required for train/validation/test splitting")

    counts = _partition_counts(len(entries), fractions)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(entries))
    shuffled = [entries[int(index)] for index in order]

    train_end = int(counts[0])
    validation_end = train_end + int(counts[1])
    partition_entries = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    partitions = {
        name: _store_from_entries(store.num_objectives, values)
        for name, values in partition_entries.items()
    }

    fingerprints = {
        name: [_context_fingerprint(context) for context, _ in values]
        for name, values in partition_entries.items()
    }
    _assert_disjoint(fingerprints)
    manifest: dict[str, Any] = {
        "version": 1,
        "seed": int(seed),
        "fractions": {
            "train": float(train_fraction),
            "validation": float(validation_fraction),
            "test": float(test_fraction),
        },
        "source_contexts": int(store.context_count()),
        "source_vertices": int(store.total_vertex_count()),
        "splits": {
            name: {
                "contexts": int(partitions[name].context_count()),
                "vertices": int(partitions[name].total_vertex_count()),
                "context_fingerprints": values,
            }
            for name, values in fingerprints.items()
        },
    }
    return SupportStoreSplit(
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
        manifest=manifest,
    )


def _partition_counts(size: int, fractions: np.ndarray) -> np.ndarray:
    raw_counts = fractions * size
    counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(size - int(np.sum(counts)))
    fractional_order = np.argsort(-(raw_counts - counts), kind="stable")
    for index in fractional_order[:remainder]:
        counts[int(index)] += 1

    for empty_index in np.flatnonzero(counts == 0):
        donor_index = int(np.argmax(counts))
        if counts[donor_index] <= 1:
            raise ValueError("Unable to create three non-empty support partitions")
        counts[donor_index] -= 1
        counts[int(empty_index)] += 1
    return counts


def _store_from_entries(
    num_objectives: int,
    entries: list[tuple[np.ndarray, np.ndarray]],
) -> WeightSetStore:
    partition = WeightSetStore(num_objectives=num_objectives)
    for context, vertices in entries:
        for vertex in vertices:
            partition.observe_certified_weight(context, vertex)
    return partition


def _context_fingerprint(context: np.ndarray) -> str:
    canonical = np.asarray(context, dtype="<f4").reshape(-1)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _assert_disjoint(fingerprints: dict[str, list[str]]) -> None:
    names = tuple(fingerprints)
    for left_index, left_name in enumerate(names):
        left = set(fingerprints[left_name])
        for right_name in names[left_index + 1 :]:
            overlap = left.intersection(fingerprints[right_name])
            if overlap:
                raise AssertionError(
                    f"Support split leakage between {left_name} and {right_name}: {sorted(overlap)}"
                )
