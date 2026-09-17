from __future__ import annotations

import numpy as np
import pytest

from utils.mdn_support_data import split_weight_set_store
from utils.weight_set_store import WeightSetStore


def _store(context_count: int = 10) -> WeightSetStore:
    store = WeightSetStore(num_objectives=2)
    for index in range(context_count):
        context = np.full(8, index / 100.0, dtype=np.float32)
        first = 0.2 + 0.6 * index / max(context_count - 1, 1)
        store.observe_certified_weight(
            context,
            np.array([first, 1.0 - first], dtype=np.float32),
        )
    return store


def test_support_split_is_deterministic_complete_and_context_disjoint():
    first = split_weight_set_store(_store(), seed=17)
    second = split_weight_set_store(_store(), seed=17)

    assert first.manifest == second.manifest
    assert first.train.context_count() + first.validation.context_count() + first.test.context_count() == 10
    assert first.train.total_vertex_count() + first.validation.total_vertex_count() + first.test.total_vertex_count() == 10

    split_keys = [
        {tuple(context.tolist()) for context, _ in partition.get_all_context_vertices()}
        for partition in (first.train, first.validation, first.test)
    ]
    assert split_keys[0].isdisjoint(split_keys[1])
    assert split_keys[0].isdisjoint(split_keys[2])
    assert split_keys[1].isdisjoint(split_keys[2])
    assert all(split_keys)


def test_support_split_keeps_all_vertices_for_context_together():
    store = _store(context_count=3)
    context = np.zeros(8, dtype=np.float32)
    store.observe_certified_weight(context, np.array([0.9, 0.1], dtype=np.float32))

    split = split_weight_set_store(store, seed=3)

    vertex_counts = []
    for partition in (split.train, split.validation, split.test):
        weight_set = partition.get_weight_set(context)
        vertex_counts.append(0 if weight_set is None else len(weight_set.vertices))
    assert sorted(vertex_counts) == [0, 0, 2]


def test_support_split_rejects_too_few_contexts_and_invalid_fractions():
    with pytest.raises(ValueError, match="At least three"):
        split_weight_set_store(_store(context_count=2))
    with pytest.raises(ValueError, match="must sum to 1"):
        split_weight_set_store(
            _store(context_count=3),
            train_fraction=0.8,
            validation_fraction=0.15,
            test_fraction=0.15,
        )
