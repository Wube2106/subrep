from __future__ import annotations

import json

import numpy as np
import torch

from generator.mdn import MotiveDecompositionNetwork
from generator.train_mdn_support import run_support_head_experiment
from utils.weight_set_store import WeightSetStore


def test_support_experiment_writes_reproducible_split_and_reports(tmp_path):
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    base_checkpoint = tmp_path / "base.pth"
    torch.save({"model_state_dict": model.state_dict()}, base_checkpoint)

    store = WeightSetStore(num_objectives=2)
    for index in range(6):
        first = 0.25 + index * 0.1
        store.observe_certified_weight(
            np.full(8, index / 10.0, dtype=np.float32),
            np.array([first, 1.0 - first], dtype=np.float32),
        )
    store_path = tmp_path / "weights.json"
    store.save(store_path)
    output_dir = tmp_path / "evaluation"
    output_checkpoint = tmp_path / "support.pth"

    report = run_support_head_experiment(
        base_checkpoint_path=base_checkpoint,
        weight_store_path=store_path,
        output_dir=output_dir,
        output_checkpoint_path=output_checkpoint,
        max_epochs=2,
        early_stopping_patience=1,
        seed=9,
        device="cpu",
    )

    assert report["status"] == "completed"
    assert output_checkpoint.exists()
    assert (output_dir / "support_train.json").exists()
    assert (output_dir / "support_validation.json").exists()
    assert (output_dir / "support_test.json").exists()
    assert (output_dir / "split_manifest.json").exists()
    assert (output_dir / "support_experiment.json").exists()
    assert (output_dir / "support_test_report.md").exists()

    serialized = json.loads((output_dir / "support_experiment.json").read_text(encoding="utf-8"))
    assert serialized["test"]["arms"]["learned"]["cds_agreement_rate"] is None
    assert serialized["test"]["arms"]["learned"]["support_value_mse"] >= 0.0
    assert "train_mean_constant" in serialized["test"]["arms"]
