"""Bridge-level tests for shared-scale epsilon normalization (Approach A)."""

import numpy as np
import pytest

from bridge.controller import MetaMoController
from bridge.epsilon_scaling import EpsilonScalingConfig
from bridge.governor import FakeGovernor
from bridge.protocol import GovernorSignal
from generator.mdn import MotiveDecompositionNetwork
from utils.mdn_contracts import CandidateSkillRecord
from utils.mdn_runtime_pipeline import (
    RuntimeCertificationPipeline,
    RuntimePipelineConfig,
)
from utils.weight_set_store import WeightSetStore


def make_pds_pipeline():
    return RuntimeCertificationPipeline(
        model=MotiveDecompositionNetwork(input_dim=8, num_objectives=2),
        weight_store=WeightSetStore(num_objectives=2),
        config=RuntimePipelineConfig(
            gate_type="PDS",
            use_cvar=False,
            train_support_after_certify=False,
        ),
    )

def make_candidate():
    return CandidateSkillRecord(
        skill_id="borderline_candidate",
        delta_r=19.8,
        delta_n=(-20.0, -20.0),
        is_certified=False,
        gate_type="PDS",
    )

def make_signal(epsilon=0.1):
    return GovernorSignal(
        weights=np.array([0.5, 0.5], dtype=np.float32),
        pds_epsilon=epsilon,
        cvar_tail_level=0.1,
    )

def test_certify_rejects_duplicate_skill_ids():
    controller = MetaMoController(
        FakeGovernor([make_signal()]),
        make_pds_pipeline(),
        epsilon_scaling=EpsilonScalingConfig(enabled=True),
    )

    candidates = [make_candidate(), make_candidate()]

    with pytest.raises(ValueError, match="unique skill_id"):
        controller.certify(
            context=np.array([0.1] * 8, dtype=np.float32),
            candidate_skills=candidates,
            baseline_stats={},
            signal=make_signal(),
        )
def test_scaling_disabled_preserves_original_pds_values():
    pipeline = make_pds_pipeline()
    controller = MetaMoController(
        FakeGovernor([make_signal()]),
        pipeline,
        epsilon_scaling=EpsilonScalingConfig(enabled=False),
    )

    (result,) = controller.certify(
        context=np.array([0.1] * 8, dtype=np.float32),
        candidate_skills=[make_candidate()],
        baseline_stats={
            "baseline_payoff": 100.0,
            "baseline_motives": np.array([0.5, 0.2], dtype=np.float32),
        },
        signal=make_signal(),
    )

    assert result.is_certified is False
    assert result.delta_r == pytest.approx(19.8)
    assert result.delta_n == pytest.approx((-20.0, -20.0))
    assert result.epsilon == pytest.approx(0.1)
def test_approach_a_changes_borderline_pds_admission():
    pipeline = make_pds_pipeline()
    controller = MetaMoController(
        FakeGovernor([make_signal()]),
        pipeline,
        epsilon_scaling=EpsilonScalingConfig(
            enabled=True,
        ),
    )

    (result,) = controller.certify(
        context=np.array([0.1] * 8, dtype=np.float32),
        candidate_skills=[make_candidate()],
        baseline_stats={
            "baseline_payoff": 100.0,
            "baseline_motives": np.array([0.5, 0.2], dtype=np.float32),
        },
        signal=make_signal(),
    )

    assert result.is_certified is True
    assert result.delta_r == pytest.approx(19.8)
    assert result.delta_n == pytest.approx((-20.0, -20.0))
    assert result.epsilon == pytest.approx(0.1)
def test_crashed_candidate_is_rejected_even_when_scaling_would_admit_it():
    pipeline = make_pds_pipeline()
    controller = MetaMoController(
        FakeGovernor([make_signal()]),
        pipeline,
        epsilon_scaling=EpsilonScalingConfig(
            enabled=True,
        ),
    )

    crashed_candidate = CandidateSkillRecord(
        skill_id="crashed_candidate",
        delta_r=19.8,
        delta_n=(-20.0, -20.0),
        is_certified=False,
        gate_type="PDS",
        metadata={"crashed": True},
    )

    (result,) = controller.certify(
        context=np.array([0.1] * 8, dtype=np.float32),
        candidate_skills=[crashed_candidate],
        baseline_stats={
            "baseline_payoff": 100.0,
            "baseline_motives": np.array([0.5, 0.2], dtype=np.float32),
        },
        signal=make_signal(),
    )

    assert result.is_certified is False
def test_selection_uses_original_values_after_scaling():
    class AdmitAllPipeline:
        def certify_candidate_skills(
            self,
            *,
            context,
            candidate_skills,
            baseline_stats,
            weights_used=None,
            cvar_confidence=None,
        ):
            from dataclasses import replace

            return [
                replace(candidate, is_certified=True)
                for candidate in candidate_skills
            ]

    candidates = [
        CandidateSkillRecord(
            skill_id="payoff_skill",
            delta_r=110.0,
            delta_n=(0.0, 0.0),
            is_certified=False,
            gate_type="PDS",
        ),
        CandidateSkillRecord(
            skill_id="motive_skill",
            delta_r=0.0,
            delta_n=(200.0, 200.0),
            is_certified=False,
            gate_type="PDS",
        ),
    ]

    signal = make_signal()
    controller = MetaMoController(
        FakeGovernor([signal]),
        AdmitAllPipeline(),
        epsilon_scaling=EpsilonScalingConfig(
            enabled=True,
        ),
    )

    records = controller.certify(
        context=np.array([0.1] * 8, dtype=np.float32),
        candidate_skills=candidates,
        baseline_stats={},
        signal=signal,
    )

    selected_id, selected_score = controller.select(records, signal)

    assert selected_id == "motive_skill"
    assert selected_score == pytest.approx(200.0)
    assert records[0].delta_r == pytest.approx(110.0)
    assert records[1].delta_n == pytest.approx((200.0, 200.0))
def test_epsilon_change_changes_admission_with_scaling_enabled():
    pipeline = make_pds_pipeline()
    signals = [make_signal(0.10), make_signal(0.00)]
    governor = FakeGovernor(signals)

    controller = MetaMoController(
        governor,
        pipeline,
        epsilon_scaling=EpsilonScalingConfig(
            enabled=True,
        ),
    )

    # After Approach A scaling:
    # delta_r = 0.05, delta_n = (-0.1, -0.1).
    # PDS margin = -0.05, between epsilon 0.10 and epsilon 0.00.
    candidate = CandidateSkillRecord(
        skill_id="epsilon_borderline",
        delta_r=0.05 * 82.7272727,
        delta_n=(-0.1 * 235.1594902, -0.1 * 235.1594902),
        is_certified=False,
        gate_type="PDS",
    )

    context = np.array([0.1] * 8, dtype=np.float32)
    baseline_stats = {
        "baseline_payoff": 100.0,
        "baseline_motives": np.array([0.5, 0.2], dtype=np.float32),
    }

    first = controller.step(
        context=context,
        candidate_skills=[candidate],
        baseline_stats=baseline_stats,
        outcome_for=lambda record: __import__(
            "bridge.protocol", fromlist=["SkillOutcome"]
        ).SkillOutcome(
            delta_r=0.0,
            delta_n=np.zeros(2),
            admitted=record is not None,
        ),
    )

    second = controller.step(
        context=context,
        candidate_skills=[candidate],
        baseline_stats=baseline_stats,
        outcome_for=lambda record: __import__(
            "bridge.protocol", fromlist=["SkillOutcome"]
        ).SkillOutcome(
            delta_r=0.0,
            delta_n=np.zeros(2),
            admitted=record is not None,
        ),
    )

    assert first.pds_epsilon == pytest.approx(0.10)
    assert first.admitted_count == 1
    assert second.pds_epsilon == pytest.approx(0.00)
    assert second.admitted_count == 0
def test_scripted_epsilon_sequence_changes_admitted_set():
    from bridge.protocol import SkillOutcome

    epsilons = [0.50, 0.30, 0.10, 0.00]
    signals = [make_signal(epsilon) for epsilon in epsilons]

    controller = MetaMoController(
        FakeGovernor(signals),
        make_pds_pipeline(),
        epsilon_scaling=EpsilonScalingConfig(
            enabled=True,
        ),
    )

    candidate = CandidateSkillRecord(
        skill_id="epsilon_borderline",
        delta_r=0.05 * 82.7272727,
        delta_n=(-0.1 * 235.1594902, -0.1 * 235.1594902),
        is_certified=False,
        gate_type="PDS",
    )

    context = np.array([0.1] * 8, dtype=np.float32)
    baseline_stats = {
        "baseline_payoff": 100.0,
        "baseline_motives": np.array([0.5, 0.2], dtype=np.float32),
    }

    results = []
    for _ in epsilons:
        record = controller.step(
            context=context,
            candidate_skills=[candidate],
            baseline_stats=baseline_stats,
            outcome_for=lambda selected: SkillOutcome(
                delta_r=0.0,
                delta_n=np.zeros(2),
                admitted=selected is not None,
            ),
        )
        results.append(record)

    assert [record.pds_epsilon for record in results] == pytest.approx(epsilons)
    assert [record.admitted_count for record in results] == [1, 1, 1, 0]
def test_scripted_epsilon_sequence_preserves_raw_baseline_admission():
    from bridge.protocol import SkillOutcome

    epsilons = [0.50, 0.30, 0.10, 0.00]
    signals = [make_signal(epsilon) for epsilon in epsilons]

    controller = MetaMoController(
        FakeGovernor(signals),
        make_pds_pipeline(),
        epsilon_scaling=EpsilonScalingConfig(enabled=False),
    )

    candidate = CandidateSkillRecord(
        skill_id="epsilon_borderline",
        delta_r=0.05 * 82.7272727,
        delta_n=(-0.1 * 235.1594902, -0.1 * 235.1594902),
        is_certified=False,
        gate_type="PDS",
    )

    context = np.array([0.1] * 8, dtype=np.float32)
    baseline_stats = {
        "baseline_payoff": 100.0,
        "baseline_motives": np.array([0.5, 0.2], dtype=np.float32),
    }

    results = []
    for _ in epsilons:
        results.append(
            controller.step(
                context=context,
                candidate_skills=[candidate],
                baseline_stats=baseline_stats,
                outcome_for=lambda selected: SkillOutcome(
                    delta_r=0.0,
                    delta_n=np.zeros(2),
                    admitted=selected is not None,
                ),
            )
        )

    assert [record.pds_epsilon for record in results] == pytest.approx(epsilons)
    assert [record.admitted_count for record in results] == [0, 0, 0, 0]
def test_approach_a_supports_six_objectives():
    from bridge.epsilon_scaling import apply_epsilon_scale

    result = apply_epsilon_scale(
        delta_r=82.7272727,
        delta_n=(235.1594902, 470.3189804, -235.1594902,
                 117.5797451, 0.0, -470.3189804),
        epsilon=0.3,
        payoff_scale=82.7272727,
        motive_scale=235.1594902,
    )

    assert result.delta_r == pytest.approx(1.0)
    assert result.delta_n == pytest.approx(
        (1.0, 2.0, -1.0, 0.5, 0.0, -2.0)
    )
    assert result.epsilon == pytest.approx(0.3)
@pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.3, 0.5])
def test_approach_a_crash_veto_holds_at_every_epsilon(epsilon):
    class AdmitAllPipeline:
        def certify_candidate_skills(
            self,
            *,
            context,
            candidate_skills,
            baseline_stats,
            weights_used=None,
            cvar_confidence=None,
        ):
            from dataclasses import replace

            return [
                replace(candidate, is_certified=True)
                for candidate in candidate_skills
            ]

    crashed_candidate = CandidateSkillRecord(
        skill_id="crashed_candidate",
        delta_r=19.8,
        delta_n=(-20.0, -20.0),
        is_certified=False,
        gate_type="PDS",
        metadata={"crashed": True},
    )

    signal = make_signal(epsilon)
    controller = MetaMoController(
        FakeGovernor([signal]),
        AdmitAllPipeline(),
        epsilon_scaling=EpsilonScalingConfig(
            enabled=True,
        ),
    )

    (result,) = controller.certify(
        context=np.array([0.1] * 8, dtype=np.float32),
        candidate_skills=[crashed_candidate],
        baseline_stats={},
        signal=signal,
    )

    assert result.is_certified is False
@pytest.mark.parametrize("bad_scale", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
@pytest.mark.parametrize("field", ["payoff_scale", "motive_scale"])
def test_epsilon_config_rejects_invalid_shared_scales(field, bad_scale):
    with pytest.raises(ValueError):
        EpsilonScalingConfig(**{field: bad_scale})
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("delta_r", float("nan")),
        ("delta_r", float("inf")),
        ("delta_n", (1.0, float("nan"))),
        ("delta_n", (1.0, float("-inf"))),
        ("epsilon", float("nan")),
        ("epsilon", float("inf")),
        ("epsilon", -0.1),
    ],
)
def test_apply_epsilon_scale_rejects_invalid_candidate_values(field, value):
    from bridge.epsilon_scaling import apply_epsilon_scale

    arguments = {
        "delta_r": 1.0,
        "delta_n": (1.0, 2.0),
        "epsilon": 0.1,
        "payoff_scale": 1.0,
        "motive_scale": 1.0,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        apply_epsilon_scale(**arguments)
def test_approach_a_rejects_overflowed_scaled_values():
    from bridge.epsilon_scaling import apply_epsilon_scale

    with pytest.raises(ValueError, match="non-finite values"):
        apply_epsilon_scale(
            delta_r=1e308,
            delta_n=(1.0, 2.0),
            epsilon=0.1,
            payoff_scale=1e-308,
            motive_scale=1.0,
        )
