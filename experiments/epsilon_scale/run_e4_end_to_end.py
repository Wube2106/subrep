"""E4: scripted epsilon comparison through the real LunarLander loop."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))

from bridge.controller import MetaMoController
from bridge.epsilon_scaling import EpsilonScalingConfig, apply_epsilon_scale
from bridge.governor import FakeGovernor
from bridge.protocol import GovernorSignal
from data_collector.collect_candidate_sets import (
    CandidatePolicy,
    build_default_candidate_policies,
)
from env.lunar_lander_rollout import (
    NOOP_BASELINE,
    episode_outcome,
    evaluate_candidates,
    execute_candidate,
)
from env.lunar_lander_wrapper import SubRepEnv
from generator.mdn import MotiveDecompositionNetwork
from utils.mdn_contracts import CandidateSkillRecord
from utils.mdn_runtime_pipeline import (
    RuntimeCertificationPipeline,
    RuntimePipelineConfig,
)
from utils.weight_set_store import WeightSetStore


SEED = 1
CONTROLLER_SEED = 42
EPSILONS = [0.5, 0.3, 0.1, 0.0]
WEIGHTS = np.array([0.5, 0.5], dtype=np.float64)

# Fixed calibration scales from the completed E2 experiment.
PAYOFF_SCALE = 82.7272727
MOTIVE_SCALE = 235.1594902

CHECKPOINT = REPO / "models" / "pilot_ppo.pt"
CSV_PATH = ROOT / "e4_end_to_end_steps.csv"
JSON_PATH = ROOT / "e4_end_to_end_summary.json"


def make_pipeline():
    torch.manual_seed(CONTROLLER_SEED)
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    model.eval()

    return RuntimeCertificationPipeline(
        model=model,
        weight_store=WeightSetStore(num_objectives=2),
        config=RuntimePipelineConfig(
            gate_type="PDS",
            use_cvar=False,
            train_support_after_certify=False,
        ),
    )


def make_governor():
    signals = [
        GovernorSignal(
            weights=WEIGHTS.copy(),
            pds_epsilon=epsilon,
            cvar_tail_level=0.5,
        )
        for epsilon in EPSILONS
    ]
    return FakeGovernor(signals, repeat_last=True)


def run_mode(env, policies, candidate_records, baseline_stats, enabled):
    mode = "after_scaling_on" if enabled else "before_scaling_off"

    controller = MetaMoController(
        governor=make_governor(),
        pipeline=make_pipeline(),
        seed=CONTROLLER_SEED,
        epsilon_scaling=EpsilonScalingConfig(
            enabled=enabled,
            payoff_scale=PAYOFF_SCALE,
            motive_scale=MOTIVE_SCALE,
        ),
    )

    policy_by_id = {policy.skill_id: policy for policy in policies}
    # This candidate has deliberately scripted certification values.
    # Its physical rollout uses the deterministic PPO policy and is reported
    # separately; these values are NOT presented as measured PPO improvements.
    policy_by_id["borderline_scripted"] = policy_by_id["ppo_deterministic"]
    captured = {}
    original_certify = controller.certify

    # Observe the records returned by the controller's real certification
    # path. No direct call to the pipeline is used for admission decisions.
    def capture_certification(**kwargs):
        records = original_certify(**kwargs)
        captured["records"] = records
        return records

    controller.certify = capture_certification

    rows = []
    for step, epsilon in enumerate(EPSILONS):
        executed = {}

        def outcome_for(selected):
            policy = (
                NOOP_BASELINE
                if selected is None
                else policy_by_id[selected.skill_id]
            )
            delta_r, delta_n, info = execute_candidate(
                env, policy, seed=SEED
            )
            executed.update(info)
            executed["executed_policy"] = policy.skill_id
            executed["delta_r"] = float(delta_r)
            executed["delta_n"] = np.asarray(delta_n).tolist()

            return episode_outcome(
                delta_r,
                delta_n,
                info,
                admitted=selected is not None,
            )

        step_record = controller.step(
            context=np.asarray(baseline_stats["info"]["start_obs"]),
            candidate_skills=candidate_records,
            baseline_stats=baseline_stats,
            outcome_for=outcome_for,
            seed=SEED,
        )

        certified_records = captured["records"]
        admitted_ids = sorted(
            record.skill_id
            for record in certified_records
            if record.is_certified
        )

        print(
            "{} step={} epsilon={:.2f} weights={} admitted={} "
            "selected={} landing={} crashed={}".format(
                mode,
                step_record.step,
                step_record.pds_epsilon,
                np.asarray(step_record.weights).tolist(),
                admitted_ids,
                step_record.selected_skill_id,
                executed.get("landing_success", False),
                executed.get("crashed", False),
            )
        )

        for record in certified_records:
            original = next(
                candidate
                for candidate in candidate_records
                if candidate.skill_id == record.skill_id
            )
            raw_margin = float(original.delta_r) + min(
                float(value) for value in original.delta_n
            )
            scaled = apply_epsilon_scale(
                delta_r=float(original.delta_r),
                delta_n=original.delta_n,
                epsilon=float(step_record.pds_epsilon),
                payoff_scale=PAYOFF_SCALE,
                motive_scale=MOTIVE_SCALE,
            )
            scaled_margin = scaled.delta_r + min(scaled.delta_n)
            effective_margin = record.admission_margin

            print(
                "  candidate={} raw_margin={:.6f} "
                "approach_A_margin={:.6f} "
                "pipeline_margin={} admitted={}".format(
                    record.skill_id,
                    raw_margin,
                    scaled_margin,
                    (
                        "n/a"
                        if effective_margin is None
                        else "{:.6f}".format(effective_margin)
                    ),
                    record.is_certified,
                )
            )

            rows.append({
                "mode": mode,
                "step": step_record.step,
                "epsilon": step_record.pds_epsilon,
                "weights": json.dumps(
                    np.asarray(step_record.weights).tolist()
                ),
                "skill_id": record.skill_id,
                "delta_r": original.delta_r,
                "delta_n": json.dumps(list(original.delta_n)),
                "raw_pds_margin": raw_margin,
                "effective_pipeline_margin": effective_margin,
                "margin_before_fix": raw_margin,
                "margin_after_fix": scaled_margin,
                "actual_pipeline_margin": effective_margin,
                "admitted": record.is_certified,
                "candidate_landing_success": (
                    None
                    if original.metadata.get("scripted_certification", False)
                    else original.metadata.get("landing_success")
                ),
                "candidate_crashed": (
                    None
                    if original.metadata.get("scripted_certification", False)
                    else original.metadata.get("crashed")
                ),
                "certification_values_scripted": bool(
                    original.metadata.get("scripted_certification", False)
                ),
                "selected_skill": step_record.selected_skill_id,
                "executed_policy": executed.get("executed_policy"),
                "executed_landing_success": executed.get(
                    "landing_success", False
                ),
                "executed_crashed": executed.get("crashed", False),
                "executed_steps": executed.get("steps"),
            })

    return rows


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            "Required PPO checkpoint not found: {}".format(CHECKPOINT)
        )

    env = SubRepEnv(seed=SEED)
    try:
        policies = build_default_candidate_policies(
            env, pilot_checkpoint=str(CHECKPOINT)
        )

        # Score the real candidates once. Both runs reuse these exact records,
        # baseline statistics, and starting observation.
        baseline_stats, candidate_records = evaluate_candidates(
            env, policies, seed=SEED
        )

        # A controlled borderline case makes the epsilon boundary observable.
        # Approach A gives it an effective margin of about -0.02461:
        # admitted at epsilon 0.5/0.3/0.1, rejected at epsilon 0.0.
        # Its physical execution policy is PPO deterministic, but its
        # certification values are explicitly synthetic.
        candidate_records.append(
            CandidateSkillRecord(
                skill_id="borderline_scripted",
                delta_r=5.0,
                delta_n=(-20.0, -20.0),
                is_certified=False,
                gate_type="PDS",
                metadata={
                    "scripted_certification": True,
                },
            )
        )

        print("Candidate measurements (scripted candidate is labelled):")
        for candidate in candidate_records:
            print(
                "  {}: delta_r={:.6f}, delta_n={}, raw_margin={:.6f}, "
                "landing={}, crashed={}".format(
                    candidate.skill_id,
                    candidate.delta_r,
                    list(candidate.delta_n),
                    candidate.delta_r + min(candidate.delta_n),
                    (
                        "not_measured"
                        if candidate.metadata.get("scripted_certification", False)
                        else candidate.metadata.get("landing_success")
                    ),
                    (
                        "not_measured"
                        if candidate.metadata.get("scripted_certification", False)
                        else candidate.metadata.get("crashed")
                    ),
                )
            )
            if candidate.metadata.get("scripted_certification"):
                print(
                    "    NOTE: scripted certification values; "
                    "not a measured PPO improvement."
                )

        rows = []
        rows.extend(
            run_mode(
                env, policies, candidate_records, baseline_stats,
                enabled=False,
            )
        )
        rows.extend(
            run_mode(
                env, policies, candidate_records, baseline_stats,
                enabled=True,
            )
        )

        with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "experiment": "E4 scripted end-to-end LunarLander comparison",
            "seed": SEED,
            "epsilon_sequence": EPSILONS,
            "weights": WEIGHTS.tolist(),
            "approach": "A",
            "use_cvar": False,
            "pipeline_gate": "PDS",
            "candidate_measurements_evaluated_once": True,
            "scripted_borderline_candidate": {
                "skill_id": "borderline_scripted",
                "delta_r": 5.0,
                "delta_n": [-20.0, -20.0],
                "physical_policy": "ppo_deterministic",
                "certification_values_are_synthetic": True,
            },
            "same_candidates_and_context_in_both_runs": True,
            "payoff_scale": PAYOFF_SCALE,
            "motive_scale": MOTIVE_SCALE,
            "admitted_by_epsilon": {},
            "selected_by_step": {},
            "executed_outcomes_by_step": {},
            "csv": str(CSV_PATH),
        }

        for mode in ("before_scaling_off", "after_scaling_on"):
            mode_rows = [row for row in rows if row["mode"] == mode]
            summary["admitted_by_epsilon"][mode] = {}
            summary["selected_by_step"][mode] = {}
            summary["executed_outcomes_by_step"][mode] = {}

            for epsilon in EPSILONS:
                summary["admitted_by_epsilon"][mode][str(epsilon)] = sorted({
                    row["skill_id"]
                    for row in mode_rows
                    if row["epsilon"] == epsilon and row["admitted"]
                })

            for step in range(len(EPSILONS)):
                step_rows = [
                    row for row in mode_rows if row["step"] == step
                ]
                first = step_rows[0]
                summary["selected_by_step"][mode][str(step)] = (
                    first["selected_skill"]
                )
                summary["executed_outcomes_by_step"][mode][str(step)] = {
                    "policy": first["executed_policy"],
                    "landing_success": first["executed_landing_success"],
                    "crashed": first["executed_crashed"],
                    "steps": first["executed_steps"],
                }

        JSON_PATH.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        print("\nCSV saved:", CSV_PATH)
        print("Summary saved:", JSON_PATH)
    finally:
        env.close()


if __name__ == "__main__":
    main()
