"""E4: real MetaMo epsilon changes with PPO and crash-prone candidates."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

test_path = ROOT / "tests" / "test_lunar_lander_integration.py"
spec = importlib.util.spec_from_file_location("lunar_lander_integration", test_path)
integration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integration)

from bridge.controller import MetaMoController
from bridge.epsilon_scaling import EpsilonScalingConfig
from env.lunar_lander_metamo import make_lunar_lander_governor
from env.lunar_lander_rollout import (
    NOOP_BASELINE,
    appraisal_scales,
    episode_outcome,
    evaluate_candidates,
    execute_candidate,
)
from env.lunar_lander_wrapper import SubRepEnv


OUTPUT_DIR = ROOT / "experiments" / "epsilon_scale"
CSV_PATH = OUTPUT_DIR / "e4_real_metamo_steps.csv"
JSON_PATH = OUTPUT_DIR / "e4_real_metamo_summary.json"

SEED = 1
NUM_STEPS = 8
CONTROLLER_SEED = 42


def main() -> None:
    env = SubRepEnv(seed=SEED)

    try:
        # This builder returns PPO policies and the fixed-action/random
        # candidates together, with no duplicate skill IDs.
        candidates = integration.ppo_candidates(env)

        baseline_stats, measured_records = evaluate_candidates(
            env, candidates, seed=SEED
        )

        payoff_scale, motive_scale = appraisal_scales(measured_records)

        governor = make_lunar_lander_governor(
            payoff_scale=payoff_scale,
            motive_scale=motive_scale,
        )
        controller = MetaMoController(
            governor,
            integration.make_pipeline(),
            seed=CONTROLLER_SEED,
            epsilon_scaling=EpsilonScalingConfig(
                enabled=True,
                    payoff_scale=82.7272727,
                motive_scale=235.1594902,
            ),
        )

        # Capture the exact records certified inside controller.step().
        certification_capture = {}
        original_certify = controller.certify

        def capture_certification(**kwargs):
            records = original_certify(**kwargs)
            certification_capture["records"] = records
            return records

        controller.certify = capture_certification

        policies = {candidate.skill_id: candidate for candidate in candidates}
        context = np.asarray(baseline_stats["info"]["start_obs"])

        rows = []
        step_summaries = []

        for step in range(NUM_STEPS):
            signal_before = governor.signal()
            modulators_before = governor.modulator_summary()

            # Capture actual certification records at the current epsilon.
            # The controller step below repeats certification with the same
            # signal and seed, then performs normal selection and execution.
            episode_info = {}

            def outcome_for(selected):
                policy = (
                    NOOP_BASELINE
                    if selected is None
                    else policies[selected.skill_id]
                )

                delta_r, delta_n, info = execute_candidate(
                    env, policy, seed=SEED
                )
                episode_info.update(info)

                return episode_outcome(
                    delta_r,
                    delta_n,
                    info,
                    admitted=selected is not None,
                )

            step_record = controller.step(
                context=context,
                candidate_skills=measured_records,
                baseline_stats=baseline_stats,
                outcome_for=outcome_for,
                seed=CONTROLLER_SEED,
            )
            certified_records = certification_capture["records"]

            epsilon_used = float(step_record.pds_epsilon)
            weights_used = np.asarray(step_record.weights).tolist()

            admitted_ids = []
            candidate_details = []

            for record in certified_records:
                admitted = bool(record.is_certified)

                if admitted:
                    admitted_ids.append(record.skill_id)

                detail = {
                    "skill_id": record.skill_id,
                    "is_certified": admitted,
                    "admission_margin": record.admission_margin,
                    "epsilon": record.epsilon,
                    "gate_type": record.gate_type,
                    "delta_r": record.delta_r,
                    "delta_n": list(record.delta_n),
                }
                candidate_details.append(detail)

                rows.append(
                    {
                        "step": step,
                        "epsilon_used": epsilon_used,
                        "weights": json.dumps(weights_used),
                        **detail,
                        "selected_skill_id": step_record.selected_skill_id,
                        "selected_score": step_record.selected_score,
                        "securing_before": modulators_before["securing"],
                        "approach_before": modulators_before["approach"],
                        "securing_after": governor.modulator_summary()["securing"],
                        "approach_after": governor.modulator_summary()["approach"],
                        "landing_success": episode_info.get("landing_success"),
                        "crashed": episode_info.get("crashed"),
                    }
                )

            modulators_after = governor.modulator_summary()
            epsilon_after = float(governor.signal().pds_epsilon)

            step_summaries.append(
                {
                    "step": step,
                    "epsilon_used": epsilon_used,
                    "epsilon_after": epsilon_after,
                    "weights": weights_used,
                    "modulators_before": modulators_before,
                    "modulators_after": modulators_after,
                    "candidate_certifications": candidate_details,
                    "admitted_candidates": sorted(admitted_ids),
                    "selected_skill_id": step_record.selected_skill_id,
                    "selected_score": step_record.selected_score,
                    "landing_success": episode_info.get("landing_success"),
                    "crashed": episode_info.get("crashed"),
                }
            )

            print(
                f"step={step} "
                f"epsilon={epsilon_used:.6f} "
                f"epsilon_next={epsilon_after:.6f} "
                f"admitted={sorted(admitted_ids)} "
                f"selected={step_record.selected_skill_id} "
                f"landing={episode_info.get('landing_success')} "
                f"crashed={episode_info.get('crashed')}"
            )

        with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "experiment": "E4 real MetaMo governor with PPO and fixed-action candidates",
            "seed": SEED,
            "steps": NUM_STEPS,
            "candidate_ids": list(policies),
            "candidate_context_fixed": True,
            "candidate_measurements_fixed": True,
            "environment_outcomes_are_real": True,
            "certification_results_are_actual_pipeline_records": True,
            "payoff_scale": float(payoff_scale),
            "motive_scale": float(motive_scale),
            "steps_detail": step_summaries,
            "csv": str(CSV_PATH),
        }

        JSON_PATH.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print(f"\nCSV saved: {CSV_PATH}")
        print(f"Summary saved: {JSON_PATH}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
