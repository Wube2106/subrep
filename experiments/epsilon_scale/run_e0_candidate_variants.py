from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from bridge.budget import BudgetCoefficients
from data_collector.collect_candidate_sets import (
    CandidatePolicy,
    PpoNoisyActions,
    PpoThenSideTradeoff,
    build_default_candidate_policies,
)
from env.lunar_lander_rollout import DEFAULT_GAMMA, evaluate_candidates
from env.lunar_lander_wrapper import SubRepEnv
from pilot.rl_pilot import RLPilot


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "experiments" / "epsilon_scale" / "results"
CHECKPOINT = ROOT / "models" / "pilot_ppo.pt"
SEEDS = tuple(range(1, 41))

EPSILON_MAX = float(BudgetCoefficients().epsilon_max)
EPSILONS = tuple(sorted(set((0.0, 0.05, 0.1, 0.2, 0.3, EPSILON_MAX))))


def epsilon_column(epsilon: float) -> str:
    return "admitted_eps_" + format(epsilon, "g").replace(".", "_")


def build_variant_policies(env: SubRepEnv) -> tuple[CandidatePolicy, ...]:
    """Build fresh stateful policy instances for the new experiment."""
    pilot = RLPilot.load(str(CHECKPOINT), map_location="cpu")
    action_space = env.env.action_space

    return (
        CandidatePolicy(
            "ppo_noise_01",
            PpoNoisyActions(pilot, noise_std=0.1, action_space=action_space),
        ),
        CandidatePolicy(
            "ppo_noise_025",
            PpoNoisyActions(pilot, noise_std=0.25, action_space=action_space),
        ),
        CandidatePolicy(
            "ppo_switch_50",
            PpoThenSideTradeoff(pilot, switch_step=50, side_action=1),
        ),
        CandidatePolicy(
            "ppo_switch_100",
            PpoThenSideTradeoff(pilot, switch_step=100, side_action=1),
        ),
        CandidatePolicy(
            "ppo_switch_200",
            PpoThenSideTradeoff(pilot, switch_step=200, side_action=1),
        ),
    )


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {CHECKPOINT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "e0_candidate_variants.csv"
    json_path = OUTPUT_DIR / "e0_candidate_variants_summary.json"

    rows: list[dict[str, Any]] = []
    env = SubRepEnv(seed=SEEDS[0])

    try:
        defaults = build_default_candidate_policies(
            env,
            pilot_checkpoint=str(CHECKPOINT),
            map_location="cpu",
        )
        variants = build_variant_policies(env)
        policies = defaults + variants

        for index, seed in enumerate(SEEDS, start=1):
            print(f"[Candidate variants] Seed {seed} ({index}/{len(SEEDS)})", flush=True)

            baseline_stats, records = evaluate_candidates(
                env,
                policies,
                seed=seed,
                gamma=DEFAULT_GAMMA,
                max_steps=None,
            )

            for record in records:
                delta_r = float(record.delta_r)
                delta_n = tuple(float(value) for value in record.delta_n)
                worst_index = min(range(len(delta_n)), key=delta_n.__getitem__)
                raw_score = delta_r + delta_n[worst_index]
                metadata = record.metadata

                row: dict[str, Any] = {
                    "seed": seed,
                    "candidate_id": record.skill_id,
                    "baseline_id": record.baseline_id or baseline_stats.get(
                        "baseline_id", "noop"
                    ),
                    "delta_r": delta_r,
                    "delta_n_safety": delta_n[0],
                    "delta_n_fuel": delta_n[1],
                    "raw_score_full_simplex": raw_score,
                    "worst_case_objective": (
                        "Safety" if worst_index == 0
                        else "Fuel" if worst_index == 1
                        else f"objective_{worst_index}"
                    ),
                    "landing_success": metadata.get("landing_success"),
                    "crashed": metadata.get("crashed"),
                    "truncated": metadata.get("truncated"),
                    "steps": metadata.get("steps"),
                    "gamma": DEFAULT_GAMMA,
                }

                for epsilon in EPSILONS:
                    row[epsilon_column(epsilon)] = raw_score >= -epsilon

                rows.append(row)

    finally:
        env.close()

    if not rows:
        raise RuntimeError("No candidate results were collected.")

    fieldnames = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    candidate_summaries: dict[str, Any] = {}
    for candidate_id in sorted({str(row["candidate_id"]) for row in rows}):
        subset = [row for row in rows if row["candidate_id"] == candidate_id]
        scores = [float(row["raw_score_full_simplex"]) for row in subset]
        summary: dict[str, Any] = {
            "records": len(subset),
            "landing_successes": sum(bool(row["landing_success"]) for row in subset),
            "crashes": sum(bool(row["crashed"]) for row in subset),
            "score_min": min(scores),
            "score_median": statistics.median(scores),
            "score_max": max(scores),
            "scores_in_minus_0_5_to_0": sum(-0.5 <= score < 0.0 for score in scores),
            "epsilon_flips": sum(
                row[epsilon_column(0.0)] != row[epsilon_column(EPSILON_MAX)]
                for row in subset
            ),
            "admissions_by_epsilon": {
                format(epsilon, "g"): sum(bool(row[epsilon_column(epsilon)]) for row in subset)
                for epsilon in EPSILONS
            },
        }
        candidate_summaries[candidate_id] = summary

    output = {
        "experiment": "LunarLander candidate variants; raw full-simplex scores",
        "seeds": list(SEEDS),
        "gamma": DEFAULT_GAMMA,
        "max_steps": None,
        "epsilons": list(EPSILONS),
        "records": len(rows),
        "candidates": candidate_summaries,
    }
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved summary: {json_path}")
    print(f"Total records: {len(rows)}")
    for candidate_id, summary in candidate_summaries.items():
        print(
            f"{candidate_id:22s} "
            f"landings={summary['landing_successes']:2d} "
            f"crashes={summary['crashes']:2d} "
            f"flips={summary['epsilon_flips']:2d} "
            f"near_zero={summary['scores_in_minus_0_5_to_0']:2d}"
        )


if __name__ == "__main__":
    main()
