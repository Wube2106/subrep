from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from bridge.budget import BudgetCoefficients
from env.lunar_lander_rollout import (
    DEFAULT_GAMMA,
    evaluate_candidates,
)
from env.lunar_lander_wrapper import SubRepEnv
from data_collector.collect_candidate_sets import build_default_candidate_policies


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "experiments" / "epsilon_scale" / "results"
CHECKPOINT = ROOT / "models" / "pilot_ppo.pt"
SEEDS = tuple(range(41, 141))

EPSILON_MAX = float(BudgetCoefficients().epsilon_max)
EPSILONS = tuple(sorted(set((0.0, 0.05, 0.1, 0.2, 0.3, EPSILON_MAX))))


def epsilon_column(epsilon: float) -> str:
    """Create a stable CSV column name for an epsilon value."""
    return "admitted_eps_" + format(epsilon, "g").replace(".", "_")


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"PPO checkpoint not found: {CHECKPOINT}\n"
            "Check that models/pilot_ppo.pt exists in this worktree."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "e0_extended_seeds_candidates.csv"
    json_path = OUTPUT_DIR / "e0_extended_seeds_summary.json"

    rows: list[dict[str, Any]] = []
    candidate_names: set[str] = set()

    env = SubRepEnv(seed=SEEDS[0])
    try:
        policies = build_default_candidate_policies(
            env,
            pilot_checkpoint=str(CHECKPOINT),
            map_location="cpu",
        )
        candidate_names = {policy.skill_id for policy in policies}

        for index, seed in enumerate(SEEDS, start=1):
            print(f"[E0] Seed {seed} ({index}/{len(SEEDS)})", flush=True)

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

                # Full-simplex PDS score: delta_r + min_i(delta_n_i).
                # This is the raw score before adding epsilon.
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
        raise RuntimeError("E0 produced no candidate records; check the rollout setup.")

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    scores = [float(row["raw_score_full_simplex"]) for row in rows]
    band_count = sum(-EPSILON_MAX <= score < 0.0 for score in scores)
    flips = [
        row for row in rows
        if row[epsilon_column(0.0)] != row[epsilon_column(EPSILON_MAX)]
    ]
    seeds_with_flips = sorted({int(row["seed"]) for row in flips})

    summary: dict[str, Any] = {
        "experiment": "Extended-seed raw LunarLander candidate margins",
        "seeds": list(SEEDS),
        "gamma": DEFAULT_GAMMA,
        "max_steps": None,
        "candidate_policies": sorted(candidate_names),
        "candidate_record_count": len(rows),
        "epsilon_values": list(EPSILONS),
        "epsilon_max": EPSILON_MAX,
        "score_definition": "delta_r + min(delta_n)",
        "score_interpretation": (
            "Full-simplex raw PDS score before epsilon; "
            "not a claim that every runtime MDN support region uses this exact score."
        ),
        "score_statistics": {
            "min": min(scores),
            "max": max(scores),
            "mean": statistics.fmean(scores),
            "median": statistics.median(scores),
            "count_in_epsilon_flip_band": band_count,
            "fraction_in_epsilon_flip_band": band_count / len(scores),
            "flip_band_definition": f"[-{EPSILON_MAX}, 0)",
        },
        "admission_counts_by_epsilon": {
            format(epsilon, "g"): sum(
                bool(row[epsilon_column(epsilon)]) for row in rows
            )
            for epsilon in EPSILONS
        },
        "flips_between_epsilon_0_and_epsilon_max": len(flips),
        "fraction_flipping_between_epsilon_0_and_epsilon_max": (
            len(flips) / len(rows)
        ),
        "seeds_with_at_least_one_flip": seeds_with_flips,
        "number_of_seeds_with_flips": len(seeds_with_flips),
        "crash_records": sum(bool(row["crashed"]) for row in rows),
        "landing_success_records": sum(
            bool(row["landing_success"]) for row in rows
        ),
        "artifacts": {
            "raw_csv": str(csv_path.relative_to(ROOT)),
            "summary_json": str(json_path.relative_to(ROOT)),
        },
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print("\n=== E0 Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nRaw records: {csv_path}")
    print(f"Summary:     {json_path}")


if __name__ == "__main__":
    main()
