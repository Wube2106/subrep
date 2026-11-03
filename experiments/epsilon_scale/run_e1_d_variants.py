from __future__ import annotations

import csv
import json
import statistics
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from bridge.budget import BudgetCoefficients
from experiments.epsilon_scale.validation import load_and_validate_candidates

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "epsilon_scale" / "results"
INPUT = RESULTS / "e0_candidate_variants.csv"
BASELINE_INPUT = RESULTS / "baseline_returns.csv"

TRAIN_SEEDS = set(range(1, 21))
TEST_SEEDS = set(range(21, 41))
EPSILONS = (
    0.0, 0.05, 0.1, 0.2, 0.3,
    float(BudgetCoefficients().epsilon_max),
)


def truth(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def main() -> None:
    all_rows, train, test = load_and_validate_candidates(INPUT)

    with BASELINE_INPUT.open(newline="", encoding="utf-8") as f:
        baseline_rows = list(csv.DictReader(f))

    if not baseline_rows:
        raise RuntimeError("Baseline dataset is empty.")

    required_columns = {"seed", "baseline_return"}
    missing_columns = required_columns - set(baseline_rows[0])
    if missing_columns:
        raise RuntimeError(
            f"Baseline dataset is missing columns: {sorted(missing_columns)}"
        )

    baseline_by_seed = {}

    for row in baseline_rows:
        try:
            seed = int(row["seed"])
            baseline_return = float(row["baseline_return"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid baseline record: {row!r}"
            ) from exc

        if seed in baseline_by_seed:
            raise RuntimeError(f"Duplicate baseline seed: {seed}")

        if not math.isfinite(baseline_return):
            raise RuntimeError(
                f"Non-finite baseline return for seed {seed}."
            )

        baseline_by_seed[seed] = abs(baseline_return)

    expected_seeds = set(range(1, 41))
    if set(baseline_by_seed) != expected_seeds:
        raise RuntimeError(
            "Baseline seed mismatch. "
            f"Missing: {sorted(expected_seeds - set(baseline_by_seed))}; "
            f"unexpected: {sorted(set(baseline_by_seed) - expected_seeds)}."
        )

    rows = []
    for source in test:
        seed = int(source["seed"])
        delta_r = float(source["delta_r"])
        delta_n = min(
            float(source["delta_n_safety"]),
            float(source["delta_n_fuel"]),
        )
        raw_score = delta_r + delta_n
        baseline_scale = baseline_by_seed[seed]

        row = {
            "seed": seed,
            "candidate_id": source["candidate_id"],
            "delta_r": delta_r,
            "delta_n_safety": float(source["delta_n_safety"]),
            "delta_n_fuel": float(source["delta_n_fuel"]),
            "baseline_return_abs": baseline_scale,
            "raw_score": raw_score,
            "crashed": source["crashed"],
            "landing_success": source["landing_success"],
        }

        for epsilon in EPSILONS:
            column = (
                f"D_admitted_eps_{format(epsilon, 'g').replace('.', '_')}"
            )
            row[column] = raw_score >= -baseline_scale * epsilon

        rows.append(row)

    crash_rows = [r for r in rows if truth(r["crashed"])]
    landing_rows = [r for r in rows if truth(r["landing_success"])]

    admissions = {}
    for epsilon in EPSILONS:
        column = f"D_admitted_eps_{format(epsilon, 'g').replace('.', '_')}"
        admissions[format(epsilon, "g")] = sum(
            bool(row[column]) for row in rows
        )

    flips_by_seed = defaultdict(int)
    monotonicity_violations = 0

    for row in rows:
        decisions = [
            bool(row[
                f"D_admitted_eps_{format(e, 'g').replace('.', '_')}"
            ])
            for e in EPSILONS
        ]

        if decisions[0] != decisions[-1]:
            flips_by_seed[str(row["seed"])] += 1

        for lower, higher in zip(decisions, decisions[1:]):
            if lower and not higher:
                monotonicity_violations += 1

    epsilon_neutral = 0.1
    epsilon_max = EPSILONS[-1]

    # Additional research metrics.
    records_by_seed = {}
    for row in rows:
        seed = str(row["seed"])
        records_by_seed[seed] = records_by_seed.get(seed, 0) + 1

    per_seed_flip_rates = {
        seed: flips_by_seed.get(seed, 0) / count * 100
        for seed, count in sorted(records_by_seed.items())
    }
    flip_rates = list(per_seed_flip_rates.values())

    neutral_column = "D_admitted_eps_0_1"
    max_column = (
        "D_admitted_eps_"
        + format(epsilon_max, "g").replace(".", "_")
    )

    recovered_landings = [
        row for row in landing_rows
        if not row[neutral_column] and row[max_column]
    ]

    summary = {
        "experiment": "E1-D: relative epsilon using baseline return size",
        "formula": "raw_score >= -epsilon * abs(baseline_return)",
        "baseline_policy": "noop",
        "baseline_return_min": min(baseline_by_seed.values()),
        "baseline_return_max": max(baseline_by_seed.values()),
        "baseline_return_mean": sum(baseline_by_seed.values())
        / len(baseline_by_seed),
        "training_seeds": sorted(TRAIN_SEEDS),
        "heldout_seeds": sorted(TEST_SEEDS),
        "heldout_record_count": len(rows),
        "epsilon_values": list(EPSILONS),
        "admissions_by_epsilon": admissions,
        "flips_epsilon_0_to_max": sum(flips_by_seed.values()),
        "flip_counts_by_seed": dict(sorted(flips_by_seed.items())),
        "crash_records": len(crash_rows),
        "crash_admissions_at_epsilon_max": sum(
            bool(r[
                f"D_admitted_eps_{format(epsilon_max, 'g').replace('.', '_')}"
            ])
            for r in crash_rows
        ),
        "landing_records": len(landing_rows),
        "landing_admissions_at_epsilon_0_1": sum(
            bool(r["D_admitted_eps_0_1"])
            for r in landing_rows
        ),
        "monotonicity_violations": monotonicity_violations,
        "flip_percentage_overall": (
            sum(flips_by_seed.values()) / len(rows) * 100
        ),
        "per_seed_flip_rates_percentage": per_seed_flip_rates,
        "seed_stability": {
            "minimum_flip_rate": min(flip_rates),
            "median_flip_rate": statistics.median(flip_rates),
            "mean_flip_rate": statistics.fmean(flip_rates),
            "maximum_flip_rate": max(flip_rates),
        },
        "landing_recovery_0_1_to_max": {
            "recovered_landings": len(recovered_landings),
            "successful_landings": len(landing_rows),
            "recovery_percentage": (
                len(recovered_landings) / len(landing_rows) * 100
                if landing_rows else 0.0
            ),
        },
        "important_limit": (
            "Offline full-simplex score comparison only; "
            "not runtime MDN-constrained certification. "
            "Baseline return magnitude is measured per seed."
        ),
    }

    csv_path = RESULTS / "e1_d_variants_heldout_rerun.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = RESULTS / "e1_d_variants_summary_rerun.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"\nCandidate results: {csv_path}")
    print(f"Summary:           {json_path}")


if __name__ == "__main__":
    main()
