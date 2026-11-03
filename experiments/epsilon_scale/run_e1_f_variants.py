from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from bridge.budget import BudgetCoefficients
from experiments.epsilon_scale.validation import load_and_validate_candidates


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "epsilon_scale" / "results"
INPUT = RESULTS / "e0_candidate_variants.csv"

TRAIN_SEEDS = set(range(1, 21))
TEST_SEEDS = set(range(21, 41))
EPSILONS = (
    0.0,
    0.05,
    0.1,
    0.2,
    0.3,
    float(BudgetCoefficients().epsilon_max),
)


def truth(value):
    return str(value).strip().lower() == "true"


def main():
    _, train, test = load_and_validate_candidates(INPUT)

    output_rows = []

    for source in test:
        dr = float(source["delta_r"])
        dn = (
            float(source["delta_n_safety"]),
            float(source["delta_n_fuel"]),
        )

        score = dr + min(dn)

        row = {
            "seed": int(source["seed"]),
            "candidate_id": source["candidate_id"],
            "delta_r": dr,
            "delta_n_safety": dn[0],
            "delta_n_fuel": dn[1],
            "crashed": source["crashed"],
            "landing_success": source["landing_success"],
            "F_raw_score": score,
        }

        for epsilon in EPSILONS:
            column = (
                "F_raw_admitted_eps_"
                f"{format(epsilon, 'g').replace('.', '_')}"
            )
            row[column] = score >= -epsilon

        output_rows.append(row)

    scores = [r["F_raw_score"] for r in output_rows]
    crashes = [r for r in output_rows if truth(r["crashed"])]
    landings = [
        r for r in output_rows if truth(r["landing_success"])
    ]

    admissions = {}
    for epsilon in EPSILONS:
        admissions[format(epsilon, "g")] = sum(
            r["F_raw_score"] >= -epsilon
            for r in output_rows
        )

    flips_by_seed = {}
    monotonicity_violations = 0

    for row in output_rows:
        score = row["F_raw_score"]
        decisions = [score >= -epsilon for epsilon in EPSILONS]

        if decisions[0] != decisions[-1]:
            seed = str(row["seed"])
            flips_by_seed[seed] = flips_by_seed.get(seed, 0) + 1

        for lower, higher in zip(decisions, decisions[1:]):
            if lower and not higher:
                monotonicity_violations += 1

    # Additional research metrics.
    records_by_seed = {}
    for row in output_rows:
        seed = str(row["seed"])
        records_by_seed[seed] = records_by_seed.get(seed, 0) + 1

    per_seed_flip_rates = {
        seed: flips_by_seed.get(seed, 0) / count * 100
        for seed, count in sorted(records_by_seed.items())
    }
    flip_rates = list(per_seed_flip_rates.values())

    epsilon_0_1_column = (
        "F_raw_admitted_eps_" + format(0.1, "g").replace(".", "_")
    )
    epsilon_max_column = (
        "F_raw_admitted_eps_"
        + format(EPSILONS[-1], "g").replace(".", "_")
    )

    recovered_landings = [
        row for row in landings
        if not row[epsilon_0_1_column]
        and row[epsilon_max_column]
    ]

    summaries = {
        "score_min": min(scores),
        "score_median": statistics.median(scores),
        "score_max": max(scores),
        "admissions_by_epsilon": admissions,
        "flips_epsilon_0_to_max": sum(flips_by_seed.values()),
        "flip_counts_by_seed": dict(sorted(flips_by_seed.items())),
        "crash_records": len(crashes),
        "crash_admissions_at_epsilon_max": sum(
            r["F_raw_score"] >= -EPSILONS[-1]
            for r in crashes
        ),
        "landing_records": len(landings),
        "landing_admissions_at_epsilon_0_1": sum(
            r["F_raw_score"] >= -0.1
            for r in landings
        ),
        "monotonicity_violations": monotonicity_violations,
        "flip_percentage_overall": (
            sum(flips_by_seed.values()) / len(output_rows) * 100
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
            "successful_landings": len(landings),
            "recovery_percentage": (
                len(recovered_landings) / len(landings) * 100
                if landings else 0.0
            ),
        },
    }

    csv_path = RESULTS / "e1_f_variants_heldout.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(output_rows[0]),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "experiment": "E1 F raw variant on expanded 11-policy candidate set",
        "input": str(INPUT.relative_to(ROOT)),
        "calibration_seeds": sorted(TRAIN_SEEDS),
        "heldout_seeds": sorted(TEST_SEEDS),
        "calibration_records": len(train),
        "heldout_records": len(output_rows),
        "epsilon_values": list(EPSILONS),
        "method": summaries,
        "important_limit": (
            "Offline full-simplex score comparison only; "
            "not runtime MDN-constrained certification."
        ),
        "heldout_csv": str(csv_path.relative_to(ROOT)),
    }

    json_path = RESULTS / "e1_f_variants_summary.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"\nHeld-out CSV: {csv_path}")
    print(f"Summary:      {json_path}")


if __name__ == "__main__":
    main()
