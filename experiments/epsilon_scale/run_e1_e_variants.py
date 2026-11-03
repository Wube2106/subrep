from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from bridge.budget import BudgetCoefficients
from experiments.epsilon_scale.validation import load_and_validate_candidates

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "epsilon_scale" / "results"
INPUT = RESULTS / "e0_candidate_variants.csv"

TRAIN_SEEDS = set(range(1, 21))
TEST_SEEDS = set(range(21, 41))

EPSILONS = (
    0.0, 0.05, 0.1, 0.2, 0.3,
    float(BudgetCoefficients().epsilon_max),
)


def truth(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def average_ranks(values: list[float]) -> list[float]:
    """Assign ascending average ranks, giving tied values equal ranks."""
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1

        average_rank = ((i + 1) + j) / 2.0

        for position in range(i, j):
            original_index = ordered[position][0]
            ranks[original_index] = average_rank

        i = j

    return ranks


def epsilon_column(epsilon: float) -> str:
    return f"E_admitted_eps_{format(epsilon, 'g').replace('.', '_')}"


def main() -> None:
    all_rows, train, test = load_and_validate_candidates(INPUT)

    grouped = defaultdict(list)
    for source in test:
        grouped[int(source["seed"])].append(source)

    rows = []

    for seed, candidates in sorted(grouped.items()):
        scores = [float(r["delta_r"]) + min(
            float(r["delta_n_safety"]),
            float(r["delta_n_fuel"]),
        ) for r in candidates]

        ranks = average_ranks(scores)
        candidate_count = len(candidates)

        for source, raw_score, rank in zip(candidates, scores, ranks):
            if candidate_count == 1:
                rank_margin = 0.0
            else:
                rank_margin = (rank - candidate_count) / (
                    candidate_count - 1
                )

            row = {
                "seed": seed,
                "candidate_id": source["candidate_id"],
                "raw_score": raw_score,
                "rank": rank,
                "candidate_count": candidate_count,
                "rank_margin": rank_margin,
                "crashed": source["crashed"],
                "landing_success": source["landing_success"],
            }

            for epsilon in EPSILONS:
                row[epsilon_column(epsilon)] = (
                    rank_margin >= -epsilon
                )

            rows.append(row)

    crash_rows = [r for r in rows if truth(r["crashed"])]
    landing_rows = [r for r in rows if truth(r["landing_success"])]

    admissions = {
        format(e, "g"): sum(bool(r[epsilon_column(e)]) for r in rows)
        for e in EPSILONS
    }

    flips_by_seed = defaultdict(int)
    monotonicity_violations = 0

    for row in rows:
        decisions = [bool(row[epsilon_column(e)]) for e in EPSILONS]

        if decisions[0] != decisions[-1]:
            flips_by_seed[str(row["seed"])] += 1

        for lower, higher in zip(decisions, decisions[1:]):
            if lower and not higher:
                monotonicity_violations += 1

    neutral_column = epsilon_column(0.1)
    max_column = epsilon_column(EPSILONS[-1])

    landing_recovered = [
        r for r in landing_rows
        if not r[neutral_column] and r[max_column]
    ]

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

    summary = {
        "experiment": "E1-E: per-seed rank/quantile-based margins",
        "formula": (
            "rank_margin = (average_ascending_rank - candidate_count)"
            " / (candidate_count - 1); "
            "admit if rank_margin >= -epsilon"
        ),
        "ranking_scope": "within each seed, across candidate policies",
        "tie_handling": "average rank",
        "training_seeds": sorted(TRAIN_SEEDS),
        "heldout_seeds": sorted(TEST_SEEDS),
        "heldout_record_count": len(rows),
        "candidates_per_seed": sorted({
            len(candidates) for candidates in grouped.values()
        }),
        "epsilon_values": list(EPSILONS),
        "admissions_by_epsilon": admissions,
        "flips_epsilon_0_to_max": sum(flips_by_seed.values()),
        "flip_counts_by_seed": dict(sorted(flips_by_seed.items())),
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
        "crash_records": len(crash_rows),
        "crash_admissions_at_epsilon_max": sum(
            bool(r[max_column]) for r in crash_rows
        ),
        "landing_records": len(landing_rows),
        "landing_admissions_at_epsilon_0_1": sum(
            bool(r[neutral_column]) for r in landing_rows
        ),
        "landing_recoveries_from_0_1_to_max": len(landing_recovered),
        "landing_recovery_percentage": (
            len(landing_recovered) / len(landing_rows) * 100
            if landing_rows else 0.0
        ),
        "recovered_landing_records": [
            {
                "seed": r["seed"],
                "candidate_id": r["candidate_id"],
                "raw_score": r["raw_score"],
                "rank_margin": r["rank_margin"],
            }
            for r in landing_recovered
        ],
        "monotonicity_violations": monotonicity_violations,
        "important_limit": (
            "Relative ranking depends on the candidate set. "
            "Adding or removing candidates can change admissions. "
            "This offline full-simplex comparison is not runtime "
            "MDN-constrained certification."
        ),
    }

    csv_path = RESULTS / "e1_e_variants_heldout_rerun.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = RESULTS / "e1_e_variants_summary_rerun.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"\nCandidate results: {csv_path}")
    print(f"Summary:           {json_path}")


if __name__ == "__main__":
    main()
