from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from bridge.budget import BudgetCoefficients
from experiments.epsilon_scale.validation import load_and_validate_candidates


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "epsilon_scale" / "results"
INPUT = RESULTS / "e0_candidate_variants.csv"
CSV_OUTPUT = RESULTS / "e2_scale_stability.csv"
JSON_OUTPUT = RESULTS / "e2_scale_stability_summary.json"

EPSILONS = (
    0.0, 0.05, 0.1, 0.2, 0.3,
    float(BudgetCoefficients().epsilon_max),
)
SCALE_FLOOR = 10.0
MOTIVE_COLUMNS = ("delta_n_safety", "delta_n_fuel")


def truth(value):
    return str(value).strip().lower() == "true"


def mean_abs(rows, column):
    return max(
        SCALE_FLOOR,
        statistics.fmean(abs(float(row[column])) for row in rows),
    )


def calculate_scales(rows):
    payoff = mean_abs(rows, "delta_r")

    motive = max(
        SCALE_FLOOR,
        statistics.fmean(
            statistics.fmean(
                abs(float(row[column])) for column in MOTIVE_COLUMNS
            )
            for row in rows
        ),
    )

    safety = mean_abs(rows, "delta_n_safety")
    fuel = mean_abs(rows, "delta_n_fuel")

    return {
        "payoff": payoff,
        "motive": motive,
        "safety": safety,
        "fuel": fuel,
    }


def admission_columns(name, mode):
    return [
        f"{name}_{mode}_admitted_eps_{format(e, 'g').replace('.', '_')}"
        for e in EPSILONS
    ]


def check_monotonicity(rows, columns):
    violations = 0

    for row in rows:
        decisions = [bool(row[column]) for column in columns]
        for lower, higher in zip(decisions, decisions[1:]):
            if lower and not higher:
                violations += 1

    return violations


def scale_statistics(values):
    values = [float(value) for value in values]
    mean = statistics.fmean(values)
    stddev = statistics.pstdev(values)

    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": mean,
        "max": max(values),
        "population_stddev": stddev,
        "coefficient_of_variation": stddev / mean if mean else None,
    }


def main():
    _, train, test = load_and_validate_candidates(INPUT)

    # Fixed scales use calibration seeds 1-20 only.
    fixed_scales = calculate_scales(train)

    # Per-seed scales use only that seed's 11 candidates.
    by_seed = defaultdict(list)
    for source in test:
        by_seed[int(source["seed"])].append(source)

    output_rows = []
    per_seed_scales = {}

    for seed in sorted(by_seed):
        candidates = by_seed[seed]
        current_scales = calculate_scales(candidates)
        per_seed_scales[str(seed)] = current_scales

        for source in candidates:
            dr = float(source["delta_r"])
            safety = float(source["delta_n_safety"])
            fuel = float(source["delta_n_fuel"])
            raw_margin = dr + min(safety, fuel)

            row = {
                "seed": seed,
                "candidate_id": source["candidate_id"],
                "delta_r": dr,
                "delta_n_safety": safety,
                "delta_n_fuel": fuel,
                "crashed": truth(source["crashed"]),
                "landing_success": truth(source["landing_success"]),
            }

            for mode, scales in (
                ("fixed", fixed_scales),
                ("per_seed", current_scales),
            ):
                # Approach A: normalized payoff + worst normalized motive.
                a_score = (
                    dr / scales["payoff"]
                    + min(
                        safety / scales["motive"],
                        fuel / scales["motive"],
                    )
                )
                row[f"A_{mode}_score"] = a_score

                for epsilon in EPSILONS:
                    key = (
                        f"A_{mode}_admitted_eps_"
                        f"{format(epsilon, 'g').replace('.', '_')}"
                    )
                    row[key] = a_score >= -epsilon

                # Approach B: independently normalized objectives.
                b_score = (
                    dr / scales["payoff"]
                    + min(
                        safety / scales["safety"],
                        fuel / scales["fuel"],
                    )
                )
                row[f"B_{mode}_score"] = b_score

                for epsilon in EPSILONS:
                    key = (
                        f"B_{mode}_admitted_eps_"
                        f"{format(epsilon, 'g').replace('.', '_')}"
                    )
                    row[key] = b_score >= -epsilon

                # Approach C: raw margin with scaled epsilon threshold.
                for variant, scale_name in (
                    ("C_payoff", "payoff"),
                    ("C_motive", "motive"),
                ):
                    row[f"{variant}_{mode}_score"] = raw_margin
                    factor = scales[scale_name]

                    for epsilon in EPSILONS:
                        key = (
                            f"{variant}_{mode}_admitted_eps_"
                            f"{format(epsilon, 'g').replace('.', '_')}"
                        )
                        row[key] = raw_margin >= -factor * epsilon

            for scale_name in ("payoff", "motive", "safety", "fuel"):
                row[f"fixed_{scale_name}_scale"] = fixed_scales[scale_name]
                row[f"per_seed_{scale_name}_scale"] = current_scales[scale_name]

            output_rows.append(row)

    # Save candidate-level results.
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    # Summarize scale variation across evaluation seeds.
    scale_summary = {}

    for scale_name in ("payoff", "motive", "safety", "fuel"):
        values = [
            per_seed_scales[str(seed)][scale_name]
            for seed in sorted(by_seed)
        ]

        scale_summary[scale_name] = {
            "fixed_calibration_value": fixed_scales[scale_name],
            "per_seed_evaluation_values": scale_statistics(values),
        }

    # Compare admissions at every epsilon.
    methods = ("A", "B", "C_payoff", "C_motive")
    admission_summary = {}
    monotonicity_summary = {}

    for method in methods:
        admission_summary[method] = {}
        monotonicity_summary[method] = {}

        for mode in ("fixed", "per_seed"):
            columns = admission_columns(method, mode)
            monotonicity_summary[method][mode] = check_monotonicity(
                output_rows, columns
            )

        for epsilon in EPSILONS:
            suffix = f"eps_{format(epsilon, 'g').replace('.', '_')}"
            fixed_key = f"{method}_fixed_admitted_{suffix}"
            per_seed_key = f"{method}_per_seed_admitted_{suffix}"

            fixed_count = sum(bool(row[fixed_key]) for row in output_rows)
            per_seed_count = sum(bool(row[per_seed_key]) for row in output_rows)

            flips_by_seed = defaultdict(int)
            totals_by_seed = defaultdict(int)

            for row in output_rows:
                seed = int(row["seed"])
                totals_by_seed[seed] += 1

                if bool(row[fixed_key]) != bool(row[per_seed_key]):
                    flips_by_seed[seed] += 1

            rates = [
                flips_by_seed[seed] / totals_by_seed[seed]
                for seed in sorted(totals_by_seed)
            ]

            changed = sum(flips_by_seed.values())

            admission_summary[method][format(epsilon, "g")] = {
                "fixed_admissions": fixed_count,
                "per_seed_admissions": per_seed_count,
                "admission_count_difference": per_seed_count - fixed_count,
                "fixed_vs_per_seed_changed_pairs": changed,
                "changed_pair_percentage": 100.0 * changed / len(output_rows),
                "per_seed_flip_rate_min": min(rates),
                "per_seed_flip_rate_median": statistics.median(rates),
                "per_seed_flip_rate_max": max(rates),
            }

    summary = {
        "experiment": "E2 fixed versus per-seed scale stability",
        "input": str(INPUT.relative_to(ROOT)),
        "calibration_seeds": list(range(1, 21)),
        "evaluation_seeds": sorted(by_seed),
        "calibration_records": len(train),
        "evaluation_records": len(test),
        "candidates_per_evaluation_seed": len(next(iter(by_seed.values()))),
        "epsilon_values": list(EPSILONS),
        "scale_floor": SCALE_FLOOR,
        "scale_method": (
            "Mean absolute value. A motive scale is the mean of each "
            "candidate's mean absolute Safety/Fuel margin."
        ),
        "fixed_scales": fixed_scales,
        "scale_variation": scale_summary,
        "per_seed_scales": per_seed_scales,
        "admission_comparison": admission_summary,
        "monotonicity_violations": monotonicity_summary,
        "interpretation_note": (
            "Per-seed scales are calculated independently from the 11 "
            "candidates for each held-out seed. This is an offline "
            "candidate-set proxy, not a runtime controller integration."
        ),
    }

    with JSON_OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("E2 scale stability experiment complete.")
    print(f"CSV: {CSV_OUTPUT}")
    print(f"JSON: {JSON_OUTPUT}")

    print("\nFixed calibration scales:")
    for name, value in fixed_scales.items():
        print(f"  {name}: {value:.6f}")

    print("\nPer-seed scale ranges:")
    for name, values in scale_summary.items():
        stats = values["per_seed_evaluation_values"]
        print(
            f"  {name}: min={stats['min']:.6f}, "
            f"median={stats['median']:.6f}, max={stats['max']:.6f}"
        )

    print("\nFixed vs per-seed admission changes:")
    for method, values in admission_summary.items():
        print(f"  {method}:")
        for epsilon, metrics in values.items():
            print(
                f"    epsilon={epsilon}: "
                f"changed={metrics['fixed_vs_per_seed_changed_pairs']}/220 "
                f"({metrics['changed_pair_percentage']:.2f}%), "
                f"admissions={metrics['fixed_admissions']} -> "
                f"{metrics['per_seed_admissions']}"
            )

    print("\nMonotonicity violations:")
    print(json.dumps(monotonicity_summary, indent=2))


if __name__ == "__main__":
    main()
