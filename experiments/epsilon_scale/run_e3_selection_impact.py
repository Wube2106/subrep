from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "epsilon_scale" / "results"
E0_INPUT = RESULTS / "e0_candidate_variants.csv"

EPSILONS = ("0", "0_05", "0_1", "0_2", "0_3", "0_5")

# Each approach points to its E1 admission decisions.
APPROACH_FILES = {
    "A": ("e1_a_variants_heldout.csv", "A_normalized"),
    "B": ("e1_b_variants_heldout_rerun.csv", "B"),
    "C_payoff": ("e1_c_variants_heldout.csv", "C_payoff_scale"),
    "C_motive": ("e1_c_variants_heldout.csv", "C_motive_scale"),
    "D": ("e1_d_variants_heldout_rerun.csv", "D"),
    "E": ("e1_e_variants_heldout_rerun.csv", "E"),
    "F_raw": ("e1_f_variants_heldout.csv", "F_raw"),
}

# Fixed evaluation weights in [Safety, Fuel] order.
WEIGHTS = (0.5, 0.5)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(value) -> bool:
    return str(value).strip().lower() == "true"


def key_for(row: dict) -> tuple[int, str]:
    return int(row["seed"]), row["candidate_id"]


def selection_score(row: dict) -> float:
    return (
        float(row["delta_r"])
        + WEIGHTS[0] * float(row["delta_n_safety"])
        + WEIGHTS[1] * float(row["delta_n_fuel"])
    )


def load_admissions(
    filename: str,
    prefix: str,
    expected_keys: set[tuple[int, str]],
) -> dict[tuple[int, str], dict[str, bool]]:
    path = RESULTS / filename
    rows = read_csv(path)
    admissions = {}

    for row in rows:
        key = key_for(row)
        if key in admissions:
            raise ValueError(f"Duplicate candidate in {filename}: {key}")

        admissions[key] = {
            epsilon: as_bool(
                row[f"{prefix}_admitted_eps_{epsilon}"]
            )
            for epsilon in EPSILONS
        }

    actual_keys = set(admissions)
    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        raise ValueError(
            f"Candidate mismatch in {filename}: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    return admissions


def choose_candidate(
    candidates: list[dict],
    admissions: dict[tuple[int, str], dict[str, bool]],
    seed: int,
    epsilon: str,
) -> dict | None:
    eligible = [
        row
        for row in candidates
        if admissions[key_for(row)][epsilon]
    ]

    if not eligible:
        return None

    # Highest score wins; lexical candidate_id breaks exact ties.
    return sorted(
        eligible,
        key=lambda row: (-selection_score(row), row["candidate_id"]),
    )[0]


def describe_selection(row: dict | None) -> dict:
    if row is None:
        return {
            "candidate_id": "no_candidate",
            "selection_score": "",
            "crashed": "",
            "landing_success": "",
        }

    return {
        "candidate_id": row["candidate_id"],
        "selection_score": selection_score(row),
        "crashed": as_bool(row["crashed"]),
        "landing_success": as_bool(row["landing_success"]),
    }


def compare_selections(
    baseline: dict | None,
    variant: dict | None,
) -> str:
    if baseline is None and variant is None:
        return "both_no_candidate"
    if baseline is None:
        return "baseline_no_candidate"
    if variant is None:
        return "variant_no_candidate"
    if baseline["candidate_id"] == variant["candidate_id"]:
        return "same_candidate"
    return "different_candidate"


def main():
    e0_rows = read_csv(E0_INPUT)
    heldout_rows = [
        row for row in e0_rows
        if 21 <= int(row["seed"]) <= 40
    ]

    if len(heldout_rows) != 220:
        raise ValueError(
            f"Expected 220 held-out E0 records, got {len(heldout_rows)}"
        )

    expected_keys = {key_for(row) for row in heldout_rows}
    if len(expected_keys) != len(heldout_rows):
        raise ValueError("E0 contains duplicate (seed, candidate_id) keys")

    candidates_by_seed = defaultdict(list)
    for row in heldout_rows:
        candidates_by_seed[int(row["seed"])].append(row)

    if set(candidates_by_seed) != set(range(21, 41)):
        raise ValueError("Expected held-out seeds 21 through 40")

    for seed, candidates in candidates_by_seed.items():
        if len(candidates) != 11:
            raise ValueError(
                f"Seed {seed} has {len(candidates)} candidates, expected 11"
            )

    admission_data = {}
    for approach, (filename, prefix) in APPROACH_FILES.items():
        admission_data[approach] = load_admissions(
            filename,
            prefix,
            expected_keys,
        )

    output_rows = []
    summary_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for approach, admissions in admission_data.items():
        for epsilon in EPSILONS:
            for seed in sorted(candidates_by_seed):
                candidates = candidates_by_seed[seed]

                baseline = choose_candidate(
                    candidates,
                    admission_data["F_raw"],
                    seed,
                    epsilon,
                )
                selected = choose_candidate(
                    candidates,
                    admissions,
                    seed,
                    epsilon,
                )

                baseline_info = describe_selection(baseline)
                selected_info = describe_selection(selected)
                comparison = compare_selections(baseline, selected)

                row = {
                    "approach": approach,
                    "epsilon": epsilon.replace("_", "."),
                    "seed": seed,
                    "baseline_candidate_id": baseline_info["candidate_id"],
                    "baseline_selection_score": baseline_info["selection_score"],
                    "baseline_crashed": baseline_info["crashed"],
                    "baseline_landing_success": baseline_info["landing_success"],
                    "selected_candidate_id": selected_info["candidate_id"],
                    "selected_selection_score": selected_info["selection_score"],
                    "selected_crashed": selected_info["crashed"],
                    "selected_landing_success": selected_info["landing_success"],
                    "comparison": comparison,
                }
                output_rows.append(row)
                summary_counts[approach][epsilon][comparison] += 1

    output_path = RESULTS / "e3_selection_impact.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "experiment": "E3 selection impact of epsilon-scale admission approaches",
        "heldout_seeds": list(range(21, 41)),
        "heldout_candidate_records": len(heldout_rows),
        "candidates_per_seed": 11,
        "selection_formula": (
            "delta_r + 0.5 * delta_n_safety + 0.5 * delta_n_fuel"
        ),
        "selection_weights": {
            "Safety": WEIGHTS[0],
            "Fuel": WEIGHTS[1],
        },
        "selection_rule": (
            "Choose the admitted candidate with the highest selection score; "
            "break exact ties by lexical candidate_id."
        ),
        "baseline": (
            "F raw-units admission decisions at the same epsilon."
        ),
        "comparison_categories": {
            "same_candidate": "Both approaches select the same candidate.",
            "different_candidate": (
                "Both select a candidate, but the candidate IDs differ."
            ),
            "baseline_no_candidate": (
                "The raw-units baseline has no eligible candidate."
            ),
            "variant_no_candidate": (
                "The tested approach has no eligible candidate."
            ),
            "both_no_candidate": (
                "Neither approach has an eligible candidate."
            ),
        },
        "results": {},
        "output_csv": str(output_path.relative_to(ROOT)),
        "important_limit": (
            "Offline selection-impact comparison using fixed weights. "
            "It is not a measurement of dynamic MetaMo runtime selection."
        ),
    }

    for approach in APPROACH_FILES:
        summary["results"][approach] = {}

        for epsilon in EPSILONS:
            counts = summary_counts[approach][epsilon]
            same = counts["same_candidate"]
            different = counts["different_candidate"]
            baseline_none = counts["baseline_no_candidate"]
            variant_none = counts["variant_no_candidate"]
            both_none = counts["both_no_candidate"]
            both_selected = same + different

            summary["results"][approach][epsilon.replace("_", ".")] = {
                "same_candidate": same,
                "different_candidate": different,
                "baseline_no_candidate": baseline_none,
                "variant_no_candidate": variant_none,
                "both_no_candidate": both_none,
                "candidate_change_rate_when_both_selected_pct": (
                    different / both_selected * 100
                    if both_selected else None
                ),
                "seeds_with_both_selected": both_selected,
                "total_seeds": 20,
            }

    json_path = RESULTS / "e3_selection_impact_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print("E3 selection-impact experiment completed.")
    print(f"Held-out seeds: {len(candidates_by_seed)}")
    print(f"Candidate records: {len(heldout_rows)}")
    print(f"Selection weights: {WEIGHTS} [Safety, Fuel]")
    print("\nDifferent selected candidates vs raw-units baseline:")
    for approach in APPROACH_FILES:
        print(f"\n{approach}:")
        for epsilon in EPSILONS:
            result = summary["results"][approach][epsilon.replace("_", ".")]
            print(
                f"  epsilon={epsilon.replace('_', '.')}: "
                f"different={result['different_candidate']}/20, "
                f"same={result['same_candidate']}/20, "
                f"variant_no_candidate={result['variant_no_candidate']}, "
                f"baseline_no_candidate={result['baseline_no_candidate']}, "
                f"both_no_candidate={result['both_no_candidate']}"
            )

    print(f"\nCSV:     {output_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
