from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import math


EXPECTED_SEEDS = set(range(1, 41))
CALIBRATION_SEEDS = set(range(1, 21))
HELDOUT_SEEDS = set(range(21, 41))
EXPECTED_CANDIDATES_PER_SEED = 11
EXPECTED_RECORDS = 440


def load_and_validate_candidates(path: Path):
    """Load the frozen E0 dataset and validate its seed/candidate structure."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != EXPECTED_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORDS} records; found {len(rows)}."
        )

    required_columns = {
        "seed",
        "candidate_id",
        "delta_r",
        "delta_n_safety",
        "delta_n_fuel",
        "crashed",
        "landing_success",
    }

    if not rows:
        raise RuntimeError("Candidate dataset is empty.")

    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise RuntimeError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )

    grouped = defaultdict(list)
    seen_pairs = set()

    for row in rows:
        try:
            seed = int(row["seed"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid seed value: {row['seed']!r}"
            ) from exc

        candidate_id = row["candidate_id"].strip()

        if seed not in EXPECTED_SEEDS:
            raise RuntimeError(f"Unexpected seed: {seed}")

        if not candidate_id:
            raise RuntimeError(f"Empty candidate_id for seed {seed}")

        pair = (seed, candidate_id)
        if pair in seen_pairs:
            raise RuntimeError(f"Duplicate seed/candidate pair: {pair}")

        seen_pairs.add(pair)
        grouped[seed].append(candidate_id)

        # Fail early if numerical fields are missing or invalid.
        for column in (
            "delta_r",
            "delta_n_safety",
            "delta_n_fuel",
        ):
            try:
                value = float(row[column])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid {column} for {pair}: {row[column]!r}"
                ) from exc

            if not math.isfinite(value):
                raise RuntimeError(
                    f"Non-finite {column} for {pair}: {row[column]!r}"
                )

    actual_seeds = set(grouped)
    if actual_seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            "Seed mismatch. "
            f"Missing: {sorted(EXPECTED_SEEDS - actual_seeds)}; "
            f"unexpected: {sorted(actual_seeds - EXPECTED_SEEDS)}."
        )

    expected_candidate_ids = set(grouped[1])

    for seed in sorted(EXPECTED_SEEDS):
        candidate_ids = grouped[seed]

        if len(candidate_ids) != EXPECTED_CANDIDATES_PER_SEED:
            raise RuntimeError(
                f"Seed {seed} has {len(candidate_ids)} records; "
                f"expected {EXPECTED_CANDIDATES_PER_SEED}."
            )

        if set(candidate_ids) != expected_candidate_ids:
            raise RuntimeError(
                f"Seed {seed} has a different candidate set."
            )

    train = [
        row for row in rows
        if int(row["seed"]) in CALIBRATION_SEEDS
    ]
    test = [
        row for row in rows
        if int(row["seed"]) in HELDOUT_SEEDS
    ]

    if len(train) != 220 or len(test) != 220:
        raise RuntimeError(
            f"Expected 220 calibration and 220 held-out records; "
            f"got {len(train)} and {len(test)}."
        )

    return rows, train, test
