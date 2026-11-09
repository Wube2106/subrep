from __future__ import annotations

import csv
import statistics
from pathlib import Path

from env.lunar_lander_rollout import DEFAULT_GAMMA, NOOP_BASELINE, run_episode
from env.lunar_lander_wrapper import SubRepEnv


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "experiments" / "epsilon_scale" / "results"
SEEDS = tuple(range(1, 41))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "baseline_returns.csv"

    env = SubRepEnv(seed=SEEDS[0])
    rows = []

    try:
        for index, seed in enumerate(SEEDS, start=1):
            baseline_payoff, _, info = run_episode(
                env,
                NOOP_BASELINE,
                seed=seed,
                gamma=DEFAULT_GAMMA,
                max_steps=None,
            )

            rows.append({
                "seed": seed,
                "baseline_id": NOOP_BASELINE.skill_id,
                "baseline_return": float(baseline_payoff),
                "baseline_return_abs": abs(float(baseline_payoff)),
                "steps": info["steps"],
                "landing_success": info["landing_success"],
                "crashed": info["crashed"],
            })

            print(
                f"Seed {seed}: baseline return = {baseline_payoff:.4f}",
                flush=True,
            )
    finally:
        env.close()

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    returns = [row["baseline_return"] for row in rows]
    magnitudes = [row["baseline_return_abs"] for row in rows]

    print("\nBaseline return summary")
    print(f"Count: {len(rows)}")
    print(f"Minimum: {min(returns):.4f}")
    print(f"Maximum: {max(returns):.4f}")
    print(f"Mean: {statistics.mean(returns):.4f}")
    print(f"Median absolute return: {statistics.median(magnitudes):.4f}")
    print(f"Minimum absolute return: {min(magnitudes):.4f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
