"""E5: controlled Minecraft PDS comparison, raw vs Approach A."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make repository packages importable when this file is run directly.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from certification.pds_test import PDSGate
from env.minecraft_rollout import (
    DEFAULT_HORIZON,
    appraisal_scales,
    evaluate_candidates,
)
from env.minecraft_stub import MinecraftStubEnv

SEED = 42
EPSILONS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.65, 0.66, 0.7, 0.99, 1.0, 2.3, 2.31)
GAMMA = 0.99


def admitted(gate: PDSGate, delta_r: float, delta_n: np.ndarray) -> bool:
    result = gate.admit(delta_r, delta_n)

    if isinstance(result, bool):
        return result
    if isinstance(result, tuple):
        return bool(result[0])
    if isinstance(result, dict) and "admitted" in result:
        return bool(result["admitted"])
    if hasattr(result, "admitted"):
        return bool(result.admitted)

    return bool(result)


def main() -> None:
    env = MinecraftStubEnv()
    try:
        env.reset(seed=SEED)

        baseline, candidates = evaluate_candidates(
            env,
            horizon=DEFAULT_HORIZON,
            gamma=GAMMA,
        )

        payoff_scale, motive_scale = appraisal_scales(candidates)

        print("=" * 78)
        print("E5 — Minecraft PDS: baseline vs Approach A")
        print("=" * 78)
        print(f"Seed: {SEED}")
        print(f"Candidates: {len(candidates)}")
        print(f"Baseline payoff: {baseline['baseline_payoff']:.6f}")
        print(f"Baseline motives: {baseline['baseline_motives']}")
        print(f"Payoff scale: {payoff_scale:.6f}")
        print(f"Motive scale: {motive_scale:.6f}")
        print("Approach A: delta_r / payoff_scale; delta_n / motive_scale")
        print("Epsilon is unchanged in both conditions.")
        print()

        print("Candidate inputs:")
        for candidate in candidates:
            print(
                f"  {candidate.skill_id:<22} "
                f"raw delta_r={candidate.delta_r:+.6f} "
                f"raw min(delta_n)={np.min(candidate.delta_n):+.6f} "
                f"scaled delta_r={candidate.delta_r / payoff_scale:+.6f} "
                f"scaled min(delta_n)="
                f"{np.min(candidate.delta_n) / motive_scale:+.6f}"
            )

        print()
        print(
            f"{'epsilon':>9} | {'Baseline':>15} | "
            f"{'Approach A':>15} | {'Changed':>8}"
        )
        print("-" * 58)

        for epsilon in EPSILONS:
            baseline_gate = PDSGate(epsilon=epsilon)
            scaled_gate = PDSGate(epsilon=epsilon)

            baseline_skills = []
            scaled_skills = []

            for candidate in candidates:
                raw_n = np.asarray(candidate.delta_n, dtype=float)
                scaled_r = float(candidate.delta_r) / payoff_scale
                scaled_n = raw_n / motive_scale

                if admitted(
                    baseline_gate,
                    float(candidate.delta_r),
                    raw_n,
                ):
                    baseline_skills.append(candidate.skill_id)

                if admitted(
                    scaled_gate,
                    scaled_r,
                    scaled_n,
                ):
                    scaled_skills.append(candidate.skill_id)

            changed = len(set(baseline_skills) ^ set(scaled_skills))

            print(
                f"{epsilon:9.3f} | "
                f"{len(baseline_skills):>2}/{len(candidates):<2} "
                f"{str(baseline_skills):<11} | "
                f"{len(scaled_skills):>2}/{len(candidates):<2} "
                f"{str(scaled_skills):<11} | "
                f"{changed:>8}"
            )

        print()
        print("Interpretation:")
        print("- Baseline uses the original candidate improvements.")
        print("- Approach A divides payoff and all six objective improvements")
        print("  by the same environment-specific scales.")
        print("- Both conditions use exactly the same epsilon values.")
        print("- This isolates PDS admission; CVaR is not involved.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
