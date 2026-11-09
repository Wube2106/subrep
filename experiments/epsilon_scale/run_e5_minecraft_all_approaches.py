"""E5 extension: controlled Minecraft PDS comparison for approaches A-F."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from certification.pds_test import PDSGate
from bridge.experimental.epsilon_scaling_original import apply_epsilon_scale
from bridge.experimental.rank_based_admission import rank_based_epsilon_admission
from env.minecraft_rollout import (
    DEFAULT_HORIZON,
    appraisal_scales,
    evaluate_candidates,
)
from env.minecraft_stub import SPEC_TRADE_DELTA_R, MinecraftStubEnv

SEED = 42
GAMMA = 0.99
EPSILONS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.65, 0.66, 0.7, 0.99, 1.0, 2.3, 2.31)


def admitted(epsilon, delta_r, delta_n):
    return bool(PDSGate(epsilon=epsilon).admit(delta_r, delta_n))


def main():
    env = MinecraftStubEnv()
    try:
        env.reset(seed=SEED)
        baseline, candidates = evaluate_candidates(
            env, horizon=DEFAULT_HORIZON, gamma=GAMMA
        )

        payoff_scale, motive_scale = appraisal_scales(candidates)
        objective_matrix = np.asarray(
            [candidate.delta_n for candidate in candidates], dtype=float
        )
        objective_scales = np.maximum(
            np.mean(np.abs(objective_matrix), axis=0),
            SPEC_TRADE_DELTA_R,
        )

        baseline_return = float(baseline["baseline_payoff"])

        print("=" * 100)
        print("E5 EXTENSION — MINECRAFT STUB: RAW PDS VS APPROACHES A–F")
        print("=" * 100)
        print(f"Environment: deterministic six-objective Minecraft-shaped stub")
        print(f"Seed: {SEED} | Horizon: {DEFAULT_HORIZON} | Gamma: {GAMMA}")
        print(f"Candidates: {len(candidates)}")
        print(f"Baseline task return: {baseline_return:.8f}")
        print(f"Payoff scale: {payoff_scale:.8f}")
        print(f"Shared motive scale: {motive_scale:.8f}")
        print(f"Objective scales B: {np.round(objective_scales, 8)}")
        print()

        print("Candidate raw improvements:")
        for c in candidates:
            print(
                f"  {c.skill_id:<22} delta_r={c.delta_r:+.8f} "
                f"min(delta_n)={min(c.delta_n):+.8f} "
                f"raw_margin={c.delta_r + min(c.delta_n):+.8f}"
            )

        methods = ("RAW", "A", "B", "C_PAYOFF", "C_MOTIVE", "D", "E", "F")
        print()
        print("Admission counts by epsilon:")
        print(f"{'epsilon':>8} | " + " | ".join(f"{m:>10}" for m in methods))
        print("-" * (12 + 13 * len(methods)))

        for epsilon in EPSILONS:
            decisions = {method: [] for method in methods}

            raw_r = [float(c.delta_r) for c in candidates]
            raw_n = [np.asarray(c.delta_n, dtype=float) for c in candidates]

            for c, r, n in zip(candidates, raw_r, raw_n):
                decisions["RAW"].append(admitted(epsilon, r, n))
                decisions["F"].append(admitted(epsilon, r, n))

                for method in ("A", "B", "C_PAYOFF", "C_MOTIVE"):
                    result = apply_epsilon_scale(
                        method=method,
                        delta_r=r,
                        delta_n=n,
                        epsilon=epsilon,
                        payoff_scale=payoff_scale,
                        motive_scale=motive_scale,
                        objective_scales=objective_scales,
                    )
                    decisions[method].append(
                        admitted(result.epsilon, result.delta_r,
                                 np.asarray(result.delta_n, dtype=float))
                    )

            # D requires a meaningful nonzero baseline return. This stub's
            # Idle baseline earns zero task reward, so D is undefined here.
            if abs(baseline_return) > 1e-12:
                for c, r, n in zip(candidates, raw_r, raw_n):
                    result = apply_epsilon_scale(
                        method="D",
                        delta_r=r,
                        delta_n=n,
                        epsilon=epsilon,
                        baseline_return=baseline_return,
                    )
                    decisions["D"].append(
                        admitted(result.epsilon, result.delta_r,
                                 np.asarray(result.delta_n, dtype=float))
                    )

            # E is batch-relative: rank all candidates together.
            _, e_decisions = rank_based_epsilon_admission(
                delta_r_values=raw_r,
                delta_n_values=raw_n,
                epsilon=epsilon,
            )
            decisions["E"] = list(e_decisions)

            counts = []
            for method in methods:
                if method == "D" and abs(baseline_return) <= 1e-12:
                    counts.append("N/A")
                else:
                    counts.append(
                        f"{sum(decisions[method])}/{len(candidates)}"
                    )

            print(
                f"{epsilon:8.3f} | "
                + " | ".join(f"{count:>10}" for count in counts)
            )

        print()
        print("Method definitions:")
        print("  RAW: original delta_r and delta_n; unchanged epsilon.")
        print("  A: divide delta_r by payoff scale and all delta_n by shared motive scale.")
        print("  B: divide delta_r by payoff scale and each delta_n by its own objective scale.")
        print("  C_PAYOFF: multiply epsilon by payoff scale.")
        print("  C_MOTIVE: multiply epsilon by shared motive scale.")
        print("  D: multiply epsilon by absolute baseline task return.")
        print("  E: batch-relative average-rank admission.")
        print("  F: unchanged raw PDS baseline.")
        print()
        print("Interpretation and limitations:")
        print("  This is a controlled PDS-only comparison; CVaR and final")
        print("  MetaMo controller selection are not included.")
        print("  D is N/A because the Idle baseline task return is zero.")
        print("  RAW and F should have identical admission counts.")
        print("  E is batch-relative and should not be interpreted as a")
        print("  per-candidate transformation.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
