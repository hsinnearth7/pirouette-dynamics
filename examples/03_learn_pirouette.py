"""Learn to sustain a pirouette and test for emergent expert biomechanics (Tier C).

Run: python examples/03_learn_pirouette.py   (~1 minute; CEM, no GPU, no extra deps)

Trains a linear policy with the Cross-Entropy Method and compares it to a passive baseline,
then runs an ablation that removes the anterior-posterior emphasis from the aesthetic reward.
The learned policy should (1) actively balance instead of toppling, and (2) hold a small AP
COM-COP inclination close to Lin et al. (2019)'s experienced-dancer value, while the ablated
(symmetric-reward) policy drifts toward the larger, novice-like inclination.
"""
from __future__ import annotations

import numpy as np

from pirouette_dynamics.env import PirouetteEnv
from pirouette_dynamics.reward import RewardWeights
from pirouette_dynamics.train import LinearPolicy, RolloutStats, cem_train, evaluate, n_params_for

EVAL_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


def _row(name: str, stats: RolloutStats) -> str:
    return (
        f"  {name:24s} return {stats.total_return:7.2f} | revs {stats.revolutions:4.2f} | "
        f"AP incl {stats.peak_inclination_ap_deg:4.1f} deg | ML incl "
        f"{stats.peak_inclination_ml_deg:4.2f} deg | {'TOPPLED' if stats.toppled else 'survived'}"
    )


def main() -> None:
    env = PirouetteEnv()
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    passive = LinearPolicy(obs_dim, act_dim, np.zeros(n_params_for(obs_dim, act_dim)))

    print("Training the aesthetic-reward (AP-weighted) policy with CEM ...")
    expert = cem_train(lambda: PirouetteEnv(), iterations=25, population=48, init_std=1.0, seed=0)

    print("Training the ablation (symmetric aesthetic reward) policy ...")
    sym_weights = RewardWeights(aesthetic_ap=0.3, aesthetic_ml=0.3)
    symmetric = cem_train(
        lambda: PirouetteEnv(weights=sym_weights),
        iterations=25,
        population=48,
        init_std=1.0,
        seed=0,
    )

    print("\nEmergence comparison (mean over 8 held-out reset seeds):")
    print(_row("passive (no control)", evaluate(env, passive, EVAL_SEEDS)))
    print(_row("learned, AP-weighted", evaluate(env, expert.policy, EVAL_SEEDS)))
    print(_row("learned, symmetric (ablate)", evaluate(PirouetteEnv(weights=sym_weights),
                                                        symmetric.policy, EVAL_SEEDS)))

    print("\nLin et al. (2019) PRE-phase AP inclination range: experienced 2.77 deg, "
          "novice 5.62 deg.")
    print("The AP-weighted policy actively balances and lands near the experienced value;")
    print("removing the AP emphasis pushes it toward the larger, novice-like inclination.")
    print("(Consistent with, not proof of, expert biomechanics -- see docs/LIMITATIONS.md.)")


if __name__ == "__main__":
    main()
