"""Reward shaping for the learned-control task (Tier C).

The reward encodes what a "good" pirouette is, so an optimiser can discover the control that
produces it:

- **keep turning**: reward accrued per revolution while upright;
- **aesthetic (AP > ML)**: penalise the COM-COP inclination, weighting the anterior-posterior
  axis more than medio-lateral. Lin et al. (2019) found the expert-novice difference is larger
  in AP (Cohen's d 0.50-1.18 over the significant phases) than in ML (0.43-0.65); the
  asymmetric weight makes the agent suppress AP inclination more than ML;
- **elegance**: penalise the trunk lean that the hip strategy incurs;
- **economy**: a small penalty on control effort (coordination economy);
- **toppling**: a large terminal penalty when the body passes the hop-onset angle.

Splitting this out keeps the reward a pure, testable function of the step outcome.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RewardWeights:
    """Weights for the shaped pirouette reward (per revolution / per degree-second)."""

    turn: float = 4.0
    aesthetic_ap: float = 1.0
    aesthetic_ml: float = 0.3
    lean: float = 0.5
    energy: float = 0.02
    topple_penalty: float = 40.0


def step_reward(
    spin_rev_s: float,
    inclination_ap_deg: float,
    inclination_ml_deg: float,
    trunk_lean_deg: float,
    control_effort: NDArray[np.float64],
    dt: float,
    weights: RewardWeights,
) -> float:
    """Per-step reward. `control_effort` is the vector of normalised torques (each in [-1, 1])."""
    reward = weights.turn * spin_rev_s * dt
    reward -= weights.aesthetic_ap * abs(inclination_ap_deg) * dt
    reward -= weights.aesthetic_ml * abs(inclination_ml_deg) * dt
    reward -= weights.lean * abs(trunk_lean_deg) * dt
    reward -= weights.energy * float(np.sum(np.square(control_effort))) * dt
    return reward
