"""Shaped reward: signs and monotonicity."""
from __future__ import annotations

import numpy as np

from pirouette_dynamics.reward import RewardWeights, step_reward

_W = RewardWeights()
_NO_EFFORT = np.zeros(3)


def test_turning_upright_is_rewarded() -> None:
    r = step_reward(1.7, 0.0, 0.0, 0.0, _NO_EFFORT, dt=0.01, weights=_W)
    assert r > 0


def test_inclination_reduces_reward() -> None:
    upright = step_reward(1.7, 0.0, 0.0, 0.0, _NO_EFFORT, 0.01, _W)
    tilted = step_reward(1.7, 5.0, 0.0, 0.0, _NO_EFFORT, 0.01, _W)
    assert tilted < upright


def test_ap_penalised_more_than_ml() -> None:
    # The asymmetric weight is the point: a given AP inclination costs more than the same ML.
    ap = step_reward(1.7, 5.0, 0.0, 0.0, _NO_EFFORT, 0.01, _W)
    ml = step_reward(1.7, 0.0, 5.0, 0.0, _NO_EFFORT, 0.01, _W)
    assert ap < ml


def test_effort_and_lean_reduce_reward() -> None:
    base = step_reward(1.7, 0.0, 0.0, 0.0, _NO_EFFORT, 0.01, _W)
    effortful = step_reward(1.7, 0.0, 0.0, 0.0, np.array([1.0, 1.0, 1.0]), 0.01, _W)
    leaning = step_reward(1.7, 0.0, 0.0, 5.0, _NO_EFFORT, 0.01, _W)
    assert effortful < base
    assert leaning < base
