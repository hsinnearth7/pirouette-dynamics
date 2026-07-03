"""CEM learner: policy plumbing and that learning beats the passive baseline."""
from __future__ import annotations

import numpy as np

from pirouette_dynamics.env import PirouetteEnv
from pirouette_dynamics.train import (
    LinearPolicy,
    cem_train,
    evaluate,
    n_params_for,
    rollout,
)


def _fast_env() -> PirouetteEnv:
    return PirouetteEnv(max_time=1.5)


def test_linear_policy_output_bounds() -> None:
    obs_dim, act_dim = 7, 4
    rng = np.random.default_rng(0)
    policy = LinearPolicy(obs_dim, act_dim, rng.normal(size=n_params_for(obs_dim, act_dim)))
    action = policy.act(rng.normal(size=obs_dim))
    assert action.shape == (act_dim,)
    assert np.all(np.abs(action) <= 1.0)


def test_rollout_returns_stats() -> None:
    env = _fast_env()
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    zero = LinearPolicy(obs_dim, act_dim, np.zeros(n_params_for(obs_dim, act_dim)))
    stats = rollout(env, zero, 0)
    assert stats.steps > 0
    assert stats.revolutions > 0


def test_cem_learns_to_beat_passive() -> None:
    result = cem_train(_fast_env, iterations=8, population=16, eval_seeds=(0, 1), seed=0)
    assert len(result.history) == 8

    env = _fast_env()
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    passive = LinearPolicy(obs_dim, act_dim, np.zeros(n_params_for(obs_dim, act_dim)))

    seeds = [0, 1, 2, 3]
    learned_return = evaluate(env, result.policy, seeds).total_return
    passive_return = evaluate(env, passive, seeds).total_return
    assert learned_return > passive_return
