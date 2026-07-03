"""PPO integration (optional): skipped unless stable-baselines3 is installed."""
from __future__ import annotations

import pytest


def test_ppo_trains_and_produces_usable_policy() -> None:
    pytest.importorskip("stable_baselines3")
    from pirouette_dynamics.env import PirouetteEnv
    from pirouette_dynamics.ppo import PPOPolicy, train_ppo
    from pirouette_dynamics.train import rollout

    policy = train_ppo(lambda: PirouetteEnv(max_time=1.0), total_timesteps=2000, seed=0)
    assert isinstance(policy, PPOPolicy)

    env = PirouetteEnv(max_time=1.0)
    obs, _ = env.reset(seed=0)
    action = policy.act(obs)
    assert action.shape == (4,)

    stats = rollout(env, policy, seed=0)
    assert stats.steps > 0
