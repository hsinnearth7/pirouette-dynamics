"""PirouetteEnv: Gymnasium API, determinism, and passive-vs-restoring behaviour."""
from __future__ import annotations

import numpy as np

from pirouette_dynamics.env import PirouetteEnv
from pirouette_dynamics.train import LinearPolicy, n_params_for, rollout


def _zero_policy(env: PirouetteEnv) -> LinearPolicy:
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    return LinearPolicy(obs_dim, act_dim, np.zeros(n_params_for(obs_dim, act_dim)))


def _restoring_policy(env: PirouetteEnv) -> LinearPolicy:
    # Negative feedback on tilt and rate for both axes; a hand-crafted stabiliser.
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    w = np.zeros((act_dim, obs_dim))
    # Positive gains: torque opposes gravity (I*theta_ddot = M g l sin(theta) - u).
    w[0, 0], w[0, 1] = 3.0, 1.0  # ankle_ap from theta_ap, omega_ap
    w[1, 2], w[1, 3] = 3.0, 1.0  # ankle_ml from theta_ml, omega_ml
    params = np.concatenate([w.reshape(-1), np.zeros(act_dim)])
    return LinearPolicy(obs_dim, act_dim, params)


def test_spaces_shapes() -> None:
    env = PirouetteEnv()
    assert env.observation_space.shape == (7,)
    assert env.action_space.shape == (4,)


def test_reset_is_seed_deterministic() -> None:
    obs_a, _ = PirouetteEnv().reset(seed=7)
    obs_b, _ = PirouetteEnv().reset(seed=7)
    assert np.allclose(obs_a, obs_b)


def test_step_contract() -> None:
    env = PirouetteEnv()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs.astype(np.float32))
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    keys = {"revolutions", "inclination_ap_deg", "inclination_ml_deg", "trunk_lean_deg"}
    assert keys <= set(info)


def test_passive_body_topples() -> None:
    env = PirouetteEnv()
    stats = rollout(env, _zero_policy(env), seed=0)
    assert stats.toppled


def test_restoring_policy_outlasts_passive() -> None:
    env = PirouetteEnv()
    passive = rollout(env, _zero_policy(env), seed=0)
    restoring = rollout(env, _restoring_policy(env), seed=0)
    assert restoring.steps > passive.steps
    assert restoring.peak_inclination_ap_deg < passive.peak_inclination_ap_deg
    assert not restoring.toppled
