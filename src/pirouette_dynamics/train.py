"""Learned control for the pirouette (Tier C), dependency-free.

A linear policy mapping the 7-d observation to the 4-d action is optimised by the
Cross-Entropy Method (CEM): sample a population of parameter vectors, keep the elite by
episode return, refit a Gaussian, repeat. CEM needs only numpy, so the learned-control result
is fully reproducible without a heavyweight RL dependency. The same `PirouetteEnv` also
accepts PPO/SAC from stable-baselines3 when it is installed (see examples/03).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from pirouette_dynamics.env import PirouetteEnv

Params = NDArray[np.float64]
EnvFactory = Callable[[], PirouetteEnv]


class Policy(Protocol):
    """Anything that maps an observation to an action: LinearPolicy, a PPO wrapper, etc."""

    def act(self, obs: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class LinearPolicy:
    """Action = tanh(W @ obs + b), flattened into one parameter vector for CEM."""

    obs_dim: int
    act_dim: int
    params: Params

    @property
    def n_params(self) -> int:
        return self.act_dim * (self.obs_dim + 1)

    def act(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        w = self.params[: self.act_dim * self.obs_dim].reshape(self.act_dim, self.obs_dim)
        b = self.params[self.act_dim * self.obs_dim :]
        return np.tanh(w @ obs + b)


def n_params_for(obs_dim: int, act_dim: int) -> int:
    return act_dim * (obs_dim + 1)


@dataclass(frozen=True)
class RolloutStats:
    """Summary of one episode."""

    total_return: float
    revolutions: float
    peak_inclination_ap_deg: float
    peak_inclination_ml_deg: float
    peak_lean_deg: float
    toppled: bool
    steps: int


def rollout(env: PirouetteEnv, policy: Policy, seed: int) -> RolloutStats:
    """Run one deterministic episode under `policy`."""
    obs, _ = env.reset(seed=seed)
    total = 0.0
    peak_ap = peak_ml = peak_lean = 0.0
    revolutions = 0.0
    toppled = False
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy.act(obs))
        total += reward
        peak_ap = max(peak_ap, abs(float(info["inclination_ap_deg"])))
        peak_ml = max(peak_ml, abs(float(info["inclination_ml_deg"])))
        peak_lean = max(peak_lean, float(info["trunk_lean_deg"]))
        revolutions = float(info["revolutions"])
        steps += 1
        if terminated or truncated:
            toppled = terminated
            break
    return RolloutStats(total, revolutions, peak_ap, peak_ml, peak_lean, toppled, steps)


def evaluate(env: PirouetteEnv, policy: Policy, seeds: list[int]) -> RolloutStats:
    """Average rollout statistics over several reset seeds."""
    runs = [rollout(env, policy, s) for s in seeds]
    n = len(runs)
    return RolloutStats(
        total_return=float(np.mean([r.total_return for r in runs])),
        revolutions=float(np.mean([r.revolutions for r in runs])),
        peak_inclination_ap_deg=float(np.mean([r.peak_inclination_ap_deg for r in runs])),
        peak_inclination_ml_deg=float(np.mean([r.peak_inclination_ml_deg for r in runs])),
        peak_lean_deg=float(np.mean([r.peak_lean_deg for r in runs])),
        toppled=bool(sum(r.toppled for r in runs) > n / 2),
        steps=int(np.mean([r.steps for r in runs])),
    )


@dataclass(frozen=True)
class CEMResult:
    """Converged policy (the final distribution mean) plus the per-iteration elite return."""

    policy: LinearPolicy
    history: list[float]


def cem_train(
    make_env: EnvFactory,
    iterations: int = 20,
    population: int = 48,
    elite_frac: float = 0.25,
    init_std: float = 1.0,
    eval_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    seed: int = 0,
) -> CEMResult:
    """Optimise a linear policy by the Cross-Entropy Method.

    The returned policy is the final distribution mean, which is more robust than any single
    sampled point. `history` is the per-iteration mean elite return, which should rise as the
    agent learns to stay upright and keep turning. Fitness averages over several reset seeds so
    the elite are policies that survive many disturbances, not lucky ones.
    """
    env = make_env()
    obs_dim = int(env.observation_space.shape[0])  # type: ignore[index]
    act_dim = int(env.action_space.shape[0])  # type: ignore[index]
    dim = n_params_for(obs_dim, act_dim)
    n_elite = max(2, int(population * elite_frac))
    seeds = list(eval_seeds)

    rng = np.random.default_rng(seed)
    mean = np.zeros(dim)
    std = np.full(dim, init_std)
    history: list[float] = []

    for _ in range(iterations):
        samples = rng.normal(mean, std, size=(population, dim))
        returns = np.array(
            [evaluate(env, LinearPolicy(obs_dim, act_dim, s), seeds).total_return for s in samples]
        )
        elite_idx = np.argsort(returns)[-n_elite:]
        elite = samples[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-3
        history.append(float(returns[elite_idx].mean()))

    return CEMResult(policy=LinearPolicy(obs_dim, act_dim, mean), history=history)
