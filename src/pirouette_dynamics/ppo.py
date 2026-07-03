"""Optional PPO training via stable-baselines3 (the `rl` extra).

The bundled CEM learner in `train.py` is the default and needs no heavyweight dependency;
this module offers PPO for those who install ``pip install 'pirouette-dynamics[rl]'``. The
trained model is wrapped in the same ``.act(obs)`` interface as `LinearPolicy`, so
`train.evaluate` / `train.rollout` work on a PPO policy unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pirouette_dynamics.train import EnvFactory


@dataclass(frozen=True)
class PPOPolicy:
    """Adapter exposing a stable-baselines3 model through the `Policy` protocol."""

    model: Any

    def act(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        action, _ = self.model.predict(obs.astype(np.float32), deterministic=True)
        return np.asarray(action, dtype=float)


def train_ppo(
    make_env: EnvFactory,
    total_timesteps: int = 100_000,
    seed: int = 0,
) -> PPOPolicy:
    """Train PPO on the pirouette env and return a `.act()`-compatible policy.

    Raises a clear error if the `rl` extra (stable-baselines3) is not installed.
    """
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "PPO needs the optional dependency; install with: "
            "pip install 'pirouette-dynamics[rl]'"
        ) from exc

    model = PPO("MlpPolicy", make_env(), seed=seed, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    return PPOPolicy(model=model)
