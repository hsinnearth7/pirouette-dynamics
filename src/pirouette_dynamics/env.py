"""A Gymnasium environment for learning to sustain a pirouette (Tier C).

The agent balances a toppling, spinning body and modulates its spin, earning reward for
revolutions turned while keeping the COM-COP inclination small (with the anterior-posterior
axis weighted more than medio-lateral, after Lin 2019). It is the reusable RL artifact: any
learner -- the bundled CEM search, or PPO/SAC from stable-baselines3 -- can be pointed at it.

State (normalised observation, 7-d):
    [theta_ap, theta_dot_ap, theta_ml, theta_dot_ml, spin_rev/1.7, arm_radius, phase]
Action (4-d, each in [-1, 1]):
    [ankle_torque_ap, ankle_torque_ml, arm_retract_rate, hip_torque_ap]

Physics reuses the Tier A/B model: an inverted-pendulum tilt in the AP and ML planes with a
bounded ankle torque, a hip torque with a trunk-lean cost, and angular-momentum conservation
for the spin (pulling the arms in lowers the vertical-axis inertia and speeds the turn).
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from pirouette_dynamics.anthropometry import GRAVITY, build_body
from pirouette_dynamics.inertia import assemble_inertia
from pirouette_dynamics.metrics import com_cop_inclination_deg
from pirouette_dynamics.reward import RewardWeights, step_reward

Obs = NDArray[np.float64]
Action = NDArray[np.float64]

_HIP_CAPACITY_RATIO = 0.35
_HIP_LEAN_AT_CAPACITY_DEG = 9.0
_MAX_ARM_RATE = 4.0
_BASELINE_SPIN_REV_S = 1.7
_RATE_NORM = 5.0  # rad/s scale for observation normalisation


class PirouetteEnv(gym.Env[Obs, Action]):
    """Balance-and-spin a pirouette; reward turning while keeping inclination small."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        base_of_support_m: float = 0.06,
        weights: RewardWeights | None = None,
        init_tilt_ap_deg: float = 3.0,
        init_tilt_ml_deg: float = 0.5,
        dt: float = 0.01,
        n_substeps: int = 5,
        max_time: float = 3.0,
        max_revolutions: float = 5.0,
        hop_onset_deg: float = 9.3,
    ) -> None:
        super().__init__()
        self.weights = weights if weights is not None else RewardWeights()
        self.dt = dt
        self.n_substeps = n_substeps
        self.max_time = max_time
        self.max_revolutions = max_revolutions
        self.hop_onset_rad = math.radians(hop_onset_deg)
        self.init_tilt_ap_rad = math.radians(init_tilt_ap_deg)
        self.init_tilt_ml_rad = math.radians(init_tilt_ml_deg)

        inertia_first = assemble_inertia(build_body(posture="retire_arms_first"))
        inertia_second = assemble_inertia(build_body(posture="retire_arms_second"))
        self._inertia = inertia_first
        self.i_tilt_ap = inertia_first.tilt_moment_ap
        self.i_tilt_ml = inertia_first.tilt_moment_ml
        self.i_zz_min = inertia_first.spin_moment  # arms in -> smaller vertical inertia
        self.i_zz_max = inertia_second.spin_moment  # arms out -> larger
        self.mgl = inertia_first.mass * GRAVITY * inertia_first.com_height

        self.tau_bos = inertia_first.mass * GRAVITY * base_of_support_m
        self.tau_hip = _HIP_CAPACITY_RATIO * self.mgl
        self.lean_per_torque = math.radians(_HIP_LEAN_AT_CAPACITY_DEG) / self.tau_hip
        self.angular_momentum = self.i_zz_max * _BASELINE_SPIN_REV_S * 2.0 * math.pi

        high = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0], dtype=np.float32)
        low = np.array([-3.0, -3.0, -3.0, -3.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._theta_ap = 0.0
        self._omega_ap = 0.0
        self._theta_ml = 0.0
        self._omega_ml = 0.0
        self._arm = 1.0
        self._time = 0.0
        self._revolutions = 0.0

    def _spin_rev_s(self) -> float:
        i_zz = self.i_zz_min + self._arm * (self.i_zz_max - self.i_zz_min)
        return self.angular_momentum / i_zz / (2.0 * math.pi)

    def _observation(self) -> Obs:
        return np.array(
            [
                self._theta_ap / self.hop_onset_rad,
                self._omega_ap / _RATE_NORM,
                self._theta_ml / self.hop_onset_rad,
                self._omega_ml / _RATE_NORM,
                self._spin_rev_s() / _BASELINE_SPIN_REV_S,
                self._arm,
                self._time / self.max_time,
            ],
            dtype=np.float64,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Obs, dict[str, Any]]:
        super().reset(seed=seed)
        ap, ml = self.init_tilt_ap_rad, self.init_tilt_ml_rad
        self._theta_ap = float(self.np_random.uniform(-ap, ap))
        self._theta_ml = float(self.np_random.uniform(-ml, ml))
        self._omega_ap = 0.0
        self._omega_ml = 0.0
        self._arm = 1.0
        self._time = 0.0
        self._revolutions = 0.0
        return self._observation(), {}

    def step(self, action: Action) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        ankle_ap = float(a[0]) * self.tau_bos
        ankle_ml = float(a[1]) * self.tau_bos
        arm_rate = float(a[2]) * _MAX_ARM_RATE
        hip_ap = float(a[3]) * self.tau_hip

        u_ap = ankle_ap + hip_ap
        dt_sub = self.dt / self.n_substeps
        for _ in range(self.n_substeps):
            acc_ap = (self.mgl * math.sin(self._theta_ap) - u_ap) / self.i_tilt_ap
            acc_ml = (self.mgl * math.sin(self._theta_ml) - ankle_ml) / self.i_tilt_ml
            self._omega_ap += acc_ap * dt_sub
            self._omega_ml += acc_ml * dt_sub
            self._theta_ap += self._omega_ap * dt_sub
            self._theta_ml += self._omega_ml * dt_sub

        self._arm = float(np.clip(self._arm + arm_rate * self.dt, 0.0, 1.0))
        self._time += self.dt
        spin_rev_s = self._spin_rev_s()
        self._revolutions += spin_rev_s * self.dt

        incl_ap = float(com_cop_inclination_deg(self._inertia, self._theta_ap, ankle_ap))
        incl_ml = float(com_cop_inclination_deg(self._inertia, self._theta_ml, ankle_ml))
        lean_deg = math.degrees(self.lean_per_torque * abs(hip_ap))
        effort = np.array([a[0], a[1], a[3]], dtype=float)
        reward = step_reward(
            spin_rev_s, incl_ap, incl_ml, lean_deg, effort, self.dt, self.weights
        )

        toppled = (
            abs(self._theta_ap) > self.hop_onset_rad or abs(self._theta_ml) > self.hop_onset_rad
        )
        if toppled:
            reward -= self.weights.topple_penalty
        truncated = self._time >= self.max_time or self._revolutions >= self.max_revolutions

        info = {
            "revolutions": self._revolutions,
            "inclination_ap_deg": incl_ap,
            "inclination_ml_deg": incl_ml,
            "trunk_lean_deg": lean_deg,
        }
        return self._observation(), reward, toppled, truncated, info
