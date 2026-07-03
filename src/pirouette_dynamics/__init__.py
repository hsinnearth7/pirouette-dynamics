"""pirouette-dynamics: physics-based simulation and learned control of ballet turns.

Layers (see README):
- anthropometry: De Leva segment masses + a parametric standing/retire posture.
- inertia: assemble the whole-body inertia tensor about a pivot (parallel axis).
- rigidbody: the toppling equation of motion (gravity-driven inverted spinning body).
- topple: high-level analyses that reproduce Lott & Laws (2012) balance thresholds.
- control: ankle vs hip balance strategies that close the loop on the toppling body.
- metrics: the COM-COP inclination angle (Lin 2019), shared with the measurement tool.
- env / reward / train: a Gymnasium task and a CEM learner for the pirouette (Tier C).
"""
from __future__ import annotations

from pirouette_dynamics.anthropometry import BodySegment, build_body, list_postures
from pirouette_dynamics.control import (
    ControlledResult,
    Controller,
    ankle_controller,
    hip_controller,
    integrate_controlled,
    max_recoverable_tilt_deg,
    revolutions_with_control,
    stability_domain,
)
from pirouette_dynamics.env import PirouetteEnv
from pirouette_dynamics.inertia import RigidBodyInertia, assemble_inertia
from pirouette_dynamics.metrics import com_cop_inclination_deg
from pirouette_dynamics.reward import RewardWeights, step_reward
from pirouette_dynamics.rigidbody import ToppleResult, integrate_topple
from pirouette_dynamics.topple import (
    HOP_ONSET_ANGLE_DEG,
    critical_initial_tilt_deg,
    revolutions_to_topple,
)
from pirouette_dynamics.train import (
    CEMResult,
    LinearPolicy,
    RolloutStats,
    cem_train,
    evaluate,
    rollout,
)

__all__ = [
    "HOP_ONSET_ANGLE_DEG",
    "BodySegment",
    "CEMResult",
    "ControlledResult",
    "Controller",
    "LinearPolicy",
    "PirouetteEnv",
    "RewardWeights",
    "RigidBodyInertia",
    "RolloutStats",
    "ToppleResult",
    "ankle_controller",
    "assemble_inertia",
    "build_body",
    "cem_train",
    "com_cop_inclination_deg",
    "critical_initial_tilt_deg",
    "evaluate",
    "hip_controller",
    "integrate_controlled",
    "integrate_topple",
    "list_postures",
    "max_recoverable_tilt_deg",
    "revolutions_to_topple",
    "revolutions_with_control",
    "rollout",
    "stability_domain",
    "step_reward",
]

__version__ = "0.1.0"
