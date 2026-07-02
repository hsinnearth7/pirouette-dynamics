"""pirouette-dynamics: physics-based simulation and learned control of ballet turns.

Layers (see README):
- anthropometry: De Leva segment masses + a parametric standing/retire posture.
- inertia: assemble the whole-body inertia tensor about a pivot (parallel axis).
- rigidbody: the toppling equation of motion (gravity-driven inverted spinning body).
- topple: high-level analyses that reproduce Lott & Laws (2012) balance thresholds.
- metrics: the COM-COP inclination angle (Lin 2019), shared with the measurement tool.
"""
from __future__ import annotations

from pirouette_dynamics.anthropometry import BodySegment, build_body, list_postures
from pirouette_dynamics.inertia import RigidBodyInertia, assemble_inertia
from pirouette_dynamics.metrics import com_cop_inclination_deg
from pirouette_dynamics.rigidbody import ToppleResult, integrate_topple
from pirouette_dynamics.topple import (
    HOP_ONSET_ANGLE_DEG,
    critical_initial_tilt_deg,
    revolutions_to_topple,
)

__all__ = [
    "HOP_ONSET_ANGLE_DEG",
    "BodySegment",
    "RigidBodyInertia",
    "ToppleResult",
    "assemble_inertia",
    "build_body",
    "com_cop_inclination_deg",
    "critical_initial_tilt_deg",
    "integrate_topple",
    "list_postures",
    "revolutions_to_topple",
]

__version__ = "0.1.0"
