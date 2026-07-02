"""pirouette-dynamics: physics-based simulation and learned control of ballet turns.

See README.md for the layered design (physics core -> control -> learned control).
"""
from __future__ import annotations

from pirouette_dynamics.anthropometry import BodySegment, build_body, list_postures
from pirouette_dynamics.inertia import RigidBodyInertia, assemble_inertia
from pirouette_dynamics.rigidbody import ToppleResult, integrate_topple

__all__ = [
    "BodySegment",
    "RigidBodyInertia",
    "ToppleResult",
    "assemble_inertia",
    "build_body",
    "integrate_topple",
    "list_postures",
]

__version__ = "0.1.0"
