"""pirouette-dynamics: physics-based simulation and learned control of ballet turns.

See README.md for the layered design (physics core -> control -> learned control).
"""
from __future__ import annotations

from pirouette_dynamics.anthropometry import BodySegment, build_body, list_postures

__all__ = [
    "BodySegment",
    "build_body",
    "list_postures",
]

__version__ = "0.1.0"
