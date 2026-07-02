"""Assemble a whole-body rigid-body inertia tensor about a pivot (the supporting foot).

Given the segmented body from `anthropometry.build_body`, this combines segment masses,
COM positions, and own inertia tensors into:
- total mass M and whole-body COM,
- the inertia tensor about an arbitrary pivot (parallel-axis theorem, full 3D),
- convenience scalars the toppling model needs: COM height above the pivot, the spin-axis
  moment (vertical), and the tilt-axis moment (horizontal).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pirouette_dynamics.anthropometry import BodySegment, Mat3, Vec3


def _parallel_axis(mass: float, r: Vec3) -> Mat3:
    """Inertia shift of a point mass at offset r: m (|r|^2 I - r r^T)."""
    return mass * (float(r @ r) * np.eye(3) - np.outer(r, r))


@dataclass(frozen=True)
class RigidBodyInertia:
    """Whole-body inertial summary about a chosen pivot point."""

    mass: float
    com: Vec3
    pivot: Vec3
    inertia_pivot: Mat3  # 3x3 tensor about the pivot, world axes
    inertia_com: Mat3  # 3x3 tensor about the whole-body COM, world axes

    @property
    def com_height(self) -> float:
        """Vertical (z) distance of the COM above the pivot, i.e. the pendulum length l."""
        return float(self.com[2] - self.pivot[2])

    @property
    def spin_moment(self) -> float:
        """Moment of inertia about the vertical (z) axis through the pivot (the L = I*w axis)."""
        return float(self.inertia_pivot[2, 2])

    @property
    def tilt_moment_ap(self) -> float:
        """Moment resisting an anterior-posterior tilt (rotation about the ML / y axis)."""
        return float(self.inertia_pivot[1, 1])

    @property
    def tilt_moment_ml(self) -> float:
        """Moment resisting a medio-lateral tilt (rotation about the AP / x axis)."""
        return float(self.inertia_pivot[0, 0])

    def principal_moments(self) -> Vec3:
        """Sorted eigenvalues of the inertia tensor about the COM (principal moments)."""
        vals = np.linalg.eigvalsh(self.inertia_com)
        return np.sort(vals)


def assemble_inertia(
    segments: Sequence[BodySegment],
    pivot: Vec3 | None = None,
) -> RigidBodyInertia:
    """Combine segments into a whole-body inertia about `pivot` (default: floor origin).

    The pivot defaults to the coordinate origin, which `anthropometry` places at the floor
    under the supporting foot.
    """
    if pivot is None:
        pivot = np.zeros(3)
    pivot = np.asarray(pivot, dtype=float)

    mass = float(sum(s.mass for s in segments))
    com = np.sum([s.mass * s.com for s in segments], axis=0) / mass

    inertia_pivot = np.zeros((3, 3))
    inertia_com = np.zeros((3, 3))
    for s in segments:
        inertia_pivot += s.inertia_com + _parallel_axis(s.mass, s.com - pivot)
        inertia_com += s.inertia_com + _parallel_axis(s.mass, s.com - com)

    return RigidBodyInertia(
        mass=mass,
        com=com,
        pivot=pivot,
        inertia_pivot=inertia_pivot,
        inertia_com=inertia_com,
    )
