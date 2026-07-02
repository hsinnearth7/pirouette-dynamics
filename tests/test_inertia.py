"""Whole-body inertia assembly: parallel-axis correctness, plausible bands, skater effect."""
from __future__ import annotations

import numpy as np
import pytest

from pirouette_dynamics.anthropometry import BodySegment, build_body
from pirouette_dynamics.inertia import assemble_inertia


def test_point_mass_parallel_axis() -> None:
    # A single 2 kg point mass 0.5 m up the z-axis: I about origin = m d^2 about x and y, 0 about z.
    seg = BodySegment("pt", 2.0, np.array([0.0, 0.0, 0.5]), np.zeros((3, 3)))
    body = assemble_inertia([seg])
    expected = 2.0 * 0.5**2
    assert body.inertia_pivot[0, 0] == pytest.approx(expected)
    assert body.inertia_pivot[1, 1] == pytest.approx(expected)
    assert body.inertia_pivot[2, 2] == pytest.approx(0.0, abs=1e-12)
    assert body.com_height == pytest.approx(0.5)


def test_spin_moment_in_dancer_band() -> None:
    # Guilford's honours thesis measured whole-body vertical-axis MOI ~1.72 kg m^2. Our
    # arms-first retire should sit in a plausible dancer band around that anchor.
    body = assemble_inertia(build_body())
    assert 0.7 < body.spin_moment < 2.5


def test_tilt_moment_dominates_spin_moment() -> None:
    # Tilting about the foot swings the whole mass on a ~0.9 m arm, so the tilt moment is
    # far larger than the near-axis spin moment.
    body = assemble_inertia(build_body())
    assert body.tilt_moment_ap > 20.0 * body.spin_moment


def test_arms_second_raises_spin_moment() -> None:
    # Opening the arms increases the vertical-axis inertia (the figure-skater effect).
    first = assemble_inertia(build_body(posture="retire_arms_first"))
    second = assemble_inertia(build_body(posture="retire_arms_second"))
    assert second.spin_moment > first.spin_moment


def test_principal_moments_satisfy_triangle_inequality() -> None:
    body = assemble_inertia(build_body())
    i1, i2, i3 = body.principal_moments()
    assert i1 > 0
    assert i1 + i2 >= i3 - 1e-9
