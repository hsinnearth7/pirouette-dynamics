"""Reproduce the Lott & Laws (2012) balance thresholds from the physics model."""
from __future__ import annotations

import pytest

from pirouette_dynamics.anthropometry import build_body
from pirouette_dynamics.inertia import assemble_inertia
from pirouette_dynamics.topple import (
    critical_initial_tilt_deg,
    initial_tilt_for_revolutions_numeric,
    reproduce_lott_laws,
    revolutions_to_topple,
)


@pytest.fixture
def dancer() -> object:
    return assemble_inertia(build_body(stature=1.593, mass=51.54))


def test_two_turns_need_sub_degree_start(dancer: object) -> None:
    # Lott & Laws headline: a rigid body needs to begin < 1 deg from vertical for > 2 turns.
    crit = critical_initial_tilt_deg(dancer, target_revolutions=2.0)  # type: ignore[arg-type]
    assert 0.05 < crit < 1.0


def test_critical_tilt_decreases_with_more_turns(dancer: object) -> None:
    one = critical_initial_tilt_deg(dancer, 1.0)  # type: ignore[arg-type]
    two = critical_initial_tilt_deg(dancer, 2.0)  # type: ignore[arg-type]
    three = critical_initial_tilt_deg(dancer, 3.0)  # type: ignore[arg-type]
    assert one > two > three


def test_closed_form_matches_numeric(dancer: object) -> None:
    # The small-angle closed form must agree with root-finding on the full integration.
    for n in (1.0, 2.0, 3.0):
        closed = critical_initial_tilt_deg(dancer, n)  # type: ignore[arg-type]
        numeric = initial_tilt_for_revolutions_numeric(dancer, n)  # type: ignore[arg-type]
        assert closed == pytest.approx(numeric, rel=0.05)


def test_more_initial_tilt_gives_fewer_revolutions(dancer: object) -> None:
    revs_small = revolutions_to_topple(dancer, 0.3)  # type: ignore[arg-type]
    revs_large = revolutions_to_topple(dancer, 3.0)  # type: ignore[arg-type]
    assert revs_small > revs_large


def test_report_fields_are_physical(dancer: object) -> None:
    rep = reproduce_lott_laws(dancer)  # type: ignore[arg-type]
    assert rep.tau > 0
    assert rep.com_height > 0
    assert rep.tilt_moment > rep.spin_moment
    assert rep.critical_tilt_2rev_deg < 1.0
