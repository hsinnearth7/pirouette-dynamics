"""Balance control: base-of-support limit, ankle vs hip authority, closed-loop behaviour."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pirouette_dynamics.anthropometry import build_body
from pirouette_dynamics.control import (
    ankle_controller,
    hip_controller,
    integrate_controlled,
    max_recoverable_tilt_deg,
    revolutions_with_control,
    stability_domain,
)
from pirouette_dynamics.inertia import assemble_inertia
from pirouette_dynamics.topple import revolutions_to_topple


@pytest.fixture
def dancer() -> object:
    return assemble_inertia(build_body())


def test_recoverable_tilt_grows_with_base_of_support(dancer: object) -> None:
    pointe = max_recoverable_tilt_deg(dancer, ankle_controller(dancer, 0.02))  # type: ignore[arg-type]
    demi = max_recoverable_tilt_deg(dancer, ankle_controller(dancer, 0.06))  # type: ignore[arg-type]
    flat = max_recoverable_tilt_deg(dancer, ankle_controller(dancer, 0.10))  # type: ignore[arg-type]
    assert pointe < demi < flat
    assert pointe < 1.5  # ankle strategy on pointe can hold barely ~1 degree


def test_hip_has_more_authority_than_pointe_ankle(dancer: object) -> None:
    ankle = ankle_controller(dancer, 0.02)  # type: ignore[arg-type]
    hip = hip_controller(dancer)  # type: ignore[arg-type]
    assert hip.tau_max > ankle.tau_max
    assert max_recoverable_tilt_deg(dancer, hip) > max_recoverable_tilt_deg(dancer, ankle)  # type: ignore[arg-type]


def test_controller_torque_saturates(dancer: object) -> None:
    ankle = ankle_controller(dancer, 0.02)  # type: ignore[arg-type]
    assert ankle.torque(0.0, 0.0) == pytest.approx(0.0)
    assert ankle.torque(math.radians(30.0), 0.0) == pytest.approx(ankle.tau_max)
    assert ankle.torque(math.radians(-30.0), 0.0) == pytest.approx(-ankle.tau_max)


def test_demi_ankle_stabilises_small_tilt(dancer: object) -> None:
    res = integrate_controlled(dancer, ankle_controller(dancer, 0.06), math.radians(1.0))  # type: ignore[arg-type]
    assert res.stabilized
    assert res.topple_time is None
    assert abs(res.theta[-1]) < math.radians(0.2)


def test_pointe_ankle_fails_beyond_its_limit(dancer: object) -> None:
    res = integrate_controlled(dancer, ankle_controller(dancer, 0.02), math.radians(2.0))  # type: ignore[arg-type]
    assert not res.stabilized
    assert res.topple_time is not None


def test_hip_recovers_where_pointe_ankle_fails(dancer: object) -> None:
    res = integrate_controlled(dancer, hip_controller(dancer), math.radians(3.0))  # type: ignore[arg-type]
    assert res.stabilized
    # The recovery is bought with a visible trunk lean, absent from the ankle strategy.
    assert res.peak_lean_deg > 0.0


def test_ankle_strategy_has_no_trunk_lean(dancer: object) -> None:
    res = integrate_controlled(dancer, ankle_controller(dancer, 0.06), math.radians(1.0))  # type: ignore[arg-type]
    assert np.all(res.trunk_lean == 0.0)


def test_control_delays_topple_versus_passive(dancer: object) -> None:
    # Even when it cannot fully stabilise, a saturated ankle torque buys extra revolutions.
    passive = revolutions_to_topple(dancer, 2.0)  # type: ignore[arg-type]
    controlled = revolutions_with_control(dancer, ankle_controller(dancer, 0.02), 2.0)  # type: ignore[arg-type]
    assert controlled > passive


def test_strong_controller_reports_infinite_revolutions(dancer: object) -> None:
    assert revolutions_with_control(dancer, hip_controller(dancer), 2.0) == float("inf")  # type: ignore[arg-type]


def test_stability_domain_shape_and_origin(dancer: object) -> None:
    tilt = np.array([0.5, 2.0, 5.0])
    rate = np.array([-20.0, 0.0, 20.0])
    grid = stability_domain(dancer, hip_controller(dancer), tilt, rate)  # type: ignore[arg-type]
    assert grid.shape == (3, 3)
    assert bool(grid[0, 1])  # small tilt, zero rate -> stabilised
