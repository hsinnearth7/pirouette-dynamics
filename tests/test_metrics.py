"""COM-COP inclination metric (Lin 2019 definition)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pirouette_dynamics.anthropometry import GRAVITY, build_body
from pirouette_dynamics.inertia import assemble_inertia
from pirouette_dynamics.metrics import com_cop_inclination_deg


@pytest.fixture
def dancer() -> object:
    return assemble_inertia(build_body())


def test_no_ankle_torque_inclination_equals_tilt(dancer: object) -> None:
    # With the COP fixed at the pivot, the COM-COP inclination is just the body tilt.
    incl = com_cop_inclination_deg(dancer, math.radians(5.0))  # type: ignore[arg-type]
    assert float(incl) == pytest.approx(5.0, abs=1e-6)


def test_supporting_torque_reduces_inclination(dancer: object) -> None:
    # A COP shift that lands under the COM zeroes the inclination.
    theta = math.radians(5.0)
    length = dancer.com_height  # type: ignore[attr-defined]
    mg = dancer.mass * GRAVITY  # type: ignore[attr-defined]
    full_support = mg * length * math.sin(theta)  # torque putting COP under the COM
    incl = com_cop_inclination_deg(dancer, theta, full_support)  # type: ignore[arg-type]
    assert float(incl) == pytest.approx(0.0, abs=1e-6)


def test_accepts_array_input(dancer: object) -> None:
    thetas = np.radians(np.array([0.0, 2.0, 4.0]))
    incl = com_cop_inclination_deg(dancer, thetas)  # type: ignore[arg-type]
    assert incl.shape == (3,)
    assert np.all(np.diff(incl) > 0)
