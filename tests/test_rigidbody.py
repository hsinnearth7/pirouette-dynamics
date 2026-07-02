"""Rigid-body core: conservation for the free top; cosh law for the topple integrator."""
from __future__ import annotations

import numpy as np
import pytest

from pirouette_dynamics.anthropometry import build_body
from pirouette_dynamics.inertia import assemble_inertia
from pirouette_dynamics.rigidbody import (
    integrate_free_top,
    integrate_topple,
    small_angle_topple,
    topple_time_constant,
)


def test_free_top_conserves_angular_momentum_and_energy() -> None:
    # Torque-free asymmetric top: |L| in body frame and rotational KE must stay constant.
    principal = np.array([0.8, 1.0, 1.5])
    omega0 = np.array([2.0, 0.3, 1.0])
    _t, omega = integrate_free_top(principal, omega0, duration=5.0)

    def ang_mom_norm(w: np.ndarray) -> np.ndarray:
        return np.linalg.norm(principal * w, axis=1)

    def kinetic(w: np.ndarray) -> np.ndarray:
        return 0.5 * np.sum(principal * w**2, axis=1)

    lmag = ang_mom_norm(omega)
    e = kinetic(omega)
    assert np.allclose(lmag, lmag[0], rtol=1e-6)
    assert np.allclose(e, e[0], rtol=1e-6)


def test_topple_matches_small_angle_cosh() -> None:
    # For a sub-degree start the nonlinear integration should track theta0 cosh(t/tau).
    body = assemble_inertia(build_body())
    tau = topple_time_constant(body)
    theta0 = np.deg2rad(0.3)
    res = integrate_topple(body, theta0, duration=0.5, hop_onset_rad=np.deg2rad(50))
    analytic = small_angle_topple(theta0, res.t, tau)
    assert np.allclose(res.theta, analytic, rtol=2e-3)


def test_topple_is_monotonic_and_hits_threshold() -> None:
    body = assemble_inertia(build_body())
    res = integrate_topple(body, np.deg2rad(1.0), duration=10.0)
    assert np.all(np.diff(res.theta) >= -1e-9)  # tilt only grows from rest
    assert res.reached_hop
    assert res.theta[-1] == pytest.approx(res.hop_onset_rad, abs=1e-3)


def test_time_constant_is_sub_second() -> None:
    body = assemble_inertia(build_body())
    assert 0.2 < topple_time_constant(body) < 0.6
