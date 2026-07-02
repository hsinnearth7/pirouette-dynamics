"""Rigid-body dynamics: gravity-driven toppling, plus a torque-free Euler integrator.

Toppling model
--------------
A pirouette spins slowly (~1.7 rev/s in Lott & Laws 2012), well below the "sleeping top"
condition, so gyroscopic terms are small and the dancer topples essentially like an inverted
physical pendulum pivoting at the supporting foot:

    I_tilt * theta_ddot = M * g * l * sin(theta)

where theta is the tilt of the body axis from vertical, l is the COM height above the pivot,
and I_tilt is the whole-body moment about the horizontal tilt axis through the pivot. The
motion is unstable: a small initial offset grows as cosh(t / tau) with tau = sqrt(I_tilt /
(M g l)). This is the model Lott & Laws integrate to get topple angle vs time.

`integrate_free_top` is the torque-free rigid body (Euler's equations). It carries no ballet
content; it exists so the test-suite can assert that the rigid-body machinery conserves
angular momentum and energy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from pirouette_dynamics.anthropometry import GRAVITY
from pirouette_dynamics.inertia import RigidBodyInertia

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ToppleResult:
    """Trajectory of a toppling body until it reaches the hop-onset angle (or time runs out)."""

    t: Array  # time samples (s)
    theta: Array  # tilt angle from vertical (rad)
    tau: float  # topple time-constant sqrt(I_tilt / (M g l)) (s)
    hop_onset_rad: float  # threshold that ended the run
    topple_time: float | None  # time to reach hop_onset, or None if never reached

    @property
    def reached_hop(self) -> bool:
        return self.topple_time is not None


def topple_time_constant(inertia: RigidBodyInertia, tilt_axis: str = "ap") -> float:
    """tau = sqrt(I_tilt / (M g l)); the e-folding-ish time of the cosh growth."""
    i_tilt = inertia.tilt_moment_ap if tilt_axis == "ap" else inertia.tilt_moment_ml
    denom = inertia.mass * GRAVITY * inertia.com_height
    if denom <= 0:
        raise ValueError("COM must be above the pivot (positive com_height) to topple.")
    return math.sqrt(i_tilt / denom)


def small_angle_topple(theta0_rad: float, t: Array, tau: float) -> Array:
    """Analytic small-angle solution theta(t) = theta0 * cosh(t / tau)."""
    return theta0_rad * np.cosh(t / tau)


def integrate_topple(
    inertia: RigidBodyInertia,
    initial_tilt_rad: float,
    duration: float,
    tilt_axis: str = "ap",
    hop_onset_rad: float = np.deg2rad(9.3),
    max_step: float = 0.002,
) -> ToppleResult:
    """Integrate the nonlinear inverted-pendulum topple until `hop_onset_rad` or `duration`.

    Starts from rest at `initial_tilt_rad`. Uses a terminal event at the hop-onset angle so
    the returned `topple_time` is the instant the dancer would have to hop to recover.
    """
    i_tilt = inertia.tilt_moment_ap if tilt_axis == "ap" else inertia.tilt_moment_ml
    k = inertia.mass * GRAVITY * inertia.com_height / i_tilt  # = 1 / tau^2

    def rhs(_t: float, y: Array) -> list[float]:
        theta, omega = y
        return [omega, k * np.sin(theta)]

    def hop_event(_t: float, y: Array) -> float:
        return float(y[0] - hop_onset_rad)

    hop_event.terminal = True  # type: ignore[attr-defined]
    hop_event.direction = 1.0  # type: ignore[attr-defined]

    sol = solve_ivp(
        rhs,
        (0.0, duration),
        [initial_tilt_rad, 0.0],
        method="RK45",
        events=hop_event,
        max_step=max_step,
        rtol=1e-9,
        atol=1e-12,
        dense_output=False,
    )
    topple_time = float(sol.t_events[0][0]) if sol.t_events[0].size > 0 else None
    return ToppleResult(
        t=sol.t,
        theta=sol.y[0],
        tau=math.sqrt(1.0 / k),
        hop_onset_rad=hop_onset_rad,
        topple_time=topple_time,
    )


def integrate_free_top(
    principal_moments: Array,
    omega0: Array,
    duration: float,
    n_samples: int = 400,
) -> tuple[Array, Array]:
    """Torque-free rigid body via Euler's equations. Returns (t, omega[body frame]).

    Used only to verify the rigid-body core conserves angular momentum and kinetic energy.
    """
    i1, i2, i3 = (float(v) for v in principal_moments)

    def rhs(_t: float, w: Array) -> list[float]:
        w1, w2, w3 = w
        return [
            (i2 - i3) / i1 * w2 * w3,
            (i3 - i1) / i2 * w3 * w1,
            (i1 - i2) / i3 * w1 * w2,
        ]

    t_eval = np.linspace(0.0, duration, n_samples)
    sol = solve_ivp(
        rhs,
        (0.0, duration),
        np.asarray(omega0, dtype=float),
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    return sol.t, sol.y.T
