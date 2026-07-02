"""Balance control: bounded ankle and hip strategies stabilising the toppling body.

Tier A showed the passive body topples. Tier B closes the loop with a restoring torque:

    I_tilt * theta_ddot = M g l sin(theta) - u,      u = saturate(kp*theta + kd*theta_dot)

Two strategies differ in where the torque comes from and what bounds it:

- **Ankle strategy** (expert-like, elegant): torque about the ankle, but the centre of
  pressure can only move within the base of support, so ``|u| <= M g r_bos``. The largest tilt
  it can hold statically is ``arcsin(r_bos / l)`` -- about 1 deg on pointe, a few degrees on a
  flat foot. This is the mechanical reason a pirouette is so unstable.
- **Hip strategy** (novice-like at landing, per Lin 2019): the trunk is thrown at the hip,
  giving more corrective authority than the tiny pointe base allows, but at the cost of a
  visible trunk lean (tracked as a side-effect), i.e. a larger COM-COP excursion.

The model is a single tilt degree of freedom; the hip lean is reported as a cost readout, not
a second simulated link (see docs/LIMITATIONS.md).
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
class Controller:
    """A saturated PD balance controller with an optional trunk-lean cost per unit torque."""

    name: str
    kp: float  # proportional gain (N m / rad)
    kd: float  # derivative gain (N m s / rad)
    tau_max: float  # torque saturation (N m)
    lean_per_torque: float = 0.0  # trunk lean (rad) produced per N m of corrective torque

    def torque(self, theta: float, theta_dot: float) -> float:
        raw = self.kp * theta + self.kd * theta_dot
        return float(np.clip(raw, -self.tau_max, self.tau_max))


def _tilt_moment(inertia: RigidBodyInertia, tilt_axis: str) -> float:
    return inertia.tilt_moment_ap if tilt_axis == "ap" else inertia.tilt_moment_ml


def _gravity_torque_scale(inertia: RigidBodyInertia) -> float:
    """M g l, the peak-per-radian gravitational toppling torque coefficient."""
    return inertia.mass * GRAVITY * inertia.com_height


def _pd_gains(
    inertia: RigidBodyInertia,
    stiffness_ratio: float,
    damping_ratio: float,
    tilt_axis: str,
) -> tuple[float, float]:
    """PD gains for a target closed-loop stiffness and damping.

    kp = stiffness_ratio * M g l gives net restoring stiffness (kp - M g l); kd is set to
    `damping_ratio` times critical damping of that net stiffness.
    """
    mgl = _gravity_torque_scale(inertia)
    kp = stiffness_ratio * mgl
    net_stiffness = kp - mgl
    inertia_tilt = _tilt_moment(inertia, tilt_axis)
    kd = damping_ratio * 2.0 * math.sqrt(inertia_tilt * net_stiffness)
    return kp, kd


def ankle_controller(
    inertia: RigidBodyInertia,
    base_of_support_m: float = 0.06,
    stiffness_ratio: float = 2.0,
    damping_ratio: float = 1.0,
    tilt_axis: str = "ap",
) -> Controller:
    """Ankle strategy: torque bounded by the base of support (default demi-pointe, 6 cm)."""
    kp, kd = _pd_gains(inertia, stiffness_ratio, damping_ratio, tilt_axis)
    tau_max = inertia.mass * GRAVITY * base_of_support_m
    return Controller(
        name=f"ankle(bos={base_of_support_m * 100:.0f}cm)",
        kp=kp,
        kd=kd,
        tau_max=tau_max,
    )


def hip_controller(
    inertia: RigidBodyInertia,
    torque_capacity_ratio: float = 0.35,
    lean_at_capacity_deg: float = 9.0,
    stiffness_ratio: float = 2.0,
    damping_ratio: float = 1.0,
    tilt_axis: str = "ap",
) -> Controller:
    """Hip strategy: larger torque authority (fraction of M g l) but with a trunk-lean cost."""
    kp, kd = _pd_gains(inertia, stiffness_ratio, damping_ratio, tilt_axis)
    tau_max = torque_capacity_ratio * _gravity_torque_scale(inertia)
    lean_per_torque = math.radians(lean_at_capacity_deg) / tau_max
    return Controller(
        name=f"hip(cap={torque_capacity_ratio:.2f}Mgl)",
        kp=kp,
        kd=kd,
        tau_max=tau_max,
        lean_per_torque=lean_per_torque,
    )


def max_recoverable_tilt_deg(
    inertia: RigidBodyInertia,
    controller: Controller,
) -> float:
    """Largest tilt the controller can hold statically: arcsin(tau_max / (M g l)).

    Beyond this the gravitational torque exceeds the controller's saturation and toppling is
    inevitable regardless of gains. This is the sharpest expression of the base-of-support
    limit for the ankle strategy.
    """
    ratio = controller.tau_max / _gravity_torque_scale(inertia)
    return math.degrees(math.asin(min(1.0, ratio)))


@dataclass(frozen=True)
class ControlledResult:
    """Closed-loop trajectory of a controlled toppling body."""

    t: Array
    theta: Array
    torque: Array
    trunk_lean: Array  # hip-strategy side-effect (rad); zero for the ankle strategy
    controller: Controller
    hop_onset_rad: float
    topple_time: float | None

    @property
    def stabilized(self) -> bool:
        """True if the body never reached hop onset and the tilt is decaying to near-vertical."""
        return self.topple_time is None and abs(self.theta[-1]) < 0.5 * abs(self.theta[0])

    @property
    def peak_tilt_deg(self) -> float:
        return float(np.rad2deg(np.max(np.abs(self.theta))))

    @property
    def peak_lean_deg(self) -> float:
        return float(np.rad2deg(np.max(np.abs(self.trunk_lean))))


def integrate_controlled(
    inertia: RigidBodyInertia,
    controller: Controller,
    initial_tilt_rad: float,
    initial_rate_rad_s: float = 0.0,
    duration: float = 6.0,
    tilt_axis: str = "ap",
    hop_onset_rad: float = math.radians(9.3),
    n_samples: int = 600,
) -> ControlledResult:
    """Integrate the closed-loop toppling dynamics under `controller`."""
    inertia_tilt = _tilt_moment(inertia, tilt_axis)
    mgl = _gravity_torque_scale(inertia)

    def rhs(_t: float, y: Array) -> list[float]:
        theta, omega = y
        u = controller.torque(theta, omega)
        return [omega, (mgl * math.sin(theta) - u) / inertia_tilt]

    def hop_event(_t: float, y: Array) -> float:
        return float(abs(y[0]) - hop_onset_rad)

    hop_event.terminal = True  # type: ignore[attr-defined]

    t_eval = np.linspace(0.0, duration, n_samples)
    sol = solve_ivp(
        rhs,
        (0.0, duration),
        [initial_tilt_rad, initial_rate_rad_s],
        method="RK45",
        t_eval=t_eval,
        events=hop_event,
        rtol=1e-9,
        atol=1e-12,
    )
    theta = sol.y[0]
    omega = sol.y[1]
    torque = np.array([controller.torque(th, om) for th, om in zip(theta, omega, strict=True)])
    trunk_lean = controller.lean_per_torque * np.abs(torque)
    topple_time = float(sol.t_events[0][0]) if sol.t_events[0].size > 0 else None
    return ControlledResult(
        t=sol.t,
        theta=theta,
        torque=torque,
        trunk_lean=trunk_lean,
        controller=controller,
        hop_onset_rad=hop_onset_rad,
        topple_time=topple_time,
    )


def revolutions_with_control(
    inertia: RigidBodyInertia,
    controller: Controller,
    initial_tilt_deg: float,
    spin_rev_s: float = 1.7,
    tilt_axis: str = "ap",
    duration: float = 6.0,
) -> float:
    """Revolutions completed before toppling, or inf if the controller stabilises the turn."""
    res = integrate_controlled(
        inertia,
        controller,
        initial_tilt_rad=math.radians(initial_tilt_deg),
        duration=duration,
        tilt_axis=tilt_axis,
    )
    if res.stabilized:
        return float("inf")
    if res.topple_time is None:
        return float("inf")
    return spin_rev_s * res.topple_time


def stability_domain(
    inertia: RigidBodyInertia,
    controller: Controller,
    tilt_grid_deg: Array,
    rate_grid_deg_s: Array,
    tilt_axis: str = "ap",
    duration: float = 6.0,
) -> Array:
    """Boolean grid [len(tilt), len(rate)] of whether each initial state is stabilised.

    Traces the controller's basin of attraction in the (tilt, tilt-rate) plane.
    """
    out = np.zeros((len(tilt_grid_deg), len(rate_grid_deg_s)), dtype=bool)
    for i, th0 in enumerate(tilt_grid_deg):
        for j, w0 in enumerate(rate_grid_deg_s):
            res = integrate_controlled(
                inertia,
                controller,
                initial_tilt_rad=math.radians(float(th0)),
                initial_rate_rad_s=math.radians(float(w0)),
                duration=duration,
                tilt_axis=tilt_axis,
            )
            out[i, j] = res.stabilized
    return out
