"""High-level pirouette-balance analyses that reproduce Lott & Laws (2012).

Key results this reproduces (JDMS 16(4):167-174, and the deep-notes on disk):
- A rigid dancer must begin within a **sub-degree** offset of vertical to complete more than
  ~2 revolutions before toppling.
- Dancers hop to recover once the body tilt reaches ~**9.3 deg** from vertical.
- Baseline spin rate ~**1.7 rev/s**.

The point of the module: turn these published numbers into functions whose output can be
asserted against the paper, so the physics core is validated with no motion-capture data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from pirouette_dynamics.inertia import RigidBodyInertia
from pirouette_dynamics.rigidbody import integrate_topple, topple_time_constant

HOP_ONSET_ANGLE_DEG: float = 9.3  # Lott & Laws (2012) mean hop-onset tilt
BASELINE_SPIN_REV_S: float = 1.7  # Lott & Laws (2012) mean spin rate


def revolutions_to_topple(
    inertia: RigidBodyInertia,
    initial_tilt_deg: float,
    spin_rev_s: float = BASELINE_SPIN_REV_S,
    tilt_axis: str = "ap",
    hop_onset_deg: float = HOP_ONSET_ANGLE_DEG,
    max_duration: float = 10.0,
) -> float:
    """Revolutions completed before the tilt reaches `hop_onset_deg`.

    Integrates the toppling model, finds the time to hop onset, and multiplies by the spin
    rate. Returns `inf` if the body never reaches the threshold within `max_duration`.
    """
    result = integrate_topple(
        inertia,
        initial_tilt_rad=np.deg2rad(initial_tilt_deg),
        duration=max_duration,
        tilt_axis=tilt_axis,
        hop_onset_rad=np.deg2rad(hop_onset_deg),
    )
    if result.topple_time is None:
        return float("inf")
    return spin_rev_s * result.topple_time


def critical_initial_tilt_deg(
    inertia: RigidBodyInertia,
    target_revolutions: float,
    spin_rev_s: float = BASELINE_SPIN_REV_S,
    tilt_axis: str = "ap",
    hop_onset_deg: float = HOP_ONSET_ANGLE_DEG,
) -> float:
    """Largest starting tilt (deg) that still completes `target_revolutions` before hopping.

    This is the Lott & Laws headline: for `target_revolutions > 2` the answer is < 1 deg.
    Solved in closed form from the small-angle law theta_hop = theta0 * cosh(T / tau), which
    is accurate here because theta0 is sub-degree (well inside the linear regime).
    """
    tau = topple_time_constant(inertia, tilt_axis=tilt_axis)
    turn_duration = target_revolutions / spin_rev_s
    theta0_rad = np.deg2rad(hop_onset_deg) / np.cosh(turn_duration / tau)
    return float(np.rad2deg(theta0_rad))


def initial_tilt_for_revolutions_numeric(
    inertia: RigidBodyInertia,
    target_revolutions: float,
    spin_rev_s: float = BASELINE_SPIN_REV_S,
    tilt_axis: str = "ap",
    hop_onset_deg: float = HOP_ONSET_ANGLE_DEG,
) -> float:
    """Same as `critical_initial_tilt_deg` but solved by root-finding on the full nonlinear
    integration. Used to confirm the small-angle closed form is faithful."""

    def gap(theta0_deg: float) -> float:
        return (
            revolutions_to_topple(
                inertia, theta0_deg, spin_rev_s, tilt_axis, hop_onset_deg
            )
            - target_revolutions
        )

    # Larger initial tilt -> fewer revolutions, so gap is monotonically decreasing.
    return float(brentq(gap, 1e-4, hop_onset_deg - 1e-3, xtol=1e-6))


@dataclass(frozen=True)
class LottLawsReport:
    """Reproduced thresholds for one body/spin configuration."""

    tau: float
    com_height: float
    tilt_moment: float
    spin_moment: float
    critical_tilt_1rev_deg: float
    critical_tilt_2rev_deg: float
    critical_tilt_3rev_deg: float


def reproduce_lott_laws(
    inertia: RigidBodyInertia,
    spin_rev_s: float = BASELINE_SPIN_REV_S,
    tilt_axis: str = "ap",
) -> LottLawsReport:
    """Bundle the reproduced numbers for reporting in notebook 01."""
    return LottLawsReport(
        tau=topple_time_constant(inertia, tilt_axis=tilt_axis),
        com_height=inertia.com_height,
        tilt_moment=inertia.tilt_moment_ap if tilt_axis == "ap" else inertia.tilt_moment_ml,
        spin_moment=inertia.spin_moment,
        critical_tilt_1rev_deg=critical_initial_tilt_deg(inertia, 1.0, spin_rev_s, tilt_axis),
        critical_tilt_2rev_deg=critical_initial_tilt_deg(inertia, 2.0, spin_rev_s, tilt_axis),
        critical_tilt_3rev_deg=critical_initial_tilt_deg(inertia, 3.0, spin_rev_s, tilt_axis),
    )
