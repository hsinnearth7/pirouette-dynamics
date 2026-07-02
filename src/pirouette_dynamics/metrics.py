"""Balance metrics, shared in definition with the measurement tool ballet-com-cop-analyzer.

The COM-COP inclination angle (Lin, Su & Lin 2019, Eq. 2/3) is the angle between vertical
and the line from the centre of pressure to the centre of mass. Lin argues it, not raw COP
displacement, is the meaningful stability measure for a dynamic turn; it is also the quantity
the sibling markerless-video analyzer computes. Expressing the simulator in the same unit is
what lets a simulated turn be compared with Lin's published expert/novice numbers.
"""
from __future__ import annotations

import numpy as np

from pirouette_dynamics.anthropometry import GRAVITY
from pirouette_dynamics.inertia import RigidBodyInertia


def com_cop_inclination_deg(
    inertia: RigidBodyInertia,
    theta_rad: float | np.ndarray,
    ankle_torque: float | np.ndarray = 0.0,
) -> np.ndarray:
    """COM-COP inclination angle (deg) for a body tilted by `theta_rad`.

    In the reduced inverted-pendulum model the COM sits a horizontal distance ``l*sin(theta)``
    ahead of the pivot, at height ``l*cos(theta)``. An ankle torque shifts the COP under the
    foot by ``ankle_torque / (M g)``, reducing the COM-COP offset. With no ankle torque the
    inclination equals the tilt angle; a fully supporting ankle torque drives it to zero.
    """
    length = inertia.com_height
    mg = inertia.mass * GRAVITY
    theta = np.asarray(theta_rad, dtype=float)
    x_com = length * np.sin(theta)
    x_cop = np.asarray(ankle_torque, dtype=float) / mg
    height = length * np.cos(theta)
    inclination: np.ndarray = np.rad2deg(np.arctan2(x_com - x_cop, height))
    return inclination
