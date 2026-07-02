"""Segmental body model: De Leva (1996) masses + a parametric ballet posture.

What this module produces
-------------------------
`build_body(...)` returns a list of `BodySegment`, each carrying its mass, its 3D
centre-of-mass position, and its own inertia tensor (about its COM, in world axes).
`inertia.assemble_inertia` then combines them into a whole-body inertia tensor about a
chosen pivot (the supporting foot).

Modelling choices and honesty
-----------------------------
- Segment MASS fractions and longitudinal COM ratios are De Leva (1996) Table 4 values,
  ported verbatim from the sibling ``ballet-com-cop-analyzer`` project (they sum to 1.0
  per sex; see the test).
- Each segment's OWN inertia is modelled as a **uniform solid cylinder** defined by its two
  endpoint joints (axis + length) and a per-segment radius. This avoids transcribing De
  Leva's three-axis radii of gyration (a real error risk) while keeping the dominant term
  physically exact: the parallel-axis contribution ``m * d^2`` uses the true De Leva mass
  and COM position. For toppling about the foot, that parallel-axis term dominates, so the
  cylinder approximation of the (small) own-inertia is a documented, low-impact simplification.
  Replacing it with De Leva radii of gyration is listed as a future refinement in
  ``docs/LIMITATIONS.md``.
- Joint positions come from a parametric skeleton scaled by stature, using standard
  anthropometric segment-height ratios (Winter, *Biomechanics and Motor Control of Human
  Movement*, 4th ed.). The retire posture places the gesture leg drawn up to the support
  knee and the arms rounded, i.e. a canonical single-support turning position.

All maths is pure numpy so it is testable with no external data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Vec3 = NDArray[np.float64]
Mat3 = NDArray[np.float64]

GRAVITY: float = 9.81  # m / s^2


@dataclass(frozen=True)
class _SegmentSpec:
    """Static De Leva parameters + a cylinder radius for one anatomical segment."""

    name: str
    proximal: str
    distal: str
    mass_m: float  # mass fraction of body mass, male (De Leva 1996 Table 4)
    mass_f: float  # mass fraction of body mass, female
    com_m: float  # COM position from proximal, fraction of segment length, male
    com_f: float  # ... female
    radius_frac_stature: float  # cylinder radius as a fraction of stature (own-inertia only)
    bilateral: bool


# De Leva (1996) Table 4. Mass = %body mass / 100; COM ratio = %segment length / 100 from
# proximal. Mass + COM columns are ported from ballet-com-cop-analyzer/balletcomcop/
# anthropometry.py. `radius_frac_stature` is an anthropometric segment half-thickness used
# only for the (sub-dominant) own-inertia cylinder; see module docstring.
_SEGMENTS: tuple[_SegmentSpec, ...] = (
    _SegmentSpec("head", "mid_ear", "nose",
                 0.0694, 0.0668, 0.500, 0.500, 0.070, False),
    _SegmentSpec("trunk", "mid_shoulder", "mid_hip",
                 0.4346, 0.4257, 0.5138, 0.4964, 0.090, False),
    _SegmentSpec("upper_arm", "{s}_shoulder", "{s}_elbow",
                 0.0271, 0.0255, 0.5772, 0.5754, 0.033, True),
    _SegmentSpec("forearm_hand", "{s}_elbow", "{s}_wrist",
                 0.0223, 0.0194, 0.4574, 0.4559, 0.028, True),
    _SegmentSpec("thigh", "{s}_hip", "{s}_knee",
                 0.1416, 0.1478, 0.4095, 0.3612, 0.055, True),
    _SegmentSpec("shank", "{s}_knee", "{s}_ankle",
                 0.0433, 0.0481, 0.4395, 0.4352, 0.040, True),
    _SegmentSpec("foot", "{s}_ankle", "{s}_foot_index",
                 0.0137, 0.0129, 0.4415, 0.4014, 0.030, True),
)


@dataclass(frozen=True)
class BodySegment:
    """One assembled segment: mass (kg), world COM (m), own inertia tensor about its COM."""

    name: str
    mass: float
    com: Vec3
    inertia_com: Mat3


def total_mass_fraction(sex: str) -> float:
    """Sum of all segment mass fractions (~1.0). Ported check from the analyzer."""
    total = 0.0
    for seg in _SEGMENTS:
        m = seg.mass_m if sex == "male" else seg.mass_f
        total += m * (2 if seg.bilateral else 1)
    return total


# --- Parametric skeleton -------------------------------------------------------------
# Joint positions in a frame with origin at the floor under the support foot:
#   x = anterior (AP, forward +), y = lateral toward the gesture leg (ML), z = up.
# Heights are Winter's standing anthropometric ratios of stature H; the retire posture then
# lifts and turns out the gesture leg and rounds the arms. Support leg = right (near axis),
# gesture leg = left (+y).


def _retire_keypoints(stature: float) -> dict[str, Vec3]:
    """Right-support retire with rounded arms. Returns name -> 3D point (m)."""
    h = stature

    def p(x: float, y: float, z: float) -> Vec3:
        return np.array([x * h, y * h, z * h], dtype=float)

    kp: dict[str, Vec3] = {
        # support (right) leg: essentially vertical, close to the spin axis
        "right_ankle": p(0.0, -0.02, 0.039),
        "right_knee": p(0.0, -0.02, 0.285),
        "right_hip": p(0.0, -0.05, 0.530),
        # gesture (left) leg in retire: thigh turned out, foot drawn to the support knee
        "left_hip": p(0.0, 0.05, 0.530),
        "left_knee": p(0.02, 0.20, 0.470),
        "left_ankle": p(0.02, 0.05, 0.315),
        "left_foot_index": p(0.02, 0.0, 0.280),
        "right_foot_index": p(0.10, -0.02, 0.0),  # support toe on the floor (pivot side)
        # trunk / shoulders / head, stacked on the vertical axis
        "left_shoulder": p(0.0, 0.10, 0.818),
        "right_shoulder": p(0.0, -0.10, 0.818),
        "left_ear": p(0.0, 0.03, 0.930),
        "right_ear": p(0.0, -0.03, 0.930),
        "nose": p(0.02, 0.0, 0.930),
        # arms rounded low in front (first position): moderate radius from the axis
        "left_elbow": p(0.10, 0.18, 0.700),
        "right_elbow": p(0.10, -0.18, 0.700),
        "left_wrist": p(0.15, 0.05, 0.620),
        "right_wrist": p(0.15, -0.05, 0.620),
    }
    return kp


def _arms_second_keypoints(stature: float) -> dict[str, Vec3]:
    """Retire but with arms opened to the side (second position): larger spin-axis inertia."""
    kp = _retire_keypoints(stature)
    h = stature
    kp["left_elbow"] = np.array([0.02 * h, 0.28 * h, 0.790 * h])
    kp["right_elbow"] = np.array([0.02 * h, -0.28 * h, 0.790 * h])
    kp["left_wrist"] = np.array([0.02 * h, 0.42 * h, 0.760 * h])
    kp["right_wrist"] = np.array([0.02 * h, -0.42 * h, 0.760 * h])
    return kp


_POSTURES = {
    "retire_arms_first": _retire_keypoints,
    "retire_arms_second": _arms_second_keypoints,
}


def list_postures() -> tuple[str, ...]:
    """Names accepted by `build_body(posture=...)`."""
    return tuple(_POSTURES)


def _derive_midpoints(kp: dict[str, Vec3]) -> dict[str, Vec3]:
    out = dict(kp)
    out["mid_shoulder"] = 0.5 * (kp["left_shoulder"] + kp["right_shoulder"])
    out["mid_hip"] = 0.5 * (kp["left_hip"] + kp["right_hip"])
    out["mid_ear"] = 0.5 * (kp["left_ear"] + kp["right_ear"])
    return out


def _cylinder_inertia_com(mass: float, length: float, radius: float, axis: Vec3) -> Mat3:
    """Inertia tensor of a uniform solid cylinder about its COM, in world axes.

    Axial moment I_a = 1/2 m r^2; transverse I_t = m (L^2/12 + r^2/4). The world tensor is
    I_t (Identity - uu^T) + I_a uu^T, where u is the unit axis.
    """
    norm = float(np.linalg.norm(axis))
    u = np.array([0.0, 0.0, 1.0]) if norm == 0.0 else axis / norm
    i_axial = 0.5 * mass * radius**2
    i_trans = mass * (length**2 / 12.0 + radius**2 / 4.0)
    uut = np.outer(u, u)
    return i_trans * (np.eye(3) - uut) + i_axial * uut


def build_body(
    stature: float = 1.593,
    mass: float = 51.54,
    sex: str = "female",
    posture: str = "retire_arms_first",
) -> list[BodySegment]:
    """Assemble the segmented body for a given size and posture.

    Defaults are Lin et al. (2019) experienced-dancer means (159.3 cm, 51.54 kg, female),
    so results line up with that paper's cohort.

    Parameters
    ----------
    stature : standing height in metres.
    mass : body mass in kg.
    sex : "male" or "female" (selects the De Leva column).
    posture : one of `list_postures()`.
    """
    if sex not in ("male", "female"):
        raise ValueError(f"sex must be 'male' or 'female', got {sex!r}")
    if posture not in _POSTURES:
        raise ValueError(f"posture must be one of {list_postures()}, got {posture!r}")

    kp = _derive_midpoints(_POSTURES[posture](stature))
    segments: list[BodySegment] = []
    for spec in _SEGMENTS:
        m_frac = spec.mass_m if sex == "male" else spec.mass_f
        ratio = spec.com_m if sex == "male" else spec.com_f
        radius = spec.radius_frac_stature * stature
        sides = ("left", "right") if spec.bilateral else (None,)
        for s in sides:
            prox = spec.proximal.format(s=s) if s else spec.proximal
            dist = spec.distal.format(s=s) if s else spec.distal
            p = kp[prox]
            d = kp[dist]
            com = p + ratio * (d - p)
            length = float(np.linalg.norm(d - p))
            seg_mass = m_frac * mass
            inertia = _cylinder_inertia_com(seg_mass, length, radius, d - p)
            name = f"{spec.name}_{s}" if s else spec.name
            segments.append(BodySegment(name=name, mass=seg_mass, com=com, inertia_com=inertia))
    return segments
