"""Reproduce the Lott & Laws (2012) pirouette balance thresholds from the physics model.

Run: python examples/01_reproduce_lott_laws.py

Prints the whole-body inertia summary and the critical starting tilt for 1-3 revolutions,
which should match the paper's "sub-degree offset needed for > 2 turns" result. This is the
Tier-A deliverable: a from-first-principles reproduction with no motion-capture data.
"""
from __future__ import annotations

from pirouette_dynamics.anthropometry import build_body
from pirouette_dynamics.inertia import assemble_inertia
from pirouette_dynamics.topple import (
    BASELINE_SPIN_REV_S,
    HOP_ONSET_ANGLE_DEG,
    reproduce_lott_laws,
    revolutions_to_topple,
)


def main() -> None:
    print(f"Lott & Laws (2012) anchors: hop onset {HOP_ONSET_ANGLE_DEG} deg, "
          f"spin {BASELINE_SPIN_REV_S} rev/s\n")
    for posture in ("retire_arms_first", "retire_arms_second"):
        body = assemble_inertia(build_body(stature=1.593, mass=51.54, posture=posture))
        rep = reproduce_lott_laws(body)
        print(f"[{posture}]  (Lin 2019 experienced-dancer size: 159.3 cm, 51.54 kg)")
        print(f"  COM height l            = {rep.com_height:6.3f} m")
        print(f"  spin moment  I_zz       = {rep.spin_moment:6.3f} kg m^2   (Guilford ~1.72)")
        print(f"  tilt moment  I_tilt(AP) = {rep.tilt_moment:6.2f} kg m^2")
        print(f"  topple time-constant tau= {rep.tau:6.3f} s")
        print(f"  critical start, 1 rev   = {rep.critical_tilt_1rev_deg:6.3f} deg")
        print(f"  critical start, 2 rev   = {rep.critical_tilt_2rev_deg:6.3f} deg  <- <1 deg")
        print(f"  critical start, 3 rev   = {rep.critical_tilt_3rev_deg:6.3f} deg")
        print(f"  revolutions from 0.5deg = {revolutions_to_topple(body, 0.5):6.2f}\n")


if __name__ == "__main__":
    main()
