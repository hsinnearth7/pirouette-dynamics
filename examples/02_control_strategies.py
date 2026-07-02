"""Compare balance-control strategies for a pirouette (Tier B).

Run: python examples/02_control_strategies.py

Shows (1) how the ankle strategy's recoverable tilt is set by the base of support -- about a
degree on pointe, a few degrees on a flat foot -- and (2) how the hip strategy recovers larger
tilts at the cost of a visible trunk lean, mirroring the expert-ankle / novice-hip contrast in
Lin et al. (2019).
"""
from __future__ import annotations

import math

from pirouette_dynamics.anthropometry import build_body
from pirouette_dynamics.control import (
    ankle_controller,
    hip_controller,
    integrate_controlled,
    max_recoverable_tilt_deg,
    revolutions_with_control,
)
from pirouette_dynamics.inertia import assemble_inertia
from pirouette_dynamics.topple import revolutions_to_topple


def main() -> None:
    body = assemble_inertia(build_body())

    print("Ankle strategy: largest statically recoverable tilt vs base of support")
    for label, bos in (("pointe", 0.02), ("demi-pointe", 0.06), ("flat foot", 0.10)):
        mr = max_recoverable_tilt_deg(body, ankle_controller(body, base_of_support_m=bos))
        print(f"  {label:11s} (BOS {bos * 100:4.1f} cm) -> {mr:5.2f} deg")

    hip = hip_controller(body)
    print(f"  hip strategy            -> {max_recoverable_tilt_deg(body, hip):5.2f} deg (lean)\n")

    print("Revolutions before toppling from a 2.0 deg initial tilt (spin 1.7 rev/s):")
    passive = revolutions_to_topple(body, 2.0)
    pointe = revolutions_with_control(body, ankle_controller(body, 0.02), 2.0)
    demi = revolutions_with_control(body, ankle_controller(body, 0.06), 2.0)
    print(f"  passive (no control)     = {passive:6.2f}")
    print(f"  ankle, pointe BOS        = {pointe:6.2f}")
    print(f"  ankle, demi-pointe BOS   = {demi:6}")
    print(f"  hip strategy             = {revolutions_with_control(body, hip, 2.0):6}\n")

    print("Recovery of a 3 deg disturbance (peak tilt / peak trunk lean):")
    for name, ctrl in (("ankle demi", ankle_controller(body, 0.06)), ("hip", hip)):
        res = integrate_controlled(body, ctrl, math.radians(3.0))
        status = "stabilised" if res.stabilized else "TOPPLED"
        print(f"  {name:11s}: {status:10s}  peak tilt {res.peak_tilt_deg:4.1f} deg,"
              f" peak lean {res.peak_lean_deg:4.1f} deg")


if __name__ == "__main__":
    main()
