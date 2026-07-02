# Limitations (read before citing any number)

The model is deliberately reduced. Its results hold at that level of abstraction and no
finer.

## Tier A (physics core)

- **Single rigid body.** The dancer is one rigid body pivoting at the foot. Real dancers are
  articulated and, per Lin 2019 and Lott & Laws, actively adjust segments during the turn;
  that is exactly what Tiers B–C add. Tier-A numbers describe the *passive rigid* case only.
- **Own-inertia via uniform cylinders.** Each segment's own inertia is a uniform-cylinder
  approximation from its endpoint joints and an anthropometric radius, not De Leva's
  three-axis radii of gyration. The dominant parallel-axis term uses exact De Leva mass and
  COM, so toppling (`I_tilt`) is affected little; the spin moment `I_zz` is more sensitive to
  it. Replacing cylinders with De Leva radii of gyration is a planned refinement.
- **Parametric posture, not measured.** Joint positions come from a stature-scaled skeleton
  (Winter ratios) in one canonical retire pose, not from a specific dancer's capture. COM
  height and inertia are therefore representative, not individual.
- **Slow-spin (non-gyroscopic) topple.** Gyroscopic coupling is dropped because a 1.7 rev/s
  pirouette is far from the sleeping-top regime (see PHYSICS.md §2). For much faster turns
  (>3 rev/s) the coupling grows and the full Euler-angle top should be used.
- **Pivot idealised as a fixed point.** Foot roll, relevé/pointe contact geometry, and floor
  friction limits are not yet modelled (friction enters Tier B as the ankle-torque bound).
- **No head "spotting", no arm dynamics in the topple.** Arm position enters only through the
  inertia tensor, not as a time-varying recovery action (that is Tier B/C).

## Data and validation

- **Simulation generates its own data; there is no empirical fit here.** Validation is by
  reproducing *published* thresholds (Lott & Laws 2012; Guilford), not by fitting recorded
  motion.
- The planned Lin 2019 expert-vs-novice comparison (Tier B/C) targets **published group
  statistics**, not individual capture, and needs no video.
- Any future video cross-check would be a single-subject, 2D-markerless illustration, not a
  ground-truth validation; it is optional and explicitly not the validation backbone.
