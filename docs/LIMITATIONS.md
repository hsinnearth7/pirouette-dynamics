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

## Tier B (balance control)

- **Single tilt degree of freedom.** The controlled body still has one tilt DOF. The hip
  strategy's trunk lean is reported as a *cost readout* (`lean_per_torque * torque`), not a
  second simulated link; a genuine two-link ankle+hip model is future work.
- **Base-of-support values are representative.** `r_bos` for pointe/demi/flat (2/6/10 cm) are
  plausible anterior-posterior half-extents, not a measured pressure map, and the AP base is
  treated as symmetric (real feet are longer forward than back).
- **Phase caveat on the Lin 2019 mapping.** Lin's expert-ankle / novice-hip finding is
  specific to the landing phase, where the foot is grounded. The model's ankle-vs-hip
  contrast is *consistent with* that observation but is not a claim about mid-pointe control.
- **Controller, not muscle.** The restoring torque is an abstract joint torque with PD gains,
  not a muscle model; mapping it to muscles/EMG is the Tier D synergy step.

## Tier C (learned control)

- **Reduced 2-axis model.** The learned agent controls two tilt axes plus spin and arm
  retraction; it is not a full articulated body. The AP-vs-ML asymmetry is imposed partly
  through the disturbance (AP is perturbed more than ML) and partly through the reward weight,
  matching Lin's observation rather than deriving it from first principles.
- **"Emergence" is consistency, not proof.** That the AP-weighted policy lands near Lin's
  experienced-dancer inclination is evidence the reward and physics are set up sensibly, not a
  demonstration that real dancers optimise this reward. Claims in `examples/03` are phrased as
  "consistent with".
- **Single seed reported.** The example is deterministic at seed 0; CEM outcomes vary with
  seed and budget. The headline numbers are one reproducible run, not a mean over seeds with
  confidence intervals.
- **Simulated data only.** As in Tiers A-B, there is no empirical fit; validation is against
  Lin's *published* group statistics, which needs no motion capture.

## Data and validation

- **Simulation generates its own data; there is no empirical fit here.** Validation is by
  reproducing *published* thresholds (Lott & Laws 2012; Guilford), not by fitting recorded
  motion.
- The planned Lin 2019 expert-vs-novice comparison (Tier B/C) targets **published group
  statistics**, not individual capture, and needs no video.
- Any future video cross-check would be a single-subject, 2D-markerless illustration, not a
  ground-truth validation; it is optional and explicitly not the validation backbone.
