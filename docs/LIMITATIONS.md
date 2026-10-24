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
  friction are not modelled. Tier B bounds the ankle torque by the base of support (the COP
  must stay within `r_bos`), which is a geometric bound, not a friction limit; horizontal
  ground-reaction force, which a real hip strategy needs, is not modelled anywhere.
- **Smaller simplifications that do not change the conclusions.** The COM sits 1.8 cm AP /
  2.4 cm ML off the pivot at theta = 0, but theta = 0 is treated as balanced; `I_zz` is taken
  about the vertical through the pivot, not the COM (about 4 % high); `I_tilt` and `m g l` are
  fixed at the arms-first posture while the arm state varies; the forearm-hand segment uses the
  forearm-only COM ratio while carrying the hand mass; the support foot is flat (pivot under the
  ankle) with no relevé geometry, although `r_bos` is labelled demi-pointe.
- **No head "spotting", no arm dynamics in the topple.** Arm position enters only through the
  inertia tensor, not as a time-varying recovery action (that is Tier B/C).

## Tier B (balance control)

- **Single tilt degree of freedom.** The controlled body still has one tilt DOF. The hip
  strategy's trunk lean is reported as a *cost readout* (`lean_per_torque * torque`), not a
  second simulated link; a genuine two-link ankle+hip model is future work.
- **Hip torque acts as an external moment.** `env.py` sums `ankle_ap + hip_ap` into one
  restoring moment about the foot pivot of the single rigid body (`control.py` treats the hip
  capacity the same way). In a real body the hip torque is internal: trunk counter-rotation
  can correct balance only transiently and cannot hold the whole-body COM outside the base of
  support. The hip capacity (0.35 m g l) and the lean cost (9 deg at capacity) are assumed
  defaults with no cited source. The 20.49 deg hip ceiling and every absolute ankle-vs-hip
  torque comparison therefore follow from these assumed values, not from biomechanics. The
  combined restoring moment can reach tau_bos + tau_hip = M g (0.06 + 0.327 m), equivalent to a
  COP 39 cm from the pivot, while the inclination metric credits only the ankle part (COP within
  6 cm). With hip help, the reported inclination can therefore sit far below what the restoring
  forces imply.
- **Base-of-support values are representative and unsourced.** `r_bos` for pointe/demi/flat
  (2/6/10 cm) are assumed anterior-posterior half-extents with no cited source, not a measured
  pressure map, and the AP base is treated as symmetric (real feet are longer forward than
  back). The env default 6 cm (demi-pointe) implies a 12 cm symmetric COP range, which may be
  long for a metatarsal-to-toe contact, and the ML axis uses the same 6 cm. Like the 0.35 hip
  capacity, it sets the torque-unit ankle/hip ratio: torque ratio = command ratio x r_bos /
  (0.35 l), so at r_bos = 3 cm the in-band torque ratio would be about 0.23 to 0.31.
- **Phase caveat on the Lin 2019 mapping.** Lin et al. only suggested a hip strategy for
  novices in the ending phase of the turn, without measuring it. There is no measured
  expert-ankle / novice-hip result to reproduce, so the model's ankle-vs-hip contrast is a
  modelling hypothesis and not a claim about mid-pointe control.
- **Controller, not muscle.** The restoring torque is an abstract joint torque with PD gains,
  not a muscle model; mapping it to muscles/EMG is the Tier D synergy step.

## Tier C (learned control)

- **Reduced 2-axis model.** The learned agent controls two tilt axes plus spin and arm
  retraction; it is not a full articulated body. The AP-vs-ML asymmetry is imposed partly
  through the disturbance (initial tilt is drawn from +-3 deg in AP versus +-0.5 deg in ML) and
  partly through the reward weight, matching Lin's observation rather than deriving it from
  first principles. AP dominance is therefore partly imposed, not an independent emergence.
- **The Lin 2019 reference is a preparatory-phase range, not a single-leg peak.** The 2.77 /
  5.62 deg values (and the 4.20 deg midpoint) are Lin's *range* of COM-COP AP inclination in the
  preparatory, double-support (PRE) phase. Lin reports the single-leg turning phases as
  COM-*ankle* inclination instead. The simulation compares a per-episode *peak* |inclination|
  during single-leg spinning against that number, so the comparison mixes phase and statistic.
  The ML gap is large: Lin's ML values are 0.03 to 0.17 deg, while simulated ML inclination is
  2.5 to 6.4 deg, more than 20x higher. Sim values are "numerically comparable to a
  preparatory-phase range", not "expert-level".
- **Tilt axes have no rotating-frame terms.** AP and ML are integrated as independent planar
  pendulums, so the labels describe the dancer's orientation at turn onset only.
- **ML inclination stays higher than AP.** In the confirmatory sweep ML inclination (weight
  fixed at 0.3; the ML axis has an ankle channel only) exceeds AP inclination at every weight
  except 0.3: 2.55 to 4.62 deg in the band [0.6, 3.0], above the 4.20 deg threshold at weight
  1.5. Low AP inclination comes with higher inclination on the less penalised axis.
- **Evaluation resets are not held out.** `cem_train` averages fitness over reset seeds 0-4;
  `examples/03` and `examples/04` evaluate on seeds 0-7, so 5 of the 8 evaluation starts repeat
  training starts and only seeds 5-7 are unseen. A post-hoc held-out re-evaluation
  (`examples/05`, reset seeds 100-107, added 2026-10-15 and not part of the analysis plan) shows
  that this matters: policies topple on 28 to 55 % of held-out turns in the band weights (81 %
  at 5.0), survivors drop to 3-6 of 8, and the H2 band collapses to the single weight 0.8, so
  **H2 is not confirmed on held-out starts**. The AP means move little (1.57 to 2.47 deg for
  weights 0.6-3.0). The CEM fitness uses only 5 fixed resets, which likely causes this
  overfitting.
- **Survival and means.** A policy "survives" if it topples on at most 4 of its 8 evaluation
  resets, so a survivor may still fall on half of its turns. Even on reset seeds 0-7 the mean
  topple fraction per weight is 0.12 to 0.36 (0.75 at weight 5.0); see
  `docs/SWEEP_RESULTS_CONFIRMATORY_2026-07-04.md`. The sweep's inclination means and CIs include
  all 8 trained policies.
- **What the CIs cover.** The 95 % CIs are t-intervals over the 8 CEM training seeds, with every
  policy evaluated on the same fixed reset states, so they cover training-seed variance
  conditional on those states and leave out initial-condition sampling. There is no
  multiplicity correction across the 9 weights and 3 rules. H2 uses point means with no CI, and
  the band is chosen from the data before H1c is tested within it.
- **"Emergence" is consistency, not proof.** That the AP-weighted policy's AP inclination is
  numerically comparable to Lin's experienced-dancer preparatory-phase range is evidence the reward and physics are set up sensibly, not a
  demonstration that real dancers optimise this reward. Claims in `examples/03` are phrased as
  "consistent with".
- **Seeds.** The `examples/03` table is one deterministic run at seed 0; CEM outcomes vary
  with seed and budget. Seed-level evidence comes from the 8-seed confirmatory sweep
  (`examples/04`, `docs/SWEEP_RESULTS_CONFIRMATORY_2026-07-04.md`), which gives mean and
  t-based 95% CI per reward weight. The symmetric-reward ablation (3.6 deg, one seed) is
  below the 4.20 deg expert/novice midpoint, but the 8-seed equivalent (`aesthetic_ap` 0.3:
  3.91 +- 2.37 deg, CI 1.53 to 6.28) spans the novice value 5.62 deg, so it does not show a
  move toward novice behaviour.
- **Robust band is [0.6, 3.0], 5x, on reset seeds 0-7 only.** The earlier [0.3, 3.0] / 10x
  figure came from verdict code that ignored the contiguity rule (weight 0.4 survives only 5/8).
  The lowest H1c lower CI in band is 1.52 in command units. On held-out resets 100-107 the band
  is [0.8, 0.8] (span 1x); H1c (1.31) and the floor (1.93 deg) still pass there.
- **Ankle/hip ratio is command utilisation.** `ankle_hip_ratio_*` is mean|a_ankle| / mean|a_hip|
  on normalised commands. The environment scales ankle by `m g r_bos` and hip by
  `0.35 m g l` (capacity ratio about 5.45), so in torque units the in-band ratio is 0.46 to
  0.63 (lowest lower CI 0.28): the hip produces the larger mean absolute torque, a direction
  that follows from the assumed hip capacity 0.35 and r_bos = 6 cm, both unsourced (below about
  0.16 m g l the same commands would read as ankle-dominant). No ankle torque dominance is claimed. The command ratio is also shaped by
  the reward: the penalised inclination uses a COP that only the ankle shifts
  (`metrics.py`, `env.py` passes `ankle_ap`/`ankle_ml` only), and the lean term penalises hip
  command, so ankle preference is partly built in, not untargeted emergence.
- **Reference thresholds are inputs.** 9.3 deg hop onset and the 1.7 rev/s spin rate are taken
  from Lott & Laws (2012); the 0.498 deg critical start is a consistency check that inherits
  them, not a first-principles derivation.
- **Analysis plan not externally registered.** `docs/PREREGISTRATION_idea1_2026-07-04.md` was
  drafted before the confirmatory run but committed together with the results; no timestamped
  external registry entry exists. Treat it as a documented plan, not a registered protocol.
- **PPO cross-check not run.** No PPO result exists; learner-independence is untested.
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
