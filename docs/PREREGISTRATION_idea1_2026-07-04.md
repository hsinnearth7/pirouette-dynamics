# Pre-registration — confirmatory run (RL emergence of the expert pirouette strategy)

> The 2026-07-03 reward-weight sensitivity sweep was an exploratory pilot. This document locks
> the design, the target vs held-out distinction, and the falsifiable decision rules BEFORE the
> confirmatory run, so the result cannot be dismissed as post-hoc tuning. Any departure from
> this design is recorded as a deviation in section 7.
> Date: 2026-07-04. Status: confirmatory design LOCKED before the confirmatory run. Learner CEM
> (primary) + PPO cross-check. No new human-subject data (bucket A). Metric: whole-body two-point
> COM-COP inclination (COM vs ankle-torque-driven COP; `metrics.com_cop_inclination_deg`), which is
> NOT a bare single-pendulum tilt (the COP-shift term distinguishes it; reduced single-tilt-DOF model).

## 1. Hypotheses
- **H1 (emergence)**: A reward-only, imitation-free policy trained in `PirouetteEnv` reproduces the
  expert-dancer balance signature: (a) emergent AP COM-COP inclination near Lin et al. (2019)
  experienced value (2.77 deg) and below novice (5.62 deg); (b) AP-dominant control; (c) ankle-dominant
  torque usage.
- **H2 (robustness)**: H1(a) holds across a BROAD band of the reward AP-weight, not only at a tuned value.

## 2. Design (LOCKED)
- Env: `PirouetteEnv` defaults; `aesthetic_ml` fixed 0.3; sweep `aesthetic_ap` over
  {0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0}.
- Learner: CEM (dependency-free), 25 iterations, population 48, **N = 8 training seeds** per grid point
  (primary). **PPO (stable-baselines3) cross-check** at `aesthetic_ap` in {0.3, 1.0}, 3 seeds each.
- Evaluation: each trained policy on 8 held-out reset seeds.
- Reported per grid point: emergent AP inclination (mean, t-based 95% CI across the 8 train seeds),
  ML inclination, revolutions, **ankle/hip torque-usage ratio**, survival rate (not-toppled / 8).

## 3. Target vs held-out quantities (the emergence logic)
- **TARGET (in the reward — a low value is NOT evidence of emergence):** AP inclination (penalized via
  `aesthetic_ap`), ML inclination (`aesthetic_ml`), trunk lean (`lean`), revolutions (`turn`).
- **HELD-OUT (NOT in the reward — the genuine emergence evidence):**
  - **H-a**: the VALUE of the AP-inclination floor and whether it coincides with the human expert
    (2.77 deg). The reward drives AP inclination toward 0, so a floor at ~2.77 (not 0) is a
    physics-limited emergent property, not a target.
  - **H-b**: the ankle-vs-hip torque-usage ratio. The reward penalizes hip lean but does NOT target the
    ankle/hip split; emergent ankle-dominance (ratio > 1) is held-out.
  - **H-c**: robustness — whether the expert-side AP inclination holds across a broad weight band rather
    than monotonically tracking the weight toward 0.

## 4. Pre-specified decision rules (thresholds fixed BEFORE the run)
- **H2 robustness CONFIRMED** iff there is a contiguous `aesthetic_ap` band spanning at least a 3x range
  in which (i) mean emergent AP inclination <= 4.20 deg (the experienced-novice midpoint
  (2.77 + 5.62) / 2) AND (ii) survival >= 6/8. Otherwise not confirmed.
- **H1c ankle-dominance CONFIRMED** iff, within that band, the mean ankle/hip torque-usage ratio > 1
  with its 95% CI excluding 1.
- **Non-baked-in floor CONFIRMED** iff emergent AP inclination does NOT decrease monotonically toward 0
  across the high-weight arm (it plateaus at a floor > 1 deg for the top three weights), evidencing a
  physics limit rather than reward-tracking.
- **PPO cross-check PASSES** iff PPO at `aesthetic_ap` = 1.0 yields mean AP inclination in [1.0, 3.5] deg
  (consistent with the CEM expert-side result), confirming the finding is learner-independent.

## 5. Falsification
- H2 falsified if the expert-side band is narrower than a 3x range, or survival is poor across it.
- H1c falsified if hip usage >= ankle usage in the band.
- Floor falsified if AP inclination -> ~0 monotonically with weight (reward-tracking, not a physics floor).

## 6. Analysis (fixed)
- Mean and t-based 95% CI across the 8 training seeds per grid point; survival as a count. No post-hoc
  addition of weights, metrics, or seeds; no metric redefinition.
- The 2026-07-03 pilot (3 seeds) is exploratory and reported as such; only the 8-seed confirmatory run
  supports the confirmatory claims here.

## 7. Deviations
- **2026-10-15 (POST-HOC held-out re-evaluation; NOT in the analysis plan)**: added after the
  results were known, in response to the entry below on overlapping resets, so it is a
  post-hoc analysis and not a confirmatory test. `examples/05_heldout_reevaluation.py` retrains
  the same 9 weights x 8 CEM seeds (seeded CEM; policies saved to
  `results/confirmatory_policies.json`; on reset seeds 0-7 they reproduce the confirmatory
  survivor counts) and evaluates each policy on reset seeds 100-107, which the optimiser never
  saw (n = 8, like section 2). Results in `results/reward_sweep_heldout.csv`. The section-4 rules
  applied unchanged give: **H2 NOT confirmed** on held-out starts (widest contiguous band
  [0.8, 0.8], span 1.0x; survivors per weight 5, 4, 3, 6, 4, 3, 6, 4, 1 of 8; mean topple
  fraction 0.42, 0.44, 0.48, 0.33, 0.48, 0.55, 0.28, 0.42, 0.81 versus 0.12 to 0.36 (0.75 at 5.0)
  on seeds 0-7). H1c command-ratio lower CI in that band = 1.31 (> 1, still met as command
  utilisation; the torque-unit caveat below applies). Floor: min AP over surviving weights
  >= 1.0 = 1.93 deg (> 1, still met). Held-out AP means (4.24, 3.09, 2.47, 2.11, 1.94, 1.57,
  1.93, 1.86, 2.51 deg) are close to the confirmatory values, so the AP level is stable while
  the survival leg of H2 is not. With 32 held-out resets (100-131, `--extended`) the band is
  [2.0, 2.0], H1c 1.64, floor 1.96, same conclusions. The 2026-07-04 H2 verdict therefore
  depends on the overlap between training and evaluation resets.
- **2026-10-15 (evaluation resets not held out; wording correction, no new data)**: section 2
  promised evaluation on 8 held-out reset seeds. The run used `cem_train` with its default
  `eval_seeds = (0, 1, 2, 3, 4)` for the fitness and evaluated on `EVAL_SEEDS = range(8)`, so
  5 of the 8 evaluation starts repeat training starts and only seeds 5-7 are unseen. The sweep
  was not rerun; results are reported as "evaluated on reset seeds 0-7". The post-hoc
  held-out re-evaluation in the entry above retrains the policies and saves them.
- **2026-10-15 (H-b is not untargeted)**: section 3 lists the ankle/hip split as held-out from
  the reward. It is not: the penalised COM-COP inclination uses a COP shifted only by the ankle
  torque (`metrics.com_cop_inclination_deg`, called with `ankle_ap`/`ankle_ml`), and the lean
  term penalises every unit of hip command. Ankle-favouring command use is therefore partly
  built into the reward, and H1c is not evidence of untargeted emergence.
- **2026-10-15 (hip model assumption)**: the hip torque is summed with the ankle torque as a
  restoring moment about the foot pivot of the single rigid body, with an assumed capacity of
  0.35 m g l and 9 deg of lean at capacity (no cited source). Torque-unit ankle/hip comparisons
  depend on these values.
- **2026-10-15 (floor rule implementation)**: section 4 states the floor rule for "the top
  three weights" (2.0, 3.0, 5.0). The verdict code takes the minimum AP inclination over
  surviving weights >= 1.0 (1.0, 1.5, 2.0, 3.0; 5.0 fails survival). The verdict is the same
  either way (1.72, 1.71, 2.41 and 1.45 are all > 1 deg).
- **2026-10-15 (ML inclination, reporting addition)**: ML inclination exceeds AP at every
  weight except 0.3 (2.55 to 4.62 deg in band; above 4.20 at weight 1.5). Manuscripts now
  report it alongside AP.
- **2026-10-15 (provenance statement)**: this plan was drafted before the confirmatory run but was
  committed to git together with the results, and it was not registered with an external registry.
  "Locked" in the header therefore means locked in the author's workflow, not independently
  time-stamped.
- **2026-10-15 (interpretation correction, no new data)**: the "ankle/hip torque-usage ratio" in
  sections 2-5 is computed by `ankle_hip_usage()` on the clipped policy commands (mean |a[0]| /
  mean |a[3]|), i.e. each actuator's command as a fraction of its own capacity. The env scales them
  differently (ankle m g r_bos with r_bos = 0.06 m; hip 0.35 m g l with l = 0.934 m), a capacity
  ratio of 5.4. The confirmed H1c therefore concerns command utilization. In torque units the
  in-band ratio is about 0.46 to 0.63 (hip > ankle), so read as torque usage H1c meets its own
  falsification criterion in section 5. Manuscripts now report the ratio as command utilization
  and state the torque-unit equivalent.
- **2026-10-14 (analysis correction, no new data)**: the H2 verdict code did not enforce the contiguity
  required in section 4 (weight 0.4 fails survival at 5/8). Fixed and recomputed from the saved CSV:
  band [0.6, 3.0], span 5.0x, H2 CONFIRMED; H1c lower-CI minimum 1.52, CONFIRMED; floor unchanged
  (see the correction note in `docs/SWEEP_RESULTS_CONFIRMATORY_2026-07-04.md`). The "10x" figure in the
  2026-07-04 entry below is superseded by 5x.
- **2026-07-04**: the 8-seed CEM confirmatory run completed and PASSED all three decision rules
  (H2 band span 10x; H1c ankle/hip ratio lower-CI 1.31 > 1; floor 1.45 deg > 1). Results in
  `docs/SWEEP_RESULTS_CONFIRMATORY_2026-07-04.md`. The **PPO learner cross-check (section 2) was NOT
  run** in the same job (the background process was stopped after the CEM sweep); it remains PENDING and
  will be run separately (a lighter 40-60k-timestep version). This does not affect the CEM confirmatory
  verdicts, which stand on their own.
