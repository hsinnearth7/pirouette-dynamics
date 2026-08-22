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
- **2026-07-04**: the 8-seed CEM confirmatory run completed and PASSED all three decision rules
  (H2 band span 10x; H1c ankle/hip ratio lower-CI 1.31 > 1; floor 1.45 deg > 1). Results in
  `docs/SWEEP_RESULTS_CONFIRMATORY_2026-07-04.md`. The **PPO learner cross-check (section 2) was NOT
  run** in the same job (the background process was stopped after the CEM sweep); it remains PENDING and
  will be run separately (a lighter 40-60k-timestep version). This does not affect the CEM confirmatory
  verdicts, which stand on their own.
