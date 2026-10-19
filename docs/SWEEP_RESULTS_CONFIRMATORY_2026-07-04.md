# Confirmatory sweep result (8 seeds, planned analysis) — 2026-07-04

> Pre-registration: `docs/PREREGISTRATION_idea1_2026-07-04.md`. This 8-seed CEM run is the
> CONFIRMATORY analysis; it supersedes the 2026-07-03 3-seed pilot (`SWEEP_RESULTS_2026-07-03.md`)
> for confirmatory claims. Script: `examples/04_reward_sensitivity_sweep.py` (mode `cem`).
> **PPO learner cross-check: PENDING** (the background job was stopped after the CEM sweep).

## Result (8 training seeds; AP inclination mean with t-based 95% CI; each policy on evaluation reset seeds 0-7)

> **Correction (2026-10-15, round 3): seeds, reward shaping, hip model, ML.**
> - The evaluation resets were not held out. `cem_train` (train.py) averages fitness over reset
>   seeds 0-4 and `train_point_cem` does not override `eval_seeds`; evaluation uses
>   `EVAL_SEEDS = range(8)`. So 5 of the 8 evaluation starts repeat training starts. "Held-out"
>   below is kept only in the original text for the record.
> - Survival = toppled on at most 4 of 8 evaluation resets; means and CIs include all 8 policies.
> - H-b (ankle/hip split) is not untargeted: the penalised COM-COP inclination uses a COP that
>   only the ankle torque shifts, and the lean term penalises hip command.
> - Hip torque enters the single-body tilt equation as an external restoring moment with an
>   assumed capacity (0.35 m g l) and lean (9 deg at capacity), both without a cited source.
> - ML inclination exceeds AP at every weight except 0.3 (in band 2.55 to 4.62 deg; 4.62 > 4.20
>   at weight 1.5). See `docs/LIMITATIONS.md` and PREREGISTRATION section 7.

| aesthetic_ap | AP incl (mean +- 95% CI) | ML | revs | ankle/hip ratio | survived |
|---|---|---|---|---|---|
| 0.30 | 3.91 +- 2.37 | 3.46 | 4.69 | 3.04 +- 1.73 | 7/8 |
| 0.40 | 2.96 +- 1.35 | 3.72 | 4.63 | 2.98 +- 1.27 | 5/8 |
| 0.60 | 2.43 +- 0.83 | 3.92 | 4.60 | 3.02 +- 1.36 | 7/8 |
| 0.80 | 2.00 +- 0.66 | 3.66 | 4.70 | 2.50 +- 0.96 | 7/8 |
| **1.00** | **1.79 +- 0.77** | 3.66 | 4.61 | 2.85 +- 1.33 | 7/8 |
| 1.50 | 1.45 +- 0.42 | 4.62 | 4.46 | 2.62 +- 0.99 | 6/8 |
| 2.00 | 1.72 +- 0.43 | 2.55 | 4.84 | 3.42 +- 1.38 | 8/8 |
| 3.00 | 1.71 +- 0.48 | 3.83 | 4.65 | 2.52 +- 0.65 | 8/8 |
| 5.00 | 2.41 +- 1.60 | 6.40 | 3.69 | 3.92 +- 1.82 | 2/8 |

Reference (Lin 2019 PRE-phase AP): experienced 2.77, novice 5.62 deg. Midpoint 4.20 (expert-side threshold).
PRE is Lin's preparatory, double-support phase, and these values are *ranges* of COM-COP AP
inclination; the simulation reports a per-episode *peak* |inclination| during single-leg
spinning. Lin's ML values are 0.03 to 0.17 deg, against 2.5 to 6.4 deg here. The comparison is
numerical, across phase and statistic (see `docs/LIMITATIONS.md`, Tier C).

CI semantics: the 95 % CIs are t-intervals over the 8 CEM training seeds, with every policy
evaluated on the same 8 reset states, so they cover training-seed variance conditional on those
states and not initial-condition sampling. No multiplicity correction is applied across the 9
weights and 3 rules, and H2 uses point means without a CI.

Mean topple fraction per weight on reset seeds 0-7 (a "survived" policy may still topple on up
to 4 of 8 resets): 0.27, 0.33, 0.28, 0.27, 0.30, 0.36, 0.12, 0.27, 0.75 for weights 0.3 to 5.0.

Tested environment for the reruns (2026-10-15; bitwise identical to the July CSV): Python 3.14.0,
numpy 2.3.5, scipy 1.17.0, gymnasium 1.2.3, Windows 11. CI runs Python 3.11 and 3.12.

> **Correction (2026-10-14).** The original auto-computed H2 band below was wrong: the verdict code
> took the min and max of all passing weights and ignored that `aesthetic_ap` = 0.4 fails the survival
> rule (5/8 < 6/8), although section 4 of the pre-registration requires a *contiguous* band. Recomputed
> from `results/reward_sweep_confirmatory.csv` with the fixed code (`widest_contiguous_band` in
> `examples/04_reward_sensitivity_sweep.py`, tested in `tests/test_sweep_band.py`): band = [0.6, 3.0],
> span 5.0x (>= 3x required), so **H2 is still CONFIRMED**; H1c minimum ratio lower-CI within the band
> = 1.52 (> 1), **still CONFIRMED**; floor 1.45 deg unchanged. The original text is kept below for the
> record.

> **Correction (2026-10-15): ankle/hip ratio units.** The "ankle/hip ratio" column is
> mean|a[0]| / mean|a[3]| on clipped, normalised commands (command utilisation), not torque. The env
> scales ankle by tau_bos = m g 0.06 and hip by tau_hip = 0.35 m g l (l = 0.934 m), so hip torque
> per unit command is 5.446x larger. Torque-unit ratio = command ratio / 5.446 (constant per env, so
> the CI half-width scales identically). Computed by `python examples/04_reward_sensitivity_sweep.py
> torque` from the saved CSV into `results/reward_sweep_confirmatory_torque.csv` (original CSV
> unchanged; `results/` is gitignored, and the two confirmatory CSVs are force-added to the
> repository). In band [0.6, 3.0] the torque ratio is 0.46 to 0.63 (hip produces the larger mean
> absolute torque, a direction set by the assumed hip capacity); the lowest lower 95% CI is 0.28. Read as torque, H1c (ratio > 1) is NOT met.
> The 5x band and the 1.52 command-ratio CI remain valid for command utilisation only.
>
> | aesthetic_ap | command ratio | torque ratio (mean +- 95% CI) |
> |---|---|---|
> | 0.30 | 3.04 | 0.558 +- 0.318 |
> | 0.40 | 2.98 | 0.548 +- 0.233 |
> | 0.60 | 3.02 | 0.555 +- 0.250 |
> | 0.80 | 2.50 | 0.459 +- 0.176 |
> | 1.00 | 2.85 | 0.524 +- 0.245 |
> | 1.50 | 2.62 | 0.481 +- 0.182 |
> | 2.00 | 3.42 | 0.628 +- 0.253 |
> | 3.00 | 2.52 | 0.463 +- 0.119 |
> | 5.00 | 3.92 | 0.719 +- 0.333 |

## Verdicts as originally auto-computed (all three PASS; see corrections above)
- **H2 robustness CONFIRMED**: expert-side band (AP <= 4.20, survival >= 6/8) = `aesthetic_ap` [0.3, 3.0],
  span 10.0x (>= 3x required).
- **H1c ankle-dominance CONFIRMED (held-out)**: the ankle/hip torque-usage ratio's lower 95%-CI bound
  within the band = 1.31 > 1. The reward penalises hip lean but never the ankle/hip split, so robust
  ankle-dominance (ratio ~2.5-3.4x) is genuine emergence, not a target.
- **Non-baked-in floor CONFIRMED**: the AP-inclination floor over surviving `aesthetic_ap` >= 1.0 is
  1.45 deg (> 1 deg), not driven to 0. The reward drives AP inclination toward 0, so a floor near the
  human expert is a physics limit, not reward-tracking.

## Honest interpretation
- On reset seeds 0-7, low AP inclination under the AP-penalised reward holds across a (5x, corrected from
  10x) contiguous band of the reward weight, not at a tuned value, with 8-seed CIs and 6-8/8 survival
  across the band. It is not an AP-dominant strategy: ML inclination exceeds AP at every weight except
  0.3. On held-out resets the band does not survive (see the post-hoc section below). The
  original text called it ankle-dominant; that holds for command utilisation only (see the 2026-10-15
  correction: in torque units the hip produces more mean absolute torque under the assumed
  capacity, and the reward itself favours the ankle; ML inclination stays above AP).
  This is a documented-plan (not externally registered) rebuttal to the "post-hoc tuning / one-number curve-fit" objection.
- The emergent AP inclination in the band (1.45-2.43 deg) is numerically at or below Lin's
  experienced-dancer value (2.77), which is a preparatory-phase range, not a single-leg peak; this
  is not a like-for-like skill comparison.
- Honest bounds: too little weight (0.3) is novice-ward (3.9 deg); too much (5.0) collapses training
  (2/8 survive; wide CI). A bounded attractor basin, not a monotone plateau.
- Caveats retained: reduced single-tilt-DOF model; peak-inclination metric; CEM learner (PPO cross-check
  pending). 8 seeds is adequate; the wide-CI extremes (0.3, 5.0) would tighten with more seeds.

## Post-hoc held-out re-evaluation (2026-10-15; NOT in the analysis plan)

Added after the results were known, because 5 of the 8 evaluation resets above are CEM training
resets. `python examples/05_heldout_reevaluation.py --retrain --extended` retrains the same 72
policies (seeded; on resets 0-7 they reproduce the table above exactly; parameters saved to
`results/confirmatory_policies.json`) and evaluates them on reset seeds the optimiser never saw.
Output: `results/reward_sweep_heldout.csv`. Run on the tested environment above, ~7 min with
10 worker processes.

| aesthetic_ap | topple frac 0-7 | topple frac 100-107 | survived 0-7 | survived 100-107 | AP 100-107 (mean +- 95% CI) | ML 100-107 (mean +- 95% CI) |
|---|---|---|---|---|---|---|
| 0.30 | 0.27 | 0.42 | 7/8 | 5/8 | 4.24 +- 2.57 | 3.96 +- 2.92 |
| 0.40 | 0.33 | 0.44 | 5/8 | 4/8 | 3.09 +- 1.53 | 4.21 +- 2.37 |
| 0.60 | 0.28 | 0.48 | 7/8 | 3/8 | 2.47 +- 0.75 | 5.03 +- 2.36 |
| 0.80 | 0.27 | 0.33 | 7/8 | 6/8 | 2.11 +- 0.65 | 4.00 +- 2.89 |
| 1.00 | 0.30 | 0.48 | 7/8 | 4/8 | 1.94 +- 0.90 | 4.74 +- 2.80 |
| 1.50 | 0.36 | 0.55 | 6/8 | 3/8 | 1.57 +- 0.41 | 5.32 +- 3.28 |
| 2.00 | 0.12 | 0.28 | 8/8 | 6/8 | 1.93 +- 0.53 | 3.27 +- 2.29 |
| 3.00 | 0.27 | 0.42 | 8/8 | 4/8 | 1.86 +- 0.49 | 4.74 +- 2.62 |
| 5.00 | 0.75 | 0.81 | 2/8 | 1/8 | 2.51 +- 1.53 | 6.76 +- 3.25 |

Section-4 rules applied unchanged:

| eval resets | H2 band (span) | H2 | H1c min lower CI (command ratio) | floor (min AP, surviving w >= 1.0) |
|---|---|---|---|---|
| 0-7 (confirmatory) | [0.6, 3.0] (5.0x) | CONFIRMED | 1.52 | 1.45 |
| 100-107 (held out, n = 8) | [0.8, 0.8] (1.0x) | **NOT confirmed** | 1.31 | 1.93 |
| 100-131 (held out, n = 32) | [2.0, 2.0] (1.0x) | **NOT confirmed** | 1.64 | 1.96 |

Reading: the AP-inclination level is stable on unseen starts, but survival is not, so the
robustness claim (H2) rests on the overlap between training and evaluation resets. The CEM fitness
uses only 5 fixed resets, which likely causes this overfitting; training on more, randomised
resets would be a new, exploratory experiment. H1c remains a command-utilisation statement only
(see the torque-unit correction above).

## Next
- Run the PPO learner cross-check (lighter: e.g. 40-60k timesteps, `aesthetic_ap`=1.0, 3 seeds) to confirm
  learner-independence, per the pre-registration.
