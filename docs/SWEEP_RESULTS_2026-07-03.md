# Reward-weight sensitivity sweep — exploratory pilot (2026-07-03)

> **Superseded; historical record only (note added 2026-10-15).** The confirmatory 8-seed run and
> its corrections are in `SWEEP_RESULTS_CONFIRMATORY_2026-07-04.md` and PREREGISTRATION section 7.
> Several readings below do not survive them: the "robust attractor" band is [0.6, 3.0] on reset
> seeds 0-7 and collapses on held-out resets (post-hoc, `examples/05`); ML inclination exceeds AP
> at most weights, so the strategy is AP-penalised rather than AP-dominant; and the floor is
> numerically comparable to Lin's preparatory-phase range, not "equal to the expert value".
> The 3-seed pilot script was not preserved: the first committed `examples/04` is already the
> 8-seed confirmatory version, so `results/reward_sensitivity_sweep.csv` cannot be regenerated
> from tracked code and the numbers below are a historical record.

> Script: `examples/04_reward_sensitivity_sweep.py`. 9 reward weights x 3 CEM
> training seeds x 25 iterations; `aesthetic_ap=1.0, seed 0` reproduces examples/03's expert config.
> **Question**: is the expert-like, AP-dominant balance strategy a robust attractor across the reward
> AP-weight, or does it appear only at one tuned value?

## Result (mean +- sd over 3 training seeds; evaluated on reset seeds 0-7, of which 0-4 also enter the CEM fitness, so not held out; corrected 2026-10-15)

| aesthetic_ap | AP:ML ratio | AP incl (deg) | ML incl (deg) | revs | survived |
|---|---|---|---|---|---|
| 0.30 | 1.0  | 3.73 +- 0.80 | 5.10 | 4.47 | 3/3 |
| 0.40 | 1.3  | 3.79 +- 2.51 | 4.40 | 4.39 | 1/3 |
| 0.60 | 2.0  | 2.30 +- 0.73 | 4.97 | 4.45 | 2/3 |
| 0.80 | 2.7  | 2.54 +- 0.45 | 5.49 | 4.42 | 2/3 |
| **1.00** | **3.3** | **2.16 +- 1.31** | 3.92 | 4.48 | 3/3 |
| 1.50 | 5.0  | 1.70 +- 0.40 | 5.46 | 4.39 | 2/3 |
| 2.00 | 6.7  | 2.01 +- 0.69 | 2.82 | 4.75 | 3/3 |
| 3.00 | 10.0 | 2.01 +- 0.55 | 2.49 | 5.00 | 3/3 |
| 5.00 | 16.7 | 3.98 +- 2.35 | 5.62 | 3.10 | 0/3 |

Reference (Lin et al. 2019 PRE-phase AP inclination): experienced 2.77 deg, novice 5.62 deg.

## Interpretation (honest)

1. **Robust interior band, not a tuned point.** For `aesthetic_ap` in [0.6, 3.0] (AP:ML ratio 2 to 10x)
   the emergent AP inclination stays at 1.7 to 2.5 deg, straddling Lin's experienced value (2.77) and
   well below novice (5.62). The expert-like AP-dominant strategy is therefore a **robust attractor over
   a broad band of the reward weight**, which is the strongest available rebuttal to the "one-number
   curve-fit / tuned" objection.
2. **A bounded basin, not monotone-to-zero.** Too little AP weight (0.3 to 0.4) leaves the policy
   novice-ward (~3.7 deg); too much (5.0) makes training **collapse** (0/3 survive; AP jumps to ~4 deg
   with a huge error bar). The reward penalises AP inclination linearly, so a naive expectation is
   AP -> 0, yet the achievable floor is ~2 deg and it coincides with the human expert value. That the
   physics-limited floor equals the expert value (not 0) is the genuinely emergent, not-baked-in result.
3. **Honest caveats.** (a) Seed-to-seed CEM training variance is non-trivial:
   several weights had only 1 or 2 of 3 seeds survive, and error bars are wide. The emergence is robust
   *on average across the band*, not deterministic every run. (b) ML inclination is noisier than AP.
   (c) 3 training seeds is thin. For a stronger version: >=8 training seeds, consider PPO (more reliable
   than CEM), and report the survival rate alongside the inclination.

## Takeaway

- Supports the "robust, not tuned" reading while honestly bounding it. The emergent,
  NOT-directly-rewarded quantities are the real result: the ~2 deg physics-limited AP floor that
  matches the human expert (the reward would prefer 0), sustained revolutions (~5), and low hip-lean.

## Next steps
- Re-run with >=8 training seeds and/or PPO for tight error bars and higher survival; report survival.
- Pre-register the multi-statistic target set (which quantities are targets vs held-out) before the
  confirmatory run.
