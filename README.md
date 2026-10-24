# pirouette-dynamics

A physics-based simulator and a learned (CEM) controller for the balance of a ballet
**pirouette**. The guiding question:

> How many revolutions can a dancer complete before toppling, what control strategy
> maximises that, and does the learned-optimal strategy reproduce the biomechanics that
> distinguish expert from novice dancers?

The project builds a pirouette as a *toppling spinning rigid body*, checks it for
consistency against published balance thresholds (two of which are model inputs, see
Tier A below), then adds control and reinforcement learning on top. It is a companion to a separate markerless-video **measurement** tool
(`ballet-com-cop-analyzer`): that project *measures* a real turn; this one *simulates,
explains, and optimises* one. No motion-capture data is required, because the simulator
generates its own.

## Status

| Tier | Content | State |
|------|---------|-------|
| **A. Physics core** | Whole-body inertia tensor (De Leva masses) + gravity-driven toppling ODE | **done** |
| **B. Balance control** | bounded ankle vs hip strategies, COM-COP inclination metric, stability domains | **done** |
| **C. Learned control** | `gymnasium` env + CEM policy (PPO optional) with an aesthetic reward | **done** |
| D. Bridges (optional) | joint-torque muscle synergies (NMF); qualitative video cross-check | planned |

## What Tier A reproduces

Running `python examples/01_reproduce_lott_laws.py` (Lin 2019 experienced-dancer size,
159.3 cm / 51.54 kg, arms-first retire):

```
COM height l            =  0.934 m
spin moment  I_zz       =  1.046 kg m^2   (Guilford honours thesis measured ~1.72)
tilt moment  I_tilt(AP) = 49.88 kg m^2
topple time-constant tau=  0.325 s
critical start, 2 rev   =  0.498 deg      <- Lott & Laws: sub-degree needed for >2 turns
```

- **Lott & Laws (2012)**: a rigid dancer must begin within **< 1 deg** of vertical to clear
  more than two turns, and hops to recover at a body tilt of **~9.3 deg**. These are
  *inputs* taken from Lott & Laws: 9.3 deg is the hop-onset parameter, and the 1.7 rev/s
  spin rate comes from the same source. The sub-degree critical start (0.498 deg) is
  therefore a consistency check of the toppling model given those inputs, not an
  independent first-principles derivation. Opening the arms raises the vertical-axis inertia toward Guilford's measured
  **1.72 kg·m²** (the figure-skater effect).

## What Tier B shows

`python examples/02_control_strategies.py` closes the loop with a restoring torque and
contrasts the two textbook balance strategies:

```
Ankle strategy: largest statically recoverable tilt vs base of support
  pointe      (BOS  2.0 cm) ->  1.23 deg
  demi-pointe (BOS  6.0 cm) ->  3.68 deg
  flat foot   (BOS 10.0 cm) ->  6.15 deg
  hip strategy            -> 20.49 deg (with trunk lean)
```

- The ankle strategy's recoverable tilt is capped at `arcsin(r_bos / l)`: about **1 deg on
  pointe**, a few degrees flat. That is the mechanical reason a pirouette topples so readily
  and needs the hip strategy or an active hop.
- The **hip strategy** recovers far larger tilts but pays for it with a visible **trunk lean**
  (a larger COM-COP inclination). In this model the hip torque is added to the ankle torque as
  a restoring moment about the foot pivot of the single rigid body, with an assumed capacity of
  0.35 m g l and 9 deg of lean at capacity (no cited source). A real hip strategy is internal and
  transient, so the 20.49 deg figure is a consequence of that assumption, not a statically
  achievable tilt. Lin et al. (2019) only suggested a hip strategy for
  novices in the ending phase of the turn and did not measure it, so this is a modelling
  contrast, not a replication of a measured expert-ankle / novice-hip finding. The shared `metrics.com_cop_inclination_deg` is defined exactly as Lin's
  Eq. 2/3, so simulated turns are comparable with that paper's published numbers.

## What Tier C shows

`python examples/03_learn_pirouette.py` trains a linear policy with the Cross-Entropy Method
(no GPU, no extra dependency) to balance and keep turning, then ablates the reward. Averaged
over 8 evaluation reset seeds 0-7 (deterministic, seed 0). These are not held out: CEM fitness
averages over reset seeds 0-4, so only seeds 5-7 are unseen in training.

```
  passive (no control)         return -36.6 | revs 1.6 | AP incl 8.4 deg | ML incl 3.4 deg | TOPPLED
  learned, AP-weighted reward  return +17.4 | revs 5.0 | AP incl 2.0 deg | ML incl 0.5 deg | survived
  learned, symmetric (ablate)  return +17.3 | revs 5.0 | AP incl 3.6 deg | ML incl 3.0 deg | survived
```

Three things are observed. None is tuned per result, but the setup builds in part of the
AP-dominant pattern (the initial-tilt range is +-3 deg in AP versus +-0.5 deg in ML, and the
reward has an AP term):

- **Active balance.** This seed-0 policy toppled on none of the 8 resets (nor on held-out
  resets 100-107) and completes all five turns, echoing Lott & Laws's point that dancers
  recover *during* the turn rather than holding a perfect start. It is one of the better
  seeds: across the 8-seed sweep, policies at this weight topple on 30 % of resets 0-7 and on
  48 % of held-out resets.
- **Low AP inclination in this run.** With the AP-weighted aesthetic reward the policy holds
  AP inclination at **2.0 deg**, numerically comparable to Lin's experienced-dancer value
  (2.77 deg, a preparatory-phase range rather than a single-leg peak), and ML at 0.5 deg. The
  low ML value is a seed-0 result: across 8 seeds at the same weight ML averages 3.66 deg
  (per seed 0.4 to 6.6), above the AP mean of 1.79, so the run does not show the AP-dominant
  signature Lin reports. Lin's ML values (0.03 to 0.17 deg) are also far below the simulated
  ones.
- **Reward shapes inclination.** Removing the AP emphasis (symmetric reward) raises AP
  inclination to 3.6 deg in this single-seed run. That is still below the 4.20 deg midpoint
  between Lin's experienced (2.77) and novice (5.62) values. Across 8 seeds the equivalent
  sweep weight (`aesthetic_ap` 0.3) gives 3.91 +- 2.37 deg (95% CI 1.53 to 6.28), an interval
  that spans the novice value, so the ablation is inconclusive about the novice-to-expert
  axis.

`python examples/04_reward_sensitivity_sweep.py` repeats the training over a grid of AP
reward weights with 8 CEM seeds each. The analysis plan (`docs/PREREGISTRATION_idea1_2026-07-04.md`)
was drafted before the run but committed together with the results and not registered
externally; its deviations are logged in section 7. Expert-side AP inclination (at most
4.20 deg, at least 6/8 policies "surviving", where a survivor may still topple on up to half
of its 8 resets; mean topple fraction 0.12 to 0.36 per weight in band) holds over the
contiguous band
`aesthetic_ap` in [0.6, 3.0] (a 5x span; an earlier 10x figure was a verdict-code error,
corrected 2026-10-14). The ankle/hip ratio is reported as normalised-command
utilisation (2.50 to 3.42 in band, lowest lower CI 1.52). It is not an untargeted quantity:
the penalised COM-COP inclination uses a COP that only the ankle torque shifts, and the lean
term penalises every unit of hip command, so the reward itself favours the ankle. In torque
units, which divide by the hip-to-ankle capacity factor 5.45 (`tau_hip / tau_bos`), the
in-band ratio is 0.46 to 0.63 (lowest lower CI 0.28), i.e. the hip produces the larger mean
absolute torque. That ordering follows from the assumed hip capacity (0.35 m g l) and
r_bos = 6 cm, neither with a cited source; the command ratio must not be read as torque dominance by the ankle. ML inclination
(weight fixed at 0.3, ankle-only control) exceeds AP inclination at every weight except 0.3
(2.55 to 4.62 deg in band, above 4.20 at weight 1.5). Each policy is evaluated on reset seeds
0-7, of which 0-4 also enter the CEM fitness, so the sweep is not a held-out test.

`python examples/05_heldout_reevaluation.py` is a **post-hoc** check, not part of the analysis
plan: it retrains the same 72 policies (seeded, so identical; saved to
`results/confirmatory_policies.json`) and evaluates them on reset seeds 100-107, which training
never saw. There the policies topple on 28 to 55 % of turns at the band weights, survivors fall
to 3-6 of 8, and the widest contiguous band shrinks to the single weight 0.8, so **H2
(robustness across a broad band) is not confirmed on held-out starts**. AP means barely move
(1.57 to 2.47 deg for weights 0.6-3.0), and H1c (command-ratio lower CI 1.31) and the floor
(1.93 deg) still pass. Results: `results/reward_sweep_heldout.csv`; see PREREGISTRATION
section 7.

This is *consistent with*, not proof of, expert biomechanics (see `docs/LIMITATIONS.md`). PPO
(stable-baselines3) is available as `pirouette_dynamics.ppo.train_ppo` with the `rl` extra,
but the PPO cross-check has **not been run**, so no PPO result is claimed here; CEM is the
default because it is dependency-free and reproducible.

## Install

```bash
pip install -e ".[viz,dev]"      # core + plots + test tooling
pip install -e ".[rl]"           # optional: PPO via stable-baselines3
pytest                           # physics conservation, thresholds, control, learning
```

Requires Python >= 3.11, numpy, scipy and gymnasium (a core dependency: the package imports
the env at load time); matplotlib for plots; stable-baselines3 only for the optional PPO path.

## Layout

```
src/pirouette_dynamics/
  anthropometry.py  # De Leva segment masses + a parametric retire posture -> segments
  inertia.py        # whole-body inertia tensor about the supporting foot (parallel axis)
  rigidbody.py      # toppling equation of motion + a torque-free Euler integrator
  topple.py         # revolutions-to-topple, critical start angle, Lott & Laws report
  control.py        # bounded ankle vs hip balance strategies, stability domains
  metrics.py        # COM-COP inclination angle (Lin 2019), shared with the analyzer
  env.py            # Gymnasium PirouetteEnv (balance + spin + aesthetic reward)
  reward.py         # shaped reward: turning, AP>ML inclination, elegance, economy
  train.py          # dependency-free CEM policy search + rollout/evaluate
  ppo.py            # optional PPO adapter (stable-baselines3)
examples/           # 01 reproduce Lott & Laws, 02 control strategies, 03 learn to turn,
                    # 04 reward sensitivity sweep (plan in docs/PREREGISTRATION, not externally registered),
                    # 05 post-hoc held-out re-evaluation of the 04 policies
tests/              # pytest
docs/               # PHYSICS.md, METHOD.md, LIMITATIONS.md, PREREGISTRATION_idea1_2026-07-04.md,
                    # SWEEP_RESULTS_CONFIRMATORY_2026-07-04.md, SWEEP_RESULTS_2026-07-03.md (pilot)
results/            # force-added sweep CSVs and saved policies (the rest of results/ is gitignored)
```

## What it demonstrates

Rigid-body dynamics and the inertia tensor (physics / classical mechanics), ODE integration
and a closed-form `cosh` growth law (applied maths), a segmental anthropometric model
(biomechanics), and a tested, typed, CI-checked Python package (software engineering).
Tiers B and C add control theory and reinforcement learning. Each tier is grounded in the
pirouette literature cited in `docs/METHOD.md`.

See `docs/LIMITATIONS.md` for an explicit list of the modelling simplifications.
