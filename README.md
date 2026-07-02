# pirouette-dynamics

A physics-based simulator and (later) a learned controller for the balance of a ballet
**pirouette**. The guiding question:

> How many revolutions can a dancer complete before toppling, what control strategy
> maximises that, and does the learned-optimal strategy reproduce the biomechanics that
> distinguish expert from novice dancers?

The project builds a pirouette as a *toppling spinning rigid body*, reproduces published
balance thresholds from first principles, then adds control and reinforcement learning on
top. It is a companion to a separate markerless-video **measurement** tool
(`ballet-com-cop-analyzer`): that project *measures* a real turn; this one *simulates,
explains, and optimises* one. No motion-capture data is required, because the simulator
generates its own.

## Status

| Tier | Content | State |
|------|---------|-------|
| **A. Physics core** | Whole-body inertia tensor (De Leva masses) + gravity-driven toppling ODE | **done** |
| **B. Balance control** | bounded ankle vs hip strategies, COM-COP inclination metric, stability domains | **done** |
| C. Learned control | `gymnasium` env + PPO that maximises revolutions with an aesthetic reward | planned |
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
  more than two turns, and hops to recover at a body tilt of **~9.3 deg**. Both fall out of
  the model. Opening the arms raises the vertical-axis inertia toward Guilford's measured
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
  (a larger COM-COP inclination), consistent with the expert-ankle / novice-hip contrast in
  Lin et al. (2019). The shared `metrics.com_cop_inclination_deg` is defined exactly as Lin's
  Eq. 2/3, so simulated turns are comparable with that paper's published numbers.

## Install

```bash
pip install -e ".[viz,dev]"
pytest            # physics conservation + threshold reproduction
```

Requires Python >= 3.11, numpy, scipy (matplotlib for plots; torch/gymnasium for Tier C).

## Layout

```
src/pirouette_dynamics/
  anthropometry.py  # De Leva segment masses + a parametric retire posture -> segments
  inertia.py        # whole-body inertia tensor about the supporting foot (parallel axis)
  rigidbody.py      # toppling equation of motion + a torque-free Euler integrator
  topple.py         # revolutions-to-topple, critical start angle, Lott & Laws report
examples/01_reproduce_lott_laws.py
tests/              # pytest
docs/               # PHYSICS.md, METHOD.md, LIMITATIONS.md
```

## What it demonstrates

Rigid-body dynamics and the inertia tensor (physics / classical mechanics), ODE integration
and a closed-form `cosh` growth law (applied maths), a segmental anthropometric model
(biomechanics), and a tested, typed, CI-checked Python package (software engineering).
Tiers B and C add control theory and reinforcement learning. Each tier is grounded in the
pirouette literature cited in `docs/METHOD.md`.

See `docs/LIMITATIONS.md` for an explicit list of the modelling simplifications.
