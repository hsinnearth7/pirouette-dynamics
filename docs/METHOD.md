# Method and literature map

Each module is anchored in the pirouette biomechanics / physics literature. Deep notes for
every source live in the sibling reference library (`12_舞蹈生物力學_DanceBiomechanics`).

| Module | Method | Primary sources |
|--------|--------|-----------------|
| `anthropometry.py` | 12-segment body, De Leva mass fractions + COM ratios; uniform-cylinder own inertia; Winter stature ratios for the skeleton | De Leva 1996 *J Biomech* 29:1223; Winter, *Biomechanics and Motor Control of Human Movement* 4e |
| `inertia.py` | whole-body inertia tensor about the foot by the parallel-axis theorem; principal moments | classical rigid-body mechanics (e.g. Tong, *Classical Dynamics*) |
| `rigidbody.py` | inverted-pendulum topple ODE; torque-free Euler equations for conservation checks | Lott & Laws 2012 *J Dance Med Sci* 16(4):167 |
| `topple.py` | revolutions-to-topple; sub-degree critical start angle; `cosh` growth law | Lott & Laws 2012; Guilford honours thesis (whole-body MOI 1.72 kg·m²) |

## Planned (Tiers B–C) and their anchors

- **Ankle vs hip strategy, AP-dominant control** — Lin, Su & Lin 2019 *Front Bioeng
  Biotechnol* 7:290. Expert dancers keep a small anterior-posterior COM–COP inclination and
  use an ankle strategy; novices show larger AP inclination, overshooting, and a hip
  strategy. These become the control-strategy comparison (Tier B) and the emergence test for
  the learned policy (Tier C). Reproducing Lin's *published* expert/novice inclination
  numbers is the validation backbone and needs no motion capture.
- **Angular-momentum generation as the success variable** — the TWU triple-turn thesis and
  the Guilford thesis: peak longitudinal angular momentum and its generation rate separate
  successful from failed turns. These shape the Tier-C reward.
- **Aesthetic reward, AP > ML weighting** — Lin 2019 effect sizes (Cohen's d 0.87–1.18 for
  AP vs 0.43–0.65 for ML) motivate an asymmetric inclination penalty in the reward.

## Metric compatibility with the measurement tool

`metrics.py` (Tier B) will compute the COM–COP inclination angle with the same definition as
Lin 2019 Eq. (2)/(3) and the sibling `ballet-com-cop-analyzer`, so a simulated turn and a
measured turn are expressed in one common quantity.
