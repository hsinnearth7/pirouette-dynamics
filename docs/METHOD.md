# Method and literature map

Each module is anchored in the pirouette biomechanics / physics literature. Deep notes for
every source live in the sibling reference library (`12_舞蹈生物力學_DanceBiomechanics`).

| Module | Method | Primary sources |
|--------|--------|-----------------|
| `anthropometry.py` | 12-segment body, De Leva mass fractions + COM ratios; uniform-cylinder own inertia; Winter stature ratios for the skeleton | De Leva 1996 *J Biomech* 29:1223; Winter, *Biomechanics and Motor Control of Human Movement* 4e |
| `inertia.py` | whole-body inertia tensor about the foot by the parallel-axis theorem; principal moments | classical rigid-body mechanics (e.g. Tong, *Classical Dynamics*) |
| `rigidbody.py` | inverted-pendulum topple ODE; torque-free Euler equations for conservation checks | Lott & Laws 2012 *J Dance Med Sci* 16(4):167 |
| `topple.py` | revolutions-to-topple; sub-degree critical start angle; `cosh` growth law | Lott & Laws 2012; Guilford honours thesis (whole-body MOI 1.72 kg·m²) |
| `control.py` | bounded PD ankle strategy (COP within base of support) vs hip strategy (larger authority, trunk-lean cost); closed-loop integration; stability domains | Lin, Su & Lin 2019 *Front Bioeng Biotechnol* 7:290 (ankle vs hip; AP-dominant) |
| `metrics.py` | COM-COP inclination angle, defined as in Lin 2019 Eq. 2/3 and the sibling analyzer | Lin, Su & Lin 2019; Chen & Chou 2010 |
| `env.py` / `reward.py` | Gymnasium task: balance two tilt axes + modulate spin; reward = turning - AP>ML inclination - lean - effort, terminal topple penalty | Lin 2019 (AP-dominant); Lott & Laws 2012 (active recovery) |
| `train.py` / `ppo.py` | Cross-Entropy Method policy search (default, dependency-free); optional PPO via stable-baselines3 | Rubinstein CEM; Schulman et al. 2017 (PPO) |

## Anchors for the current and planned tiers

- **Ankle vs hip strategy, AP-dominant control (Tier B, done)** — Lin, Su & Lin 2019 *Front
  Bioeng Biotechnol* 7:290. Expert dancers keep a small anterior-posterior COM–COP
  inclination and use an ankle strategy; novices show larger AP inclination, overshooting,
  and a hip strategy. `control.py` implements the two strategies; reproducing Lin's
  *published* expert/novice inclination numbers is the planned validation backbone (Tier C)
  and needs no motion capture.
- **Aesthetic reward, AP > ML weighting (Tier C, done)** — Lin 2019 effect sizes (Cohen's d
  0.87–1.18 for AP vs 0.43–0.65 for ML) motivate the asymmetric inclination penalty in
  `reward.py`. In `examples/03` the AP-weighted policy converges near Lin's experienced-dancer
  AP inclination; removing the asymmetry moves it toward the novice value. Reproducing those
  published numbers is the video-free validation backbone.
- **Angular-momentum generation as the success variable** — the TWU triple-turn thesis and
  the Guilford thesis: peak longitudinal angular momentum and its generation rate separate
  successful from failed turns. A candidate refinement to the Tier-C reward.

## Metric compatibility with the measurement tool

`metrics.py` computes the COM–COP inclination angle with the same definition as Lin 2019 Eq.
(2)/(3) and the sibling `ballet-com-cop-analyzer`, so a simulated turn and a measured turn are
expressed in one common quantity.
