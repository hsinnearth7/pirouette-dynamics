# Physics of the toppling model (Tier A)

## 1. The dancer as a spinning rigid body on a fixed pivot

A pirouette turns about a near-vertical axis on one supporting foot. We model the dancer as
a single rigid body pivoting at the supporting foot's floor contact. Two moments of inertia
about that pivot matter:

- **spin moment** `I_zz` — about the vertical axis. The mass sits close to this axis, so
  `I_zz` is small (~1-1.7 kg·m²). It sets the spin rate for a given angular momentum,
  `L = I_zz · ω`, and is what a dancer changes by pulling the arms in (the figure-skater
  effect: smaller `I_zz` → larger `ω`).
- **tilt moment** `I_tilt` — about a horizontal axis through the foot. Tilting swings the
  whole body on a ~0.9 m arm, so `I_tilt` is ~50 kg·m², roughly 50× larger than `I_zz`.

Both are assembled in `inertia.py` from the segmental body (`anthropometry.py`) via the
parallel-axis theorem:

```
I_pivot = Σ_i [ I_i(about its own COM) + m_i (|r_i|² Identity − r_i r_iᵀ) ]
```

The parallel-axis term uses the exact De Leva (1996) segment mass and COM position; only the
(sub-dominant) own-inertia term uses the uniform-cylinder approximation (see LIMITATIONS).

## 2. Why a pirouette topples like an inverted pendulum

A fast-spinning top stays upright ("sleeping top") when `(I_zz ω)² > 4 I_tilt M g l`. A
pirouette spins slowly — Lott & Laws report **1.7 rev/s** (ω ≈ 10.7 rad/s). With `I_zz ≈ 1`,
`I_tilt ≈ 50`, `M ≈ 51.5`, `l ≈ 0.93`: the left side ≈ 115, the right side ≈ 94,000. The
top is nowhere near sleeping, so gyroscopic terms are negligible and the body topples as a
plain inverted physical pendulum about the tilt axis:

```
I_tilt · θ̈ = M g l · sin θ            (θ = tilt from vertical)
```

This is exactly the rigid-body equation Lott & Laws integrate for topple angle vs time.

## 3. The sub-degree result, in closed form

Linearising (`sin θ ≈ θ`) gives unstable growth from rest:

```
θ(t) = θ₀ · cosh(t / τ),      τ = √( I_tilt / (M g l) )
```

For the default dancer, `τ ≈ 0.325 s`. A dancer hops to recover at `θ_hop ≈ 9.3°`
(Lott & Laws). To finish `N` revolutions at spin rate `f` the turn lasts `T = N / f`, so the
largest tolerable starting offset is

```
θ₀,crit = θ_hop / cosh(T / τ)
```

- N = 2 at 1.7 rev/s: `T = 1.176 s`, `cosh(T/τ) ≈ 18.7`, `θ₀,crit ≈ 0.50°`.
- N = 3: `θ₀,crit ≈ 0.08°`.

So **more than two rigid-body turns demands a sub-degree initial alignment** — below normal
postural sway. `topple.py` computes this both in closed form and by root-finding on the full
nonlinear integration; the tests assert they agree. This is Lott & Laws's central point and
motivates Tiers B–C: real dancers do not hold a perfect initial alignment, they *actively
recover during the turn*, which is what a controller (and then a learned policy) will model.

## 4. Validation anchors

| Quantity | Model | Source |
|----------|-------|--------|
| Hop-onset tilt | 9.3° (input threshold) | Lott & Laws 2012, mean 9.3° ± 1.9° |
| Spin rate | 1.7 rev/s | Lott & Laws 2012, mean |
| Critical start for >2 turns | ~0.5° (< 1°) | Lott & Laws 2012, "sub-degree" |
| Vertical-axis MOI `I_zz` | 1.05 (arms first) – 1.62 (arms second) kg·m² | Guilford honours thesis, 1.72 ± 0.1 |

The torque-free Euler integrator in `rigidbody.py` is not ballet-specific; it exists so the
test-suite can confirm the rigid-body core conserves angular momentum and kinetic energy.
