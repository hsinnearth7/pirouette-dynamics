# pirouette-dynamics

A physics-based simulator and (later) a learned controller for the balance of a ballet
**pirouette**. The guiding question:

> How many revolutions can a dancer complete before toppling, what control strategy
> maximises that, and does the learned-optimal strategy reproduce the biomechanics that
> distinguish expert from novice dancers?

The project models a pirouette as a toppling spinning rigid body, reproduces published
balance thresholds from first principles, then adds control and reinforcement learning.

## Install

```bash
pip install -e ".[viz,dev]"
pytest
```

Requires Python >= 3.11, numpy, scipy.

_Under active construction; see `docs/` for the physics and method notes._
