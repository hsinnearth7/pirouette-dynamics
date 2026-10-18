"""Ankle/hip ratio: command utilisation vs torque units (review finding, 2026-10-15)."""
import csv
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pirouette_dynamics.env import PirouetteEnv

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "sweep", _ROOT / "examples" / "04_reward_sensitivity_sweep.py")
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)


class _FixedPolicy:
    """Constant ankle/hip commands; the other channels are zero."""

    def __init__(self, ankle: float, hip: float) -> None:
        self._a = np.array([ankle, 0.0, 0.0, hip], dtype=float)

    def act(self, obs):
        return self._a


def test_capacity_factor_is_tau_ratio_from_env() -> None:
    env = PirouetteEnv()
    assert sweep.capacity_factor(env) == pytest.approx(env.tau_hip / env.tau_bos)
    assert 5.0 < sweep.capacity_factor(env) < 6.0  # ~5.446 for the default anthropometry


def test_command_to_torque_conversion() -> None:
    assert sweep.command_to_torque_ratio(5.0, 5.0) == pytest.approx(1.0)
    assert sweep.command_to_torque_ratio(2.5, 5.446) == pytest.approx(2.5 / 5.446)


def test_rollout_torque_ratio_equals_command_ratio_over_factor() -> None:
    env = PirouetteEnv()
    pol = _FixedPolicy(0.6, 0.2)
    cmd = sweep.ankle_hip_command_ratio(env, pol, [0, 1])
    tq = sweep.ankle_hip_torque_ratio(env, pol, [0, 1])
    assert cmd == pytest.approx(3.0)
    assert tq == pytest.approx(cmd / sweep.capacity_factor(env))
    assert sweep.ankle_hip_usage is sweep.ankle_hip_command_ratio  # backward-compatible alias


def test_sidecar_scales_saved_results_and_keeps_source() -> None:
    src = _ROOT / "results" / "reward_sweep_confirmatory.csv"
    before = src.read_bytes()
    factor = sweep.capacity_factor(PirouetteEnv())
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "t.csv"
        out = sweep.write_torque_sidecar(str(src), str(dst), factor)
        with open(dst, newline="", encoding="utf-8") as fh:
            assert len(list(csv.DictReader(fh))) == len(out)
    assert src.read_bytes() == before
    first = out[0]
    assert first["torque_ratio_mean"] == pytest.approx(first["command_ratio_mean"] / factor)
