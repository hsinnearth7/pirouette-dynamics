"""Planned H2 band must be contiguous (docs/PREREGISTRATION_idea1_2026-07-04.md, section 4)."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "sweep", Path(__file__).resolve().parents[1] / "examples" / "04_reward_sensitivity_sweep.py")
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)


def _row(w: float, ap: float, surv: int) -> dict:
    return {"aesthetic_ap": w, "ap_incl_mean": ap, "n_survived": surv}


def test_failing_weight_splits_the_band() -> None:
    rows = [_row(0.3, 3.9, 7), _row(0.4, 3.0, 5), _row(0.6, 2.4, 7), _row(1.0, 1.8, 7),
            _row(3.0, 1.7, 8), _row(5.0, 2.4, 2)]
    band = sweep.widest_contiguous_band(rows)
    assert [r["aesthetic_ap"] for r in band] == [0.6, 1.0, 3.0]


def test_no_passing_weight_gives_empty_band() -> None:
    assert sweep.widest_contiguous_band([_row(1.0, 5.0, 8), _row(2.0, 1.0, 3)]) == []


def test_widest_span_wins_over_longest_count() -> None:
    rows = [_row(0.1, 1.0, 8), _row(0.2, 1.0, 8), _row(0.25, 1.0, 8), _row(0.3, 9.0, 8),
            _row(1.0, 1.0, 8), _row(10.0, 1.0, 8)]
    band = sweep.widest_contiguous_band(rows)
    assert [r["aesthetic_ap"] for r in band] == [1.0, 10.0]
