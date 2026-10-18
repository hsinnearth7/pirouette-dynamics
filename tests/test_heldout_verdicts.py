"""Verdict and summary logic of the post-hoc held-out re-evaluation (examples/05)."""
import importlib.util
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "heldout", Path(__file__).resolve().parents[1] / "examples" / "05_heldout_reevaluation.py")
heldout = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(heldout)


def _row(w: float, ap: float, surv: int, ratio: float = 2.5, ci: float = 1.0) -> dict:
    return {"aesthetic_ap": w, "ap_incl_mean": ap, "n_survived": surv,
            "ankle_hip_ratio_mean": ratio, "ankle_hip_ratio_ci95": ci}


def test_wide_band_confirms_all_three_rules() -> None:
    rows = [_row(0.3, 3.9, 7), _row(0.4, 3.0, 5), _row(0.6, 2.4, 7, 3.0, 1.4),
            _row(1.0, 1.8, 7, 2.9, 1.3), _row(3.0, 1.7, 8, 2.5, 0.6), _row(5.0, 2.4, 2)]
    v = heldout.verdicts(rows)
    assert (v["band_lo"], v["band_hi"]) == (0.6, 3.0)
    assert v["band_span"] == pytest.approx(5.0)
    assert v["h2_confirmed"]
    assert v["h1c_min_lower_ci"] == pytest.approx(1.6)
    assert v["h1c_confirmed"]
    assert v["floor_min_ap"] == pytest.approx(1.7)
    assert v["floor_confirmed"]


def test_single_weight_band_fails_h2_but_keeps_h1c_and_floor() -> None:
    rows = [_row(0.6, 2.5, 3), _row(0.8, 2.1, 6, 2.12, 0.81), _row(1.0, 1.9, 4),
            _row(2.0, 1.93, 6), _row(3.0, 1.9, 4)]
    v = heldout.verdicts(rows)
    assert v["band_lo"] == 0.8  # 2.0 also passes, but 1.0 breaks contiguity; 0.8 is first
    assert v["band_span"] == pytest.approx(1.0)
    assert not v["h2_confirmed"]
    assert v["h1c_min_lower_ci"] == pytest.approx(1.31)
    assert v["h1c_confirmed"]
    assert v["floor_min_ap"] == pytest.approx(1.93)  # only 2.0 survives among weights >= 1.0
    assert v["floor_confirmed"]


def test_no_passing_weight_gives_nan_band_and_no_h1c() -> None:
    v = heldout.verdicts([_row(1.0, 5.0, 8), _row(2.0, 0.5, 3)])
    assert math.isnan(v["band_lo"]) and v["band_span"] == 0.0
    assert not v["h2_confirmed"] and not v["h1c_confirmed"]
    assert v["floor_min_ap"] == pytest.approx(5.0)  # floor is checked independently of the band
    assert v["floor_confirmed"]


def test_floor_fails_when_ap_tracks_weight_to_zero() -> None:
    v = heldout.verdicts([_row(1.0, 0.8, 8), _row(2.0, 0.4, 8)])
    assert not v["floor_confirmed"]


def test_summarise_counts_survivors_and_topple_fraction() -> None:
    evals = [{"aesthetic_ap": 1.0, "seed": s, "eval_set": "held100_107", "topple_frac": tf,
              "ap": ap, "ml": 3.0, "toppled": tf > 0.5, "ratio": 2.0}
             for s, (tf, ap) in enumerate([(0.0, 1.0), (0.25, 2.0), (0.625, 3.0)])]
    evals.append({**evals[0], "eval_set": "eval0_7"})  # other sets are ignored
    (row,) = heldout.summarise(evals, "held100_107")
    assert row["n_survived"] == 2 and row["n_seeds"] == 3 and row["n_resets"] == 8
    assert row["topple_frac_mean"] == pytest.approx((0.0 + 0.25 + 0.625) / 3)
    assert row["ap_incl_mean"] == pytest.approx(2.0)
    assert row["ml_incl_ci95"] == pytest.approx(0.0)


def test_heldout_sets_do_not_overlap_cem_fitness_resets() -> None:
    fitness = {0, 1, 2, 3, 4}
    assert not fitness & set(heldout.EVAL_SETS["held100_107"])
    assert len(heldout.EVAL_SETS["held100_107"]) == 8
