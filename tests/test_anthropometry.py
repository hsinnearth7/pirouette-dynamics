"""Body model: De Leva mass closure, assembly sanity, and posture effects."""
from __future__ import annotations

import numpy as np
import pytest

from pirouette_dynamics.anthropometry import build_body, list_postures, total_mass_fraction


@pytest.mark.parametrize("sex", ["male", "female"])
def test_mass_fractions_sum_to_one(sex: str) -> None:
    # De Leva segment masses must close to the whole body (ported invariant from analyzer).
    assert total_mass_fraction(sex) == pytest.approx(1.0, abs=0.01)


def test_build_body_segment_count_and_total_mass() -> None:
    segments = build_body(stature=1.593, mass=51.54)
    # head + trunk (single) + 5 bilateral pairs = 12 -> 2 + 5*2 = 12 segments.
    assert len(segments) == 12
    total = sum(s.mass for s in segments)
    assert total == pytest.approx(51.54, rel=0.01)


def test_com_height_in_anthropometric_band() -> None:
    from pirouette_dynamics.inertia import assemble_inertia

    body = assemble_inertia(build_body(stature=1.593, mass=51.54))
    # Standing/retire whole-body COM sits near ~0.55 of stature above the floor pivot.
    assert 0.45 * 1.593 < body.com_height < 0.62 * 1.593


def test_invalid_sex_and_posture_raise() -> None:
    with pytest.raises(ValueError):
        build_body(sex="other")
    with pytest.raises(ValueError):
        build_body(posture="grand_jete")


def test_segment_inertia_tensors_are_symmetric_psd() -> None:
    for s in build_body():
        assert np.allclose(s.inertia_com, s.inertia_com.T)
        assert np.all(np.linalg.eigvalsh(s.inertia_com) >= -1e-12)


def test_all_listed_postures_build() -> None:
    for name in list_postures():
        assert len(build_body(posture=name)) == 12
