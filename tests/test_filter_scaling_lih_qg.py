"""
Tests for examples/filter_scaling_lih_qg.py (RESEARCH_NOTES §50): the Z-basis
filter reaches the H2 error but not the LiH error, which sits in the X/Y groups.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

import filter_scaling_lih_qg as S  # noqa: E402


@pytest.fixture(scope="module")
def h2():
    return S.analyse("H2", "all_to_all")


@pytest.fixture(scope="module")
def lih():
    return S.analyse("LiH", "all_to_all", 1)


def test_h2_error_sits_in_the_z_group_and_the_filter_reaches_it(h2):
    assert h2["groups"] == 5
    assert abs(h2["z_raw"]) > 0.8 * abs(h2["total_raw"])
    assert abs(h2["total_filter"]) < 0.5 * abs(h2["total_raw"])


def test_lih_error_sits_in_the_xy_groups(lih):
    assert lih["groups"] == 17 and lih["two_qubit_gates"] == 20
    assert abs(lih["off_raw"]) > 0.9 * abs(lih["total_raw"])
    assert abs(lih["total_filter"] - lih["total_raw"]) < 0.2 * abs(lih["total_raw"])
    assert lih["ceiling_exact_z"] == pytest.approx(lih["off_raw"])


def test_kept_fraction_rescaling_overcorrects(lih):
    assert lih["off_rescaled"] < 0 < lih["off_raw"]
    assert abs(lih["off_rescaled"]) > abs(lih["off_raw"])
    assert lih["shrink_from_kept"] < lih["shrink_actual"]


def test_noise_floor_and_witness(lih):
    assert lih["floor"] == pytest.approx(15 / 64)
    assert lih["ideal_mean_qg_z"] == pytest.approx(1 / 3)
    assert lih["mean_qg_z"] < lih["ideal_mean_qg_z"]
