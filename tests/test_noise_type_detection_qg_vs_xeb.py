"""
Tests for examples/noise_type_detection_qg_vs_xeb.py: T1-vs-unital
detection from counts with qg features vs XEB / HOP.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from noise_type_detection_qg_vs_xeb import (  # noqa: E402
    STRONG,
    WEAK,
    build_dataset,
    ideal_mean_qg_z_spread,
    t1_detection_accuracy,
)


@pytest.fixture(scope="module")
def strong():
    return build_dataset(STRONG)


@pytest.fixture(scope="module")
def weak():
    return build_dataset(WEAK)


@pytest.mark.parametrize("n_shots", [100, 1000])
def test_finding_a_mean_qg_z_detects_strong_t1_better_than_standard_benchmarks(strong, n_shots):
    qg = t1_detection_accuracy(strong, n_shots, "mean_qg_z")
    assert qg > 0.85
    for other in ("xeb", "hop", "qg_s"):
        assert qg > t1_detection_accuracy(strong, n_shots, other) + 0.1


def test_finding_a_accuracy_grows_with_noise_strength(strong):
    low = t1_detection_accuracy(strong, 1000, "mean_qg_z", strength_range=(0.0, 0.1))
    high = t1_detection_accuracy(strong, 1000, "mean_qg_z", strength_range=(0.25, 0.51))
    assert high > 0.95
    assert low < high - 0.15


@pytest.mark.parametrize("feature", ["mean_qg_z", "qg_s", "xeb", "hop"])
def test_finding_b_weak_noise_is_undetectable_even_with_many_shots(weak, feature):
    assert t1_detection_accuracy(weak, 100_000, feature) == pytest.approx(2 / 3, abs=0.02)


def test_finding_b_circuit_to_circuit_spread_of_ideal_mean_qg_z():
    sd, lo, hi = ideal_mean_qg_z_spread()
    assert sd == pytest.approx(0.139, abs=0.001)
    assert lo < -0.3 and hi > 0.3
