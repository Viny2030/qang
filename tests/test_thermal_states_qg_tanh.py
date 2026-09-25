"""
Tests for examples/thermal_states_qg_tanh.py: thermal states in qg units,
qg_Z = tanh(beta h).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

from thermal_states_qg_tanh import (  # noqa: E402
    check_identities,
    effective_temperature_mK,
    estimate_temperature,
    heat_capacity_from_qg,
    ising_row,
    mean_field_qg,
    optimal_probe_qg,
    preparation_angle,
    thermometry_table,
)


def test_single_qubit_thermodynamics_is_exact_in_qg():
    assert check_identities() < 1e-10


def test_preparation_angle_is_arccos_tanh():
    for b in (0.0, 0.3, 1.0, 4.0):
        assert preparation_angle(b) == pytest.approx(math.acos(math.tanh(b)), abs=1e-12)


def test_optimal_thermometer_is_the_schottky_peak():
    qs = optimal_probe_qg()
    assert qs * math.atanh(qs) == pytest.approx(1.0, abs=1e-12)
    assert 2 * math.atanh(qs) == pytest.approx(2.3994, abs=1e-4)
    grid = np.linspace(0.01, 0.99, 981)
    best = grid[np.argmax([heat_capacity_from_qg(q) for q in grid])]
    assert best == pytest.approx(qs, abs=2e-3)


def test_nernst_limit():
    for q in (0.999, 0.99999):
        assert heat_capacity_from_qg(q) < 0.05
    assert heat_capacity_from_qg(0.99999) < heat_capacity_from_qg(0.999)


def test_raw_thermometer_fails_and_restricted_posteriors_do_not():
    assert estimate_temperature(10, 10, "raw") == 0.0
    assert math.isinf(estimate_temperature(5, 10, "raw"))
    for m in ("haar", "jeffreys"):
        for k in (0, 5, 10):
            t = estimate_temperature(k, 10, m)
            assert 0.0 < t < math.inf
    r = thermometry_table(0.99, 100)
    assert r["raw"]["failure"] > 0.5
    assert r["haar"]["failure"] == 0.0 and r["jeffreys"]["failure"] == 0.0


def test_estimators_agree_with_many_shots():
    r = thermometry_table(optimal_probe_qg(), 400)
    errs = [r[m]["median_rel_error"] for m in ("raw", "haar", "jeffreys")]
    assert max(errs) - min(errs) < 0.01
    assert max(errs) < 0.1


def test_effective_qubit_temperature():
    assert effective_temperature_mK(0.98, 5.0) == pytest.approx(52.2, abs=0.2)


def test_classical_ising_entropy_is_the_qg_decomposition():
    for b in (0.1, 0.5, 2.0):
        r = ising_row(b, n=4)
        assert r["gap_bits"] == pytest.approx(0.0, abs=1e-10)


def test_transverse_field_breaks_the_decomposition():
    weak, strong = ising_row(1.5, n=4, g=0.5), ising_row(1.5, n=4, g=1.5)
    assert 0 < weak["gap_bits"] < strong["gap_bits"]


def test_mean_field_overshoots_the_exact_ring():
    r = ising_row(0.4, n=6)
    assert r["qg_free"] < r["qg_exact"] < r["qg_mean_field"]
    assert mean_field_qg(0.4, 0.0, 1.0) == pytest.approx(math.tanh(0.4))


def test_purification_circuit_matches_tanh():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    from thermal_states_qg_tanh import sampled_qg

    for b in (0.2, 1.0):
        assert sampled_qg(b, shots=20000) == pytest.approx(math.tanh(b), abs=0.02)
