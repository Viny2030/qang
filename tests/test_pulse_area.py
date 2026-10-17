"""Tests for qang.pulse_area (§160)."""

import math

import numpy as np
import pytest

from qang import pulse_area as PA


def test_minimal_area_angle_range():
    for t in np.linspace(-20, 20, 81):
        for period in (2 * math.pi, math.pi):
            r, k = PA.minimal_area_angle(t, period)
            assert -period / 2 < r <= period / 2 + 1e-12
            assert r + k * period == pytest.approx(t)


def test_random_amplitude_damping_formula():
    rng = np.random.default_rng(0)
    eps = rng.normal(0, 0.05, 200000)
    for th in (0.5, 2.0, 5.0):
        assert np.mean(np.cos(th * (1 + eps))) == pytest.approx(PA.rx_qg_z_random_amplitude(th, 0.05), abs=3e-3)
    assert PA.qg_sensitivity(math.pi / 2) == pytest.approx(-1.0)


def test_rewrite_preserves_the_unitary_and_lowers_the_area():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    rng = np.random.default_rng(1)
    for _ in range(20):
        qc = QuantumCircuit(3)
        for _ in range(6):
            a, b = (int(x) for x in rng.choice(3, 2, replace=False))
            qc.rx(rng.uniform(-9, 9), a)
            qc.rzz(rng.uniform(-9, 9), a, b)
            qc.ryy(rng.uniform(-9, 9), a, b)
        m = PA.minimal_area_circuit(qc)
        assert np.allclose(Operator(m).data, Operator(qc).data, atol=1e-10)
        assert PA.area(m) <= PA.area(qc) + 1e-12
