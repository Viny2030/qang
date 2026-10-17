"""Tests for qang.fisher (§159)."""

import math

import numpy as np
import pytest

from qang import fisher as F


def test_one_qubit_fisher_is_flat_in_the_angle():
    for th in (0.2, 0.9, 1.6, 2.5):
        assert F.fisher_theta(th) == pytest.approx(1.0, abs=1e-12)
        assert F.fisher_qg(math.cos(th)) == pytest.approx(1 / math.sin(th) ** 2, rel=1e-12)
        assert F.cramer_rao_qg(math.cos(th), 100) == pytest.approx(math.sin(th) ** 2 / 100)


def test_one_evaluation_cannot_give_the_derivative():
    for th in (0.3, 1.1):
        for s in (0.4, math.pi / 2, 2.0):
            f1, f2, d1, d2 = F.indistinguishable_pair(th, s)
            assert f1(th + s) == pytest.approx(f2(th + s), abs=1e-12)
            assert abs(d1 - d2) > 1e-3
            assert F.shift_gradient(f1, th, s) == pytest.approx(d1, abs=1e-12)
            assert F.shift_gradient(f2, th, s) == pytest.approx(d2, abs=1e-12)


def test_real_ansatz_metric_diagonal_is_one_quarter_and_blocks_are_exact():
    rng = np.random.default_rng(1)
    for n, L in ((3, 2), (4, 3)):
        th = rng.uniform(0, 2 * math.pi, n * L)
        G = F.full_metric(th, n, L)
        B = F.block_metric(th, n, L)
        assert np.allclose(np.diag(G), 0.25, atol=1e-12)
        mask = B != 0
        assert np.allclose(G[mask], B[mask], atol=1e-12)


def test_gradient_matches_finite_difference():
    n, L = 3, 2
    H = F.tfim_hamiltonian(n)
    th = np.random.default_rng(2).uniform(0, 2 * math.pi, n * L)
    g = F.gradient(th, H, n, L)
    for i in range(th.size):
        e = np.zeros_like(th)
        e[i] = 1e-6
        fd = (F.energy(th + e, H, n, L) - F.energy(th - e, H, n, L)) / 2e-6
        assert g[i] == pytest.approx(fd, abs=1e-6)


def test_block_metric_from_exact_counts():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    n, L = 3, 2
    th = np.random.default_rng(3).uniform(0, 2 * math.pi, n * L)
    exact = F.block_metric(th, n, L)
    for layer in range(L):
        qc = F.y_basis_circuit(th, n, L, layer)
        qc.remove_final_measurements()
        c = {k: int(round(v * 1e9)) for k, v in Statevector(qc).probabilities_dict().items()}
        blk = exact[layer * n:(layer + 1) * n, layer * n:(layer + 1) * n]
        assert np.allclose(F.block_metric_from_counts(c, n), blk, atol=1e-6)
