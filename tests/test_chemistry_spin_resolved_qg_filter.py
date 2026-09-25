"""
Tests for examples/chemistry_spin_resolved_qg_filter.py: one qg filter
(total N) vs two (N_up, N_down) on H2.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("scipy")

from chemistry_spin_resolved_qg_filter import (  # noqa: E402
    SPIN_MASK,
    TOTAL_MASK,
    UP,
    DOWN,
    register_qg,
    run_case,
)
from chemistry_qg_symmetry_witness import FCI_ENERGY, H2_JW, ansatz, optimal_angle  # noqa: E402

SEEDS = (11, 12, 13)


def _mean(noise):
    runs = [run_case(noise, 0.0, 20000, s) for s in SEEDS]
    out = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    for k in ("readout", "total", "spin"):
        out[k] = 1e3 * (out[k] - FCI_ENERGY)
    return out


def test_masks():
    as_bits = lambda m: sorted(format(i, "04b") for i in range(16) if m[i])  # noqa: E731
    assert as_bits(SPIN_MASK) == ["0011", "0110", "1001", "1100"]
    assert as_bits(TOTAL_MASK & ~SPIN_MASK) == ["0101", "1010"]


def test_hamiltonian_conserves_each_spin_number():
    from qiskit.quantum_info import SparsePauliOp

    for qubits in (UP, DOWN):
        n_op = SparsePauliOp.from_list(
            [("IIII", len(qubits) / 2)] + [("".join("Z" if 3 - k == q else "I" for k in range(4)), -0.5) for q in qubits])
        comm = (H2_JW @ n_op - n_op @ H2_JW).simplify(atol=1e-12)
        assert np.allclose(comm.coeffs, 0.0)


def test_ideal_state_has_zero_qg_in_each_spin_register():
    from qiskit.quantum_info import Statevector

    p = np.abs(Statevector(ansatz(optimal_angle())).data) ** 2
    assert register_qg(p, UP) == pytest.approx(0.0, abs=1e-12)
    assert register_qg(p, DOWN) == pytest.approx(0.0, abs=1e-12)
    assert np.sum(p * SPIN_MASK) == pytest.approx(1.0)


def test_spin_filters_help_under_depolarizing_noise():
    r = _mean(("depolarizing", 0.10))
    assert r["spin_leak"] > 0.03
    assert r["spin"] < 0.8 * r["total"]
    assert r["qg_up"] < 0 < r["qg_down"]


def test_filters_coincide_when_there_is_no_spin_leak():
    r = _mean(("dephasing", 0.10))
    assert r["spin_leak"] == pytest.approx(0.0, abs=1e-12)
    assert r["spin"] == pytest.approx(r["total"], abs=1e-9)
