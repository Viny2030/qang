"""
Tests for examples/hubbard_trotter_qg_filters.py: 1D Hubbard Trotter
dynamics with the N and spin-resolved qg filters vs ZNE.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("scipy")

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Operator, SparsePauliOp  # noqa: E402
from scipy.linalg import expm  # noqa: E402

import hubbard_trotter_qg_filters as H  # noqa: E402


def _number_op(qubits):
    n = H.N_QUBITS
    terms = [("I" * n, len(qubits) / 2)]
    for q in qubits:
        s = ["I"] * n
        s[n - 1 - q] = "Z"
        terms.append(("".join(s), -0.5))
    return SparsePauliOp.from_list(terms).to_matrix()


def test_hamiltonian_conserves_both_spin_numbers():
    h = H.hamiltonian().to_matrix()
    for qubits in (H.UP, H.DOWN):
        n = _number_op(qubits)
        assert np.abs(h @ n - n @ h).max() < 1e-12


def test_trotter_step_converges_to_exact_propagator():
    errs = []
    for dt in (0.02, 0.01):
        qc = QuantumCircuit(H.N_QUBITS)
        H.trotter_step(qc, dt=dt)
        u, e = Operator(qc).data, expm(-1j * dt * H.hamiltonian().to_matrix())
        ph = np.vdot(e.ravel(), u.ravel())
        errs.append(np.abs(u - ph / abs(ph) * e).max())
    assert errs[1] < 1e-3 and errs[0] / errs[1] == pytest.approx(4, rel=0.1)  # O(dt^2) per step


def test_initial_observables_and_symmetry_of_ideal_output():
    p0 = H.ideal_probabilities(0)
    assert H.charge_imbalance(p0) == pytest.approx(1.0)
    assert H.double_occupancy(p0) == pytest.approx(0.5)
    for steps in (1, 3):
        qg_up, qg_down, kept_n, kept_s = H.spin_witnesses(H.ideal_probabilities(steps))
        assert qg_up == pytest.approx(0, abs=1e-12) and qg_down == pytest.approx(0, abs=1e-12)
        assert kept_s == pytest.approx(1.0)
    pe = H.exact_probabilities(2 * H.DT)
    assert H.charge_imbalance(pe) == pytest.approx(H.charge_imbalance(H.ideal_probabilities(2)), abs=0.01)


def test_filters_and_noise_floor():
    uniform = np.full(2**H.N_QUBITS, 1 / 2**H.N_QUBITS)
    assert H.double_occupancy(H.spin_filter(uniform)) == pytest.approx(0.25)
    assert H.charge_imbalance(H.spin_filter(uniform)) == pytest.approx(0.0, abs=1e-12)
    kept = H.spin_filter(uniform)
    assert kept[(H._N_UP != 2) | (H._N_DOWN != 2)].sum() == 0
    assert H.n_filter(uniform)[(H._N_UP + H._N_DOWN) != 4].sum() == 0


def test_folding_preserves_the_unitary():
    from qiskit import transpile

    qc = transpile(H.trotter_circuit(1, measure=False), basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
    assert Operator(H.fold_two_qubit(qc, 3)).equiv(Operator(qc))
    with pytest.raises(ValueError):
        H.fold_two_qubit(qc, 2)


def test_all_to_all_spin_filters_and_zne():
    t = H.error_table((4, 8), seeds=(11, 12, 13), device="all_to_all")
    e8 = t[8]
    d = lambda m: e8[("double_occ", m)][0]  # noqa: E731
    assert abs(e8["qg_up"]) < 0.01 and abs(e8["qg_down"]) < 0.01  # unital noise: no drift
    assert e8["kept_spin"] < 0.6 and e8["spin_leak"] > 0.04
    assert d("spin_filter") < 0.85 * d("n_filter") < 0.85 * d("raw")
    assert d("zne_spin") < d("raw") / 3


def test_fake_brisbane_routing_creates_spin_leak_that_the_spin_filter_removes():
    pytest.importorskip("qiskit_ibm_runtime")
    t = H.error_table((1, 2), seeds=(11, 12, 13), device="brisbane")
    for steps in (1, 2):
        e = t[steps]
        assert e["spin_leak"] > 0.05
        for obs in ("double_occ", "imbalance"):
            assert e[(obs, "spin_filter")][0] < e[(obs, "n_filter")][0] < e[(obs, "raw")][0]
        assert e[("double_occ", "zne_spin")][0] < 0.02
    assert max(t[2]["qg_up"], t[2]["qg_down"]) > 0.03  # T1 drift towards +1 is visible
