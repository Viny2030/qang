"""
Tests for qang.phase.QangPhi, the formalized qg_Phi closed-form phase unit
(Future Research Direction, "Deferred" note at the end of Section 6 of the
paper; GitHub issue: "Formalize qg_Phi: a closed-form phase unit").

Three things are checked:

  1. Pure algebra: qg_Phi(phi) = e^(i*2*pi*phi), and unlike qg_Z / qg_S,
     it is invertible over its FULL domain [0, 1) -- round-tripping
     phi -> value -> phi never needs a branch choice or half-domain
     restriction.
  2. The named anchor points reproduce the standard single-qubit phase
     gates (I, T, S, Z, S-dagger) exactly, analogous to Table 1 (qg_Z)
     in the paper.
  3. Independent cross-check against a real circuit: a Hadamard test on a
     controlled-phase gate CP(theta) applied to an eigenstate |1>
     recovers, via two qg_Z-valued measurements (the real and imaginary
     parts), exactly the same phi that QangPhi predicts from theta alone
     -- validating the unit against circuit simulation, not just its own
     algebra, the same way qg_Z was validated against the Grover
     statevector trajectory in Section 4.3.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import pytest

from qang.phase import QangPhi, ANCHOR_POINTS, anchor_value

qiskit = pytest.importorskip("qiskit")

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qang.multiqubit import per_qubit_qg_z


# --------------------------------------------------------------------- #
# 1. pure algebra: closed form + full-domain invertibility
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("phi", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999])
def test_value_matches_closed_form(phi):
    qg = QangPhi(phi)
    expected = complex(math.cos(2 * math.pi * phi), math.sin(2 * math.pi * phi))
    assert abs(qg.value - expected) < 1e-12


@pytest.mark.parametrize("phi", [0.0, 0.1, 0.25, 0.33, 0.5, 0.75, 0.9])
def test_full_domain_round_trip_no_branch_needed(phi):
    """Unlike Qang.to_theta() (qg_Z) or Qang.theta_from_entropic() (qg_S),
    QangPhi.from_complex() needs no branch argument -- the whole point of
    Section 6's 'Deferred' note: this unit is invertible everywhere."""
    qg = QangPhi(phi)
    recovered = QangPhi.from_complex(qg.value)
    assert recovered.phi == pytest.approx(phi, abs=1e-9)


def test_negative_and_out_of_range_phi_normalizes_to_canonical_range():
    assert QangPhi(-0.25).phi == pytest.approx(0.75, abs=1e-9)
    assert QangPhi(1.25).phi == pytest.approx(0.25, abs=1e-9)


def test_from_complex_rejects_non_unit_modulus():
    with pytest.raises(ValueError):
        QangPhi.from_complex(0.5 + 0.5j)


def test_radians_round_trip_is_exact():
    for theta in [0.0, 0.3, math.pi, 2.1, 2 * math.pi - 0.01]:
        qg = QangPhi.from_radians(theta)
        assert qg.to_radians() == pytest.approx(theta % (2 * math.pi), abs=1e-9)


# --------------------------------------------------------------------- #
# 2. named anchor points reproduce the standard phase gates
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name, expected",
    [
        ("identity", 1.0 + 0.0j),
        ("t_gate", complex(math.cos(math.pi / 4), math.sin(math.pi / 4))),
        ("s_gate", 1.0j),
        ("z_gate", -1.0 + 0.0j),
        ("s_dagger", -1.0j),
    ],
)
def test_anchor_points_match_standard_gates(name, expected):
    assert abs(anchor_value(name) - expected) < 1e-9


def test_unknown_anchor_raises():
    with pytest.raises(ValueError):
        anchor_value("not_a_real_gate")


# --------------------------------------------------------------------- #
# 3. cross-check against a Hadamard-test circuit
# --------------------------------------------------------------------- #
def _qg_z_profile(qc: QuantumCircuit, n_qubits: int):
    sv = Statevector.from_instruction(qc).reverse_qargs().data
    return per_qubit_qg_z(sv, n_qubits=n_qubits)


def _hadamard_test(theta: float, imag_part: bool) -> float:
    """
    Standard Hadamard test for the phase e^{i*theta} of CP(theta) acting on
    the eigenstate |1> (qubit 1). Reading out qg_Z of the ancilla (qubit 0)
    gives cos(theta) directly (imag_part=False) or sin(theta)
    (imag_part=True, via an extra Sdg before the final Hadamard).
    """
    qc = QuantumCircuit(2)
    qc.x(1)  # target = |1>, an eigenstate of CP(theta) with eigenvalue e^{i*theta}
    qc.h(0)
    if imag_part:
        qc.sdg(0)
    qc.cp(theta, 0, 1)
    qc.h(0)
    return _qg_z_profile(qc, n_qubits=2)[0]


@pytest.mark.parametrize("phi", [0.0, 0.125, 0.25, 0.5, 0.75, 0.3, 0.9])
def test_hadamard_test_circuit_matches_qang_phi(phi):
    theta = phi * 2.0 * math.pi
    real_part = _hadamard_test(theta, imag_part=False)
    imag_part = _hadamard_test(theta, imag_part=True)

    expected = QangPhi(phi).value
    assert real_part == pytest.approx(expected.real, abs=1e-9)
    assert imag_part == pytest.approx(expected.imag, abs=1e-9)

    recovered_phi = QangPhi.from_complex(complex(real_part, imag_part)).phi
    assert recovered_phi == pytest.approx(phi, abs=1e-9)


if __name__ == "__main__":
    print("Run via `pytest tests/test_phase.py -v` for full parametrized coverage.")
    qg = QangPhi(0.25)
    assert abs(qg.value - 1.0j) < 1e-12
    assert QangPhi.from_complex(qg.value).phi == pytest.approx(0.25, abs=1e-9)
    print("Smoke check passed.")
