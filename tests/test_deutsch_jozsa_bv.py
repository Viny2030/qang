"""
Tests for the Deutsch-Jozsa and Bernstein-Vazirani algorithms, reformulated
in terms of qg_Z (Future Research Direction, Section 6.2 of the paper;
GitHub issue: "Verify qg_Z reformulation of Deutsch-Jozsa and
Bernstein-Vazirani").

Both algorithms terminate in a computational-basis measurement of the input
register that resolves a population bias -- the same measurement structure
already treated for Grover's algorithm in Section 4.3 of the paper. This
file shows that qang.multiqubit.per_qubit_qg_z of the final input register
gives each algorithm's answer directly, with no separate decoding step:

  - Deutsch-Jozsa: constant f  -> qg_Z = +1.0 for every input qubit.
                   balanced f  -> at least one input qubit has qg_Z = -1.0.
  - Bernstein-Vazirani: qg_Z of qubit i directly encodes secret bit s_i:
                   qg_Z = +1.0 -> s_i = "0",  qg_Z = -1.0 -> s_i = "1".

Both algorithms are exact (no shot noise): every input qubit's qg_Z lands
on a pole (+-1.0) to machine precision, never an intermediate value.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qang.multiqubit import per_qubit_qg_z


def _qg_z_profile(qc: QuantumCircuit, n_qubits: int):
    """qg_Z of every qubit, indexed to match Qiskit's own qubit numbering
    (reverse_qargs() undoes Qiskit's little-endian statevector convention,
    so profile[i] here is qubit i's own qg_Z, not the reversed index)."""
    sv = Statevector.from_instruction(qc).reverse_qargs().data
    return per_qubit_qg_z(sv, n_qubits=n_qubits)


def _dj_oracle(qc: QuantumCircuit, n: int, kind: str):
    """Append a Deutsch-Jozsa oracle to a circuit with n input qubits
    (0..n-1) plus one ancilla (qubit n).

    kind:
      "constant_0"       f(x) = 0 for all x            (no gates)
      "constant_1"       f(x) = 1 for all x            (X on ancilla)
      "balanced_parity"  f(x) = x_0 xor ... xor x_{n-1} (CX from every input)
      "balanced_single"  f(x) = x_0                     (CX from qubit 0 only)
    """
    if kind == "constant_0":
        return
    if kind == "constant_1":
        qc.x(n)
        return
    if kind == "balanced_parity":
        for i in range(n):
            qc.cx(i, n)
        return
    if kind == "balanced_single":
        qc.cx(0, n)
        return
    raise ValueError(f"unknown oracle kind {kind!r}")


def _deutsch_jozsa_circuit(n: int, kind: str) -> QuantumCircuit:
    qc = QuantumCircuit(n + 1)
    qc.x(n)
    qc.h(range(n + 1))
    _dj_oracle(qc, n, kind)
    qc.h(range(n))
    return qc


def _bernstein_vazirani_circuit(secret: str) -> QuantumCircuit:
    n = len(secret)
    qc = QuantumCircuit(n + 1)
    qc.x(n)
    qc.h(range(n + 1))
    for i, bit in enumerate(secret):
        if bit == "1":
            qc.cx(i, n)
    qc.h(range(n))
    return qc


@pytest.mark.parametrize("kind", ["constant_0", "constant_1"])
def test_deutsch_jozsa_constant_gives_all_plus_one_qg_z(kind):
    n = 3
    qc = _deutsch_jozsa_circuit(n, kind)
    profile = _qg_z_profile(qc, n_qubits=n + 1)[:n]  # input qubits only
    for qg_z in profile:
        assert qg_z == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("kind", ["balanced_parity", "balanced_single"])
def test_deutsch_jozsa_balanced_gives_at_least_one_minus_one_qg_z(kind):
    n = 3
    qc = _deutsch_jozsa_circuit(n, kind)
    profile = _qg_z_profile(qc, n_qubits=n + 1)[:n]
    # exactness of the DJ promise: every input qubit is on a pole, never a mix
    for qg_z in profile:
        assert abs(abs(qg_z) - 1.0) < 1e-9
    assert any(qg_z == pytest.approx(-1.0, abs=1e-9) for qg_z in profile)


@pytest.mark.parametrize("secret", ["101", "0000", "11111", "010101"])
def test_bernstein_vazirani_recovers_secret_from_qg_z(secret):
    n = len(secret)
    qc = _bernstein_vazirani_circuit(secret)
    profile = _qg_z_profile(qc, n_qubits=n + 1)[:n]

    recovered_bits = []
    for qg_z in profile:
        assert abs(abs(qg_z) - 1.0) < 1e-9  # exact pole, no ambiguity
        recovered_bits.append("0" if qg_z > 0 else "1")
    recovered_secret = "".join(recovered_bits)

    assert recovered_secret == secret


if __name__ == "__main__":
    print("Run via `pytest tests/test_deutsch_jozsa_bv.py -v` for full parametrized coverage.")
    # A quick, non-parametrized smoke check:
    assert all(
        v == pytest.approx(1.0, abs=1e-9)
        for v in _qg_z_profile(_deutsch_jozsa_circuit(3, "constant_0"), 4)[:3]
    )
    n = len("101")
    profile = _qg_z_profile(_bernstein_vazirani_circuit("101"), n + 1)[:n]
    recovered = "".join("0" if v > 0 else "1" for v in profile)
    assert recovered == "101"
    print("Smoke check passed.")
