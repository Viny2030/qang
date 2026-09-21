"""
Quantum Phase Estimation (QPE), connected to qg_Phi: closes the loop on
qang.phase (the QangPhi unit formalized from the paper's Section 6
"Deferred" note) by using it in an actual multi-qubit algorithm, rather
than only as a standalone closed-form unit.

Standard QPE (Kitaev, 1995; textbook form e.g. Nielsen & Chuang, Section
5.2) estimates the phase phi of an eigenvalue e^{i*2*pi*phi} of a
unitary U, given one of its eigenstates, using n counting qubits
initialized to |+>, controlled-U^(2^k) applications, and an inverse
Quantum Fourier Transform. The n-qubit counting register's most likely
measured bitstring y gives phi_estimate = y / 2^n, exact whenever
phi * 2^n is an integer, and within 1 / 2^n of the true phase otherwise.

The connection made here: running QPE on U = a phase gate P(2*pi*phi)
with its |1> eigenstate recovers exactly the phi_estimate that
qang.phase.QangPhi already takes as its defining parameter --
QangPhi(phi_estimate).value should reproduce U's own eigenvalue
e^{i*2*pi*phi} to the precision QPE allows. In particular, running QPE
on the T, S, and Z gates (three of qang.phase.ANCHOR_POINTS' four
non-trivial anchors) recovers their anchor phi values (1/8, 1/4, 1/2)
*exactly* (probability 1, zero estimation error) whenever the counting
register has enough qubits to represent them exactly in binary -- QPE
is doing algorithmically, on an actual circuit, exactly what
qang.phase.QangPhi.from_complex() does by closed-form phase extraction.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector

from qang.phase import ANCHOR_POINTS, QangPhi


def qpe_circuit(n_counting: int, phase: float) -> QuantumCircuit:
    """
    QPE circuit for U = P(2*pi*phase) (a single-qubit phase gate),
    starting the target qubit in U's own |1> eigenstate. Counting
    qubits are 0..n_counting-1; the target qubit is n_counting.
    """
    if n_counting < 1:
        raise ValueError(f"n_counting must be >= 1, got {n_counting}.")
    target = n_counting
    qc = QuantumCircuit(n_counting + 1, name=f"qpe_{n_counting}")
    qc.x(target)  # |1>, the eigenstate of P(theta) with eigenvalue e^{i*theta}
    for q in range(n_counting):
        qc.h(q)
    for k in range(n_counting):
        angle = 2.0 * np.pi * phase * (2 ** k)
        qc.cp(angle, k, target)
    qc.append(QFTGate(n_counting).inverse(), list(range(n_counting)))
    return qc


def decode_phase_estimate(qc: QuantumCircuit, n_counting: int) -> Tuple[float, float]:
    """
    Deterministic decoding (exact statevector simulation, no sampling):
    returns (phi_estimate, probability_of_that_outcome), reading off the
    most probable n_counting-bit outcome of the counting register.
    """
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities(qargs=list(range(n_counting)))
    y = int(np.argmax(probs))
    return y / (2 ** n_counting), float(probs[y])


def qpe_estimate(n_counting: int, phase: float) -> Tuple[float, float]:
    """Convenience wrapper: build the QPE circuit for ``phase`` and
    decode its phase estimate in one call."""
    qc = qpe_circuit(n_counting, phase)
    return decode_phase_estimate(qc, n_counting)


def qpe_recovers_qang_phi(n_counting: int, phase: float) -> Tuple[QangPhi, complex]:
    """
    Run QPE, build a QangPhi from the recovered phi_estimate, and return
    (that QangPhi, the true eigenvalue e^{i*2*pi*phase} it should match).
    """
    phi_estimate, _probability = qpe_estimate(n_counting, phase)
    qg_phi = QangPhi(phi_estimate)
    true_eigenvalue = complex(np.exp(1j * 2.0 * np.pi * phase))
    return qg_phi, true_eigenvalue


if __name__ == "__main__":
    print("QPE on the qang.phase.ANCHOR_POINTS gates (exact recovery expected):")
    for name, phi_true in ANCHOR_POINTS.items():
        if phi_true == 0.0:
            continue  # identity: trivial, no counting qubits needed
        n_counting = 4
        qg_phi, true_eigenvalue = qpe_recovers_qang_phi(n_counting, phi_true)
        print(
            f"  {name:10s}  phi_true={phi_true:.4f}  "
            f"QPE qg_Phi={qg_phi.value:+.4f}  true={true_eigenvalue:+.4f}  "
            f"|diff|={abs(qg_phi.value - true_eigenvalue):.2e}"
        )
    print()

    print("QPE on a generic (non-exact) phase, error bounded by 1/2^n_counting:")
    phase = 0.3
    for n_counting in [4, 6, 8, 10]:
        phi_estimate, prob = qpe_estimate(n_counting, phase)
        bound = 1.0 / (2 ** n_counting)
        print(
            f"  n_counting={n_counting:2d}  phi_estimate={phi_estimate:.6f}  "
            f"error={abs(phi_estimate - phase):.2e}  bound=1/2^n={bound:.2e}  "
            f"P(this outcome)={prob:.4f}"
        )
