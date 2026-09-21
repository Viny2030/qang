"""
Randomized Benchmarking (RB), reformulated in terms of qg_Z: the second
connection between the qang framework and an industry-standard
benchmarking protocol (after Quantum Volume / qg_S in
examples/quantum_volume_qg_s.py).

Standard single-qubit Clifford RB (Magesan, Gambetta, & Emerson, 2011,
"Scalable and Robust Randomized Benchmarking of Quantum Processes",
Phys. Rev. Lett. 106, 180504) works by running sequences of m random
Clifford gates followed by the one Clifford that inverts the whole
sequence, so the noiseless circuit is always the identity. Averaged over
many random sequences, the survival probability decays as
p(m) = A * f^m + B, and the decay rate f gives the average gate
fidelity F_avg = (1 + f) / 2 (single qubit, d = 2) -- a Clifford-
sequence-independent measure of the underlying per-gate noise, because
the Clifford twirl converts an arbitrary noise channel into an
effective depolarizing channel.

The reformulation here: for the idealized case where the per-gate noise
already *is* a depolarizing channel epsilon(rho) = (1 - p) rho + p * I/2,
the survival probability is exactly the qg_Z of the final state
(P(0) - P(1) after the recovery gate), and it has an EXACT closed form
that random Clifford Benchmarking's own headline formula only
approximates in general:

    qg_Z(m) = (1 - p) ** (m + 1)        (m random Cliffords + 1 recovery)

This holds bit-for-bit regardless of which random Cliffords were drawn
(demonstrated below by sweeping the random seed at fixed m and finding
zero variation), because a depolarizing channel shrinks the Bloch
vector's magnitude by a fixed factor independent of its direction, and
every Clifford in between is a pure rotation that preserves that
magnitude -- so only the *total* number of noisy gates matters, not
their order or identity. This is exactly why RB's own randomization
step works in the first place (it is designed to convert whatever the
real per-gate noise is into something behaving like this idealized
case); here, starting from an already-depolarizing model, the
Clifford-independence is exact rather than an averaged-over-many-
sequences approximation.

Fitting log(qg_Z(m)) linearly in (m + 1) recovers the depolarizing
parameter f = 1 - p to machine precision from noiseless-simulation data,
and F_avg = (1 + f) / 2 then follows directly -- the same
extraction procedure real RB experiments use, applied to the
qg_Z-native quantity instead of a raw measured "0" count.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, random_clifford


def build_rb_blocks(sequence_length: int, seed: int) -> List[QuantumCircuit]:
    """
    ``sequence_length`` random single-qubit Clifford circuits, plus one
    final recovery circuit (the exact inverse of their composition), so
    that composing every block in order is the identity on |0> in the
    noiseless case (checked explicitly in tests/test_randomized_benchmarking_qg_z.py).
    """
    qc = QuantumCircuit(1)
    blocks: List[QuantumCircuit] = []
    for i in range(sequence_length):
        clifford = random_clifford(1, seed=seed * 100_000 + i)
        block = clifford.to_circuit()
        blocks.append(block)
        qc.compose(block, inplace=True)
    recovery = qc.inverse()
    blocks.append(recovery)
    return blocks


def rb_qg_z_survival(sequence_length: int, seed: int, p_depol: float) -> float:
    """
    qg_Z of the final state after running ``build_rb_blocks``'s blocks in
    order, applying the single-qubit depolarizing channel
    rho -> (1 - p_depol) * rho + p_depol * I / 2 after *every* block
    (including the recovery block) -- (m + 1) noisy applications total
    for a sequence of length m.
    """
    if not (0.0 <= p_depol <= 1.0):
        raise ValueError(f"p_depol must lie in [0, 1], got {p_depol}.")
    blocks = build_rb_blocks(sequence_length, seed)
    rho = DensityMatrix.from_label("0")
    identity_over_2 = np.eye(2, dtype=complex) / 2.0
    for block in blocks:
        rho = rho.evolve(block)
        data = (1.0 - p_depol) * rho.data + p_depol * identity_over_2
        rho = DensityMatrix(data)
    p0 = float(np.real(rho.data[0, 0]))
    p1 = float(np.real(rho.data[1, 1]))
    return p0 - p1


def fit_depolarizing_parameter(
    sequence_lengths: Sequence[int], qg_z_values: Sequence[float]
) -> float:
    """
    Recover f = 1 - p_depol from a linear fit of ln(qg_Z(m)) against
    (m + 1), i.e. of ln(f) * (m + 1) -- the same log-linear extraction
    real RB experiments perform on their averaged survival-probability
    data.
    """
    x = np.asarray(sequence_lengths, dtype=float) + 1.0
    y = np.log(np.asarray(qg_z_values, dtype=float))
    slope, _intercept = np.polyfit(x, y, 1)
    return float(np.exp(slope))


def average_gate_fidelity(f: float, dimension: int = 2) -> float:
    """Standard RB fidelity formula (Magesan et al. 2011): the average
    single-qubit gate fidelity implied by depolarizing parameter f."""
    return (1.0 + (dimension - 1) * f) / dimension


if __name__ == "__main__":
    p_depol = 0.05
    sequence_lengths = [0, 2, 5, 10, 20, 40]

    print(f"True depolarizing parameter p_depol = {p_depol} (f_true = {1 - p_depol:.4f})")
    print()
    print("Clifford-seed independence check (same m, different random sequences):")
    for m in [0, 5, 10]:
        vals = [rb_qg_z_survival(m, seed, p_depol) for seed in range(4)]
        print(f"  m={m}: qg_Z across 4 random seeds = {[f'{v:.10f}' for v in vals]}")
    print()

    qg_z_values = [rb_qg_z_survival(m, seed=0, p_depol=p_depol) for m in sequence_lengths]
    for m, qgz in zip(sequence_lengths, qg_z_values):
        print(f"  m={m:3d}  qg_Z(m) = {qgz:.6f}  (1-p)^(m+1) = {(1 - p_depol) ** (m + 1):.6f}")

    f_fit = fit_depolarizing_parameter(sequence_lengths, qg_z_values)
    print()
    print(f"Fitted f = {f_fit:.8f}  (true f = {1 - p_depol:.8f})")
    print(f"Fitted average gate fidelity F_avg = {average_gate_fidelity(f_fit):.8f}")
