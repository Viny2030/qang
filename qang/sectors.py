"""
qang.sectors -- conserved-sector tools for the qg filter (RESEARCH_NOTES §20, §60, §66, §68).

The register-mean witness and the shot-level filter of qang rest on a conserved
Hamming weight N. Two practical questions follow, and this module answers them
without any noise model:

  * ``filter_distribution``: the filter as an operation on an outcome
    distribution (or counts): keep the outcomes with Hamming weight N and
    renormalize; also returns the kept fraction.
  * ``sector_exposure``: how far a circuit's intermediate states leave the
    conserved sector right after its multi-qubit gates, from a noiseless
    statevector simulation. A decay that happens while part of the state is
    outside the sector can be rotated back into it and pass the filter
    (§60): the exposure is the population at risk.

Only ``sector_exposure`` needs Qiskit; importing the module does not.
"""

from __future__ import annotations

import numpy as np


def hamming_weights(n_qubits: int) -> np.ndarray:
    """Hamming weight of every basis index (bit i of the index = qubit i)."""
    idx = np.arange(2**n_qubits)
    return ((idx[:, None] >> np.arange(n_qubits)) & 1).sum(axis=1)


def filter_distribution(probs, n_qubits: int, weight: int):
    """The qg filter as an operation: keep outcomes of Hamming weight ``weight``.

    ``probs`` may be probabilities or counts over the 2**n_qubits outcomes.
    Returns (filtered distribution, normalized to 1, and the kept fraction).
    """
    p = np.asarray(probs, dtype=float)
    keep = hamming_weights(n_qubits) == weight
    kept = p[keep].sum()
    total = p.sum()
    out = np.where(keep, p, 0.0)
    if kept > 0:
        out = out / kept
    return out, float(kept / total) if total > 0 else 0.0


def sector_exposure(circuit, weight: int, min_qubits: int = 2):
    """Population outside the Hamming-weight-``weight`` sector right after each
    gate acting on at least ``min_qubits`` qubits (noiseless statevector).

    Returns a dict with the mean and maximum exposure over those gates and the
    per-gate list. A circuit whose multi-qubit gates all conserve the weight at
    every point has exposure 0; basis changes around two-qubit gates raise it.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    n = circuit.num_qubits
    outside = hamming_weights(n) != weight
    sv = Statevector.from_label("0" * n)
    per_gate = []
    for inst in circuit.data:
        op = inst.operation
        if op.name in ("measure", "barrier", "save_probabilities", "save_statevector", "save_density_matrix"):
            continue
        qidx = [circuit.find_bit(q).index for q in inst.qubits]
        sub = QuantumCircuit(n)
        sub.append(op, qidx)
        sv = sv.evolve(sub)
        if len(qidx) >= min_qubits:
            per_gate.append(float(np.sum(np.abs(sv.data[outside]) ** 2)))
    arr = np.array(per_gate) if per_gate else np.zeros(1)
    return {"mean": float(arr.mean()), "max": float(arr.max()), "per_gate": per_gate}
