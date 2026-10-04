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

  * ``SectorExposurePass``: the same screen as a Qiskit transpiler analysis
    pass; it writes the result to ``property_set["sector_exposure"]``, so it
    can sit in a PassManager next to the passes that choose a compilation.

  * ``echo_transfer_matrix`` and ``unmix_sector``: echo calibration of the
    errors the filter cannot see, those that move an excitation inside the
    sector (§105). Echo circuits (each sector basis state, the circuit and its
    inverse) give the transfer matrix M between sector states; inverting
    M^power on the filtered distribution mitigates the in-sector error, as
    readout mitigation does for bit flips.

Only ``sector_exposure`` and ``SectorExposurePass`` need Qiskit; importing the
module does not (``unmix_sector`` uses SciPy).
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


try:  # the analysis pass needs Qiskit; the rest of the module does not
    from qiskit.transpiler.basepasses import AnalysisPass as _AnalysisPass

    _QISKIT = True
except ImportError:  # pragma: no cover - exercised only when qiskit is absent
    _QISKIT = False

if _QISKIT:

    class SectorExposurePass(_AnalysisPass):
        """Transpiler analysis pass: records ``sector_exposure`` of the circuit
        being compiled in ``property_set["sector_exposure"]`` (RFC v2, Section 3.3).

        It does not change the circuit. Statevector cost: use it on circuits of
        up to about 20 qubits. The exposure ranks compilations by how much T1
        error the Hamming-weight filter will let through (RESEARCH_NOTES §68); it
        is a screen, not an error model."""

        def __init__(self, weight: int, min_qubits: int = 2):
            super().__init__()
            self.weight = weight
            self.min_qubits = min_qubits

        def run(self, dag):
            from qiskit.converters import dag_to_circuit

            self.property_set["sector_exposure"] = sector_exposure(dag_to_circuit(dag), self.weight, self.min_qubits)
            return dag

else:  # pragma: no cover

    class SectorExposurePass:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("SectorExposurePass requires Qiskit: pip install qiskit")


# --------------------------------------------------------------------- #
# echo calibration of in-sector errors (§105)
# --------------------------------------------------------------------- #
def sector_states(n_qubits: int, weight: int) -> np.ndarray:
    """Basis indices of the weight-``weight`` sector, in increasing order."""
    return np.where(hamming_weights(n_qubits) == weight)[0]


def echo_transfer_matrix(echo_probs, n_qubits: int, weight: int, prepared=None) -> np.ndarray:
    """Transfer matrix between sector states from echo runs.

    ``echo_probs[j]`` is the outcome distribution (or counts, length 2**n) of an
    echo circuit that prepared the sector basis state ``prepared[j]``
    (default: every state of ``sector_states`` in order). Column j of the
    result is the filtered distribution of run j over the sector states, so
    M[i, j] = P(measured sector state i | prepared j). With fewer echo runs
    than sector states, the prepared columns are filled and the others are
    left as the identity (no correction for them)."""
    states = sector_states(n_qubits, weight)
    pos = {int(s): i for i, s in enumerate(states)}
    prepared = states if prepared is None else np.asarray(prepared)
    if len(prepared) != len(echo_probs):
        raise ValueError("need one echo distribution per prepared state.")
    M = np.eye(len(states))
    for s, p in zip(prepared, echo_probs):
        f, _ = filter_distribution(p, n_qubits, weight)
        M[:, pos[int(s)]] = f[states]
    return M


def matrix_power_stochastic(M, power: float) -> np.ndarray:
    """Real, non-negative, column-normalized principal power of a transfer matrix
    (power 0.5: the forward half of an echo)."""
    from scipy.linalg import fractional_matrix_power

    P = np.clip(np.real(fractional_matrix_power(np.asarray(M, float), power)), 0.0, None)
    return P / P.sum(axis=0, keepdims=True)


def unmix_sector(probs, M, n_qubits: int, weight: int, power: float = 0.5) -> np.ndarray:
    """Filter ``probs`` to the sector and undo the in-sector errors measured by
    ``echo_transfer_matrix``: non-negative least squares for M^power x = f,
    normalized. Returns a full-length distribution (zero outside the
    sector). power 0.5 models a circuit with half the echo's gates; 1.0 uses
    M itself."""
    from scipy.optimize import nnls

    states = sector_states(n_qubits, weight)
    f, _ = filter_distribution(probs, n_qubits, weight)
    A = np.asarray(M, float) if power == 1.0 else matrix_power_stochastic(M, power)
    x, _ = nnls(A, f[states])
    out = np.zeros(2**n_qubits)
    out[states] = x / x.sum() if x.sum() > 0 else 1.0 / len(states)
    return out
