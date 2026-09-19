"""
quang.multiqubit — generalization of the qang unit to multi-qubit
tensor-product projection profiles.

This addresses the second half of Future Research Direction #4: "... and to
multi-qubit systems via tensor-product projection profiles."

Two complementary generalizations, both reducing exactly to the paper's
single-qubit qg_Z / qg_S when n_qubits == 1:

  * per-qubit local qang    ``per_qubit_qg_z`` / ``marginal_qg_s`` reduce the
                             full n-qubit state to each individual qubit's
                             reduced density matrix (partial trace) and
                             report that qubit's own qg_Z / qg_S -- a
                             "projection profile" across the register.
  * joint / marginal entropy ``joint_qg_s`` is the Shannon entropy of the
                             full 2^n computational-basis outcome
                             distribution (measuring every qubit); combined
                             with the per-qubit *von Neumann* marginal
                             entropy (``marginal_von_neumann_entropy``), the
                             gap between them is a direct, quantitative
                             witness of entanglement between qubits -- see
                             the Bell-state example in the test suite, where
                             every single-qubit marginal is maximally mixed
                             (qg_Z = 0) while the joint state is exactly
                             pure (von Neumann entropy 0).
"""

from __future__ import annotations

import string
from typing import List, Sequence, Union

import numpy as np

from .mixed import (
    IDENTITY_2,
    SIGMA_Z,
    density_from_statevector,
    qg_z_density,
    von_neumann_entropy,
)


def _as_density(state: Union[np.ndarray, Sequence[complex]], n_qubits: int) -> np.ndarray:
    """Accept either a 2^n statevector or a 2^n x 2^n density matrix."""
    state = np.asarray(state, dtype=complex)
    dim = 2 ** n_qubits
    if state.ndim == 1:
        if state.shape[0] != dim:
            raise ValueError(f"statevector has dim {state.shape[0]}, expected {dim} for n_qubits={n_qubits}.")
        return density_from_statevector(state)
    if state.shape != (dim, dim):
        raise ValueError(f"density matrix has shape {state.shape}, expected {(dim, dim)} for n_qubits={n_qubits}.")
    return state


def product_state(single_qubit_states: Sequence[np.ndarray]) -> np.ndarray:
    """
    Tensor-product density matrix of n independent single-qubit density
    matrices (or 2-component statevectors, which are converted first).
    """
    rho = None
    for s in single_qubit_states:
        s = np.asarray(s, dtype=complex)
        r = density_from_statevector(s) if s.ndim == 1 else s
        rho = r if rho is None else np.kron(rho, r)
    return rho


def bell_state(kind: str = "phi_plus") -> np.ndarray:
    """
    One of the four 2-qubit Bell states, as a statevector in the
    |00>, |01>, |10>, |11> basis -- the standard worked example of maximal
    entanglement, used in tests to show the joint/marginal entropy gap.
    """
    s = 1.0 / np.sqrt(2.0)
    kinds = {
        "phi_plus": np.array([s, 0, 0, s], dtype=complex),
        "phi_minus": np.array([s, 0, 0, -s], dtype=complex),
        "psi_plus": np.array([0, s, s, 0], dtype=complex),
        "psi_minus": np.array([0, s, -s, 0], dtype=complex),
    }
    if kind not in kinds:
        raise ValueError(f"kind must be one of {list(kinds)}, got {kind!r}.")
    return kinds[kind]


# --------------------------------------------------------------------- #
# partial trace
# --------------------------------------------------------------------- #
def partial_trace(rho: np.ndarray, n_qubits: int, keep: Sequence[int]) -> np.ndarray:
    """
    Reduced density matrix on the qubits in ``keep`` (0-indexed), tracing
    out every other qubit. Standard einsum-based partial trace.
    """
    keep = sorted(keep)
    if any(k < 0 or k >= n_qubits for k in keep):
        raise ValueError(f"keep indices must lie in [0, {n_qubits - 1}], got {keep}.")
    trace_out = [i for i in range(n_qubits) if i not in keep]

    letters = string.ascii_lowercase
    if 2 * n_qubits > len(letters):
        raise ValueError("partial_trace supports at most 13 qubits with this implementation.")

    row_labels = list(letters[:n_qubits])
    col_labels = list(letters[n_qubits:2 * n_qubits])
    for i in trace_out:
        col_labels[i] = row_labels[i]  # repeated index => einsum sums (traces) over it

    in_sub = "".join(row_labels) + "".join(col_labels)
    out_sub = "".join(row_labels[i] for i in keep) + "".join(col_labels[i] for i in keep)

    rho_t = rho.reshape([2] * (2 * n_qubits))
    result = np.einsum(f"{in_sub}->{out_sub}", rho_t)
    d = 2 ** len(keep)
    return result.reshape(d, d)


# --------------------------------------------------------------------- #
# per-qubit local qang ("projection profile" across the register)
# --------------------------------------------------------------------- #
def per_qubit_qg_z(state: Union[np.ndarray, Sequence[complex]], n_qubits: int) -> List[float]:
    """qg_Z for each qubit's own reduced density matrix: a length-n_qubits profile."""
    rho = _as_density(state, n_qubits)
    return [qg_z_density(partial_trace(rho, n_qubits, keep=[i])) for i in range(n_qubits)]


def marginal_qg_s(state: Union[np.ndarray, Sequence[complex]], n_qubits: int) -> List[float]:
    """
    Classical (Z-basis outcome) qg_S for each qubit's own marginal: the
    same H(p0) formula as the single-qubit qg_S, applied to each qubit's
    reduced diagonal.
    """
    rho = _as_density(state, n_qubits)
    result = []
    for i in range(n_qubits):
        reduced = partial_trace(rho, n_qubits, keep=[i])
        p0 = float(np.real(reduced[0, 0]))
        p0 = min(max(p0, 0.0), 1.0)
        p1 = 1.0 - p0
        h = 0.0
        for p in (p0, p1):
            if p > 1e-15:
                h -= p * np.log2(p)
        result.append(h)
    return result


def marginal_von_neumann_entropy(
    state: Union[np.ndarray, Sequence[complex]], n_qubits: int
) -> List[float]:
    """
    von Neumann entropy of each qubit's own reduced density matrix. For a
    product state this is 0 for every qubit (each qubit is individually
    pure); for an entangled state (e.g. a Bell pair) it is > 0 even when
    the *global* state is exactly pure -- the standard entanglement witness.
    """
    rho = _as_density(state, n_qubits)
    return [von_neumann_entropy(partial_trace(rho, n_qubits, keep=[i])) for i in range(n_qubits)]


# --------------------------------------------------------------------- #
# joint (whole-register) qg_S
# --------------------------------------------------------------------- #
def joint_qg_s(
    state: Union[np.ndarray, Sequence[complex]], n_qubits: int, normalize: bool = True
) -> float:
    """
    Shannon entropy of the full 2^n computational-basis outcome
    distribution obtained by measuring every qubit in the Z basis --
    the direct multi-qubit generalization of the single-qubit qg_S.

    normalize=True divides by log2(2^n) = n_qubits so the result stays in
    [0, 1] regardless of register size, matching qg_S's original range;
    normalize=False returns the raw Shannon entropy in bits.
    """
    rho = _as_density(state, n_qubits)
    probs = np.clip(np.real(np.diag(rho)), 0.0, 1.0)
    h = 0.0
    for p in probs:
        if p > 1e-15:
            h -= p * np.log2(p)
    if normalize and n_qubits > 1:
        h = h / n_qubits
    return float(h)
