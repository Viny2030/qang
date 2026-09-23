"""
qang.mixed — generalization of the qang unit to mixed states and POVMs.

This addresses the first half of Future Research Direction #4: "extend the
binary framework to POVMs and multi-outcome measurements, to mixed states
(where, unlike the pure-state case treated here, the von Neumann entropy of
rho is no longer trivially zero) ...".

Two generalizations are provided:

  * ``qg_z_density``        the polar qang for ANY single-qubit state, pure
                              or mixed: Tr(rho @ sigma_z). Reduces exactly to
                              qg_Z(theta) = cos(theta) for a pure state.
  * ``von_neumann_entropy`` S(rho) = -Tr(rho log2 rho), computed from rho's
                              eigenvalues. Identically 0 for any pure state
                              (as the paper's Section 2.2 notes), strictly
                              positive for a genuinely mixed state -- this is
                              the quantity qg_S in the paper explicitly is
                              *not*, so it fills the gap the paper names.
  * ``povm_outcome_probabilities`` / ``qg_s_povm``
                              the Shannon entropy of an n-outcome POVM
                              measurement on rho, generalizing qg_S beyond a
                              two-outcome Z-basis projective measurement.

All functions accept plain 2x2 (or, for multi-outcome POVMs, an element
list of 2x2) NumPy-compatible arrays, so they compose with qang.multiqubit
and with Qiskit density matrices without any conversion layer.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)

PROJECT_0 = np.array([[1, 0], [0, 0]], dtype=complex)
PROJECT_1 = np.array([[0, 0], [0, 1]], dtype=complex)


# --------------------------------------------------------------------- #
# construction helpers
# --------------------------------------------------------------------- #
def density_from_statevector(psi: Sequence[complex]) -> np.ndarray:
    """rho = |psi><psi| for a (normalized) statevector of any dimension."""
    psi = np.asarray(psi, dtype=complex).reshape(-1, 1)
    norm = float(np.real((psi.conj().T @ psi)).item())
    if abs(norm - 1.0) > 1e-6:
        raise ValueError(f"statevector must be normalized, got norm={norm:.6f}.")
    return psi @ psi.conj().T


def mix(rho_a: np.ndarray, rho_b: np.ndarray, p: float) -> np.ndarray:
    """A classical (incoherent) mixture p*rho_a + (1-p)*rho_b."""
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must lie in [0, 1], got {p}.")
    return p * np.asarray(rho_a, dtype=complex) + (1.0 - p) * np.asarray(rho_b, dtype=complex)


def is_valid_density_matrix(rho: np.ndarray, atol: float = 1e-6) -> bool:
    """Hermitian, unit trace, positive semi-definite."""
    rho = np.asarray(rho, dtype=complex)
    hermitian = np.allclose(rho, rho.conj().T, atol=atol)
    unit_trace = abs(np.trace(rho) - 1.0) < atol
    eigvals = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)
    psd = bool(np.all(eigvals >= -atol))
    return bool(hermitian and unit_trace and psd)


# --------------------------------------------------------------------- #
# polar qang for a general (pure or mixed) single-qubit state
# --------------------------------------------------------------------- #
def qg_z_density(rho: np.ndarray) -> float:
    """
    qg_Z for any single-qubit density matrix: Tr(rho @ sigma_z).

    For a pure state rho = |psi><psi| with |psi> = cos(theta/2)|0> +
    e^{i*phi} sin(theta/2)|1>, this reduces exactly to cos(theta), matching
    qang.core.Qang.from_angles(theta, mode="polar").value.
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (2, 2):
        raise ValueError(f"qg_z_density expects a single-qubit (2x2) rho, got shape {rho.shape}.")
    return float(np.real(np.trace(rho @ SIGMA_Z)))


def purity(rho: np.ndarray) -> float:
    """Tr(rho^2); 1.0 for a pure state, 1/d for the maximally mixed state of dim d."""
    rho = np.asarray(rho, dtype=complex)
    return float(np.real(np.trace(rho @ rho)))


# --------------------------------------------------------------------- #
# von Neumann entropy -- the quantity qg_S in the paper explicitly is NOT
# --------------------------------------------------------------------- #
def von_neumann_entropy(rho: np.ndarray, base: float = 2.0, atol: float = 1e-12) -> float:
    """
    S(rho) = -Tr(rho log_base rho), computed from rho's eigenvalues.

    Identically 0 for any pure state (rank-1 rho); this is exactly the fact
    the paper's Section 2.2 uses to justify why qg_S is instead defined as
    the *classical* Shannon entropy of the measurement-outcome distribution.
    von_neumann_entropy is the quantity that fills the "mixed states" half
    of Future Research Direction #4.
    """
    rho = np.asarray(rho, dtype=complex)
    eigvals = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)
    eigvals = np.clip(np.real(eigvals), 0.0, 1.0)
    s = 0.0
    for lam in eigvals:
        if lam > atol:
            s -= lam * (np.log(lam) / np.log(base))
    return float(s)


# --------------------------------------------------------------------- #
# POVM generalization -- the "multi-outcome measurements" half of
# Future Research Direction #4
# --------------------------------------------------------------------- #
def standard_z_povm() -> List[np.ndarray]:
    """The two-outcome projective Z-basis POVM {|0><0|, |1><1|}."""
    return [PROJECT_0, PROJECT_1]


def is_valid_povm(elements: Sequence[np.ndarray], atol: float = 1e-6) -> bool:
    """Each element positive semi-definite and they sum to the identity."""
    total = sum(np.asarray(e, dtype=complex) for e in elements)
    dim = total.shape[0]
    if not np.allclose(total, np.eye(dim, dtype=complex), atol=atol):
        return False
    for e in elements:
        eigvals = np.linalg.eigvalsh((e + e.conj().T) / 2.0)
        if np.any(eigvals < -atol):
            return False
    return True


def povm_outcome_probabilities(rho: np.ndarray, povm: Sequence[np.ndarray]) -> List[float]:
    """p_i = Tr(rho @ E_i) for each POVM element E_i (Born's rule, generalized)."""
    rho = np.asarray(rho, dtype=complex)
    probs = [float(np.real(np.trace(rho @ np.asarray(e, dtype=complex)))) for e in povm]
    total = sum(probs)
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"POVM outcome probabilities must sum to 1, got {total:.6f}.")
    return [max(0.0, p) for p in probs]


def qg_s_povm(rho: np.ndarray, povm: Sequence[np.ndarray], normalize: bool = True) -> float:
    """
    Generalized entropic qang: the Shannon entropy of an n-outcome POVM
    measurement on rho. With ``povm=standard_z_povm()`` and a pure-state rho,
    this reduces exactly to qg_S(theta) from qang.core.

    normalize=True divides by log2(n_outcomes) so the result stays in
    [0, 1] regardless of how many outcomes the POVM has, matching qg_S's
    original [0, 1] range; normalize=False returns the raw Shannon entropy
    in bits (range [0, log2(n_outcomes)]).
    """
    probs = povm_outcome_probabilities(rho, povm)
    n = len(probs)
    h = 0.0
    for p in probs:
        if p > 1e-15:
            h -= p * np.log2(p)
    if normalize and n > 1:
        h = h / np.log2(n)
    return float(h)


def trine_povm() -> List[np.ndarray]:
    """
    A canonical 3-outcome (non-projective) single-qubit POVM: three
    equally-spaced rank-1 elements (2/3)|v_k><v_k| in the X-Z plane, used
    as a worked example of the n-outcome generalization beyond n=2.
    """
    elements = []
    for k in range(3):
        angle = 2.0 * np.pi * k / 3.0
        v = np.array([[np.cos(angle / 2.0)], [np.sin(angle / 2.0)]], dtype=complex)
        elements.append((2.0 / 3.0) * (v @ v.conj().T))
    return elements
