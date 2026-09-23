"""
qang.multiqubit — generalization of the qang unit to multi-qubit
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

Finite-shot qg_S (Miller-Madow bias correction)
-------------------------------------------------
``joint_qg_s`` needs the full state (or density matrix), which is only
ever available in simulation. On real hardware -- or any shot-based
simulator -- you only ever have a finite number of measurement counts,
and the plug-in (maximum-likelihood) Shannon entropy estimator built
directly from those counts is well known to be negatively biased: it
systematically UNDERESTIMATES the true entropy, worst when the shot
budget N is small relative to the number of possible outcomes 2^n_qubits
(exactly the regime that matters as a benchmarked circuit grows). See
examples/quantum_volume_qg_s_finite_shots.py for a direct, quantitative
measurement of that bias and of how much the Miller-Madow correction
below closes it. ``shannon_entropy_miller_madow_bits`` and
``joint_qg_s_from_counts`` require nothing about the true distribution --
only the observed counts and the total shot number -- which is exactly
the information available from a real device.

qg-native correlation (a mutual-information-like invariant)
---------------------------------------------------------------
qang.core's docstring establishes an exact, branch-free identity between
qg_Z and qg_S for any single qubit: qg_S = H((1 + qg_Z) / 2) (see
``qang.core.qg_s_from_qg_z``). Applied per qubit, via partial trace, this
means ``per_qubit_qg_z`` and ``marginal_qg_s`` are always two views of
the SAME per-qubit information -- for every qubit of every state, product
or entangled, pure or mixed (see this module's test suite for a check of
this identity against random Haar states, not just the canonical
examples below).

The joint (whole-register) qg_S is where correlations between qubits
enter, and ``qg_correlation`` makes that entry a single, always
non-negative number:

    qg_correlation(state) = sum_i marginal_qg_s(state)[i]  -  joint_qg_s(state, normalize=False)

This is never negative -- classical Shannon entropy is SUBADDITIVE,
H(X_1, ..., X_n) <= sum_i H(X_i), with equality iff the qubits' Z-basis
outcomes are statistically independent -- so ``qg_correlation`` is exactly
0 for any product state and strictly positive whenever the joint
measurement-outcome distribution carries information that no single
qubit's own marginal does (e.g. a Bell or GHZ state). It is the same
"gap" this module's Bell-state test already demonstrates qualitatively
(every marginal maximally mixed, the joint state exactly pure), now
named, always well-defined, and true in general rather than for one
worked example.
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


def ghz_state(n_qubits: int) -> np.ndarray:
    """
    The n-qubit GHZ state (|00...0> + |11...1>) / sqrt(2), as a statevector
    -- the general-n complement to bell_state (ghz_state(2) is exactly
    bell_state("phi_plus")). The canonical example of a state whose
    per-qubit marginals are all maximally mixed (qg_Z = 0, marginal
    qg_S = 1 bit each) while the joint measurement outcome carries only 1
    bit total, regardless of n -- see qg_correlation below.
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}.")
    dim = 2 ** n_qubits
    sv = np.zeros(dim, dtype=complex)
    s = 1.0 / np.sqrt(2.0)
    sv[0] = s
    sv[dim - 1] = s
    return sv


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


# --------------------------------------------------------------------- #
# finite-shot qg_S (see this module's docstring, "Finite-shot qg_S")
# --------------------------------------------------------------------- #
def shannon_entropy_plugin_bits(counts: Union[dict, Sequence[int]]) -> float:
    """
    Plug-in (maximum-likelihood) Shannon entropy estimate, in bits, from
    observed outcome counts -- a dict mapping outcome to count, or a
    plain sequence of per-outcome counts. Known to be negatively biased
    at finite sample size (see shannon_entropy_miller_madow_bits for a
    first-order correction).
    """
    values = list(counts.values()) if isinstance(counts, dict) else list(counts)
    n = float(sum(values))
    if n <= 0:
        raise ValueError("counts must contain at least one observed shot.")
    h = 0.0
    for c in values:
        if c <= 0:
            continue
        p = c / n
        h -= p * np.log2(p)
    return float(h)


def shannon_entropy_miller_madow_bits(counts: Union[dict, Sequence[int]]) -> float:
    """
    Miller-Madow bias-corrected Shannon entropy estimate, in bits:

        H_MM = H_plugin + (K_observed - 1) / (2 * N * ln(2))

    K_observed is the number of DISTINCT outcomes actually observed in
    this sample (never the full 2^n_qubits alphabet size -- an outcome
    that happens not to appear contributes nothing), and N is the total
    shot count. This is the classic first-order correction (Miller, 1955;
    Madow, 1948) for the plug-in estimator's negative bias; it reduces,
    but does not eliminate, that bias -- see
    examples/quantum_volume_qg_s_finite_shots.py for a direct, quantified
    measurement of by how much, across a range of shot budgets.
    """
    values = list(counts.values()) if isinstance(counts, dict) else list(counts)
    n = float(sum(values))
    if n <= 0:
        raise ValueError("counts must contain at least one observed shot.")
    k_observed = sum(1 for c in values if c > 0)
    h_plugin = shannon_entropy_plugin_bits(counts)
    correction = (k_observed - 1) / (2.0 * n * np.log(2.0))
    return float(h_plugin + correction)


def joint_qg_s_from_counts(
    counts: Union[dict, Sequence[int]],
    n_qubits: int,
    normalize: bool = True,
    bias_correction: str = "miller_madow",
) -> float:
    """
    Finite-shot estimate of joint_qg_s, built directly from observed
    measurement counts -- a dict mapping outcome to count, or a plain
    sequence of per-outcome counts -- exactly the information available
    from real hardware or a shot-based simulator, as opposed to
    joint_qg_s, which needs the full state and is only ever available in
    simulation.

    bias_correction: "miller_madow" (default; see
    shannon_entropy_miller_madow_bits) or "none" (the uncorrected
    plug-in estimate, provided for direct comparison).
    """
    if bias_correction == "miller_madow":
        h = shannon_entropy_miller_madow_bits(counts)
    elif bias_correction == "none":
        h = shannon_entropy_plugin_bits(counts)
    else:
        raise ValueError(f"bias_correction must be 'miller_madow' or 'none', got {bias_correction!r}.")
    if normalize and n_qubits > 1:
        h = h / n_qubits
    return float(h)


# --------------------------------------------------------------------- #
# qg-native correlation (see this module's docstring,
# "qg-native correlation (a mutual-information-like invariant)")
# --------------------------------------------------------------------- #
def qg_correlation(state: Union[np.ndarray, Sequence[complex]], n_qubits: int) -> float:
    """
    sum_i marginal_qg_s(state)[i]  -  joint_qg_s(state, n_qubits, normalize=False)

    Always >= 0 (subadditivity of Shannon entropy), exactly 0 for any
    product state, and strictly positive exactly when the qubits' Z-basis
    measurement outcomes are statistically correlated -- see this module's
    docstring for the full derivation and ghz_state for a family of states
    with a known closed-form value (n_qubits - 1 bits, for n_qubits >= 1).
    """
    marg = marginal_qg_s(state, n_qubits)
    joint = joint_qg_s(state, n_qubits, normalize=False)
    return float(sum(marg) - joint)


# --------------------------------------------------------------------- #
# Relaxation bias: a T1-aware companion to joint qg_S
# --------------------------------------------------------------------- #
# joint_qg_s rises with noise only when the noise channel's fixed point
# is the maximally mixed state (depolarizing, dephasing). Amplitude
# damping (T1) drives the register towards |00...0> instead, so under
# strong enough T1 decay qg_S FALLS again, all the way to 0 -- see
# examples/quantum_volume_qg_s_realistic_noise.py, Findings C and D.
# The register-averaged qg_Z below tells the two regimes apart: unital
# noise pulls it towards 0, while T1 decay pulls it towards +1.
#
# Being a mean of per-qubit <sigma_z> values, it is LINEAR in the state
# (a Hilbert-Schmidt inner product with (1/n) * sum_i Z_i), so its
# finite-shot estimator below is an exactly unbiased sample mean -- unlike
# the entropy-based qg_S, which needs the Miller-Madow correction above.
def mean_qg_z(state: Union[np.ndarray, Sequence[complex]], n_qubits: int) -> float:
    """
    Register-averaged qg_Z, (1/n) * sum_i Tr(rho_i sigma_z): the
    "relaxation bias" of the register. +1 means every qubit is in |0>
    (the fixed point of amplitude damping), 0 is what any unital noise
    channel (depolarizing, dephasing) drives it towards.

    Independent of qubit ordering, since it averages over all qubits.
    """
    return float(np.mean(per_qubit_qg_z(state, n_qubits)))


def mean_qg_z_from_counts(counts: Union[dict, Sequence[int]], n_qubits: int) -> float:
    """
    Finite-shot estimate of mean_qg_z from measurement counts: the
    average, over shots and qubits, of +1 for each measured 0 and -1 for
    each measured 1. Exactly unbiased at any shot count (it is a sample
    mean).

    ``counts`` is either a dict mapping outcome to count -- the outcome
    being a bitstring such as Qiskit's ``"0101"`` (spaces allowed) or an
    integer basis index -- or a plain sequence of per-outcome counts
    indexed by basis index. Qubit ordering does not matter, since only
    the total number of 1s in each outcome is used.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1.")
    items = counts.items() if isinstance(counts, dict) else enumerate(counts)

    total_shots = 0
    total_ones = 0
    for outcome, c in items:
        c = int(c)
        if c < 0:
            raise ValueError("counts must be non-negative.")
        if c == 0:
            continue
        if isinstance(outcome, str):
            bits = outcome.replace(" ", "")
            if len(bits) != n_qubits or set(bits) - {"0", "1"}:
                raise ValueError(f"outcome {outcome!r} is not a {n_qubits}-bit bitstring.")
            ones = bits.count("1")
        else:
            idx = int(outcome)
            if not 0 <= idx < 2**n_qubits:
                raise ValueError(f"outcome index {idx} out of range for {n_qubits} qubits.")
            ones = bin(idx).count("1")
        total_shots += c
        total_ones += c * ones

    if total_shots == 0:
        raise ValueError("counts contain no shots.")
    # mean over shots and qubits of (+1 for a 0, -1 for a 1)
    return float(1.0 - 2.0 * total_ones / (total_shots * n_qubits))
