"""
qang.interop — read qg quantities from the measurement results of any quantum
SDK, with one qubit-order convention (RESEARCH_NOTES §130).

Each SDK orders the bits of an outcome differently: Qiskit writes qubit 0 at
the RIGHT of a bitstring (and separates registers with spaces), while Cirq,
PennyLane, Amazon Braket and pytket put qubit 0 FIRST. Mixing them silently
swaps qubits. This module converts every result to one canonical form,

    {(b_0, b_1, ..., b_{n-1}): count}      b_i = outcome of qubit i,

and computes on it the qang read-outs used throughout the library:

  * ``qg_z_profile``   per-qubit qg_Z = <Z_i> with its shot-noise error;
  * ``weight_filter``  the qg filter: keep the shots of a given Hamming weight
                       (the sector of a weight-conserving circuit), with the
                       kept fraction;
  * ``joint_qg_s``     the measurement-outcome entropy of the register
                       (Miller-Madow corrected by default).

Supported inputs (no SDK is imported unless its object is passed):
  Qiskit   counts dict (``result.get_counts()``) with ``sdk="qiskit"``, or a
           SamplerV2 ``BitArray`` (``pub_result.data.<creg>``);
  Cirq     ``cirq.Result`` (all measurement keys, in key order) or a
           ``result.histogram`` with ``sdk="cirq"`` and ``n_qubits``;
  PennyLane  the dict of ``qml.counts()`` with ``sdk="pennylane"``;
  Braket   a task result (``measurement_counts``) or that dict with
           ``sdk="braket"``;
  pytket   a ``BackendResult`` (``get_counts()``) or its dict of tuples.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .multiqubit import shannon_entropy_miller_madow_bits, shannon_entropy_plugin_bits

__all__ = [
    "canonical_counts",
    "qg_z_profile",
    "weight_filter",
    "filtered_qg_z_profile",
    "joint_qg_s",
]

Canonical = Dict[Tuple[int, ...], int]
_FIRST = ("cirq", "pennylane", "braket", "pytket", "big")      # qubit 0 first
_LAST = ("qiskit", "little")                                   # qubit 0 last


def _from_strings(d, qubit0_last: bool) -> Canonical:
    out: Canonical = {}
    for key, c in d.items():
        bits = str(key).replace(" ", "")
        if set(bits) - {"0", "1"}:
            raise ValueError(f"outcome {key!r} is not a bitstring.")
        t = tuple(int(b) for b in (bits[::-1] if qubit0_last else bits))
        out[t] = out.get(t, 0) + int(c)
    return out


def _from_ints(d, n_qubits: int, qubit0_last: bool) -> Canonical:
    out: Canonical = {}
    for key, c in d.items():
        bits = format(int(key), f"0{n_qubits}b")
        t = tuple(int(b) for b in (bits[::-1] if qubit0_last else bits))
        out[t] = out.get(t, 0) + int(c)
    return out


def canonical_counts(result, sdk: Optional[str] = None, n_qubits: Optional[int] = None) -> Canonical:
    """Counts as {(b_0, ..., b_{n-1}): count}, qubit 0 first, from any supported SDK result (see module doc)."""
    # objects first (duck typing, no imports)
    if hasattr(result, "get_bitstrings") and hasattr(result, "num_bits"):          # Qiskit BitArray
        return _from_strings(result.get_counts(), qubit0_last=True)
    if hasattr(result, "measurements") and hasattr(result, "histogram") and isinstance(result.measurements, dict):
        import numpy as np                                                       # cirq.Result
        cols = [np.asarray(result.measurements[k]) for k in sorted(result.measurements)]
        bits = np.concatenate([c.reshape(c.shape[0], -1) for c in cols], axis=1).astype(int)
        out: Canonical = {}
        for row in map(tuple, bits):
            out[row] = out.get(row, 0) + 1
        return out
    if hasattr(result, "measurement_counts"):                                     # Braket task result
        return _from_strings(result.measurement_counts, qubit0_last=False)
    if hasattr(result, "get_counts") and not isinstance(result, dict):
        c = result.get_counts()
        if c and isinstance(next(iter(c)), tuple):                                 # pytket BackendResult
            return {tuple(int(b) for b in k): int(v) for k, v in c.items()}
        return _from_strings(c, qubit0_last=True)                                  # Qiskit Result
    if not isinstance(result, dict) or not result:
        raise ValueError("unsupported or empty result.")
    first = next(iter(result))
    if isinstance(first, tuple):                                                    # pytket dict / canonical
        return {tuple(int(b) for b in k): int(v) for k, v in result.items()}
    if sdk is None:
        raise ValueError("a counts dict needs sdk= ('qiskit', 'cirq', 'pennylane', 'braket' or 'pytket'): "
                         "they order the qubits differently.")
    sdk = sdk.lower()
    if sdk not in _FIRST + _LAST:
        raise ValueError(f"unknown sdk {sdk!r}.")
    last = sdk in _LAST
    if isinstance(first, str):
        return _from_strings(result, qubit0_last=last)
    if n_qubits is None:
        raise ValueError("integer outcomes need n_qubits.")
    return _from_ints(result, n_qubits, qubit0_last=last)


def _n_and_shots(counts: Canonical) -> Tuple[int, int]:
    n = len(next(iter(counts)))
    shots = sum(counts.values())
    if shots <= 0:
        raise ValueError("counts contain no shots.")
    return n, shots


def qg_z_profile(counts: Canonical) -> List[Tuple[float, float]]:
    """[(qg_Z_i, standard error)] per qubit, qubit 0 first: qg_Z = 1 - 2 P(b_i = 1), error sqrt((1 - qg^2)/N)."""
    n, shots = _n_and_shots(counts)
    out = []
    for i in range(n):
        ones = sum(c for k, c in counts.items() if k[i] == 1)
        q = 1.0 - 2.0 * ones / shots
        out.append((q, math.sqrt(max(1.0 - q * q, 0.0) / shots)))
    return out


def weight_filter(counts: Canonical, weight: int) -> Tuple[Canonical, float]:
    """The qg filter: keep the outcomes with Hamming weight ``weight``. Returns (kept counts, kept fraction)."""
    _, shots = _n_and_shots(counts)
    kept = {k: c for k, c in counts.items() if sum(k) == weight}
    return kept, sum(kept.values()) / shots


def filtered_qg_z_profile(counts: Canonical, weight: int) -> List[Tuple[float, float]]:
    """qg_z_profile of the shots kept by the filter (raises if no shot is kept)."""
    kept, frac = weight_filter(counts, weight)
    if frac == 0:
        raise ValueError(f"no shot has weight {weight}.")
    return qg_z_profile(kept)


def joint_qg_s(counts: Canonical, correction: str = "miller_madow") -> float:
    """Measurement-outcome entropy of the register in qg units (bits / n), plug-in or Miller-Madow."""
    n, _ = _n_and_shots(counts)
    vals = list(counts.values())
    if correction == "miller_madow":
        h = shannon_entropy_miller_madow_bits(vals)
    elif correction == "none":
        h = shannon_entropy_plugin_bits(vals)
    else:
        raise ValueError("correction must be 'miller_madow' or 'none'.")
    return h / n
