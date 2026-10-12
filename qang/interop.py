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

Qiskit primitives (§131):
  * ``sampler_qg``            the qg_Z profile, the filtered profile and the
                              kept fraction of every parameter set of a
                              SamplerV2 pub result, as arrays;
  * ``qg_z_observables``      the n observables whose EstimatorV2 values ARE
                              the qg_Z of each qubit;
  * ``filtered_observables``  the weight-w projector P_w and the n products
                              P_w Z_i as Pauli sums, so that EstimatorV2
                              returns the filtered read-out directly;
  * ``estimator_filtered_qg`` (kept fraction, filtered qg_Z) from those values:
                              filtered qg_Z_i = <P_w Z_i> / <P_w>.
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
    "sampler_qg",
    "qg_z_observables",
    "filtered_observables",
    "estimator_filtered_qg",
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


# ---------------------------------------------------------------- Qiskit primitives (§131)


def _bitarray(pub_result, creg: Optional[str]):
    data = getattr(pub_result, "data", pub_result)
    if hasattr(data, "get_bitstrings"):
        return data
    names = list(data.keys()) if hasattr(data, "keys") else [k for k in vars(data) if not k.startswith("_")]
    if creg is None:
        if len(names) != 1:
            raise ValueError(f"the pub result has registers {names}: pass creg=.")
        creg = names[0]
    return getattr(data, creg)


def sampler_qg(pub_result, weight: Optional[int] = None, creg: Optional[str] = None) -> dict:
    """qg read-out of a Qiskit SamplerV2 pub result, for every parameter set of the pub.

    Returns a dict of numpy arrays, ``shape`` being the pub's shape:
      "qg", "se"   shape + (n,)  per-qubit qg_Z and its error (qubit 0 first);
      "kept"       shape         fraction of shots of Hamming weight ``weight`` (1 without a filter);
    with ``weight`` given, "qg" and "se" are those of the kept shots (nan where none is kept).
    """
    import numpy as np

    bits = _bitarray(pub_result, creg)
    shape, n = tuple(bits.shape), int(bits.num_bits)
    qg, se, kept = np.full(shape + (n,), np.nan), np.full(shape + (n,), np.nan), np.ones(shape)
    for idx in np.ndindex(*shape) if shape else [()]:
        c = _from_strings(bits.get_counts(loc=idx) if shape else bits.get_counts(), qubit0_last=True)
        if weight is not None:
            c, kept[idx] = weight_filter(c, weight)
            if not c:
                continue
        prof = qg_z_profile(c)
        qg[idx] = [q for q, _ in prof]
        se[idx] = [e for _, e in prof]
    return {"qg": qg, "se": se, "kept": kept}


def qg_z_observables(n_qubits: int) -> List[str]:
    """Pauli labels (Qiskit order, qubit 0 rightmost) of Z_0, ..., Z_{n-1}: EstimatorV2 returns qg_Z directly."""
    return ["I" * (n_qubits - 1 - i) + "Z" + "I" * i for i in range(n_qubits)]


def _diag_to_paulis(diag, n: int) -> Dict[str, float]:
    """Pauli-Z expansion {label: coeff} of a diagonal operator given by its 2^n diagonal (Qiskit index order)."""
    import numpy as np

    c = np.asarray(diag, float).copy()
    h = 1
    while h < len(c):                                      # fast Walsh-Hadamard transform
        for i in range(0, len(c), 2 * h):
            a, b = c[i:i + h].copy(), c[i + h:i + 2 * h].copy()
            c[i:i + h], c[i + h:i + 2 * h] = a + b, a - b
        h *= 2
    c /= 2 ** n
    out = {}
    for s, v in enumerate(c):
        if abs(v) > 1e-12:
            out["".join("Z" if (s >> (n - 1 - k)) & 1 else "I" for k in range(n))] = float(v)
    return out


def filtered_observables(n_qubits: int, weight: int) -> List[Dict[str, float]]:
    """[P_w, P_w Z_0, ..., P_w Z_{n-1}] as {Pauli label: coeff} (Qiskit order), for EstimatorV2.

    P_w projects on the computational states of Hamming weight w. Both are diagonal, so each is a sum of
    Z strings; the expansion has up to 2^n terms (fine for the register sizes of current devices' tests,
    n <= ~12). Pass the list as one observables array of an Estimator pub.
    """
    import numpy as np

    if not 0 <= weight <= n_qubits:
        raise ValueError(f"weight must be in [0, {n_qubits}].")
    idx = np.arange(2 ** n_qubits)
    w = np.array([bin(i).count("1") for i in idx])
    mask = (w == weight).astype(float)
    obs = [_diag_to_paulis(mask, n_qubits)]
    for i in range(n_qubits):
        z = 1.0 - 2.0 * ((idx >> i) & 1)
        obs.append(_diag_to_paulis(mask * z, n_qubits))
    return obs


def estimator_filtered_qg(evs):
    """(kept fraction, filtered qg_Z array) from EstimatorV2 values of ``filtered_observables`` (last axis)."""
    import numpy as np

    evs = np.asarray(evs, float)
    kept = evs[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return kept, evs[..., 1:] / kept[..., None]
