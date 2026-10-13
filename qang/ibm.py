"""
qang.ibm — qg read-outs on IBM backends and IBM Runtime (§137).

Three pieces that make qang a drop-in on IBM devices (real ones through
qiskit-ibm-runtime, or their fake backends in local mode):

  * ``WeightPreservationCheck``  a Qiskit transpiler analysis pass: does every
    gate of the circuit conserve the Hamming weight (the excitation number)?
    If so the qg filter applies. Results go to the pass manager's property set.
  * ``readout_errors``           the published, per-qubit, asymmetric readout
    errors (P(1|0), P(0|1)) of the physical qubits a transpiled circuit
    measures, read from the backend calibration.
  * ``ibm_qg``                   the qg read-out of a SamplerV2 pub result
    (IBM Runtime or any V2 sampler) for a transpiled circuit: per-qubit qg_Z
    raw and readout-corrected, and with the qg filter applied to the
    readout-corrected distribution.

Readout correction inverts the tensor product of the per-qubit confusion
matrices on the measured distribution (exact for independent readout
errors; 2^n entries, so n <= 14). The corrected distribution is a
quasi-probability; the qg filter keeps its weight-w part and renormalizes.

Requires Qiskit (imported on use). Qubit order everywhere: qubit 0 first,
as in qang.interop.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "WeightPreservationCheck",
    "measured_physical_qubits",
    "readout_errors",
    "readout_inverse_distribution",
    "ibm_qg",
]

# gates that conserve the excitation number for every parameter value
_WEIGHT_PRESERVING_NAMES = {
    "id", "delay", "barrier", "z", "s", "sdg", "t", "tdg", "rz", "p", "u1", "cz", "cp", "cu1", "crz", "rzz",
    "swap", "iswap", "xx_plus_yy", "ccz", "mcphase", "mcp", "cs", "csdg", "global_phase",
}


def _number_operator(k: int) -> np.ndarray:
    """Diagonal of the excitation number on k qubits (Qiskit little-endian indexing; symmetric anyway)."""
    return np.array([bin(i).count("1") for i in range(2 ** k)], float)


def _gate_preserves_weight(inst) -> Optional[bool]:
    """True/False for a gate, None if it cannot be decided (unbound parameters on an unknown gate)."""
    op = inst.operation
    if op.name in _WEIGHT_PRESERVING_NAMES:
        return True
    if op.name in ("measure", "reset"):
        return False
    try:
        from qiskit.quantum_info import Operator

        U = Operator(op).data
    except Exception:
        return None
    n = _number_operator(int(round(np.log2(U.shape[0]))))
    # U conserves N iff U N = N U, i.e. U only connects basis states of equal weight
    return bool(np.allclose(U * (n[:, None] != n[None, :]), 0, atol=1e-9))


try:
    from qiskit.transpiler.basepasses import AnalysisPass as _AnalysisPass
except ImportError:  # pragma: no cover - only without Qiskit
    _AnalysisPass = object


class WeightPreservationCheck(_AnalysisPass):
    """Transpiler analysis pass: does every gate before measurement conserve the Hamming weight?

    Sets in the property set:
      ``qg_weight_preserving``      True if every gate is known to conserve the weight, False if one does
                                    not, None if some gate could not be decided;
      ``qg_non_preserving_gates``   [(index, name, qubits)] of the gates that break it (or are undecided);
      ``qg_initial_weight``         the weight prepared from |0...0> by leading X gates, if the circuit
                                    starts that way (else None): the sector to filter on.
    Gate-wise conservation is sufficient, not necessary (X then X conserves overall but is flagged).
    """

    def run(self, dag):
        from qiskit.converters import dag_to_circuit

        qc = dag_to_circuit(dag)
        bad, undecided = [], False
        leading_x, started = set(), False
        for i, inst in enumerate(qc.data):
            name = inst.operation.name
            qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
            if name in ("measure", "barrier"):
                continue
            if not started and name == "x" and qubits[0] not in leading_x:
                leading_x.add(qubits[0])
                continue
            started = True
            ok = _gate_preserves_weight(inst)
            if ok is None:
                undecided = True
                bad.append((i, name, qubits))
            elif not ok:
                bad.append((i, name, qubits))
        self.property_set["qg_non_preserving_gates"] = bad
        if any(_gate_preserves_weight(qc.data[i]) is False for i, _, _ in bad):
            self.property_set["qg_weight_preserving"] = False
        else:
            self.property_set["qg_weight_preserving"] = None if undecided else True
        self.property_set["qg_initial_weight"] = len(leading_x) if leading_x or not bad else None
        return dag


def measured_physical_qubits(transpiled) -> List[int]:
    """Physical qubit measured into each classical bit (index = clbit) of a transpiled circuit."""
    out: Dict[int, int] = {}
    for inst in transpiled.data:
        if inst.operation.name == "measure":
            q = transpiled.find_bit(inst.qubits[0]).index
            c = transpiled.find_bit(inst.clbits[0]).index
            out[c] = q
    if not out:
        raise ValueError("the circuit measures nothing.")
    return [out[c] for c in sorted(out)]


def readout_errors(backend, physical_qubits: Sequence[int]) -> List[Tuple[float, float]]:
    """[(P(1|0), P(0|1))] per physical qubit from the backend calibration (asymmetric when published,
    else the symmetric measure error of the target)."""
    props = None
    try:
        props = backend.properties()
    except Exception:
        props = None
    out = []
    for q in physical_qubits:
        try:
            e01 = props.qubit_property(q, "prob_meas1_prep0")[0]
            e10 = props.qubit_property(q, "prob_meas0_prep1")[0]
        except Exception:
            e = backend.target["measure"][(q,)].error or 0.0
            e01 = e10 = e
        out.append((float(e01), float(e10)))
    return out


def readout_inverse_distribution(probs: np.ndarray, errors: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Invert independent per-qubit readout errors on a distribution indexed qubit 0 first
    (index bit n-1-i is qubit i). Returns a quasi-probability vector (may have small negatives)."""
    n = len(errors)
    p = np.asarray(probs, float).reshape((2,) * n)
    for i, (e01, e10) in enumerate(errors):
        M = np.array([[1 - e01, e10], [e01, 1 - e10]])          # columns: prepared 0, 1
        p = np.moveaxis(np.tensordot(np.linalg.inv(M), np.moveaxis(p, i, 0), axes=(1, 0)), 0, i)
    return p.reshape(-1)


def _profile(q: np.ndarray, n: int, shots: int):
    """Per-qubit qg_Z and shot-noise error from a (quasi-)distribution indexed qubit 0 first."""
    idx = np.arange(2 ** n)
    out = []
    for i in range(n):
        z = 1 - 2 * ((idx >> (n - 1 - i)) & 1)
        v = float(np.clip(np.dot(q, z), -1, 1))
        out.append((v, float(np.sqrt(max(1 - v * v, 0) / shots))))
    return out


def ibm_qg(pub_result, transpiled, backend, weight: Optional[int] = None, creg: Optional[str] = None) -> dict:
    """qg read-out of a SamplerV2 pub result (single parameter set) for a circuit transpiled to ``backend``.

    Returns {"raw": [(qg_Z, se)], "readout": [...], "filtered": [...] (raw shots, weight w),
             "filtered_readout": [...], "kept_raw": float, "kept_readout": float, "physical_qubits": [...]}.
    The filtered entries are present only when ``weight`` is given. Errors of corrected values are the
    shot-noise errors scaled by the confusion amplification 1 / (1 - P(1|0) - P(0|1)).
    """
    from . import interop as IO

    bits = IO._bitarray(pub_result, creg)
    counts = IO._from_strings(bits.get_counts(), qubit0_last=True)
    n = len(next(iter(counts)))
    shots = sum(counts.values())
    phys = measured_physical_qubits(transpiled)
    if len(phys) != n:
        raise ValueError(f"the result has {n} bits but the circuit measures {len(phys)} qubits.")
    errs = readout_errors(backend, phys)
    p = np.zeros(2 ** n)
    for k, c in counts.items():
        p[int("".join(map(str, k)), 2)] += c / shots
    q = readout_inverse_distribution(p, errs)
    amp = [1 / max(1 - a - b, 1e-9) for a, b in errs]
    out = {"physical_qubits": phys, "readout_errors": errs,
           "raw": IO.qg_z_profile(counts),
           "readout": [(v, se * a) for (v, se), a in zip(_profile(q, n, shots), amp)]}
    if weight is not None:
        w = np.array([bin(i).count("1") for i in range(2 ** n)])
        kept_raw, frac = IO.weight_filter(counts, weight)
        out["kept_raw"] = frac
        out["filtered"] = IO.qg_z_profile(kept_raw) if frac > 0 else None
        mass = float(q[w == weight].sum())
        out["kept_readout"] = mass
        if mass > 0:
            qf = np.where(w == weight, q, 0.0) / mass
            n_eff = max(int(round(mass * shots)), 1)
            out["filtered_readout"] = [(v, se * a) for (v, se), a in zip(_profile(qf, n, n_eff), amp)]
        else:
            out["filtered_readout"] = None
    return out
