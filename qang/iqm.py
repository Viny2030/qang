"""
qang.iqm — qg read-outs on IQM backends: IQM Resonance and IQM's fake backends (§165, §168; book chapter 31).

What qang.ibm does for IBM devices, for IQM's:

  * ``check_weight``            qang.ibm.WeightPreservationCheck run on the circuit BEFORE transpilation. IQM's
                                native single-qubit gate (``r``, a phased X rotation) does not conserve the
                                weight gate by gate, so a compiled weight-conserving circuit fails the check
                                although the circuit as a whole conserves it; check the logical circuit.
  * ``readout_errors``          [(P(1|0), P(0|1))] per physical qubit from a fake backend's error profile.
  * ``pole_circuits`` / ``readout_from_poles``   the same, measured on the device with two calibration circuits
                                (every qubit in |0>, every qubit in |1>), as §168 did. The calibration API of a
                                real IQMBackend is not read here: it could not be tested without credits.
  * ``iqm_qg``                  per-qubit qg_Z from the counts of ``IQMBackend.run``: raw, readout-corrected,
                                filtered (weight w), and filtered after readout correction.
  * ``predict_kept_fraction``   the fraction of shots the qg filter will keep, from the error profile.
  * ``best_chain``              the initial layout (the transpiler's default or a chain of the square lattice)
                                with the highest predicted kept fraction.

Readout correction inverts the per-qubit confusion matrices (qang.ibm.readout_inverse_distribution); the
corrected distribution is a quasi-probability, and the filter keeps its weight-w part and renormalizes.

Requires Qiskit; the fake backends and IQMBackend come from iqm-client[qiskit] (extra ``qang[iqm]``, in its own
environment: iqm-client pins its own Qiskit). Qubit order everywhere: qubit 0 first, as in qang.interop.
"""

from __future__ import annotations

from math import comb
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ibm import (
    WeightPreservationCheck,
    _profile,
    linear_chains,
    measured_physical_qubits,
    readout_inverse_distribution,
    uncorrectable_qubits,
)

__all__ = [
    "check_weight",
    "readout_errors",
    "pole_circuits",
    "readout_from_poles",
    "iqm_qg",
    "predict_kept_fraction",
    "best_chain",
]

# IQM names its native single-qubit gate "r" in the Qiskit target and "prx" in the error profile
_PROFILE_GATE = {"r": "prx", "prx": "prx", "cz": "cz"}


def check_weight(circuit) -> dict:
    """Run qang.ibm.WeightPreservationCheck on ``circuit`` (use the circuit before transpiling to IQM).

    Returns {"weight_preserving": True/False/None, "initial_weight": w or None, "non_preserving_gates": [...]}."""
    from qiskit.transpiler import PassManager

    pm = PassManager([WeightPreservationCheck()])
    pm.run(circuit)
    ps = pm.property_set
    return {"weight_preserving": ps["qg_weight_preserving"], "initial_weight": ps["qg_initial_weight"],
            "non_preserving_gates": ps["qg_non_preserving_gates"]}


def _error_profile(backend):
    ep = getattr(backend, "error_profile", None)
    if ep is None:
        raise ValueError("this backend publishes no error profile (a real IQMBackend): measure the readout with "
                         "pole_circuits / readout_from_poles instead.")
    return ep


def readout_errors(backend, physical_qubits: Sequence[int]) -> List[Tuple[float, float]]:
    """[(P(1|0), P(0|1))] per physical qubit (backend indices) from an IQM fake backend's error profile."""
    ep = _error_profile(backend)
    out = []
    for q in physical_qubits:
        r = ep.readout_errors[backend.index_to_qubit_name(q)]
        out.append((float(r["0"]), float(r["1"])))
    return out


def pole_circuits(n_qubits: int):
    """The two readout-calibration circuits on ``n_qubits``: all in |0> and all in |1> (X on each), measured.
    Transpile them with the same ``initial_layout`` as the experiment and ``optimization_level=0``."""
    from qiskit import QuantumCircuit

    out = []
    for flip in (False, True):
        qc = QuantumCircuit(n_qubits, n_qubits, name="pole1" if flip else "pole0")
        if flip:
            qc.x(range(n_qubits))
        qc.barrier()
        qc.measure(range(n_qubits), range(n_qubits))
        out.append(qc)
    return out


def _canonical(counts):
    from . import interop as IO

    if counts and isinstance(next(iter(counts)), str):
        return IO.canonical_counts(counts, sdk="qiskit")
    return counts


def readout_from_poles(counts_zero, counts_one) -> List[Tuple[float, float]]:
    """[(P(1|0), P(0|1))] per qubit from the counts of the two ``pole_circuits`` (Qiskit bit strings or qang
    tuples). Assumes independent readout errors: each qubit's error is its marginal on the pole."""
    c0, c1 = _canonical(counts_zero), _canonical(counts_one)
    n = len(next(iter(c0)))
    s0, s1 = sum(c0.values()), sum(c1.values())
    out = []
    for i in range(n):
        e01 = sum(c for k, c in c0.items() if k[i] == 1) / s0
        e10 = sum(c for k, c in c1.items() if k[i] == 0) / s1
        out.append((float(e01), float(e10)))
    return out


def iqm_qg(counts, transpiled, errors: Optional[Sequence[Tuple[float, float]]] = None,
           weight: Optional[int] = None) -> dict:
    """qg read-out of the counts of one circuit run with ``IQMBackend.run`` (``result.get_counts(i)``).

    ``errors`` are the readout errors of the measured qubits in clbit order (``readout_errors`` on the
    ``measured_physical_qubits`` of ``transpiled``, or ``readout_from_poles``); without them only the raw and
    filtered entries are returned. Returns {"raw", "readout", "filtered", "filtered_readout": [(qg_Z, se)],
    "kept_raw", "kept_readout", "physical_qubits"}, as qang.ibm.ibm_qg; the filtered entries need ``weight``.
    Errors of corrected values are the shot-noise errors scaled by 1 / (1 - P(1|0) - P(0|1)).
    """
    from . import interop as IO

    c = _canonical(counts)
    n = len(next(iter(c)))
    shots = sum(c.values())
    phys = measured_physical_qubits(transpiled)
    if len(phys) != n:
        raise ValueError(f"the counts have {n} bits but the circuit measures {len(phys)} qubits.")
    out: Dict[str, object] = {"physical_qubits": phys, "raw": IO.qg_z_profile(c)}
    if weight is not None:
        kept, frac = IO.weight_filter(c, weight)
        out["kept_raw"] = frac
        out["filtered"] = IO.qg_z_profile(kept) if frac > 0 else None
    if errors is None:
        return out
    if len(errors) != n:
        raise ValueError(f"need readout errors for {n} qubits, got {len(errors)}.")
    p = np.zeros(2 ** n)
    for k, v in c.items():
        p[int("".join(map(str, k)), 2)] += v / shots
    q = readout_inverse_distribution(p, errors)
    bad = uncorrectable_qubits(errors)
    amp = [1.0 if i in bad else 1 / (1 - a - b) for i, (a, b) in enumerate(errors)]
    out["readout_errors"] = list(errors)
    out["readout"] = [(v, se * a) for (v, se), a in zip(_profile(q, n, shots), amp)]
    if weight is not None:
        w = np.array([bin(i).count("1") for i in range(2 ** n)])
        mass = float(q[w == weight].sum())
        out["kept_readout"] = mass
        if mass > 0:
            qf = np.where(w == weight, q, 0.0) / mass
            out["filtered_readout"] = [(v, se * a) for (v, se), a in
                                       zip(_profile(qf, n, max(int(round(mass * shots)), 1)), amp)]
        else:
            out["filtered_readout"] = None
    return out


def predict_kept_fraction(transpiled, backend, weight: int) -> dict:
    """Predicted fraction of shots of Hamming weight ``weight`` for a weight-conserving circuit transpiled to an
    IQM fake backend, from its error profile (times in ns). Each channel becomes a probability of changing the
    weight, as in qang.ibm.predict_kept_fraction:
      single-qubit gate, depolarizing parameter p  -> p/2   (X and Y of the four Paulis move the weight);
      CZ, depolarizing parameter p                  -> 5p/8  (10 of the 16 two-qubit Paulis, as in qang.ibm);
      relaxation over the busy time t of a qubit    -> (1 - exp(-t/T1)) w/n;
      pure dephasing during the gates               -> 2z/3, z = (1 - exp(-t/T_phi))/2;
      readout                                       -> w/n P(0|1) + (1 - w/n) P(1|0).
    ``kept`` is the product S of the survivals; ``kept_mixed_floor`` = S + (1 - S) C(n, w)/2^n also counts the
    shots that land back in the sector, as for a fully mixed state (the better estimate on IBM, §138).
    """
    ep = _error_profile(backend)
    name = backend.index_to_qubit_name
    phys = measured_physical_qubits(transpiled)
    n = len(phys)
    frac = weight / n
    s_gate, busy = 1.0, {}
    for inst in transpiled.data:
        op = inst.operation.name
        if op not in _PROFILE_GATE:
            continue
        qs = tuple(transpiled.find_bit(q).index for q in inst.qubits)
        g = _PROFILE_GATE[op]
        if len(qs) == 1:
            p = ep.single_qubit_gate_depolarizing_error_parameters[g].get(name(qs[0]), 0.0)
            s_gate *= 1 - p / 2
            d = ep.single_qubit_gate_durations[g]
        else:
            table = ep.two_qubit_gate_depolarizing_error_parameters[g]
            a, b = name(qs[0]), name(qs[1])
            p = table.get((a, b), table.get((b, a), 0.0))
            s_gate *= 1 - 5 * p / 8
            d = ep.two_qubit_gate_durations[g]
        for q in qs:
            busy[q] = busy.get(q, 0.0) + d
    s_relax = s_deph = 1.0
    for q in phys:
        t1, t2, tb = ep.t1s[name(q)], ep.t2s[name(q)], busy.get(q, 0.0)
        s_relax *= 1 - (1 - np.exp(-tb / t1)) * frac
        rate = max(1 / min(t2, 2 * t1) - 1 / (2 * t1), 0.0)
        s_deph *= 1 - 2 * (1 - np.exp(-tb * rate)) / 2 / 3
    s_read = 1.0
    for e01, e10 in readout_errors(backend, phys):
        s_read *= 1 - (frac * e10 + (1 - frac) * e01)
    kept = s_gate * s_relax * s_deph * s_read
    mixed = comb(n, weight) / 2 ** n
    return {"kept": float(kept), "kept_mixed_floor": float(kept + (1 - kept) * mixed), "gates": float(s_gate),
            "relaxation": float(s_relax), "dephasing": float(s_deph), "readout": float(s_read)}


def best_chain(circuit, backend, weight: int, limit: int = 100, seed: int = 0, seed_transpiler: int = 137,
               optimization_level: int = 3) -> dict:
    """The initial layout with the highest predicted kept fraction (``kept_mixed_floor``) for ``circuit`` (measured)
    on an IQM fake backend. Candidates: the transpiler's default layout and ``limit`` chains of the coupling map
    (qang.ibm.linear_chains), each transpiled with the same settings (those of §165 by default) and scored, so
    the choice is never worse than the default by the prediction.
    Returns {"layout", "prediction", "transpiled", "default_prediction", "candidates"}."""
    from qiskit import transpile

    def score(lay):
        kw = {"initial_layout": lay} if lay is not None else {}
        tc = transpile(circuit, backend, optimization_level=optimization_level, seed_transpiler=seed_transpiler, **kw)
        return tc, predict_kept_fraction(tc, backend, weight)

    t0, p0 = score(None)
    best = {"layout": None, "prediction": p0, "transpiled": t0}
    n_cand = 1
    for chain in linear_chains(backend, circuit.num_qubits, limit=limit, seed=seed):
        tc, pr = score(chain)
        n_cand += 1
        if pr["kept_mixed_floor"] > best["prediction"]["kept_mixed_floor"]:
            best = {"layout": chain, "prediction": pr, "transpiled": tc}
    best["default_prediction"] = p0
    best["candidates"] = n_cand
    return best
