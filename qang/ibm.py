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

Also (§138):
  * ``estimator_qg``            qg_Z (and the filtered qg_Z) from an Estimator
    (IBM Runtime or any V2 estimator): the qg observables mapped through the
    circuit's layout, so device-side mitigation (TREX, ZNE) applies to them;
  * ``predict_kept_fraction``   the fraction of shots the qg filter will keep,
    predicted from the calibration before running: gate errors, relaxation
    and pure dephasing during the gates (§133), and readout;
  * ``linear_chains`` / ``best_chain``  candidate physical-qubit chains of the
    coupling map, and the one with the highest predicted kept fraction.

And qang's weight-conserving QNN (qang.qml.WeightQNN) on IBM (§139):
  * ``qnn_template``  the QNN as ONE parameterized Qiskit circuit (amplitude
    loader + trained layers, every RBS a native XX+YY gate), so a whole data
    set is a single Sampler pub: one transpilation, one job;
  * ``run_qnn``       transpile (optionally on ``best_chain``), run, and read
    the decision values raw, with the qg filter, and with the filter after
    calibrated readout correction.

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
    "estimator_qg",
    "predict_kept_fraction",
    "linear_chains",
    "best_chain",
    "best_layout",
    "plan_shots",
    "fold_two_qubit_gates",
    "zne_qg",
    "qnn_template",
    "qnn_parameter_values",
    "run_qnn",
    "unary_loader",
    "unary_orthogonal",
    "pca_rotation",
    "unary_spectrum",
    "xy_trotter",
    "xy_exact_occupations",
    "occupations",
    "xxz_trotter",
    "xxz_exact_occupations",
    "unary_overlap_circuit",
    "swap_test_circuit",
    "overlap_fidelity",
    "swap_test_fidelity",
    "add_static_zz",
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
        leading_x, touched = set(), set()
        for i, inst in enumerate(qc.data):
            name = inst.operation.name
            qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
            if name in ("measure", "barrier"):
                continue
            # an X that is the first operation on its qubit prepares an excitation from |0> (per qubit, so
            # the order in which the DAG lists gates on different qubits does not matter)
            if name == "x" and qubits[0] not in touched:
                leading_x.add(qubits[0])
                touched.add(qubits[0])
                continue
            touched.update(qubits)
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


MIN_READOUT_CONTRAST = 0.05


def uncorrectable_qubits(errors: Sequence[Tuple[float, float]]) -> List[int]:
    """Indices whose readout cannot be inverted: contrast 1 - P(1|0) - P(0|1) below MIN_READOUT_CONTRAST
    (for instance a qubit published as always reading 1). Their readout is left uncorrected (§144)."""
    return [i for i, (a, b) in enumerate(errors) if 1 - a - b < MIN_READOUT_CONTRAST]


def readout_inverse_distribution(probs: np.ndarray, errors: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Invert independent per-qubit readout errors on a distribution indexed qubit 0 first
    (index bit n-1-i is qubit i). Returns a quasi-probability vector (may have small negatives).
    Qubits whose confusion matrix is (near-)singular are left uncorrected (see uncorrectable_qubits)."""
    n = len(errors)
    p = np.asarray(probs, float).reshape((2,) * n)
    skip = set(uncorrectable_qubits(errors))
    for i, (e01, e10) in enumerate(errors):
        if i in skip:
            continue
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
    bad = uncorrectable_qubits(errs)
    amp = [1.0 if i in bad else 1 / (1 - a - b) for i, (a, b) in enumerate(errs)]
    out = {"physical_qubits": phys, "readout_errors": errs, "uncorrectable_qubits": [phys[i] for i in bad],
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


# ---------------------------------------------------------------- Estimator, prediction, layout (§138)


def estimator_qg(estimator, transpiled, weight: Optional[int] = None, n_qubits: Optional[int] = None,
                 parameter_values=None) -> dict:
    """qg_Z per virtual qubit (and, with ``weight``, the filtered qg_Z and kept fraction) from a V2 estimator.

    ``transpiled`` is the device circuit WITHOUT measurements; the qg observables of qang.interop are mapped
    through its layout. Returns {"qg": [(value, std)], "kept": float, "filtered": [values]}.
    """
    from qiskit.quantum_info import SparsePauliOp

    from . import interop as IO

    n = n_qubits or len(transpiled.layout.initial_virtual_layout(filter_ancillas=True))
    obs = [SparsePauliOp(lab) for lab in IO.qg_z_observables(n)]
    if weight is not None:
        obs += [SparsePauliOp.from_list(list(d.items())) for d in IO.filtered_observables(n, weight)]
    obs = [o.apply_layout(transpiled.layout) for o in obs]
    pub = (transpiled, obs) if parameter_values is None else (transpiled, obs, parameter_values)
    data = estimator.run([pub]).result()[0].data
    evs, stds = np.asarray(data.evs, float), np.asarray(data.stds, float)
    out = {"qg": [(float(evs[i]), float(stds[i])) for i in range(n)]}
    if weight is not None:
        kept, q = IO.estimator_filtered_qg(evs[n:])
        out["kept"], out["filtered"] = float(kept), [float(x) for x in q]
    return out


def predict_kept_fraction(transpiled, backend, weight: int) -> dict:
    """Predicted fraction of shots of Hamming weight ``weight`` for a weight-preserving circuit transpiled to
    ``backend``, from its calibration. Each channel is turned into a probability of changing the weight:
      single-qubit gate of error e   -> e        (depolarizing 2e, X/Y half of it);
      two-qubit gate of error e      -> 5e/6     (depolarizing 4e/3, 10 of 15 Paulis move the weight);
      relaxation over the busy time t of a qubit -> (1 - exp(-t/T1)) * w/n (only excited qubits decay);
      pure dephasing during the gates -> 2z/3 with z = (1 - exp(-t/T_phi))/2 (§133, twirled);
      readout                         -> w/n * P(0|1) + (1 - w/n) * P(1|0).
    The kept fraction is the product S of the survivals; shots that come back into the sector are ignored,
    so ``kept`` is a lower bound in spirit. ``kept_mixed_floor`` = S + (1 - S) C(n, w) / 2^n also counts the
    shots that land in the sector after a weight error, as for a fully mixed state; it is the better
    estimate (found after the §138 predictions were scored: mean error 0.013-0.041 instead of 0.08-0.13).
    Returns both and the parts.
    """
    t = backend.target
    phys = measured_physical_qubits(transpiled)
    n = len(phys)
    frac = weight / n
    s_gate, busy = 1.0, {}
    for inst in transpiled.data:
        name = inst.operation.name
        if name in ("measure", "barrier", "rz", "delay", "reset"):
            continue
        qs = tuple(transpiled.find_bit(q).index for q in inst.qubits)
        props = t[name].get(qs) if name in t.operation_names else None
        e = (props.error or 0.0) if props is not None else 0.0
        d = (props.duration or 0.0) if props is not None else 0.0
        s_gate *= 1 - (e if len(qs) == 1 else 5 * e / 6)
        for q in qs:
            busy[q] = busy.get(q, 0.0) + d
    s_relax = s_deph = 1.0
    for q in phys:
        qp = t.qubit_properties[q]
        tb = busy.get(q, 0.0)
        if qp is not None and qp.t1:
            s_relax *= 1 - (1 - np.exp(-tb / qp.t1)) * frac
            if qp.t2:
                rate = max(1 / min(qp.t2, 2 * qp.t1) - 1 / (2 * qp.t1), 0.0)
                s_deph *= 1 - 2 * (1 - np.exp(-tb * rate)) / 2 / 3
    s_read = 1.0
    for e01, e10 in readout_errors(backend, phys):
        s_read *= 1 - (frac * e10 + (1 - frac) * e01)
    kept = s_gate * s_relax * s_deph * s_read
    from math import comb

    mixed = comb(n, weight) / 2 ** n
    return {"kept": float(kept), "kept_mixed_floor": float(kept + (1 - kept) * mixed), "gates": float(s_gate), "relaxation": float(s_relax),
            "dephasing": float(s_deph), "readout": float(s_read)}


def linear_chains(backend, length: int, limit: int = 200, seed: int = 0) -> List[List[int]]:
    """Up to ``limit`` distinct simple paths of ``length`` physical qubits in the coupling map (random DFS)."""
    cmap = backend.target.build_coupling_map()
    nbrs: Dict[int, set] = {}
    for a, b in cmap.get_edges():
        nbrs.setdefault(a, set()).add(b)
        nbrs.setdefault(b, set()).add(a)
    rng = np.random.default_rng(seed)
    nodes = sorted(nbrs)
    seen, out = set(), []
    tries = 0
    while len(out) < limit and tries < 50 * limit:
        tries += 1
        path = [int(rng.choice(nodes))]
        while len(path) < length:
            opts = [v for v in nbrs[path[-1]] if v not in path]
            if not opts:
                break
            path.append(int(rng.choice(opts)))
        if len(path) == length:
            key = tuple(path) if path[0] < path[-1] else tuple(reversed(path))
            if key not in seen:
                seen.add(key)
                out.append(list(key))
    return out


def best_layout(circuit, backend, weight: int, limit: int = 100, seed: int = 0, seed_transpiler: int = 137,
                optimization_level: int = 1) -> dict:
    """The initial layout with the highest predicted kept fraction (``kept_mixed_floor``) for ``circuit`` AS IT
    WILL RUN (parameterized circuits are fine: the prediction uses gate errors, not angles).

    Candidates: the transpiler's own default layout and ``limit`` chains of the coupling map; each is
    transpiled with the same settings and scored, so the choice is never worse than the default by the
    prediction, and circuits that are not chains (rings, all-to-all) are scored after their routing.
    Returns {"layout", "prediction", "transpiled", "default_prediction", "candidates"}.
    """
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


def best_chain(circuit, backend, weight: int, limit: int = 100, seed: int = 0, seed_transpiler: int = 0):
    """The chain (initial layout) with the highest predicted kept fraction among ``linear_chains``.
    ``circuit`` must be measured. Returns (layout, prediction, transpiled circuit)."""
    from qiskit import transpile

    best = None
    for chain in linear_chains(backend, circuit.num_qubits, limit=limit, seed=seed):
        tc = transpile(circuit, backend, initial_layout=chain, optimization_level=1, seed_transpiler=seed_transpiler)
        pred = predict_kept_fraction(tc, backend, weight)
        if best is None or pred["kept"] > best[1]["kept"]:
            best = (chain, pred, tc)
    return best


# ---------------------------------------------------------------- shot planning (§143)


def plan_shots(transpiled, backend, weight: int, target_se: float, qg=None) -> dict:
    """Shots needed so that every filtered qg_Z has standard error <= ``target_se``, and the device time.

    The filter keeps a fraction k of the shots (``predict_kept_fraction``, kept_mixed_floor); a filtered
    qg_Z read from N k kept shots has standard error sqrt((1 - qg^2) / (N k)). With the expected qg values
    (``qg``, e.g. from a noiseless simulation) the worst qubit sets N; without them qg = 0 (worst case).
    Device time ~ N (circuit duration + repetition delay), an estimate of the QPU seconds (overheads such
    as compilation and queue are not included).
    """
    import math

    k = predict_kept_fraction(transpiled, backend, weight)["kept_mixed_floor"]
    q = np.zeros(1) if qg is None else np.asarray(qg, float)
    need = float(np.max(1 - q ** 2))
    shots = int(math.ceil(need / (k * target_se ** 2)))
    try:
        dur = float(transpiled.estimate_duration(backend.target, unit="s"))
    except Exception:
        dur = 0.0
    rep = 0.0
    try:
        rep = float(getattr(backend, "default_rep_delay", None) or backend.configuration().default_rep_delay)
    except Exception:
        rep = 0.0
    return {"shots": shots, "kept": k, "kept_shots": int(round(shots * k)), "circuit_s": dur,
            "rep_delay_s": rep, "qpu_seconds": shots * (dur + rep)}


# ---------------------------------------------------------------- ZNE with the qg filter (§142)


def fold_two_qubit_gates(transpiled, scale: int):
    """Local unitary folding of every two-qubit gate of a transpiled circuit: G -> G (G^dagger G)^k with
    scale = 2k + 1 (odd). Routing, layout and measurements are kept; barriers around each fold stop the
    folds from being simplified away. Folding keeps a weight-preserving circuit weight-preserving."""
    if scale < 1 or scale % 2 == 0:
        raise ValueError("scale must be an odd positive integer.")
    out = transpiled.copy_empty_like()
    k = (scale - 1) // 2
    for inst in transpiled.data:
        out.append(inst)
        if k and inst.operation.num_qubits == 2 and inst.operation.name not in ("barrier", "measure"):
            out.barrier(inst.qubits)
            for _ in range(k):
                out.append(inst.operation.inverse(), inst.qubits)
                out.append(inst.operation, inst.qubits)
            out.barrier(inst.qubits)
    return out


def _extrapolate(scales, values, method: str):
    x = np.asarray(scales, float)
    y = np.asarray(values, float)
    if method == "linear":
        b, a = np.polyfit(x, y, 1)
        return a
    if method == "richardson":
        # Lagrange extrapolation to 0 through all points
        tot = 0.0
        for i in range(len(x)):
            w = np.prod([x[j] / (x[j] - x[i]) for j in range(len(x)) if j != i])
            tot += w * y[i]
        return tot
    raise ValueError("method must be 'linear' or 'richardson'.")


def zne_qg(sampler, transpiled, backend, weight: int, scales=(1, 3), shots: int = 4000,
           method: str = "linear") -> dict:
    """Zero-noise extrapolation of qg_Z with folded two-qubit gates, for each read-out of ``ibm_qg``
    (raw, readout, filtered, filtered_readout). Returns {"scales", "values": {kind: [[qg per qubit] per
    scale]}, "zne": {kind: [extrapolated qg per qubit, clipped to [-1, 1]]}, "kept": [kept_raw per scale]}."""
    kinds = ("raw", "readout", "filtered", "filtered_readout")
    circs = [fold_two_qubit_gates(transpiled, s) for s in scales]
    res = sampler.run(circs, shots=shots).result()
    vals = {k: [] for k in kinds}
    kept = []
    for r, c in zip(res, circs):
        out = ibm_qg(r, c, backend, weight=weight)
        for k in kinds:
            vals[k].append([v for v, _ in out[k]])
        kept.append(out["kept_raw"])
    zne = {k: [float(np.clip(_extrapolate(scales, np.array(vals[k])[:, i], method), -1, 1))
               for i in range(len(vals[k][0]))] for k in kinds}
    return {"scales": list(scales), "values": vals, "zne": zne, "kept": kept}


# ---------------------------------------------------------------- QNN on IBM (§139)


def _check_qnn(model):
    if getattr(model, "weight", 1) != 1:
        raise ValueError("qnn_template supports the weight-1 (amplitude) encoding.")


def qnn_template(model, measure: bool = True):
    """The weight-1 WeightQNN as a parameterized circuit: parameters x[0..n-2] are the loader angles of the
    input and w[0..n_theta-1] the trained RBS angles. RBS(t) on [a, b] = XXPlusYYGate(2t, pi/2), with qang
    qubit i on Qiskit qubit n-1-i (so the integer of a Qiskit bitstring is the model's basis index)."""
    import math

    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import XXPlusYYGate

    _check_qnn(model)
    n = model.n
    q = lambda i: n - 1 - i  # noqa: E731
    xa, wa = ParameterVector("x", n - 1), ParameterVector("w", model.n_theta)
    qc = QuantumCircuit(n)
    qc.x(q(0))
    for k in range(n - 1):
        qc.append(XXPlusYYGate(2 * xa[k], math.pi / 2), [q(k), q(k + 1)])
    k = 0
    for _ in range(model.layers):
        for pairs in model.sublayers:
            for a, b in pairs:
                qc.append(XXPlusYYGate(2 * wa[k], math.pi / 2), [q(a), q(b)])
                k += 1
    if measure:
        qc.measure_all()
    return qc


def _loader_angles(v):
    import math

    v = np.asarray(v, float)
    n = len(v)
    ang = [math.atan2(np.linalg.norm(v[k + 1:]), v[k]) for k in range(n - 2)]
    ang.append(math.atan2(v[n - 1], v[n - 2]))
    return ang


def qnn_parameter_values(model, X, params=None) -> np.ndarray:
    """(S, n-1+n_theta) parameter values of ``qnn_template`` for the inputs X (order: x then w)."""
    _check_qnn(model)
    params = model.params_ if params is None else params
    X = np.atleast_2d(np.asarray(X, float))
    V = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    th = np.asarray(params[: model.n_theta], float)
    return np.array([np.concatenate([_loader_angles(v), th]) for v in V])


def _ordered_values(template, values):
    """Reorder [x..., w...] columns to the template's parameter order."""
    names = [p.name for p in template.parameters]
    n_x = sum(1 for nm in names if nm.startswith("x["))
    col = {f"x[{i}]": i for i in range(n_x)}
    col.update({f"w[{i}]": n_x + i for i in range(values.shape[1] - n_x)})
    return values[:, [col[nm] for nm in names]]


def run_qnn(model, X, backend, sampler=None, layout=None, shots: int = 1000, params=None,
            seed_transpiler: int = 137) -> dict:
    """Run a trained weight-1 WeightQNN on ``backend`` for all inputs X in one Sampler job.

    ``layout``: None (transpiler default), "best" (``best_layout`` on the template that runs; §141),
    "best_chain" (§139 behaviour) or a list of physical qubits.
    Returns {"layout": [...], "kept_raw": [...], "decision": {"raw", "qang", "qang_readout"}: arrays}.
    """
    from qiskit import transpile

    from . import interop as IO

    params = model.params_ if params is None else params
    tpl = qnn_template(model)
    if isinstance(layout, str) and layout == "best":
        layout = best_layout(tpl, backend, 1, seed_transpiler=seed_transpiler)["layout"]
    elif isinstance(layout, str) and layout == "best_chain":
        bound = tpl.assign_parameters(_ordered_values(tpl, qnn_parameter_values(model, X[:1], params))[0])
        layout = best_chain(bound, backend, 1, seed_transpiler=seed_transpiler)[0]
    kw = {"initial_layout": layout} if layout is not None else {}
    t = transpile(tpl, backend, optimization_level=1, seed_transpiler=seed_transpiler, **kw)
    vals = _ordered_values(tpl, qnn_parameter_values(model, X, params))
    if sampler is None:
        try:
            from qiskit_ibm_runtime.executor_sampler import Sampler
        except ImportError:  # pragma: no cover
            from qiskit_ibm_runtime import SamplerV2 as Sampler
        sampler = Sampler(mode=backend)
    res = sampler.run([(t, vals)], shots=shots).result()[0]
    bits = IO._bitarray(res, None)
    n = model.n
    phys = measured_physical_qubits(t)
    errs = readout_errors(backend, phys)[::-1]          # Qiskit index: clbit j is bit j from the right
    w_head, b = params[model.n_theta:model.n_theta + model.n_head], params[-1]
    raw, filt, filt_ro, kept = [], [], [], []
    for i in range(len(vals)):
        p = np.zeros(2 ** n)
        for key, c in bits.get_counts(loc=i).items():
            p[int(key.replace(" ", ""), 2)] += c
        p /= p.sum()
        q = readout_inverse_distribution(p, errs)
        raw.append(float(model.qg_z(p, qang=False)[0] @ w_head + b))
        filt.append(float(model.qg_z(p, qang=True)[0] @ w_head + b))
        filt_ro.append(float(model.qg_z(q, qang=True)[0] @ w_head + b))
        kept.append(float(sum(p[1 << j] for j in range(n))))
    return {"layout": list(layout) if layout is not None else measured_physical_qubits(t), "kept_raw": kept,
            "decision": {"raw": np.array(raw), "qang": np.array(filt), "qang_readout": np.array(filt_ro)}}


# ---------------------------------------------------------------- spectra in the weight-1 sector (§146)
#
# Unary encoding: component k of a real vector is the amplitude of the state with only Qiskit qubit k
# excited. A mixture of loaded vectors x_j with weights |x_j|^2 / sum |x|^2 is rho = C / tr C, C the
# (uncentred) second-moment matrix of the data: the density matrix quantum PCA diagonalizes. Rotating with
# the PCA basis puts eigenvalue lambda_k on qubit k, so P(qubit k excited) = lambda_k / sum(lambda).


def _rbs(qc, t, a, b):
    import math

    from qiskit.circuit.library import XXPlusYYGate

    # on (e_a, e_b): [[cos t, -sin t], [sin t, cos t]]
    qc.append(XXPlusYYGate(2 * t, math.pi / 2), [a, b])


def unary_loader(v):
    """Circuit preparing sum_k v_k |e_k> (v normalized; e_k = only Qiskit qubit k excited), n-1 RBS on a chain."""
    from qiskit import QuantumCircuit

    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    n = len(v)
    qc = QuantumCircuit(n)
    qc.x(0)
    for k, t in enumerate(_loader_angles(v)):
        _rbs(qc, t, k, k + 1)
    return qc


def unary_orthogonal(O, atol: float = 1e-12):
    """Nearest-neighbour RBS circuit acting as the real orthogonal matrix O on the weight-1 sector, up to signs
    of the output components (irrelevant to computational-basis read-out): at most n(n-1)/2 RBS."""
    from qiskit import QuantumCircuit

    O = np.asarray(O, float)
    n = O.shape[0]
    if O.shape != (n, n) or not np.allclose(O @ O.T, np.eye(n), atol=1e-8):
        raise ValueError("O must be a square real orthogonal matrix.")
    M = O.T.copy()           # reduce O^T to a diagonal with rotations on rows: R O^T = D, so O = D R
    gates = []
    for c in range(n - 1):
        for r in range(n - 1, c, -1):
            a, b = r - 1, r
            if abs(M[b, c]) < atol:
                continue
            t = np.arctan2(-M[b, c], M[a, c])
            ct, st = np.cos(t), np.sin(t)
            Ma, Mb = M[a].copy(), M[b].copy()
            M[a], M[b] = ct * Ma - st * Mb, st * Ma + ct * Mb
            gates.append((t, a, b))
    qc = QuantumCircuit(n)
    for t, a, b in gates:
        _rbs(qc, t, a, b)
    return qc


def pca_rotation(X, center: bool = True):
    """(O, eigenvalues) of the PCA of the rows of X: C = X^T X / m (after centring), eigenvalues descending,
    O = V^T so that unary_orthogonal(O) sends principal component k to Qiskit qubit k."""
    X = np.asarray(X, float)
    if center:
        X = X - X.mean(axis=0)
    C = X.T @ X / len(X)
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    return V[:, order].T, lam[order]


def unary_spectrum(outs, weights, kind: str = "filtered_readout"):
    """Estimate lambda_k / sum(lambda) from the ibm_qg outputs of the circuits loader(x_j) + rotation:
    sum_j w_j P_j(qubit k excited), P = (1 - qg_Z) / 2, w_j = |x_j|^2 / sum |x|^2. Returns (values, se)."""
    w = np.asarray(weights, float)
    w = w / w.sum()
    vals = np.array([[(1 - z) / 2 for z, _ in o[kind]] for o in outs])
    ses = np.array([[se / 2 for _, se in o[kind]] for o in outs])
    return w @ vals, np.sqrt((w ** 2) @ ses ** 2)


# ---------------------------------------------------------------- XY-chain dynamics (§147)
#
# H = J sum_k (X_k X_{k+1} + Y_k Y_{k+1}) / 2 + sum_k h_k Z_k / 2 on a chain of Qiskit qubits 0..n-1. It
# conserves the number of excitations; in the weight-1 sector it is a tight-binding (continuous-time quantum
# walk) Hamiltonian with hopping J and on-site energy -h_k. First-order Trotter step: the fields, then the odd
# bonds (1-2, 3-4, ...), then the even bonds (0-1, 2-3, ...).


def _xy_sector_hamiltonian(n, J, h):
    H = np.diag(-np.asarray(h, float)).astype(complex)
    for k in range(n - 1):
        H[k, k + 1] = H[k + 1, k] = J
    return H


def xy_trotter(n: int, J: float, h, t: float, steps: int, initial: Sequence[int] = (0,), measure: bool = True):
    """First-order Trotter circuit of the XY chain for time t in ``steps`` steps, starting with the qubits in
    ``initial`` excited. Every gate (RZ, XX+YY) conserves the weight."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate

    h = np.zeros(n) if h is None else np.asarray(h, float)
    dt = t / steps
    qc = QuantumCircuit(n)
    for q in initial:
        qc.x(q)
    for _ in range(steps):
        for k in range(n):
            if h[k] != 0:
                qc.rz(h[k] * dt, k)
        for start in (1, 0):
            for k in range(start, n - 1, 2):
                qc.append(XXPlusYYGate(2 * J * dt, 0.0), [k, k + 1])
    if measure:
        qc.measure_all()
    return qc


def xy_exact_occupations(n: int, J: float, h, t: float, initial: int = 0) -> np.ndarray:
    """Exact P(qubit k excited) at time t for one excitation starting on qubit ``initial`` (no Trotter error)."""
    from scipy.linalg import expm

    h = np.zeros(n) if h is None else h
    e = np.zeros(n, complex)
    e[initial] = 1
    return np.abs(expm(-1j * _xy_sector_hamiltonian(n, J, h) * t) @ e) ** 2


def occupations(out: dict, kind: str = "filtered_readout") -> np.ndarray:
    """P(qubit k excited) = (1 - qg_Z) / 2 from an ibm_qg output."""
    return np.array([(1 - z) / 2 for z, _ in out[kind]])


# ---------------------------------------------------------------- XXZ chain, several excitations (§148)
#
# H = sum_k [J (X_k X_{k+1} + Y_k Y_{k+1}) / 2 + V Z_k Z_{k+1} / 4] + sum_k h_k Z_k / 2. The ZZ term makes
# the excitations interact; every term conserves the weight. Each bond is one two-qubit gate,
# exp(-i dt [J (XX + YY)/2 + V ZZ/4]) (the two parts commute), so a step costs no more two-qubit gates than in
# the XY chain.


def _bond_unitary(J, V, dt):
    from scipy.linalg import expm

    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    H = J * (np.kron(X, X) + np.kron(Y, Y)) / 2 + V * np.kron(Z, Z) / 4
    return expm(-1j * dt * H)


def xxz_trotter(n: int, J: float, V: float, h, t: float, steps: int, initial: Sequence[int] = (0,),
                order: int = 1, measure: bool = True):
    """Trotter circuit of the XXZ chain. order 1: per step the fields, the odd bonds, the even bonds. order 2
    (symmetric): fields/2, odd/2, even, odd/2, fields/2 per step, adjacent half-steps merged."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate

    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    h = np.zeros(n) if h is None else np.asarray(h, float)
    dt = t / steps
    qc = QuantumCircuit(n)
    for q in initial:
        qc.x(q)

    def fields(f):
        for k in range(n):
            if h[k] != 0:
                qc.rz(h[k] * dt * f, k)

    def bonds(start, f):
        g = UnitaryGate(_bond_unitary(J, V, dt * f), label="xxz")
        for k in range(start, n - 1, 2):
            qc.append(g, [k, k + 1])   # Qiskit order: the matrix is symmetric under the swap of the pair

    if order == 1:
        for _ in range(steps):
            fields(1)
            bonds(1, 1)
            bonds(0, 1)
    else:
        fields(0.5)
        for s in range(steps):
            bonds(1, 0.5)
            bonds(0, 1)
            bonds(1, 0.5)
            fields(1 if s < steps - 1 else 0.5)
    if measure:
        qc.measure_all()
    return qc


def xxz_exact_occupations(n: int, J: float, V: float, h, t: float, initial: Sequence[int] = (0,)) -> np.ndarray:
    """Exact P(qubit k excited) at time t from the basis state with ``initial`` excited (full Hilbert space,
    n <= 12)."""
    from qiskit.quantum_info import SparsePauliOp
    from scipy.sparse.linalg import expm_multiply

    h = np.zeros(n) if h is None else np.asarray(h, float)
    terms = []
    for k in range(n - 1):
        for P in ("X", "Y"):
            lab = ["I"] * n
            lab[n - 1 - k] = lab[n - 2 - k] = P
            terms.append(("".join(lab), J / 2))
        lab = ["I"] * n
        lab[n - 1 - k] = lab[n - 2 - k] = "Z"
        terms.append(("".join(lab), V / 4))
    for k in range(n):
        lab = ["I"] * n
        lab[n - 1 - k] = "Z"
        terms.append(("".join(lab), h[k] / 2))
    H = SparsePauliOp.from_list(terms).to_matrix(sparse=True)
    psi = np.zeros(2 ** n, complex)
    psi[sum(1 << q for q in initial)] = 1
    psi = expm_multiply(-1j * t * H, psi)
    p = np.abs(psi) ** 2
    idx = np.arange(2 ** n)
    return np.array([p[(idx >> k) & 1 == 1].sum() for k in range(n)])


# ---------------------------------------------------------------- state overlaps: inversion and swap tests (§149)
#
# For unary (weight-1) states |x> = sum_k x_k |e_k>, F = |<x|y>|^2 can be read two ways:
#   inversion test: load x, undo the loader of y; P(e_0) = F. n qubits, every gate conserves the weight.
#   swap test: ancilla (Qiskit qubit 0) in |+>, x on qubits 1..n, y on n+1..2n, controlled SWAPs, H on the
#   ancilla; P(ancilla 0) = (1 + F) / 2. The controlled SWAPs conserve the total weight of the two data
#   registers (2), so the filter applies to them; the ancilla, which carries the answer, is outside it.


def _loader_rbs(qc, v, offset=0):
    for k, t in enumerate(_loader_angles(np.asarray(v, float) / np.linalg.norm(v))):
        _rbs(qc, t, offset + k, offset + k + 1)


def unary_overlap_circuit(x, y, measure: bool = True):
    """Inversion test: P(only qubit 0 excited) = |<x|y>|^2 for the unary states of x and y."""
    from qiskit import QuantumCircuit

    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have the same length.")
    qc = QuantumCircuit(n)
    qc.x(0)
    _loader_rbs(qc, x)
    inv = QuantumCircuit(n)
    _loader_rbs(inv, y)
    qc.compose(inv.inverse(), inplace=True)
    if measure:
        qc.measure_all()
    return qc


def swap_test_circuit(x, y, measure: bool = True):
    """Swap test on 2n + 1 qubits: ancilla 0, x on 1..n, y on n+1..2n. P(ancilla 0) = (1 + |<x|y>|^2) / 2."""
    from qiskit import QuantumCircuit

    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have the same length.")
    qc = QuantumCircuit(2 * n + 1)
    qc.x(1)
    qc.x(n + 1)
    _loader_rbs(qc, x, offset=1)
    _loader_rbs(qc, y, offset=n + 1)
    qc.h(0)
    for k in range(n):
        qc.cswap(0, 1 + k, n + 1 + k)
    qc.h(0)
    if measure:
        qc.measure_all()
    return qc


def _distributions(pub_result, transpiled, backend):
    """(p, q, n): measured distribution indexed by the integer of the bitstring (bit j = clbit j) and its
    readout-corrected quasi-distribution."""
    from . import interop as IO

    bits = IO._bitarray(pub_result, None)
    counts = bits.get_counts()
    n = len(next(iter(counts)).replace(" ", ""))
    p = np.zeros(2 ** n)
    for key, c in counts.items():
        p[int(key.replace(" ", ""), 2)] += c
    p /= p.sum()
    errs = readout_errors(backend, measured_physical_qubits(transpiled))[::-1]
    return p, readout_inverse_distribution(p, errs), n


def _weights(n):
    return np.array([bin(i).count("1") for i in range(2 ** n)])


def overlap_fidelity(pub_result, transpiled, backend) -> dict:
    """|<x|y>|^2 from an inversion test: raw, readout, filtered (weight-1 shots), filtered_readout."""
    p, q, n = _distributions(pub_result, transpiled, backend)
    w1 = _weights(n) == 1
    out = {"raw": float(p[1]), "readout": float(q[1])}
    out["filtered"] = float(p[1] / p[w1].sum()) if p[w1].sum() > 0 else float("nan")
    out["filtered_readout"] = float(q[1] / q[w1].sum()) if q[w1].sum() > 0 else float("nan")
    out["kept_raw"] = float(p[w1].sum())
    return out


def swap_test_fidelity(pub_result, transpiled, backend) -> dict:
    """|<x|y>|^2 = 2 P(ancilla 0) - 1 from a swap test; filtered: shots whose data registers hold weight 2."""
    p, q, n = _distributions(pub_result, transpiled, backend)
    idx = np.arange(2 ** n)
    anc0 = (idx & 1) == 0
    keep = _weights(n - 1)[idx >> 1] == 2
    out = {"raw": float(2 * p[anc0].sum() - 1), "readout": float(2 * q[anc0].sum() - 1)}
    out["filtered"] = float(2 * p[anc0 & keep].sum() / p[keep].sum() - 1) if p[keep].sum() > 0 else float("nan")
    out["filtered_readout"] = (float(2 * q[anc0 & keep].sum() / q[keep].sum() - 1) if q[keep].sum() > 0
                               else float("nan"))
    out["kept_raw"] = float(p[keep].sum())
    return out


# ---------------------------------------------------------------- always-on ZZ crosstalk (§150)


def add_static_zz(transpiled, backend, nu_hz: float, label: str = "static_zz"):
    """Copy of a transpiled circuit with always-on ZZ between coupled qubits: after every layer of the circuit,
    each coupling-map edge between two qubits the circuit uses gets exp(-i theta ZZ / 2),
    theta = 2 pi nu_hz x (layer duration), the duration being the longest calibrated gate of the layer.
    The ZZ terms are inserted as unitary gates (no noise is attached to them in Aer). They conserve the
    weight, so the qg filter cannot remove what they do in the computational frame."""
    import math

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.converters import circuit_to_dag, dag_to_circuit

    used = sorted({transpiled.find_bit(q).index for inst in transpiled.data
                   if inst.operation.name not in ("barrier", "measure") for q in inst.qubits})
    used_set = set(used)
    edges = sorted({tuple(sorted(e)) for e in backend.coupling_map.get_edges()
                    if e[0] in used_set and e[1] in used_set})
    out = QuantumCircuit(*transpiled.qregs, *transpiled.cregs)
    tgt = backend.target
    for layer in circuit_to_dag(transpiled).layers():
        sub = dag_to_circuit(layer["graph"])
        dur, has_measure = 0.0, False
        for inst in sub.data:
            name = inst.operation.name
            if name == "measure":
                has_measure = True
            qs = tuple(sub.find_bit(q).index for q in inst.qubits)
            try:
                d = tgt[name][qs].duration
            except Exception:
                d = None
            dur = max(dur, d or 0.0)
        if has_measure:
            pass
        elif nu_hz and dur > 0:
            theta = 2 * math.pi * nu_hz * dur
            g = np.diag(np.exp(-0.5j * theta * np.array([1, -1, -1, 1])))
            zz = UnitaryGate(g, label=label)
            out.compose(sub, inplace=True)
            for a, b in edges:
                out.append(zz, [a, b])
            continue
        out.compose(sub, inplace=True)
    return out
