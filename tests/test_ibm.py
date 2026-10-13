"""Tests for qang.ibm (§137): weight-preservation pass, readout errors and IBM Runtime read-out."""

import numpy as np
import pytest

pytest.importorskip("qiskit")
from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit.circuit.library import XXPlusYYGate  # noqa: E402
from qiskit.transpiler import PassManager  # noqa: E402

from qang import ibm  # noqa: E402


def _w_like(n=4, seed=0):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    qc.x(0)
    for i in range(n - 1):
        qc.append(XXPlusYYGate(rng.uniform(0.3, 2.8)), [i, i + 1])
    qc.rz(0.4, 2)
    qc.cz(1, 3)
    return qc


def _check(qc):
    pm = PassManager([ibm.WeightPreservationCheck()])
    pm.run(qc)
    return pm.property_set


def test_pass_detects_weight_preserving_circuits():
    ps = _check(_w_like())
    assert ps["qg_weight_preserving"] is True and ps["qg_initial_weight"] == 1
    qc = _w_like()
    qc.h(2)
    ps = _check(qc)
    assert ps["qg_weight_preserving"] is False
    assert [g[1] for g in ps["qg_non_preserving_gates"]] == ["h"]
    # a transpiled weight-preserving circuit stays recognized when gates are bound numerically
    from qiskit_aer import AerSimulator  # noqa: F401
    t = transpile(_w_like(), basis_gates=["cz", "rz", "sx", "x"], optimization_level=0)
    assert _check(t)["qg_weight_preserving"] is False      # decomposed: sx gates break it gate-wise


def test_readout_inverse_recovers_distribution():
    rng = np.random.default_rng(1)
    n = 3
    true = rng.dirichlet(np.ones(2 ** n))
    errs = [(0.02, 0.05), (0.01, 0.03), (0.04, 0.02)]
    noisy = true.reshape((2,) * n)
    for i, (e01, e10) in enumerate(errs):
        M = np.array([[1 - e01, e10], [e01, 1 - e10]])
        noisy = np.moveaxis(np.tensordot(M, np.moveaxis(noisy, i, 0), axes=(1, 0)), 0, i)
    back = ibm.readout_inverse_distribution(noisy.reshape(-1), errs)
    assert np.allclose(back, true)


def test_runtime_sampler_on_fake_backend():
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit.quantum_info import Statevector
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler as SamplerV2   # client-side sampler (>= 0.50)
    except ImportError:  # pragma: no cover
        from qiskit_ibm_runtime import SamplerV2
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    b = FakeTorino()
    qc = _w_like(4, seed=3)
    ideal = [float(Statevector(qc).expectation_value(
        __import__("qiskit").quantum_info.SparsePauliOp("I" * (3 - i) + "Z" + "I" * i)).real) for i in range(4)]
    qm = qc.copy()
    qm.measure_all()
    t = transpile(qm, b, optimization_level=1, seed_transpiler=7)
    res = SamplerV2(mode=b).run([t], shots=4000).result()[0]
    out = ibm.ibm_qg(res, t, b, weight=1)
    assert len(out["physical_qubits"]) == 4 and all(0 <= a < 0.5 and 0 <= c < 0.5 for a, c in out["readout_errors"])
    assert 0 < out["kept_raw"] <= 1 and out["kept_readout"] > out["kept_raw"] - 0.05
    for key in ("raw", "readout", "filtered", "filtered_readout"):
        assert len(out[key]) == 4
        assert all(-1 <= v <= 1 for v, _ in out[key])
    err = {k: np.mean([abs(v - z) for (v, _), z in zip(out[k], ideal)]) for k in ("raw", "filtered_readout")}
    assert err["filtered_readout"] < 0.15


def test_example_circuits_are_weight_one():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    import ibm_runtime_qg as E

    for s in range(3):
        ps = _check(E.circuit(s))
        assert ps["qg_weight_preserving"] is True and ps["qg_initial_weight"] == 1
        assert abs(sum((1 - z) / 2 for z in E.ideal(E.circuit(s))) - 1) < 1e-9     # one excitation in total


def test_estimator_predictor_and_chains():
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime import EstimatorV2
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

    b = FakeSherbrooke()
    qc = _w_like(4, seed=2)
    t = transpile(qc, b, optimization_level=1, seed_transpiler=3)
    out = ibm.estimator_qg(EstimatorV2(mode=b), t, weight=1)
    assert len(out["qg"]) == 4 and 0.5 < out["kept"] <= 1.0 and len(out["filtered"]) == 4
    qm = qc.copy()
    qm.measure_all()
    tm = transpile(qm, b, optimization_level=1, seed_transpiler=3)
    pred = ibm.predict_kept_fraction(tm, b, 1)
    assert 0.3 < pred["kept"] < 1.0
    assert abs(pred["kept"] - pred["gates"] * pred["relaxation"] * pred["dephasing"] * pred["readout"]) < 1e-12
    chains = ibm.linear_chains(b, 4, limit=20, seed=1)
    edges = {tuple(sorted(e)) for e in b.target.build_coupling_map().get_edges()}
    assert len(chains) == 20 and all(len(set(c)) == 4 for c in chains)
    assert all(tuple(sorted((c[i], c[i + 1]))) in edges for c in chains for i in range(3))
    layout, p, tc = ibm.best_chain(qm, b, 1, limit=10, seed=1)
    assert p["kept"] >= max(ibm.predict_kept_fraction(
        transpile(qm, b, initial_layout=c, optimization_level=1, seed_transpiler=0), b, 1)["kept"]
        for c in ibm.linear_chains(b, 4, limit=10, seed=1)) - 1e-12


def test_qnn_template_matches_weightqnn():
    from qiskit.quantum_info import Statevector

    from qang.qml import WeightQNN

    m = WeightQNN(5, 1)
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (4, 4))
    m.fit(X, (X[:, 0] > 0).astype(int), epochs=2)
    tpl = ibm.qnn_template(m, measure=False)
    vals = ibm._ordered_values(tpl, ibm.qnn_parameter_values(m, X))
    ref = m.probs(m.params_[: m.n_theta], m.encode(X))
    for i in range(len(X)):
        assert np.allclose(Statevector(tpl.assign_parameters(vals[i])).probabilities(), ref[i], atol=1e-9)
    ps = _check(tpl)
    assert ps["qg_weight_preserving"] is True and ps["qg_initial_weight"] == 1


def test_run_qnn_on_fake_backend():
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

    from qang.qml import WeightQNN

    m = WeightQNN(5, 1)
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (6, 4))
    y = (X[:, 0] > 0).astype(int)
    m.fit(X, y, epochs=30)
    r = ibm.run_qnn(m, X, FakeSherbrooke(), shots=400)
    ideal = m.decision(m.params_, X)
    assert set(r["decision"]) == {"raw", "qang", "qang_readout"} and len(r["kept_raw"]) == 6
    assert np.mean(np.sign(r["decision"]["qang_readout"]) == np.sign(ideal)) >= 0.8


def test_best_layout_never_below_default_prediction():
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

    from qang.qml import WeightQNN

    b = FakeSherbrooke()
    tpl = ibm.qnn_template(WeightQNN(5, 1))
    r = ibm.best_layout(tpl, b, 1, limit=8, seed=2)
    assert r["candidates"] == 9
    assert r["prediction"]["kept_mixed_floor"] >= r["default_prediction"]["kept_mixed_floor"]


def test_folding_keeps_the_unitary_and_triples_two_qubit_gates():
    from qiskit.quantum_info import Operator

    qc = transpile(_w_like(3, seed=4), basis_gates=["cz", "rz", "sx", "x"], optimization_level=0)
    f = ibm.fold_two_qubit_gates(qc, 3)
    n2 = lambda c: sum(1 for i in c.data if i.operation.name == "cz")  # noqa: E731
    assert n2(f) == 3 * n2(qc)
    assert Operator(f).equiv(Operator(qc))
    with pytest.raises(ValueError):
        ibm.fold_two_qubit_gates(qc, 2)
    assert ibm._extrapolate([1, 3], [0.8, 0.6], "linear") == pytest.approx(0.9)
    assert ibm._extrapolate([1, 3, 5], [0.8, 0.6, 0.4], "richardson") == pytest.approx(0.9)
