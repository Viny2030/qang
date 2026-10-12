"""Tests for qang.interop (§130): one qubit order for every SDK, and the same qg read-outs everywhere."""

import math

import numpy as np
import pytest

from qang import interop as IO

QG = [0.6, -0.2, 0.0]          # qg_Z prepared on qubits 0, 1, 2
SHOTS = 20000


def _assert_profile(counts, tol_sigma=4.5):
    prof = IO.qg_z_profile(counts)
    assert len(prof) == 3
    for (q, se), target in zip(prof, QG):
        assert abs(q - target) <= tol_sigma * max(se, 1e-3)


def test_dict_conventions_and_errors():
    # qubit 0 measured as 1, the others 0
    assert IO.canonical_counts({"001": 5}, sdk="qiskit") == {(1, 0, 0): 5}
    assert IO.canonical_counts({"0 01": 5}, sdk="qiskit") == {(1, 0, 0): 5}       # registers separated by spaces
    for sdk in ("cirq", "pennylane", "braket"):
        assert IO.canonical_counts({"100": 5}, sdk=sdk) == {(1, 0, 0): 5}
    assert IO.canonical_counts({(1, 0, 0): 5}) == {(1, 0, 0): 5}                  # pytket dict
    assert IO.canonical_counts({1: 5}, sdk="qiskit", n_qubits=3) == {(1, 0, 0): 5}
    with pytest.raises(ValueError):
        IO.canonical_counts({"001": 5})                                            # sdk required for a dict
    with pytest.raises(ValueError):
        IO.canonical_counts({"001": 5}, sdk="nope")


def test_filter_profile_entropy():
    c = {(1, 0, 0): 30, (0, 1, 0): 50, (1, 1, 0): 20}
    kept, frac = IO.weight_filter(c, 1)
    assert frac == pytest.approx(0.8) and sum(kept.values()) == 80
    prof = IO.filtered_qg_z_profile(c, 1)
    assert prof[0][0] == pytest.approx(1 - 2 * 30 / 80) and prof[2][0] == pytest.approx(1.0)
    assert 0 < IO.joint_qg_s(c, "none") < IO.joint_qg_s(c) <= 1.0
    with pytest.raises(ValueError):
        IO.filtered_qg_z_profile(c, 3)


def test_qiskit_counts_and_bitarray():
    qiskit = pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    from qiskit import transpile
    from qiskit.primitives import StatevectorSampler
    from qiskit_aer import AerSimulator

    from qang.qiskit_gate import RQangGate

    qc = qiskit.QuantumCircuit(3)
    for i, q in enumerate(QG):
        qc.append(RQangGate(q), [i])
    qc.measure_all()
    sim = AerSimulator(seed_simulator=7)
    counts = sim.run(transpile(qc, sim), shots=SHOTS).result().get_counts()
    _assert_profile(IO.canonical_counts(counts, sdk="qiskit"))
    bits = StatevectorSampler(seed=7).run([qc], shots=SHOTS).result()[0].data.meas
    _assert_profile(IO.canonical_counts(bits))


def test_qiskit_parameterized_gate():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Pauli, Statevector

    from qang.qiskit_gate import append_rqang_parameter

    p = Parameter("qg")
    qc = QuantumCircuit(1)
    append_rqang_parameter(qc, p, 0)
    for v in (-0.7, 0.0, 0.45):
        z = Statevector(qc.assign_parameters({p: v})).expectation_value(Pauli("Z")).real
        assert z == pytest.approx(v, abs=1e-9)


def test_cirq_result_and_symbol():
    cirq = pytest.importorskip("cirq")
    sympy = pytest.importorskip("sympy")
    from qang.cirq_gate import rqang_gate, rqang_symbol

    qs = cirq.LineQubit.range(3)
    circ = cirq.Circuit([rqang_gate(q).on(qs[i]) for i, q in enumerate(QG)], cirq.measure(*qs, key="m"))
    res = cirq.Simulator(seed=7).run(circ, repetitions=SHOTS)
    _assert_profile(IO.canonical_counts(res))
    s = sympy.Symbol("q")
    c1 = cirq.Circuit(rqang_symbol(s).on(qs[0]))
    v = cirq.Simulator().simulate(c1, param_resolver={"q": -0.35}).final_state_vector
    assert abs(v[0]) ** 2 - abs(v[1]) ** 2 == pytest.approx(-0.35, abs=1e-6)


def test_pennylane_counts():
    qml = pytest.importorskip("pennylane")
    from qang.pennylane_gate import rqang

    has_set_shots = hasattr(qml, "set_shots")
    dev = qml.device("default.qubit", wires=3, seed=7) if has_set_shots else \
        qml.device("default.qubit", wires=3, shots=SHOTS, seed=7)

    @qml.qnode(dev)
    def circuit():
        for i, q in enumerate(QG):
            rqang(q, wires=i)
        return qml.counts()

    if has_set_shots:
        circuit = qml.set_shots(circuit, shots=SHOTS)
    _assert_profile(IO.canonical_counts(circuit(), sdk="pennylane"))


def test_duck_typed_braket_and_pytket_results():
    class BraketLike:
        measurement_counts = {"100": 7, "000": 3}

    class TketLike:
        @staticmethod
        def get_counts():
            return {(1, 0, 0): 7, (0, 0, 0): 3}

    assert IO.canonical_counts(BraketLike()) == {(1, 0, 0): 7, (0, 0, 0): 3}
    assert IO.canonical_counts(TketLike()) == {(1, 0, 0): 7, (0, 0, 0): 3}
    assert math.isclose(IO.qg_z_profile(IO.canonical_counts(BraketLike()))[0][0], 1 - 2 * 0.7)


def test_example_w_state_agrees_across_installed_sdks():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    import interop_sdks_qg as E

    rows = E.main()
    for c in rows.values():
        assert IO.weight_filter(c, 1)[1] == 1.0
        assert IO.joint_qg_s(c) == pytest.approx(math.log2(3) / 3, abs=0.01)
        for q, se in IO.qg_z_profile(c):
            assert abs(q - 1 / 3) <= 5 * se


def test_joint_qg_s_se_matches_resampling():
    rng = np.random.default_rng(134)
    probs = np.array([0.5, 0.25, 0.15, 0.06, 0.04])
    keys = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
    shots = 2000
    draws = []
    for _ in range(2000):
        c = rng.multinomial(shots, probs)
        draws.append(IO.joint_qg_s({k: int(v) for k, v in zip(keys, c) if v}))
    c0 = {k: int(v) for k, v in zip(keys, rng.multinomial(shots, probs)) if v}
    assert IO.joint_qg_s_se(c0) == pytest.approx(np.std(draws), rel=0.12)
    assert IO.joint_qg_s_se({(0, 1): 50, (1, 0): 50}) == 0.0          # uniform: no first-order error
