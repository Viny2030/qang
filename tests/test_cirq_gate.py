import math

import pytest

cirq = pytest.importorskip("cirq")

from quang.core import Qang
from quang.cirq_gate import append_qang, full_rqang_gate, rqang_gate

PI = math.pi
SHOTS = 20000


def _measured_probabilities(circuit: "cirq.Circuit", qubit: "cirq.Qid"):
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=SHOTS)
    counts = result.histogram(key="m")
    p0 = counts.get(0, 0) / SHOTS
    p1 = counts.get(1, 0) / SHOTS
    return p0, p1


@pytest.mark.parametrize("qg_z_value", [0.5, -0.3, 0.0, 0.9])
def test_rqang_gate_prepares_expected_probabilities(qg_z_value):
    qg = Qang(qg_z_value, mode="polar")
    expected_p0, expected_p1 = qg.to_probabilities()

    q = cirq.LineQubit(0)
    circuit = cirq.Circuit([rqang_gate(qg).on(q), cirq.measure(q, key="m")])

    p0, p1 = _measured_probabilities(circuit, q)
    assert p0 == pytest.approx(expected_p0, abs=0.02)
    assert p1 == pytest.approx(expected_p1, abs=0.02)


def test_full_rqang_gate_prepares_plus_i_state():
    # theta=90deg, phi=90deg -> |+i> = (|0> + i|1>)/sqrt(2): 50/50 in Z basis
    qg = Qang.from_angles(PI / 2, phi=PI / 2, mode="polar")
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit([full_rqang_gate(qg).on(q), cirq.measure(q, key="m")])

    p0, p1 = _measured_probabilities(circuit, q)
    assert p0 == pytest.approx(0.5, abs=0.02)
    assert p1 == pytest.approx(0.5, abs=0.02)


def test_full_rqang_gate_matches_statevector():
    qg = Qang.from_angles(PI / 3, phi=0.4, mode="polar")
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit([full_rqang_gate(qg).on(q)])

    sim = cirq.Simulator()
    result = sim.simulate(circuit)
    got = result.final_state_vector
    expected = qg.to_statevector()

    # global phase can differ; compare via fidelity |<expected|got>|^2 ~= 1
    overlap = abs(sum(complex(e).conjugate() * complex(g) for e, g in zip(expected, got))) ** 2
    assert overlap == pytest.approx(1.0, abs=1e-6)


def test_append_qang_convenience_matches_full_gate():
    qg = Qang.from_angles(PI / 3, phi=0.4, mode="polar")
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit()
    append_qang(circuit, qg, q, full=True)
    circuit.append(cirq.measure(q, key="m"))

    expected_p0, expected_p1 = qg.to_probabilities()
    p0, p1 = _measured_probabilities(circuit, q)
    assert p0 == pytest.approx(expected_p0, abs=0.02)
    assert p1 == pytest.approx(expected_p1, abs=0.02)


def test_gate_rejects_entropic_mode_qang():
    qg = Qang(0.7, mode="entropic")
    with pytest.raises(ValueError):
        rqang_gate(qg)
    with pytest.raises(ValueError):
        full_rqang_gate(qg)
