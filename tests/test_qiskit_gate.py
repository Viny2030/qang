import math

import pytest

qiskit = pytest.importorskip("qiskit")

from quang.core import Qang
from quang.qiskit_gate import FullRQangGate, RQangGate, append_qang

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

PI = math.pi
SHOTS = 20000


def _measured_probabilities(qc: QuantumCircuit):
    sim = AerSimulator()
    qc_t = transpile(qc, sim)
    counts = sim.run(qc_t, shots=SHOTS).result().get_counts()
    p0 = counts.get("0", 0) / SHOTS
    p1 = counts.get("1", 0) / SHOTS
    return p0, p1


@pytest.mark.parametrize("qg_z_value", [0.5, -0.3, 0.0, 0.9])
def test_rqang_gate_prepares_expected_probabilities(qg_z_value):
    qg = Qang(qg_z_value, mode="polar")
    expected_p0, expected_p1 = qg.to_probabilities()

    qc = QuantumCircuit(1, 1)
    qc.append(RQangGate(qg), [0])
    qc.measure(0, 0)

    p0, p1 = _measured_probabilities(qc)
    assert p0 == pytest.approx(expected_p0, abs=0.02)
    assert p1 == pytest.approx(expected_p1, abs=0.02)


def test_full_rqang_gate_prepares_plus_i_state():
    # theta=90deg, phi=90deg -> |+i> = (|0> + i|1>)/sqrt(2): 50/50 in Z basis
    qg = Qang.from_angles(PI / 2, phi=PI / 2, mode="polar")
    qc = QuantumCircuit(1, 1)
    qc.append(FullRQangGate(qg), [0])
    qc.measure(0, 0)

    p0, p1 = _measured_probabilities(qc)
    assert p0 == pytest.approx(0.5, abs=0.02)
    assert p1 == pytest.approx(0.5, abs=0.02)


def test_append_qang_convenience_matches_full_gate():
    qg = Qang.from_angles(PI / 3, phi=0.4, mode="polar")
    qc = QuantumCircuit(1, 1)
    append_qang(qc, qg, 0, full=True)
    qc.measure(0, 0)

    expected_p0, expected_p1 = qg.to_probabilities()
    p0, p1 = _measured_probabilities(qc)
    assert p0 == pytest.approx(expected_p0, abs=0.02)
    assert p1 == pytest.approx(expected_p1, abs=0.02)


def test_gate_rejects_entropic_mode_qang():
    qg = Qang(0.7, mode="entropic")
    with pytest.raises(ValueError):
        RQangGate(qg)
    with pytest.raises(ValueError):
        FullRQangGate(qg)
