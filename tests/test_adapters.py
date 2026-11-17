import numpy as np
import pytest


def test_pennylane_adapter():
    qml = pytest.importorskip("pennylane")
    from qang.adapters.pennylane_adapter import qnode_qg_z
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def f(a):
        qml.RY(a, wires=0)
        return qml.probs(wires=[0, 1])

    z = qnode_qg_z(f, 0.7)
    assert np.allclose(z, [np.cos(0.7), 1.0])


def test_cirq_adapter():
    cirq = pytest.importorskip("cirq")
    from qang.adapters.cirq_adapter import qg_z_from_result
    q = cirq.LineQubit.range(1)
    r = cirq.Simulator(seed=1).run(cirq.Circuit(cirq.measure(*q, key="m")), repetitions=100)
    assert qg_z_from_result(r)[0][0] == 1.0
