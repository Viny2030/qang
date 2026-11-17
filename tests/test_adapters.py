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


def test_tensorcircuit_adapter_duck_typed():
    """Sin tensorcircuit: objeto falso con expectation_ps; R_Y(theta)|0> -> cos(theta)."""
    from qang.adapters.tensorcircuit_adapter import qg_z_from_circuit

    class Fake:
        _nqubits = 2

        def expectation_ps(self, z):
            return np.cos(0.7) if z == [0] else 1.0

    assert np.allclose(qg_z_from_circuit(Fake()), [np.cos(0.7), 1.0])
    assert np.allclose(qg_z_from_circuit(Fake(), qubits=[1]), [1.0])


def test_tensorcircuit_adapter_real():
    tc = pytest.importorskip("tensorcircuit")
    from qang.adapters.tensorcircuit_adapter import qg_z_from_circuit
    c = tc.Circuit(2)
    c.ry(0, theta=0.7)
    assert np.allclose(qg_z_from_circuit(c), [np.cos(0.7), 1.0], atol=1e-6)
