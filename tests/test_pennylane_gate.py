import cmath
import math

import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
from pennylane import numpy as pnp  # noqa: E402

from qang.core import Qang  # noqa: E402
from qang.pennylane_gate import append_qang, full_rqang, rqang  # noqa: E402

PI = math.pi
SHOTS = 20000


def _overlap(expected, got):
    return abs(sum(complex(e).conjugate() * complex(g) for e, g in zip(expected, got))) ** 2


@pytest.mark.parametrize("qg_z_value", [0.5, -0.3, 0.0, 0.9])
def test_rqang_prepares_expected_probabilities(qg_z_value):
    qg = Qang(qg_z_value, mode="polar")
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        rqang(qg, wires=0)
        return qml.probs(wires=0)

    p0, p1 = circuit()
    expected_p0, expected_p1 = qg.to_probabilities()
    assert p0 == pytest.approx(expected_p0, abs=1e-12)
    assert p1 == pytest.approx(expected_p1, abs=1e-12)


def test_rqang_with_finite_shots():
    qg = Qang(0.4, mode="polar")
    dev = qml.device("default.qubit", wires=1, seed=7)

    @qml.set_shots(SHOTS)
    @qml.qnode(dev)
    def circuit():
        rqang(qg, wires=0)
        return qml.probs(wires=0)

    p0, _ = circuit()
    assert p0 == pytest.approx(0.7, abs=0.02)


def test_full_rqang_matches_statevector():
    qg = Qang.from_angles(PI / 3, phi=0.4, mode="polar")
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        full_rqang(qg, wires=0)
        return qml.state()

    assert _overlap(qg.to_statevector(), circuit()) == pytest.approx(1.0, abs=1e-12)


def test_full_rqang_matrix_matches_qiskit_ugate_and_cirq_convention():
    # U(theta, phi, 0) = [[c, -s], [e^{i phi} s, e^{i phi} c]]: the matrix
    # qang.cirq_gate builds explicitly and Qiskit's UGate uses.
    qg = Qang.from_angles(1.1, phi=-0.7, mode="polar")
    theta, phi = qg.to_angles(unit="rad")
    c, s, e = math.cos(theta / 2), math.sin(theta / 2), cmath.exp(1j * phi)
    expected = np.array([[c, -s], [e * s, e * c]], dtype=complex)

    got = qml.matrix(full_rqang, wire_order=[0])(qg, wires=0)
    assert np.allclose(got, expected, atol=1e-12)


def test_append_qang_selects_gate():
    qg = Qang.from_angles(PI / 3, phi=0.4, mode="polar")
    with qml.tape.QuantumTape() as tape:
        append_qang(qg, wires=0, full=True)
        append_qang(qg, wires=1, full=False)
    assert [op.name for op in tape.operations] == ["U3", "RY"]


@pytest.mark.parametrize("qg_z_value", [-0.9, -0.2, 0.0, 0.35, 0.8])
def test_gradient_flows_to_qg_z_exactly(qg_z_value):
    # <Z> = qg_Z by definition, so d<Z>/d(qg_Z) = 1 everywhere in the
    # interior, even though d theta/d(qg_Z) = -1/sin(theta) is not.
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, diff_method="backprop")
    def z_expval(qg):
        rqang(qg, wires=0)
        return qml.expval(qml.PauliZ(0))

    qg = pnp.array(qg_z_value, requires_grad=True)
    assert z_expval(qg) == pytest.approx(qg_z_value, abs=1e-12)
    assert qml.grad(z_expval)(qg) == pytest.approx(1.0, abs=1e-9)


def test_gradient_through_phi_and_qg_z_for_x_expval():
    # <X> = sin(theta) cos(phi) = sqrt(1 - qg^2) cos(phi)
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, diff_method="parameter-shift")
    def x_expval(qg, phi):
        full_rqang(qg, wires=0, phi=phi)
        return qml.expval(qml.PauliX(0))

    qg0, phi0 = 0.3, 0.5
    qg = pnp.array(qg0, requires_grad=True)
    phi = pnp.array(phi0, requires_grad=True)
    d_qg, d_phi = qml.grad(x_expval)(qg, phi)

    root = math.sqrt(1 - qg0**2)
    assert d_qg == pytest.approx(-qg0 / root * math.cos(phi0), abs=1e-9)
    assert d_phi == pytest.approx(-root * math.sin(phi0), abs=1e-9)


def test_optimize_directly_in_qg_space():
    # Drive <Z> to a target by gradient descent on qg_Z itself.
    dev = qml.device("default.qubit", wires=1)
    target = -0.4

    @qml.qnode(dev)
    def z_expval(qg):
        rqang(qg, wires=0)
        return qml.expval(qml.PauliZ(0))

    def cost(qg):
        return (z_expval(qg) - target) ** 2

    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    qg = pnp.array(0.6, requires_grad=True)
    for _ in range(40):
        qg = opt.step(cost, qg)
    assert float(qg) == pytest.approx(target, abs=1e-6)


def test_rejects_entropic_mode_and_out_of_range():
    with pytest.raises(ValueError):
        rqang(Qang(0.7, mode="entropic"), wires=0)
    with pytest.raises(ValueError):
        full_rqang(Qang(0.7, mode="entropic"), wires=0)
    with pytest.raises(ValueError):
        rqang(1.5, wires=0)
