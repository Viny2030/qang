"""Tests for qang.qsp (§134): QSP / weighted QSP in qg terms."""

import numpy as np
import pytest

from qang import qsp


def test_signal_and_qg():
    q = np.array([-1.0, -0.3, 0.0, 0.5, 1.0])
    x = qsp.signal_from_qg(q)
    assert np.allclose(qsp.qg_of_signal(x), q)
    U = qsp.wqsp_unitary(x, [0.0, 0.0])                     # U(x) alone
    p0 = np.abs(U[..., 0, 0]) ** 2
    assert np.allclose(2 * p0 - 1, q)                        # U(x)|0> has qg_Z = 2x^2 - 1
    with pytest.raises(ValueError):
        qsp.signal_from_qg(1.2)


def test_standard_qsp_properties():
    rng = np.random.default_rng(0)
    phis = rng.uniform(-np.pi, np.pi, 6)                     # degree 5
    x = np.linspace(-1, 1, 101)
    U = qsp.wqsp_unitary(x, phis)
    assert np.allclose(U @ np.conj(np.swapaxes(U, -1, -2)), np.eye(2))
    P = U[..., 0, 0]
    assert np.all(np.abs(P) <= 1 + 1e-12)
    # degree <= 5: a degree-5 fit in x is exact
    for part in (P.real, P.imag):
        c = np.polynomial.chebyshev.chebfit(x, part, 5)
        assert np.allclose(np.polynomial.chebyshev.chebval(x, c), part, atol=1e-10)


def test_integer_weights_equal_pruned_qsp():
    rng = np.random.default_rng(1)
    phis = rng.uniform(-np.pi, np.pi, 3)
    x = np.linspace(-1, 1, 51)
    w = qsp.wqsp_poly(x, phis, [2, 3])
    full = qsp.wqsp_poly(x, [phis[0], 0.0, phis[1], 0.0, 0.0, phis[2]])   # zero phases merge rotations
    assert np.allclose(w, full)
    a = np.arccos(x)
    assert np.allclose(qsp.wqsp_poly(x, [0.0, 0.0], [4]), np.cos(4 * a))  # Chebyshev T_4


def test_hadamard_test_is_qg():
    rng = np.random.default_rng(2)
    phis = rng.uniform(-np.pi, np.pi, 4)
    x = np.linspace(-1, 1, 21)
    p0 = qsp.hadamard_test_p0(x, phis)
    assert np.allclose(2 * p0 - 1, qsp.wqsp_poly(x, phis).real)
    k = qsp.sample_hadamard_test(x, phis, shots=200000, rng=3)
    assert np.allclose(k / 200000, p0, atol=0.006)


def test_fit_recovers_function_without_noise():
    pytest.importorskip("scipy")
    phis = np.array([0.3, -0.7, 1.1, 0.4])
    x = np.linspace(-1, 1, 41)
    shots = 10 ** 9
    k0 = np.round(qsp.hadamard_test_p0(x, phis) * shots)
    for scale in ("linear", "arcsine"):
        ph, _ = qsp.fit_phases(x, k0, shots, 4, scale=scale, starts=12, seed=4)
        assert np.allclose(qsp.wqsp_poly(x, ph).real, qsp.wqsp_poly(x, phis).real, atol=1e-4)


def test_pennylane_ops_match_numpy_and_are_differentiable():
    qml = pytest.importorskip("pennylane")
    from pennylane import numpy as pnp

    phis = np.array([0.2, -0.5, 0.9])
    w = [2, 1]
    for x in (-0.6, 0.1, 0.8):
        M = qml.matrix(lambda: qsp.wqsp_ops(x, phis, w, wires=0), wire_order=[0])()
        assert np.allclose(M, qsp.wqsp_unitary(x, phis, w))
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def c(ph):
        qsp.wqsp_ops(0.4, ph, w, wires=0)
        return qml.expval(qml.PauliZ(0))

    g = qml.grad(c)(pnp.array(phis, requires_grad=True))
    assert np.all(np.isfinite(g)) and np.any(np.abs(g) > 1e-6)


def test_example_runs_small():
    pytest.importorskip("scipy")
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    import qsp_arcsine_fit_qg as E

    e = E.errors(100, reps=2, starts=2)
    assert set(e) == {"linear", "arcsine"} and all(len(v) == 2 for v in e.values())


def test_wqsp_kan_learns_a_separable_toy():
    pytest.importorskip("scipy")
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    import wqsp_kan_qg as K

    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (80, 2))
    y = (X[:, 0] ** 2 + 0.5 * X[:, 1] > 0.4).astype(int)
    m = K.WQSPKAN(2, (1, 2)).fit(X, y, starts=3, seed=1)
    assert np.mean(m.predict(X) == y) >= 0.9
