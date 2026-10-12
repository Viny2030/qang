"""Tests for §131: qg read-out of Qiskit SamplerV2 / EstimatorV2 and the differentiable filtered qg_Z in PennyLane."""

import math

import numpy as np
import pytest

from qang import interop as IO

N = 3


def _exact_filtered(qg, weight):
    """Filtered qg_Z for a product state with per-qubit qg_Z = qg (qubit 0 first), by enumeration."""
    p1 = [(1 - q) / 2 for q in qg]
    num, den = np.zeros(len(qg)), 0.0
    for k in range(2 ** len(qg)):
        b = [(k >> (len(qg) - 1 - i)) & 1 for i in range(len(qg))]
        if sum(b) != weight:
            continue
        pr = np.prod([p1[i] if b[i] else 1 - p1[i] for i in range(len(qg))])
        den += pr
        num += pr * (1 - 2 * np.array(b))
    return den, num / den


def test_filtered_observables_are_projectors():
    for w in range(N + 1):
        obs = IO.filtered_observables(N, w)
        diag = np.zeros(2 ** N)
        for lab, c in obs[0].items():
            z = np.ones(2 ** N)
            for k, ch in enumerate(lab):
                if ch == "Z":
                    z *= 1 - 2 * ((np.arange(2 ** N) >> (N - 1 - k)) & 1)
            diag += c * z
        assert np.allclose(diag, [bin(i).count("1") == w for i in range(2 ** N)])
    assert IO.qg_z_observables(3) == ["IIZ", "IZI", "ZII"]
    with pytest.raises(ValueError):
        IO.filtered_observables(3, 4)


def _sweep_circuit():
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    from qang.qiskit_gate import append_rqang_parameter

    p = ParameterVector("qg", N)
    qc = QuantumCircuit(N)
    for i in range(N):
        append_rqang_parameter(qc, p[i], i)
    return qc, p


SWEEP = np.array([[0.6, -0.2, 0.0], [-0.5, 0.3, 0.8]])


def test_sampler_qg_sweep_and_filter():
    pytest.importorskip("qiskit")
    from qiskit.primitives import StatevectorSampler

    qc, _ = _sweep_circuit()
    qc.measure_all()
    shots = 40000
    res = StatevectorSampler(seed=131).run([(qc, SWEEP)], shots=shots).result()[0]
    out = IO.sampler_qg(res)
    assert out["qg"].shape == (2, N) and np.all(out["kept"] == 1)
    assert np.all(np.abs(out["qg"] - SWEEP) <= 4.5 * out["se"] + 1e-3)
    f = IO.sampler_qg(res, weight=1)
    for k in range(2):
        kept, q = _exact_filtered(SWEEP[k], 1)
        assert abs(f["kept"][k] - kept) < 4.5 * math.sqrt(kept * (1 - kept) / shots)
        assert np.all(np.abs(f["qg"][k] - q) <= 4.5 * f["se"][k] + 1e-3)
    # single-parameter-set pub (scalar shape) and creg selection
    qc2, p = _sweep_circuit()
    qc2.measure_all()
    r1 = StatevectorSampler(seed=1).run([(qc2, SWEEP[0])], shots=1000).result()[0]
    assert IO.sampler_qg(r1, creg="meas")["qg"].shape == (N,)


def test_estimator_qg_and_filtered_exact():
    pytest.importorskip("qiskit")
    from qiskit.primitives import StatevectorEstimator

    qc, _ = _sweep_circuit()
    est = StatevectorEstimator()
    vals = SWEEP.reshape(2, 1, N)                                  # parameter shape (2, 1)
    obs_z = IO.qg_z_observables(N)                                 # observables shape (N,) -> evs (2, N)
    evs = est.run([(qc, obs_z, vals)]).result()[0].data.evs
    assert np.allclose(evs, SWEEP, atol=1e-9)
    evs_f = est.run([(qc, IO.filtered_observables(N, 1), vals)]).result()[0].data.evs
    kept, q = IO.estimator_filtered_qg(evs_f)
    for k in range(2):
        k0, q0 = _exact_filtered(SWEEP[k], 1)
        assert kept[k] == pytest.approx(k0, abs=1e-9)
        assert np.allclose(q[k], q0, atol=1e-9)


def test_pennylane_filtered_qg_values_and_gradient():
    qml = pytest.importorskip("pennylane")
    from pennylane import numpy as pnp

    from qang.pennylane_gate import filtered_qg_z, kept_fraction, qg_z_from_probs, rqang

    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev)
    def probs(qg):
        for i in range(N):
            rqang(qg[i], wires=i)
        return qml.probs(wires=range(N))

    x = pnp.array(SWEEP[1], requires_grad=True)
    p = probs(x)
    assert np.allclose(qg_z_from_probs(p), SWEEP[1], atol=1e-9)
    k0, q0 = _exact_filtered(SWEEP[1], 1)
    assert float(kept_fraction(p, 1)) == pytest.approx(k0, abs=1e-9)
    assert np.allclose(filtered_qg_z(p, 1), q0, atol=1e-9)

    def loss(v):
        return filtered_qg_z(probs(v), 1)[0]

    g = qml.grad(loss)(x)
    eps = 1e-6
    fd = [(_exact_filtered(SWEEP[1] + eps * np.eye(N)[j], 1)[1][0]
           - _exact_filtered(SWEEP[1] - eps * np.eye(N)[j], 1)[1][0]) / (2 * eps) for j in range(N)]
    assert np.allclose(g, fd, atol=1e-6)


def test_pennylane_filtered_matches_interop_on_counts():
    qml = pytest.importorskip("pennylane")
    from qang.pennylane_gate import filtered_qg_z, rqang

    shots = 30000
    dev = qml.device("default.qubit", wires=N, seed=131)

    @qml.qnode(dev)
    def circ():
        for i in range(N):
            rqang(float(SWEEP[0][i]), wires=i)
        return qml.counts()

    if hasattr(qml, "set_shots"):
        counts = qml.set_shots(circ, shots=shots)()
    else:  # pragma: no cover
        counts = qml.QNode(circ.func, qml.device("default.qubit", wires=N, shots=shots, seed=131))()
    c = IO.canonical_counts(counts, sdk="pennylane")
    p = np.zeros(2 ** N)
    for b, n in c.items():
        p[int("".join(map(str, b)), 2)] += n / shots
    via_pl = np.asarray(filtered_qg_z(p, 1))
    via_io = np.array([q for q, _ in IO.filtered_qg_z_profile(c, 1)])
    assert np.allclose(via_pl, via_io, atol=1e-12)
