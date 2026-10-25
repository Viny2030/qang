"""Tests of qang.integrations (book, chapter 35). Each block skips when its library is not installed."""

import math

import numpy as np
import pytest


# --------------------------------------------------------------------------- scikit-learn
def _blobs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal([0.8, 0.8], 0.25, (n // 2, 2)), rng.normal([-0.8, -0.8], 0.25, (n // 2, 2))])
    y = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    return X, y


def test_sklearn_classifier_api():
    pytest.importorskip("sklearn")
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from qang.integrations.sklearn import WeightQNNClassifier

    X, y = _blobs()
    clf = WeightQNNClassifier(layers=2, epochs=40, gamma=0.05)
    assert clf.fit(X, y).score(X, y) > 0.9
    assert set(clf.predict(X)) <= {"a", "b"}
    scores = cross_val_score(make_pipeline(StandardScaler(), WeightQNNClassifier(layers=2, epochs=40)), X, y, cv=3)
    assert scores.mean() > 0.85
    assert WeightQNNClassifier(layers=2).get_params()["layers"] == 2


def test_sklearn_multiclass():
    pytest.importorskip("sklearn")
    from qang.integrations.sklearn import MultiClassQNNClassifier

    rng = np.random.default_rng(1)
    centres = np.array([[1.0, 0.0], [-0.5, 0.8], [-0.5, -0.8]])
    X = np.vstack([rng.normal(c, 0.15, (20, 2)) for c in centres])
    y = np.repeat([0, 1, 2], 20)
    clf = MultiClassQNNClassifier(layers=3, epochs=120).fit(X, y)
    assert clf.score(X, y) > 0.8
    assert clf.decision_function(X).shape == (60, 3)


# --------------------------------------------------------------------------- JAX
def test_jax_filtered_readout_jit_grad():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from qang.integrations import jax as qj

    p = jnp.array([0.05, 0.3, 0.25, 0.1, 0.2, 0.05, 0.05, 0.0])   # 3 qubits
    q, kept = jax.jit(qj.filtered_qg_z, static_argnums=(1, 2))(p, 3, 1)
    assert math.isclose(float(kept), 0.3 + 0.25 + 0.2, rel_tol=1e-6)
    from qang.integrations._readings import filtered_qg_z_from_probs
    ref, _ = filtered_qg_z_from_probs(np.asarray(p), 3, 1)
    assert np.allclose(np.asarray(q), ref, atol=1e-6)
    g = jax.grad(lambda t: qj.milliqang(qj.qg_from_p0(jnp.cos(t / 2) ** 2)))(0.7)
    assert math.isclose(float(g), -1000 * math.sin(0.7), rel_tol=1e-5)


def test_jax_through_pennylane_filtered_loss():
    jax = pytest.importorskip("jax")
    qml = pytest.importorskip("pennylane")
    jnp = jax.numpy
    jax.config.update("jax_enable_x64", True)
    from qang.integrations import jax as qj

    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev, interface="jax")
    def probs(t):
        qml.PauliX(0)
        qml.SingleExcitation(t[0], wires=[0, 1])
        qml.SingleExcitation(t[1], wires=[1, 2])
        return qml.probs(wires=range(3))

    def loss(t):
        q, _ = qj.filtered_qg_z(probs(t), 3, 1)
        return jnp.sum((q - jnp.array([0.2, -0.5, 0.3])) ** 2)

    t = jnp.array([0.4, -0.3])
    g = jax.jit(jax.grad(loss))(t)
    eps = 1e-6
    fd = [(loss(t + eps * e) - loss(t - eps * e)) / (2 * eps) for e in jnp.eye(2)]
    assert np.allclose(np.asarray(g), np.asarray(fd), atol=1e-6)


# --------------------------------------------------------------------------- QuTiP
def test_qutip_t1_filter_is_exact():
    qt = pytest.importorskip("qutip")
    from qang.integrations import qutip as qq

    w = (qt.tensor(qt.basis(2, 1), qt.basis(2, 0), qt.basis(2, 0)) + qt.tensor(qt.basis(2, 0), qt.basis(2, 1), qt.basis(2, 0))
         + qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 1))).unit()
    h = 0 * qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(2))
    res = qt.mesolve(h, w * w.dag(), [0.0, 30.0], c_ops=qq.t1_collapse(3, 50.0))
    times, qz = qq.mesolve_qg(res)
    raw_end = qz[-1]
    filt, kept = qq.filtered_qg_z(res.states[-1], 1)
    assert np.allclose(filt, 1 / 3.0, atol=1e-6)                     # noiseless W reading: 1 - 2/3
    assert math.isclose(kept, math.exp(-30 / 50), rel_tol=1e-5)       # (1 - gamma)^weight
    assert not np.allclose(raw_end, 1 / 3.0, atol=1e-2)
    r = qq.qg_readings(qt.basis(2, 0))
    assert np.allclose(r, [[0, 0, 1]])


# --------------------------------------------------------------------------- Stim
def test_stim_readings():
    stim = pytest.importorskip("stim")
    from qang.integrations import stim as qs

    c = stim.Circuit.generated("repetition_code:memory", rounds=3, distance=3, before_round_data_depolarization=0.02)
    q, e = qs.detector_qg(c, 20000, seed=1)
    assert np.all(q > 0.8) and np.all(q <= 1.0) and q.min() < 0.97     # quiet detectors sit at +1000 m-qg
    raw, _ = qs.observable_qg(c, 20000, seed=1)
    dec, _ = qs.decoded_logical_qg(c, 20000, seed=1)
    assert dec[0] > raw[0]                                            # decoding moves the logical reading to the pole
    flip = stim.Circuit("X_ERROR(0.1) 0 1\nM 0 1")
    qz, _, kept = qs.measurement_qg(flip, 50000, weight=0, seed=2)
    assert np.allclose(qz, 1.0) and abs(kept - 0.81) < 0.01


# --------------------------------------------------------------------------- SDKs
def test_qibo_rqang_and_filter():
    pytest.importorskip("qibo")
    import qibo
    from qang.integrations import qibo as qq

    qibo.set_backend("numpy")
    c = qibo.Circuit(2)
    qq.rqang(c, 0.6, 0)
    c.add(qibo.gates.M(0, 1))
    r = qq.qg_readout(c(nshots=20000), weight=1)
    assert abs(r["qg_z"][0] - 0.6) < 0.03 and abs(r["qg_z"][1] - 1.0) < 1e-12
    assert np.allclose(r["filtered_qg_z"], [-1.0, 1.0])


def test_pytket_half_turns():
    pytest.importorskip("pytket")
    from pytket import Circuit
    from qang.integrations import pytket as qp

    c = Circuit(2)
    qp.rqang(c, -0.25, 1)
    r = qp.statevector_qg(c, weight=1)
    assert np.allclose(r["qg_z"], [1.0, -0.25], atol=1e-12)
    assert math.isclose(r["kept_fraction"], 0.625, rel_tol=1e-9)


def test_braket_local_run():
    pytest.importorskip("braket.circuits")
    from braket.circuits import Circuit
    from qang.integrations import braket as qb

    c = Circuit()
    qb.rqang(c, 0.2, 0)
    c.i(1)
    r = qb.run_qg(c, 20000, weight=1)
    assert abs(r["qg_z"][0] - 0.2) < 0.03 and r["qg_z"][1] == 1.0
    assert abs(r["kept_fraction"] - 0.4) < 0.02


def test_cudaq_kernel():
    cudaq = pytest.importorskip("cudaq")
    from qang.integrations import cudaq as qc

    k = cudaq.make_kernel()
    q = k.qalloc(2)
    qc.rqang(k, q[0], -0.5)
    k.mz(q)
    r = qc.qg_readout(cudaq.sample(k, shots_count=20000), weight=1)
    assert abs(r["qg_z"][0] + 0.5) < 0.03 and r["qg_z"][1] == 1.0


# --------------------------------------------------------------------------- PyTorch
def test_torch_layer_filter_under_t1():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pennylane")
    from qang.integrations.torch import WeightQNNTorch

    torch.manual_seed(0)
    x = torch.tensor([[0.3, -0.7], [1.1, 0.2]], dtype=torch.float64)
    clean = WeightQNNTorch(2, layers=2, qang=True, t1=0.0)
    noisy = WeightQNNTorch(2, layers=2, qang=True, t1=0.1)
    raw = WeightQNNTorch(2, layers=2, qang=False, t1=0.1)
    noisy.load_state_dict(clean.state_dict())
    raw.load_state_dict(clean.state_dict())
    assert torch.allclose(noisy.readout(x), clean.readout(x), atol=1e-10)    # chapter 21, F1
    assert not torch.allclose(raw.readout(x), clean.readout(x), atol=1e-3)
    loss = clean(x).pow(2).sum()
    loss.backward()
    assert clean.qlayer.weights.grad is not None


# --------------------------------------------------------------------------- Mitiq
def _cirq_weight1(n=3):
    cirq = pytest.importorskip("cirq")
    q = cirq.LineQubit.range(n)
    c = cirq.Circuit(cirq.X(q[0]))
    for a, b, t in ((0, 1, 0.7), (1, 2, -0.4), (0, 1, 0.3)):
        c.append(cirq.givens(t).on(q[a], q[b]))
    return c


def _dm_sampler(t1=0.0, dephase=0.0):
    import cirq

    def run(circuit):
        noisy = circuit
        if t1:
            noisy = noisy.with_noise(cirq.amplitude_damp(t1))
        if dephase:
            noisy = noisy.with_noise(cirq.phase_damp(dephase))
        rho = cirq.DensityMatrixSimulator().simulate(noisy).final_density_matrix
        return np.real(np.diag(rho)).astype(float)
    return run


def test_mitiq_filter_and_zne():
    pytest.importorskip("mitiq")
    from qang.integrations import mitiq as qm

    c = _cirq_weight1()
    exact = qm.qg_executor(_dm_sampler())(c)
    filt_t1 = qm.qg_executor(_dm_sampler(t1=0.05), weight=1, qubits=[1])(c)
    raw_t1 = qm.qg_executor(_dm_sampler(t1=0.05), qubits=[1])(c)
    exact_q1 = qm.qg_executor(_dm_sampler(), qubits=[1])(c)
    assert abs(filt_t1 - exact_q1) < 1e-6                               # chapter 21, F1
    assert abs(raw_t1 - exact_q1) > 0.02
    # T1 + dephasing: the filter leaves dephasing, ZNE of the filtered reading removes most of it (F6)
    s = _dm_sampler(t1=0.05, dephase=0.03)
    f1 = qm.qg_executor(s, weight=1, qubits=[1])(c)
    fz = qm.execute_with_filter_zne(c, s, weight=1, qubits=[1], scale_factors=(1, 3, 5))
    assert abs(fz - exact_q1) < abs(f1 - exact_q1)
    assert np.isfinite(exact)


def test_mitiq_measurement_result_and_rule():
    mitiq = pytest.importorskip("mitiq")
    from qang.integrations import mitiq as qm

    r = mitiq.MeasurementResult([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]])
    assert np.allclose(qm.qg_z(r), [1 - 2 * 3 / 5, 1 - 2 * 2 / 5])
    q, kept = qm.filtered_qg_z(r, 1)
    assert math.isclose(kept, 0.6) and np.allclose(q, [1 - 2 * 2 / 3, 1 - 2 / 3])
    rule = qm.filter_rule(z_filtered=0.5, z_raw=0.3, kept=0.6, shots=1000)
    assert rule["use_filter"] and math.isclose(rule["mse_filtered"], 0.75 / 600)


# --------------------------------------------------------------------------- qiskit-experiments
def _readout_noise(e01, e10):
    from qiskit_aer.noise import NoiseModel, ReadoutError

    nm = NoiseModel()
    nm.add_readout_error(ReadoutError([[1 - e01, e01], [e10, 1 - e10]]), [0])
    return nm


def test_qiskit_experiments_heralded_readout():
    pytest.importorskip("qiskit_experiments")
    pytest.importorskip("qiskit_aer")
    from qiskit_aer import AerSimulator
    from qang.integrations.qiskit_experiments import HeraldedReadout

    backend = AerSimulator(noise_model=_readout_noise(0.015, 0.04), seed_simulator=7)
    exp = HeraldedReadout([0], backend=backend, frequency=5e9)
    exp.set_run_options(shots=200000)
    res = exp.run().block_for_results()
    df = res.analysis_results(dataframe=True)
    val = dict(zip(df["name"], df["value"]))
    assert abs(val["e01"] - 0.015) < 0.003 and abs(val["e10"] - 0.04) < 0.003
    assert val["p_excited"] < 0.004                                   # the simulator starts cold


def test_heralded_closed_form_separates_temperature():
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_experiments")
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qang.integrations.qiskit_experiments import heralded_closed_form, standard_readout

    p, e01, e10 = 0.03, 0.015, 0.04
    counts = {}
    for op in ("I", "X"):
        qc = QuantumCircuit(2, 2)                      # qubit 1 purifies a thermal qubit 0 (§25)
        qc.ry(2 * math.asin(math.sqrt(p)), 1)
        qc.cx(1, 0)
        qc.measure(0, 0)
        if op == "X":
            qc.x(0)
        qc.measure(0, 1)
        sim = AerSimulator(noise_model=_readout_noise(e01, e10), seed_simulator=3)
        counts[op] = sim.run(transpile(qc, sim), shots=400000).result().get_counts()
    est = heralded_closed_form(counts["I"], counts["X"], frequency=5e9)
    std01, _ = standard_readout(counts["I"], counts["X"])
    assert abs(est["e01"] - e01) < 0.003 and abs(est["p_excited"] - p) < 0.003
    b = 1 - e01 - e10
    assert abs(std01 - (e01 + p * b)) < 0.003                         # the standard suite: e01 + p b
    assert 60e-3 < est["T_eff"] < 80e-3                               # 69 mK at 5 GHz (§31)


# --------------------------------------------------------------------------- Qiskit Dynamics (separate env)
def test_qiskit_dynamics_area():
    pytest.importorskip("qiskit_dynamics")
    from qang.integrations.qiskit_dynamics import area_error_qg, rabi_qg
    from qang.pulse_area import rx_qg_z_systematic_amplitude

    for area in (math.pi / 2, math.pi, 2.2):
        assert abs(rabi_qg(area) - math.cos(area)) < 2e-3
        assert abs(rabi_qg(area, shape="gaussian") - math.cos(area)) < 2e-3
    assert math.isclose(area_error_qg(1.3, 0.02), rx_qg_z_systematic_amplitude(1.3, 0.02), rel_tol=1e-12)
