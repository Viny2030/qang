"""Numbers of chapter 35 of the book (qang as a module inside other libraries).

    .venv-int/Scripts/python tools/make_cap35_numbers.py          # every library but Qiskit Dynamics
    .venv-dyn/Scripts/python tools/make_cap35_numbers.py          # Qiskit Dynamics (needs qiskit<=1.3)
    (WSL) ~/qang-cudaq/bin/python tools/make_cap35_numbers.py     # CUDA-Q

Each block runs only if its library imports, and prints what the chapter quotes. No figure.
"""

import importlib
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests")))


def has(mod):
    try:
        importlib.import_module(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


def versions():
    out = {}
    for m in ("mitiq", "qiskit", "qiskit_experiments", "qiskit_dynamics", "qutip", "stim", "pymatching", "sklearn",
              "jax", "pennylane", "torch", "qibo", "pytket", "braket", "cudaq", "cirq"):
        if has(m):
            mod = importlib.import_module(m)
            out[m] = getattr(mod, "__version__", "?")
    if has("braket._sdk"):
        out["braket"] = importlib.import_module("braket._sdk").__version__
    print("versions", out)


def mitiq_block():
    from test_integrations import _cirq_weight1, _dm_sampler
    from qang.integrations import mitiq as qm

    c = _cirq_weight1()
    exact = qm.qg_executor(_dm_sampler(), qubits=[1])(c)
    rows = {"exact": exact}
    for name, s in (("T1", _dm_sampler(t1=0.05)), ("T1+deph", _dm_sampler(t1=0.05, dephase=0.03))):
        rows[name] = {
            "raw": qm.qg_executor(s, qubits=[1])(c) - exact,
            "filtered": qm.qg_executor(s, weight=1, qubits=[1])(c) - exact,
            "raw+ZNE": qm.execute_with_filter_zne(c, s, None, qubits=[1], scale_factors=(1, 3, 5)) - exact,
            "filtered+ZNE": qm.execute_with_filter_zne(c, s, 1, qubits=[1], scale_factors=(1, 3, 5)) - exact,
        }
    print("mitiq (errors in m-qg)", {k: ({kk: round(1000 * vv, 3) for kk, vv in v.items()} if isinstance(v, dict)
                                       else round(1000 * v, 1)) for k, v in rows.items()})


def experiments_block():
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from test_integrations import _readout_noise
    from qang.integrations.qiskit_experiments import heralded_closed_form, standard_readout

    e01, e10 = 0.015, 0.04
    for p in (0.0, 0.01, 0.03, 0.08):
        counts = {}
        for op in ("I", "X"):
            qc = QuantumCircuit(2, 2)
            qc.ry(2 * math.asin(math.sqrt(p)), 1)
            qc.cx(1, 0)
            qc.measure(0, 0)
            if op == "X":
                qc.x(0)
            qc.measure(0, 1)
            sim = AerSimulator(noise_model=_readout_noise(e01, e10), seed_simulator=11)
            counts[op] = sim.run(transpile(qc, sim), shots=200000).result().get_counts()
        est = heralded_closed_form(counts["I"], counts["X"], frequency=5e9)
        s01, s10 = standard_readout(counts["I"], counts["X"])
        print("experiments p=%.2f: e01 std %.4f qg %.4f | e10 std %.4f qg %.4f | p qg %.4f | T_eff %.1f mK"
              % (p, s01, est["e01"], s10, est["e10"], est["p_excited"], 1e3 * est["T_eff"]
                 if math.isfinite(est["T_eff"]) else float("nan")))


def qutip_block():
    import qutip as qt
    from qang.integrations import qutip as qq

    b = [qt.basis(2, 0), qt.basis(2, 1)]
    w = (qt.tensor(b[1], b[0], b[0]) + qt.tensor(b[0], b[1], b[0]) + qt.tensor(b[0], b[0], b[1])).unit()
    res = qt.mesolve(0 * qt.tensor([qt.qeye(2)] * 3), w * w.dag(), [0.0, 30.0], c_ops=qq.t1_collapse(3, 50.0))
    filt, kept = qq.filtered_qg_z(res.states[-1], 1)
    _, qz = qq.mesolve_qg(res)
    print("qutip W under T1 t/T1=0.6: raw m", np.round(1000 * qz[-1], 1), "filtered m", np.round(1000 * filt, 3),
          "kept", round(kept, 4), "e^-0.6", round(math.exp(-0.6), 4))


def stim_block():
    import stim
    from qang.integrations import stim as qs

    for p in (0.01, 0.03, 0.06):
        c = stim.Circuit.generated("repetition_code:memory", rounds=5, distance=5, before_round_data_depolarization=p)
        dq, _ = qs.detector_qg(c, 200000, seed=1)
        raw, _ = qs.observable_qg(c, 200000, seed=1)
        dec, _ = qs.decoded_logical_qg(c, 200000, seed=1)
        print("stim d=5 p=%.2f: detector m-qg %.1f..%.1f | logical raw %.1f decoded %.1f m-qg"
              % (p, 1000 * dq.min(), 1000 * dq.max(), 1000 * raw[0], 1000 * dec[0]))


def sklearn_block():
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons
    from qang.integrations.sklearn import WeightQNNClassifier

    X, y = make_moons(120, noise=0.15, random_state=3)
    for q in (True, False):
        s = cross_val_score(make_pipeline(StandardScaler(), WeightQNNClassifier(layers=2, epochs=60, gamma=0.15,
                                                                               qang=q)), X, y, cv=5)
        print("sklearn 5-fold, gamma 0.15, qang=%s: %.3f +- %.3f" % (q, s.mean(), s.std()))
    g = GridSearchCV(WeightQNNClassifier(layers=2, epochs=40, gamma=0.15), {"qang": [True, False]}, cv=3).fit(X, y)
    print("sklearn GridSearchCV picks", g.best_params_, round(g.best_score_, 3))


def jax_block():
    import jax
    import pennylane as qml

    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
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
    fd = jnp.array([(loss(t + 1e-6 * e) - loss(t - 1e-6 * e)) / 2e-6 for e in jnp.eye(2)])
    print("jax grad of filtered loss vs finite differences: max diff %.2e" % float(jnp.abs(g - fd).max()))


def torch_block():
    import torch
    from qang.integrations.torch import WeightQNNTorch

    torch.manual_seed(0)
    x = torch.tensor([[0.3, -0.7], [1.1, 0.2]], dtype=torch.float64)
    clean = WeightQNNTorch(2, layers=2, qang=True)
    noisy = WeightQNNTorch(2, layers=2, qang=True, t1=0.1)
    raw = WeightQNNTorch(2, layers=2, qang=False, t1=0.1)
    noisy.load_state_dict(clean.state_dict())
    raw.load_state_dict(clean.state_dict())
    print("torch T1=0.1/layer: filtered - clean %.1e, raw - clean %.1f m-qg"
          % (float((noisy.readout(x) - clean.readout(x)).abs().max()),
             1000 * float((raw.readout(x) - clean.readout(x)).abs().max())))


def sdk_block():
    out = {}
    if has("qibo"):
        import qibo
        from qang.integrations import qibo as qq
        qibo.set_backend("numpy")
        c = qibo.Circuit(2)
        qq.rqang(c, 0.6, 0)
        c.add(qibo.gates.M(0, 1))
        out["qibo qg_z[0] (0.6)"] = round(qq.qg_readout(c(nshots=100000))["qg_z"][0], 4)
    if has("pytket"):
        from pytket import Circuit
        from qang.integrations import pytket as qp
        c = Circuit(2)
        qp.rqang(c, -0.25, 1)
        out["pytket qg_z[1] (-0.25)"] = float(qp.statevector_qg(c)["qg_z"][1])
    if has("braket.circuits"):
        from braket.circuits import Circuit
        from qang.integrations import braket as qb
        c = Circuit()
        qb.rqang(c, 0.2, 0)
        c.i(1)
        out["braket qg_z[0] (0.2)"] = round(qb.run_qg(c, 100000)["qg_z"][0], 4)
    if has("cudaq"):
        import cudaq
        from qang.integrations import cudaq as qc
        k = cudaq.make_kernel()
        q = k.qalloc(2)
        qc.rqang(k, q[0], -0.5)
        k.mz(q)
        out["cudaq qg_z[0] (-0.5)"] = round(qc.qg_readout(cudaq.sample(k, shots_count=100000))["qg_z"][0], 4)
    print("sdks", out)


def dynamics_block():
    from qang.integrations.qiskit_dynamics import rabi_qg

    errs = {round(a, 3): (abs(rabi_qg(a) - math.cos(a)), abs(rabi_qg(a, shape="gaussian") - math.cos(a)))
            for a in (math.pi / 2, math.pi, 2.2)}
    print("dynamics |qg - cos A| (square, gaussian)", {k: ("%.1e" % v[0], "%.1e" % v[1]) for k, v in errs.items()})


def main():
    versions()
    for mod, block in (("mitiq", mitiq_block), ("qiskit_experiments", experiments_block), ("qutip", qutip_block),
                       ("stim", stim_block), ("sklearn", sklearn_block), ("jax", jax_block), ("torch", torch_block),
                       ("numpy", sdk_block), ("qiskit_dynamics", dynamics_block)):
        if has(mod):
            block()


if __name__ == "__main__":
    main()
