"""Tests for qang.iqm (§165, §168; book chapter 31): readout from the poles, the qg read-out of IQM counts, and,
where iqm-client is installed, the error profile, the kept-fraction prediction and the chain choice on IQM's fake
backends. Nothing here contacts IQM."""

import json
import os

import numpy as np
import pytest

pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit.library import XXPlusYYGate  # noqa: E402

from qang import iqm  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "data")


def _w_like(n=3, seed=0, measure=True):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, n if measure else 0)
    qc.x(0)
    for i in range(n - 1):
        qc.append(XXPlusYYGate(rng.uniform(0.3, 2.8)), [i, i + 1])
    if measure:
        qc.measure(range(n), range(n))
    return qc


def test_poles_of_the_garnet_run_168():
    with open(os.path.join(DATA, "iqm_milliqang_qg_run.json"), encoding="utf-8") as fh:
        c = json.load(fh)["counts"]
    (e01, e10), = iqm.readout_from_poles(c["pole0"], c["pole1"])
    assert e01 == pytest.approx(0.0055) and e10 == pytest.approx(0.029)


def test_pole_circuits():
    zero, one = iqm.pole_circuits(3)
    assert zero.count_ops().get("x", 0) == 0 and one.count_ops()["x"] == 3
    assert zero.count_ops()["measure"] == one.count_ops()["measure"] == 3


def test_check_weight_on_the_logical_circuit():
    r = iqm.check_weight(_w_like(measure=False))
    assert r["weight_preserving"] is True and r["initial_weight"] == 1
    qc = _w_like(measure=False)
    qc.rx(0.3, 1)
    assert iqm.check_weight(qc)["weight_preserving"] is False


def _exact_counts(qc, errors, scale=1e9):
    """Expected counts (qang tuples) of ``qc`` read through independent readout errors, as large floats."""
    from qiskit.quantum_info import Statevector

    n = qc.num_qubits
    p = Statevector(qc.remove_final_measurements(inplace=False)).probabilities()  # little-endian: bit i is qubit i
    p = p.reshape((2,) * n).transpose(range(n - 1, -1, -1)).reshape(-1)          # now qubit 0 first
    t = p.reshape((2,) * n)
    for i, (e01, e10) in enumerate(errors):
        M = np.array([[1 - e01, e10], [e01, 1 - e10]])
        t = np.moveaxis(np.tensordot(M, np.moveaxis(t, i, 0), axes=(1, 0)), 0, i)
    noisy = t.reshape(-1)
    exact_z = [float(np.dot(p, 1 - 2 * ((np.arange(2 ** n) >> (n - 1 - i)) & 1))) for i in range(n)]
    counts = {tuple(int(b) for b in format(k, f"0{n}b")): v * scale for k, v in enumerate(noisy) if v > 0}
    return counts, exact_z


def test_iqm_qg_undoes_the_readout_errors():
    qc = _w_like()
    errors = [(0.03, 0.06), (0.02, 0.05), (0.05, 0.04)]
    counts, z = _exact_counts(qc, errors)
    r = iqm.iqm_qg(counts, qc, errors=errors, weight=1)
    assert [v for v, _ in r["readout"]] == pytest.approx(z, abs=1e-9)
    assert [v for v, _ in r["filtered_readout"]] == pytest.approx(z, abs=1e-9)
    assert r["kept_readout"] == pytest.approx(1.0, abs=1e-9)
    assert r["kept_raw"] < 1.0
    assert max(abs(v - e) for (v, _), e in zip(r["raw"], z)) > 0.01   # the raw reading is biased


def test_iqm_qg_without_errors_and_with_qiskit_strings():
    qc = _w_like()
    r = iqm.iqm_qg({"001": 600, "010": 300, "000": 100}, qc, weight=1)   # Qiskit order: qubit 0 last
    assert "readout" not in r and r["kept_raw"] == pytest.approx(0.9)
    assert [v for v, _ in r["filtered"]] == pytest.approx([1 - 2 * 600 / 900, 1 - 2 * 300 / 900, 1.0])
    with pytest.raises(ValueError):
        iqm.iqm_qg({"01": 1}, qc)


# ------------------------------------------------------------------ IQM fake backends (iqm-client[qiskit])


@pytest.fixture(scope="module")
def apollo():
    fake = pytest.importorskip("iqm.qiskit_iqm")
    return fake.IQMFakeApollo()


def test_readout_errors_from_the_profile(apollo):
    e = iqm.readout_errors(apollo, [0, 1])
    prof = apollo.error_profile.readout_errors
    assert e[0] == (prof["QB1"]["0"], prof["QB1"]["1"]) and e[1] == (prof["QB2"]["0"], prof["QB2"]["1"])


def test_poles_on_the_fake_backend_match_its_profile(apollo):
    from qiskit import transpile

    phys = [0, 1, 4]
    ts = [transpile(p, apollo, initial_layout=phys, optimization_level=0) for p in iqm.pole_circuits(3)]
    res = apollo.run(ts, shots=20000, seed_simulator=7).result()
    measured = iqm.readout_from_poles(res.get_counts(0), res.get_counts(1))
    # the poles also see the X gate's error and relaxation, so allow a little more than shot noise
    assert np.allclose(measured, iqm.readout_errors(apollo, phys), atol=0.01)


def test_prediction_and_chain_choice(apollo):
    best = iqm.best_chain(_w_like(), apollo, 1, limit=10)
    p = best["prediction"]
    assert 0 < p["kept"] < p["kept_mixed_floor"] < 1
    assert p["kept_mixed_floor"] >= best["default_prediction"]["kept_mixed_floor"]
    assert best["candidates"] == 11


def test_a_real_backend_without_profile_is_refused():
    class Real:
        error_profile = None

    with pytest.raises(ValueError, match="pole_circuits"):
        iqm.readout_errors(Real(), [0])
