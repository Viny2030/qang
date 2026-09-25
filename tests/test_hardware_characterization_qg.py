"""
Tests for examples/hardware_characterization_qg.py: separating thermal
population (effective temperature) from asymmetric readout error with a
heralded qg sweep, vs the standard calibration suite.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("scipy")

import hardware_characterization_qg as H  # noqa: E402


def test_readout_is_affine_in_qg_and_invertible():
    a, b = H.readout_affine(0.015, 0.04)
    assert (a, b) == pytest.approx((0.025, 0.945))
    assert H.readout_errors(a, b) == pytest.approx((0.015, 0.04))
    for qg in (-1.0, 0.3, 1.0):  # measured qg = P(r=0) - P(r=1)
        p0 = (1 - 0.015) * (1 + qg) / 2 + 0.04 * (1 - qg) / 2
        assert 2 * p0 - 1 == pytest.approx(a + b * qg)


def test_effective_temperature_matches_section_25():
    assert 1e3 * float(H.t_eff_from_qg(1 - 2 * 0.01)) == pytest.approx(52.2, abs=0.1)
    assert float(H.t_eff_from_qg(1.0)) == 0.0 and np.isinf(H.t_eff_from_qg(0.0))


def test_probabilities_are_normalized_and_heralding_is_consistent():
    tr = H.truth(0.05)
    for op, t in H.qg_circuits():
        p = H.joint_probabilities(op, t, **tr)
        assert p.sum() == pytest.approx(1.0) and np.all(p >= 0)
        # marginal of the herald = single measurement of the thermal state
        assert p[0] + p[1] == pytest.approx(H.joint_probabilities("I", 0.0, **tr, herald=False)[0])


def test_closed_form_recovers_all_three_parameters_exactly():
    for p_th in (0.0, 0.02, 0.1):
        tr = H.truth(p_th, e01=0.03, e10=0.07)
        data = [(op, t, 1e9 * H.joint_probabilities(op, t, **tr)) for op, t in H.qg_circuits()]
        a, b, q = H.closed_form_t0(data)
        assert (a, b, q) == pytest.approx((tr["a"], tr["b"], tr["qg_eq"]), abs=1e-9)


def test_standard_suite_counts_thermal_population_as_readout_error():
    p_th = 0.03
    tr = H.truth(p_th)
    data = [(op, t, 1e9 * H.joint_probabilities(op, t, **tr, herald=False)) for op, t in H.standard_circuits()]
    f = H.fit_standard(data)
    b = tr["b"]
    assert f["e01"] == pytest.approx(H.TRUE["e01"] + p_th * b, abs=1e-6)
    assert f["qg_eq"] == pytest.approx(1.0, abs=1e-9)  # looks perfectly cold
    assert f["T1"] == pytest.approx(100.0, rel=1e-3) and f["T2"] == pytest.approx(70.0, rel=1e-3)


def test_qg_sweep_is_unbiased_where_the_standard_suite_is_not():
    true, s = H.benchmark(0.03, reps=20, seed=3)
    assert abs(s["qg"]["e01"][0] - true["e01"]) < 0.002
    assert abs(s["qg"]["p_th"][0] - 0.03) < 0.003
    assert abs(s["qg"]["T_eff_mK"][0] - true["T_eff_mK"]) < 3
    assert s["standard"]["e01"][0] > true["e01"] + 0.02
    assert s["standard"]["T_eff_mK"][0] == 0.0
    for k in ("T1", "T2"):
        assert abs(s["qg"][k][0] / true[k] - 1) < 0.05 and abs(s["standard"][k][0] / true[k] - 1) < 0.05


def test_non_qnd_readout_stress_keeps_the_bias_small():
    true, s = H.benchmark(0.03, reps=10, seed=4, meas_relax=0.05)
    assert abs(s["qg"]["p_th"][0] - 0.03) < 0.004 and abs(s["qg"]["e01"][0] - true["e01"]) < 0.004


def test_exact_model_matches_a_qiskit_aer_circuit():
    """Thermal start by purification (§25), heralded measurement, idle
    thermal relaxation on 'id' gates, asymmetric readout."""
    pytest.importorskip("qiskit_aer")
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, thermal_relaxation_error

    p_th, e01, e10, T1, T2, t = 0.08, 0.02, 0.05, 100.0, 70.0, 20
    tr = H.truth(p_th, e01=e01, e10=e10)
    nm = NoiseModel()
    nm.add_quantum_error(thermal_relaxation_error(T1, T2, 1.0, excited_state_population=p_th), "id", [0])
    nm.add_readout_error(ReadoutError([[1 - e01, e01], [e10, 1 - e10]]), [0])
    sim = AerSimulator(noise_model=nm)
    for op in ("I", "X", "Y90"):
        q, c = QuantumRegister(2), ClassicalRegister(2)
        qc = QuantumCircuit(q, c)
        qc.ry(float(np.arccos(1 - 2 * p_th)), 0)
        qc.cx(0, 1)                      # q0 is now thermal
        qc.measure(0, 0)                 # herald
        if op == "X":
            qc.x(0)
        elif op == "Y90":
            qc.ry(np.pi / 2, 0)
        for _ in range(t):
            qc.id(0)
        if op == "Y90":
            qc.ry(-np.pi / 2, 0)
        qc.measure(0, 1)
        shots = 40_000
        counts = sim.run(qc, shots=shots, seed_simulator=7).result().get_counts()
        emp = np.array([counts.get(k, 0) for k in ("00", "10", "01", "11")]) / shots  # (r1, r2) = (c0, c1)
        model = H.joint_probabilities(op, float(t), **tr)
        assert np.abs(emp - model).max() < 0.009, (op, emp, model)
