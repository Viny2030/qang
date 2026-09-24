"""
Tests for examples/barren_plateau_qg_local_cost.py: global cost vs the qg
local cost (1 - mean qg_Z)/2, and the classical light-cone control.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

pytest.importorskip("scipy")

from barren_plateau_qg_local_cost import (  # noqa: E402
    decay_rate,
    global_cost,
    gradient_variance,
    mean_qg_z_lightcone,
    n_params,
    output_probabilities,
    qg_local_cost,
    qg_local_cost_and_grad_lightcone,
    random_target,
    steps_to_fidelity,
    train,
    train_classical_lightcone,
    zero_gradient_fraction,
)

from qang.multiqubit import mean_qg_z  # noqa: E402


def _qiskit_state(params, target, n, L):
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    P = np.asarray(params).reshape(L + 1, n, 2)
    qc = qiskit.QuantumCircuit(n)
    for layer in range(L + 1):
        if layer > 0:
            for q in range(n - 1):
                qc.cz(q, q + 1)
        for q in range(n):
            qc.ry(P[layer, q, 0], q)
            qc.rz(P[layer, q, 1], q)
    for q in range(n):
        qc.ry(-target[q], q)
    return Statevector(qc).data


def test_simulator_matches_qiskit_and_qang_mean_qg_z():
    n, L = 3, 2
    rng = np.random.default_rng(0)
    p, tg = rng.uniform(0, 2 * np.pi, n_params(n, L)), random_target(n)
    psi = _qiskit_state(p, tg, n, L)
    probs = output_probabilities(p[None], tg, n, L)
    assert np.allclose(probs[0], np.abs(psi) ** 2, atol=1e-12)
    assert qg_local_cost(probs, n)[0] == pytest.approx((1 - mean_qg_z(psi, n)) / 2, abs=1e-12)


def test_both_costs_vanish_only_at_target():
    n, L = 4, 1
    # all rotations zero except the last Ry layer = target angles -> exact target
    tg = random_target(n)
    P = np.zeros((L + 1, n, 2))
    P[-1, :, 0] = tg
    probs = output_probabilities(P.reshape(1, -1), tg, n, L)
    assert global_cost(probs)[0] == pytest.approx(0, abs=1e-12)
    assert qg_local_cost(probs, n)[0] == pytest.approx(0, abs=1e-12)
    probs = output_probabilities(np.random.default_rng(1).uniform(0, 6, (1, n_params(n, L))), tg, n, L)
    assert global_cost(probs)[0] > 0.1 and qg_local_cost(probs, n)[0] > 0.01


def test_shallow_global_gradient_vanishes_exponentially_qg_local_polynomially():
    ns = [2, 4, 6, 8]
    vg = [gradient_variance(n, 2, "global") for n in ns]
    vl = [gradient_variance(n, 2, "qg_local") for n in ns]
    assert decay_rate(ns, vg)[0] < -1.3  # ~2^-1.8 per qubit
    assert decay_rate(ns, vl)[1] > -2.6  # ~1/n^2
    assert vl[-1] > 20 * vg[-1]


def test_deep_circuit_qg_local_cost_still_has_barren_plateau():
    v4, v8 = (gradient_variance(n, 4 * n, "qg_local") for n in (4, 8))
    assert v8 < 1e-4 and v8 < v4 / 10


def test_shot_estimated_global_gradient_is_often_exactly_zero():
    assert zero_gradient_fraction(10, 2, "global", 100, inits=20) >= 0.1
    assert zero_gradient_fraction(10, 2, "qg_local", 100, inits=10) == 0.0


def test_training_with_shots_small_n():
    fid = train(6, 2, "qg_local", shots=100, steps=60, seed=0)
    assert fid[0] < 0.05 and fid[-1] > 0.95
    assert steps_to_fidelity(fid) is not None and steps_to_fidelity(fid) < 30
    assert steps_to_fidelity(np.array([0.0, 0.1])) is None


def test_lightcone_matches_full_statevector():
    n, L = 9, 2
    rng = np.random.default_rng(3)
    p, tg = rng.uniform(0, 2 * np.pi, n_params(n, L)), random_target(n)
    probs = output_probabilities(p[None], tg, n, L)
    assert mean_qg_z_lightcone(p, tg, n, L) == pytest.approx(1 - 2 * qg_local_cost(probs, n)[0], abs=1e-12)
    c, g = qg_local_cost_and_grad_lightcone(p, tg, n, L)
    m = n_params(n, L)
    shifts = np.vstack([np.eye(m), -np.eye(m)]) * (np.pi / 2)
    cc = qg_local_cost(output_probabilities(p + shifts, tg, n, L), n)
    assert c == pytest.approx(qg_local_cost(probs, n)[0], abs=1e-12)
    assert np.allclose(g, (cc[:m] - cc[m:]) / 2, atol=1e-12)


def test_classical_lightcone_trains_beyond_statevector_reach():
    c, fid_bound, _ = train_classical_lightcone(30, 1, maxiter=300)
    assert c < 1e-9 and fid_bound > 0.999999
