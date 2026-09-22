"""
Tests for examples/quantum_volume_qg_s_finite_shots.py: the shot-budget
characterization of finite-shot qg_S (qang.multiqubit's
joint_qg_s_from_counts / shannon_entropy_miller_madow_bits) on actual
Quantum Volume circuits.

Checked:
  1. sample_counts produces valid counts dicts (non-negative, summing to
     the requested shot count, only observed outcomes present).
  2. Finding A: on a real 4-qubit QV circuit, the plug-in estimator's mean
     bias (relative to the exact, infinite-shot qg_S) shrinks as the shot
     budget grows, and Miller-Madow's bias is smaller than plug-in's at
     every budget tested.
  3. Finding B: at a fixed shot budget, the plug-in bias grows in
     magnitude as qubit count grows (more possible outcomes to resolve
     from the same number of shots), while Miller-Madow's bias stays
     much smaller throughout.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from quantum_volume_qg_s_finite_shots import mean_bias_at_shot_budget, sample_counts


def test_sample_counts_produces_valid_counts():
    rng = np.random.default_rng(0)
    probs = np.array([0.5, 0.25, 0.25, 0.0])
    counts = sample_counts(probs, n_shots=1000, rng=rng)
    assert sum(counts.values()) == 1000
    assert all(c > 0 for c in counts.values())
    assert 3 not in counts  # probability-zero outcome should essentially never appear


@pytest.mark.parametrize("n_shots", [50, 1000, 20000])
def test_mean_bias_at_shot_budget_returns_finite_values(n_shots):
    from qiskit.circuit.library import quantum_volume
    from qiskit.quantum_info import Statevector

    n_qubits = 4
    qc = quantum_volume(n_qubits, depth=n_qubits, seed=0)
    probs = Statevector.from_instruction(qc).probabilities()
    rng = np.random.default_rng(1)
    exact, bias_plugin, bias_mm = mean_bias_at_shot_budget(probs, n_qubits, n_shots, n_trials=50, rng=rng)
    assert 0.0 <= exact <= 1.0
    assert np.isfinite(bias_plugin)
    assert np.isfinite(bias_mm)


def test_finding_a_bias_shrinks_with_shot_budget_and_miller_madow_helps_throughout():
    from qiskit.circuit.library import quantum_volume
    from qiskit.quantum_info import Statevector

    n_qubits = 4
    qc = quantum_volume(n_qubits, depth=n_qubits, seed=0)
    probs = Statevector.from_instruction(qc).probabilities()
    rng = np.random.default_rng(123)

    shot_budgets = [50, 500, 5000]
    plugin_biases = []
    for n_shots in shot_budgets:
        exact, bias_plugin, bias_mm = mean_bias_at_shot_budget(probs, n_qubits, n_shots, n_trials=500, rng=rng)
        plugin_biases.append(abs(bias_plugin))
        # Miller-Madow's bias magnitude is smaller than plug-in's at every budget
        assert abs(bias_mm) < abs(bias_plugin)

    # plug-in bias magnitude shrinks monotonically as the shot budget grows
    assert plugin_biases[0] > plugin_biases[1] > plugin_biases[2]


def test_finding_b_bias_grows_with_qubit_count_at_a_fixed_shot_budget():
    from qiskit.circuit.library import quantum_volume
    from qiskit.quantum_info import Statevector

    n_shots_fixed = 1000
    plugin_biases = []
    mm_biases = []
    for nq in [3, 4, 5, 6]:
        qc = quantum_volume(nq, depth=nq, seed=0)
        probs = Statevector.from_instruction(qc).probabilities()
        rng = np.random.default_rng(7)
        exact, bias_plugin, bias_mm = mean_bias_at_shot_budget(probs, nq, n_shots_fixed, n_trials=500, rng=rng)
        plugin_biases.append(abs(bias_plugin))
        mm_biases.append(abs(bias_mm))

    # plug-in bias magnitude grows as qubit count grows at fixed shot budget
    assert plugin_biases[0] < plugin_biases[1] < plugin_biases[2] < plugin_biases[3]
    # Miller-Madow's bias stays much smaller than plug-in's, at every qubit count
    for pb, mb in zip(plugin_biases, mm_biases):
        assert mb < pb


if __name__ == "__main__":
    print("Run via `pytest tests/test_quantum_volume_qg_s_finite_shots.py -v`.")
    from qiskit.circuit.library import quantum_volume
    from qiskit.quantum_info import Statevector

    qc = quantum_volume(4, depth=4, seed=0)
    probs = Statevector.from_instruction(qc).probabilities()
    rng = np.random.default_rng(123)
    exact, bias_plugin, bias_mm = mean_bias_at_shot_budget(probs, 4, 50, n_trials=200, rng=rng)
    assert abs(bias_mm) < abs(bias_plugin)
    print("Smoke check passed.")
