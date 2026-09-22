"""
Tests for examples/quantum_volume_qg_s_vs_xeb.py: the direct comparison
between qg_S and linear XEB (cross-entropy benchmarking).

Checked:
  1. participation_ratio matches hand-computed values (uniform and
     deterministic distributions).
  2. linear_xeb_fidelity_exact matches the closed form
     (A - 1) * (1 - p_noise) under the global-depolarizing noise model,
     for a real Quantum Volume circuit's ideal distribution.
  3. linear_xeb_fidelity_exact is exactly 0 at full depolarization
     (p_noise = 1), for any circuit.
  4. linear_xeb_fidelity_from_counts converges to the exact value at a
     large shot count, and rejects empty counts.
  5. Finding A: the real 4-qubit Quantum Volume circuit used throughout
     this codebase has a participation ratio far from the idealized
     Porter-Thomas value of 2.
  6. Finding B: the finite-shot linear XEB estimator is unbiased (within
     statistical noise) at every shot budget tested, including as few as
     10 shots -- in contrast to qg_S's plug-in Shannon-entropy estimator
     (tests/test_multiqubit.py), which needs the Miller-Madow correction.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from quantum_volume_qg_s_vs_xeb import (
    linear_xeb_fidelity_exact,
    linear_xeb_fidelity_from_counts,
    mean_xeb_bias_at_shot_budget,
    mix_with_uniform,
    participation_ratio,
    sample_counts,
)


def test_participation_ratio_matches_hand_computed_value():
    # uniform over dim=4: A = dim * sum((1/dim)^2) = 1, the minimum possible value
    uniform = np.full(4, 0.25)
    assert participation_ratio(uniform) == pytest.approx(1.0, abs=1e-12)

    # a deterministic (single-outcome) distribution: A = dim * 1^2 = dim, the maximum possible value
    deterministic = np.array([1.0, 0.0, 0.0, 0.0])
    assert participation_ratio(deterministic) == pytest.approx(4.0, abs=1e-12)


def _real_qv_probs(n_qubits=4, seed=0):
    from qiskit.circuit.library import quantum_volume
    from qiskit.quantum_info import Statevector

    qc = quantum_volume(n_qubits, depth=n_qubits, seed=seed)
    return Statevector.from_instruction(qc).probabilities()


@pytest.mark.parametrize("p_noise", [0.0, 0.3, 0.7, 1.0])
def test_linear_xeb_fidelity_exact_matches_closed_form(p_noise):
    """Under the global-depolarizing mix, linear XEB is exactly linear in
    p_noise: F_XEB(p) = (A - 1) * (1 - p), where A is the circuit's own
    participation ratio (see this file's module docstring, Finding A)."""
    probs_ideal = _real_qv_probs()
    dim = len(probs_ideal)
    a = participation_ratio(probs_ideal)

    probs_actual = mix_with_uniform(probs_ideal, p_noise, dim)
    exact = linear_xeb_fidelity_exact(probs_ideal, probs_actual)
    closed_form = (a - 1.0) * (1.0 - p_noise)
    assert exact == pytest.approx(closed_form, abs=1e-9)


def test_linear_xeb_fidelity_exact_is_zero_at_full_depolarization():
    """At p_noise = 1 the actual distribution is exactly uniform, and
    linear XEB's exact value is 0 regardless of the circuit -- the
    common "noise floor" both qg_S and XEB are built to detect degradation
    away from."""
    probs_ideal = _real_qv_probs()
    dim = len(probs_ideal)
    uniform = np.full(dim, 1.0 / dim)
    assert linear_xeb_fidelity_exact(probs_ideal, uniform) == pytest.approx(0.0, abs=1e-9)


def test_linear_xeb_fidelity_from_counts_rejects_empty_counts():
    probs_ideal = _real_qv_probs()
    with pytest.raises(ValueError):
        linear_xeb_fidelity_from_counts({}, probs_ideal)


def test_linear_xeb_fidelity_from_counts_converges_to_exact_at_large_shot_count():
    probs_ideal = _real_qv_probs()
    dim = len(probs_ideal)
    probs_actual = mix_with_uniform(probs_ideal, 0.3, dim)
    exact = linear_xeb_fidelity_exact(probs_ideal, probs_actual)

    rng = np.random.default_rng(0)
    counts = sample_counts(probs_actual, n_shots=500_000, rng=rng)
    estimate = linear_xeb_fidelity_from_counts(counts, probs_ideal)
    assert estimate == pytest.approx(exact, abs=0.01)


def test_finding_a_participation_ratio_deviates_from_porter_thomas_value_for_realistic_circuits():
    """The specific 4-qubit, depth-4 Quantum Volume circuit used
    throughout this codebase is nowhere near the idealized Porter-Thomas
    participation ratio of 2 -- linear XEB's absolute scale for this
    circuit is set by A - 1 = 0.3662..., not by 1."""
    probs_ideal = _real_qv_probs(n_qubits=4, seed=0)
    a = participation_ratio(probs_ideal)
    assert a == pytest.approx(1.3662201721355798, abs=1e-9)
    assert abs(a - 2.0) > 0.5  # far from the idealized Porter-Thomas value


@pytest.mark.parametrize("n_shots", [10, 50, 500])
def test_finding_b_finite_shot_xeb_is_unbiased_even_at_tiny_shot_counts(n_shots):
    """The headline structural claim: linear XEB's finite-shot estimator
    is a sample mean of a fixed per-shot quantity, so it is exactly
    unbiased at any shot count -- verified here as "bias is within 5
    standard errors of zero", the standard statistical test for
    unbiasedness, at a shot count as small as 10."""
    probs_ideal = _real_qv_probs(n_qubits=4, seed=0)
    dim = len(probs_ideal)
    probs_actual = mix_with_uniform(probs_ideal, 0.3, dim)

    rng = np.random.default_rng(123)
    n_trials = 2000
    exact, bias, std = mean_xeb_bias_at_shot_budget(probs_ideal, probs_actual, n_shots, n_trials, rng)
    se = std / np.sqrt(n_trials)
    assert abs(bias) < 5.0 * se


if __name__ == "__main__":
    print("Run via `pytest tests/test_quantum_volume_qg_s_vs_xeb.py -v`.")
    probs_ideal = _real_qv_probs()
    dim = len(probs_ideal)
    probs_actual = mix_with_uniform(probs_ideal, 0.3, dim)
    rng = np.random.default_rng(0)
    exact, bias, std = mean_xeb_bias_at_shot_budget(probs_ideal, probs_actual, 50, 500, rng)
    se = std / np.sqrt(500)
    assert abs(bias) < 5.0 * se
    print("Smoke check passed.")
