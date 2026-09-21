"""
Tests for examples/randomized_benchmarking_qg_z.py: Randomized
Benchmarking reformulated in terms of qg_Z.

Checked:
  1. build_rb_blocks() composes to the exact identity on |0> in the
     noiseless case, for several sequence lengths and seeds -- the
     basic correctness requirement of any RB sequence construction.
  2. rb_qg_z_survival() exactly matches the closed-form
     (1 - p_depol) ** (sequence_length + 1) prediction, to floating-
     point precision, for a range of sequence lengths and depolarizing
     strengths.
  3. That value is exactly independent of which random Clifford
     sequence was drawn (several seeds at fixed sequence length and
     p_depol give identical results) -- the headline claim explained in
     the module's docstring.
  4. fit_depolarizing_parameter() recovers the true (1 - p_depol) to
     high precision from noiseless-simulation data, and
     average_gate_fidelity() then matches the textbook formula.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit.quantum_info import Statevector

from randomized_benchmarking_qg_z import (
    average_gate_fidelity,
    build_rb_blocks,
    fit_depolarizing_parameter,
    rb_qg_z_survival,
)


@pytest.mark.parametrize("sequence_length", [0, 1, 3, 7, 15])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rb_blocks_compose_to_identity_noiseless(sequence_length, seed):
    from qiskit import QuantumCircuit

    blocks = build_rb_blocks(sequence_length, seed)
    full = QuantumCircuit(1)
    for block in blocks:
        full.compose(block, inplace=True)
    sv = Statevector.from_instruction(full)
    assert np.allclose(sv.data, [1.0, 0.0], atol=1e-9)


@pytest.mark.parametrize("sequence_length", [0, 1, 3, 5, 10])
@pytest.mark.parametrize("p_depol", [0.0, 0.02, 0.05, 0.2])
def test_qg_z_survival_matches_closed_form_exactly(sequence_length, p_depol):
    value = rb_qg_z_survival(sequence_length, seed=0, p_depol=p_depol)
    expected = (1.0 - p_depol) ** (sequence_length + 1)
    assert value == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("sequence_length", [0, 3, 10])
def test_qg_z_survival_is_clifford_seed_independent(sequence_length):
    p_depol = 0.07
    values = [rb_qg_z_survival(sequence_length, seed, p_depol) for seed in range(6)]
    assert max(values) - min(values) < 1e-9


def test_rb_qg_z_survival_rejects_invalid_p_depol():
    with pytest.raises(ValueError):
        rb_qg_z_survival(3, seed=0, p_depol=1.5)


def test_fit_depolarizing_parameter_recovers_true_value():
    p_depol = 0.03
    sequence_lengths = [0, 2, 5, 10, 20, 40]
    qg_z_values = [rb_qg_z_survival(m, seed=0, p_depol=p_depol) for m in sequence_lengths]
    f_fit = fit_depolarizing_parameter(sequence_lengths, qg_z_values)
    assert f_fit == pytest.approx(1.0 - p_depol, abs=1e-9)


@pytest.mark.parametrize("p_depol", [0.01, 0.05, 0.1, 0.3])
def test_average_gate_fidelity_matches_textbook_formula(p_depol):
    f = 1.0 - p_depol
    # textbook single-qubit RB fidelity formula, d = 2: F_avg = (1 + f) / 2
    assert average_gate_fidelity(f, dimension=2) == pytest.approx((1.0 + f) / 2.0, abs=1e-12)


def test_end_to_end_fit_recovers_correct_average_gate_fidelity():
    p_depol = 0.04
    sequence_lengths = [0, 2, 5, 10, 20, 40]
    qg_z_values = [rb_qg_z_survival(m, seed=1, p_depol=p_depol) for m in sequence_lengths]
    f_fit = fit_depolarizing_parameter(sequence_lengths, qg_z_values)
    fidelity_fit = average_gate_fidelity(f_fit)
    true_fidelity = average_gate_fidelity(1.0 - p_depol)
    assert fidelity_fit == pytest.approx(true_fidelity, abs=1e-9)


if __name__ == "__main__":
    print("Run via `pytest tests/test_randomized_benchmarking_qg_z.py -v` for full coverage.")
    v = rb_qg_z_survival(5, seed=0, p_depol=0.05)
    assert v == pytest.approx(0.95 ** 6, abs=1e-9)
    print("Smoke check passed.")
