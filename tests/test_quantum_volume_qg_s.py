"""
Tests for examples/quantum_volume_qg_s.py: the first connection between
the qang framework and the industry-standard Quantum Volume benchmarking
protocol (Cross et al., 2019).

Checked:
  1. qg_S of the fully-depolarized (p_noise=1) output distribution is
     exactly 1.0 -- the theoretical maximum -- for any circuit and qubit
     count, since a uniform distribution over 2^n outcomes has maximal
     Shannon entropy by construction.
  2. Within a single circuit's noise sweep, heavy output probability
     (HOP) decreases monotonically as noise increases, while qg_S
     increases monotonically -- they move in lockstep, in opposite
     directions.
  3. A strong negative correlation between HOP and qg_S holds across
     multiple random circuits and noise levels (the headline empirical
     finding), not just for one lucky circuit instance.
  4. Ideal (noiseless) heavy output probability exceeds the standard
     QV pass threshold of 2/3 for correctly generated model circuits,
     consistent with the asymptotic Porter-Thomas prediction
     (Cross et al. 2019).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from quantum_volume_qg_s import (
    heavy_output_probability,
    heavy_set_from_ideal,
    mix_with_uniform,
    qg_s_of_distribution,
    run_noise_sweep,
)


@pytest.mark.parametrize("n_qubits,seed", [(3, 0), (4, 1), (4, 2)])
def test_fully_depolarized_output_has_maximal_qg_s(n_qubits, seed):
    noise_levels = np.array([1.0])
    rows = run_noise_sweep(n_qubits, seed, noise_levels)
    _, hop, qg_s = rows[0]
    assert qg_s == pytest.approx(1.0, abs=1e-9)
    assert hop == pytest.approx(0.5, abs=0.15)  # heavy set is close to half the outcomes


@pytest.mark.parametrize("n_qubits,seed", [(3, 0), (4, 1), (4, 3)])
def test_hop_and_qg_s_move_monotonically_in_opposite_directions(n_qubits, seed):
    noise_levels = np.linspace(0.0, 1.0, 11)
    rows = run_noise_sweep(n_qubits, seed, noise_levels)
    hops = [r[1] for r in rows]
    qgss = [r[2] for r in rows]

    # HOP non-increasing, qg_S non-decreasing, as noise increases
    assert all(hops[i] >= hops[i + 1] - 1e-9 for i in range(len(hops) - 1))
    assert all(qgss[i] <= qgss[i + 1] + 1e-9 for i in range(len(qgss) - 1))


def test_strong_negative_correlation_across_circuits_and_noise_levels():
    n_qubits = 4
    noise_levels = np.linspace(0.0, 1.0, 11)
    all_hop, all_qgs = [], []
    for seed in range(10):
        for _, hop, qgs in run_noise_sweep(n_qubits, seed, noise_levels):
            all_hop.append(hop)
            all_qgs.append(qgs)

    corr = np.corrcoef(all_hop, all_qgs)[0, 1]
    assert corr < -0.8  # strong negative correlation


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_ideal_heavy_output_probability_passes_qv_threshold(seed):
    """Correctly generated QV model circuits should pass the standard
    HOP > 2/3 threshold in the noiseless case (Cross et al. 2019's own
    asymptotic Porter-Thomas prediction is ~0.85)."""
    noise_levels = np.array([0.0])
    rows = run_noise_sweep(4, seed, noise_levels)
    _, hop_ideal, _ = rows[0]
    assert hop_ideal > 2.0 / 3.0


def test_heavy_set_and_mixing_helpers_behave_as_expected():
    probs = np.array([0.1, 0.4, 0.3, 0.2])
    heavy = heavy_set_from_ideal(probs)
    assert heavy == [1, 2]  # strictly above the median (0.25)
    assert heavy_output_probability(probs, heavy) == pytest.approx(0.7)

    mixed = mix_with_uniform(probs, p_noise=0.5, dim=4)
    assert mixed == pytest.approx([0.175, 0.325, 0.275, 0.225])
    assert mixed.sum() == pytest.approx(1.0)


if __name__ == "__main__":
    print("Run via `pytest tests/test_quantum_volume_qg_s.py -v` for full coverage.")
    rows = run_noise_sweep(4, 0, np.array([0.0, 1.0]))
    assert rows[-1][2] == pytest.approx(1.0, abs=1e-9)
    print("Smoke check passed.")
