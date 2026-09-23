"""
Tests for examples/quantum_volume_qg_s_realistic_noise.py: qg_S under
real, gate-level amplitude-damping and phase-damping noise channels
(Qiskit Aer), as opposed to the toy global-depolarizing-on-the-
distribution model used elsewhere in this codebase.

Checked:
  1. Finding A: qg_S is exactly invariant (to floating-point precision)
     under phase damping applied only at readout, at every damping
     strength tested, including full dephasing (lambda = 1.0).
  2. Finding B: the same channel applied mid-circuit instead raises qg_S
     monotonically -- it is NOT invisible there.
  3. Finding C: mid-circuit amplitude damping makes qg_S rise and then
     fall as the damping strength grows, unlike depolarizing or
     mid-circuit dephasing (which both rise monotonically); at full
     damping (gamma = 1.0) qg_S collapses to exactly 0, since total T1
     decay sends the circuit deterministically to the ground state.
  4. Finding D: mean qg_Z (qang.multiqubit.mean_qg_z) separates operating
     points that qg_S alone confuses under amplitude damping, rises
     monotonically with gamma on the reference circuit, stays near 0
     under dephasing, and is NOT monotonic on every circuit (the n=3,
     seed=1 counterexample is pinned so the claim stays honest).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
qiskit_aer = pytest.importorskip("qiskit_aer")

from qiskit.circuit.library import quantum_volume
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, depolarizing_error, phase_damping_error

from qang.multiqubit import joint_qg_s, mean_qg_z
from quantum_volume_qg_s_realistic_noise import (
    _density_matrix,
    layered_noisy_density_matrix,
    readout_only_noisy_density_matrix,
    t1_aware_profile,
)


def _qv_circuit(n_qubits=4, seed=0):
    return quantum_volume(n_qubits, depth=n_qubits, seed=seed)


@pytest.mark.parametrize("lam", [0.0, 0.3, 0.7, 1.0])
def test_finding_a_readout_only_dephasing_leaves_qg_s_exactly_unchanged(lam):
    n_qubits = 4
    qc = _qv_circuit(n_qubits)
    qg_s_ideal = joint_qg_s(_density_matrix(qc), n_qubits, normalize=True)

    rho = readout_only_noisy_density_matrix(qc, n_qubits, phase_damping_error(lam))
    qs = joint_qg_s(rho, n_qubits, normalize=True)
    assert qs == pytest.approx(qg_s_ideal, abs=1e-9)


def test_finding_b_mid_circuit_dephasing_raises_qg_s_monotonically():
    n_qubits = 4
    qc = _qv_circuit(n_qubits)
    values = []
    for lam in [0.0, 0.1, 0.4, 0.7, 1.0]:
        rho = layered_noisy_density_matrix(qc, n_qubits, phase_damping_error(lam))
        values.append(joint_qg_s(rho, n_qubits, normalize=True))
    assert values == sorted(values)
    assert values[0] < values[-1]  # a genuine, substantial rise, not noise-level drift


def test_finding_c_mid_circuit_amplitude_damping_is_not_monotonic():
    n_qubits = 4
    qc = _qv_circuit(n_qubits)
    gammas = [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    values = [
        joint_qg_s(layered_noisy_density_matrix(qc, n_qubits, amplitude_damping_error(g)), n_qubits, normalize=True)
        for g in gammas
    ]
    # rises at first...
    assert values[2] > values[0]
    # ...but falls back down well below its peak as damping approaches 1
    assert values[-1] < values[2]


def test_finding_c_full_amplitude_damping_collapses_qg_s_to_exactly_zero():
    """gamma = 1.0 sends every qubit deterministically to |0>, so the
    whole register ends up in the single, zero-entropy outcome |00...0>
    regardless of what the circuit computed."""
    n_qubits = 4
    qc = _qv_circuit(n_qubits)
    rho = layered_noisy_density_matrix(qc, n_qubits, amplitude_damping_error(1.0))
    assert joint_qg_s(rho, n_qubits, normalize=True) == pytest.approx(0.0, abs=1e-9)


def test_finding_c_contrasts_with_depolarizing_which_stays_monotonic():
    """The same mid-circuit injection method, but with depolarizing noise
    (whose fixed point IS the maximally mixed state) stays monotonically
    increasing throughout -- confirming that Finding C's non-monotonicity
    is specific to amplitude damping's different fixed point, not an
    artifact of the injection method itself."""
    n_qubits = 4
    qc = _qv_circuit(n_qubits)
    values = []
    for p in [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]:
        error_2q = depolarizing_error(p, 2)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_2q, ["unitary"])
        qc_noisy = qc.copy()
        qc_noisy.save_density_matrix()
        sim = AerSimulator(method="density_matrix", noise_model=noise_model)
        result = sim.run(qc_noisy).result()
        rho = np.asarray(result.data(0)["density_matrix"])
        values.append(joint_qg_s(rho, n_qubits, normalize=True))
    assert values == sorted(values)


if __name__ == "__main__":
    print("Run via `pytest tests/test_quantum_volume_qg_s_realistic_noise.py -v`.")
    n_qubits = 4
    qc = _qv_circuit(n_qubits)
    rho = layered_noisy_density_matrix(qc, n_qubits, amplitude_damping_error(1.0))
    assert joint_qg_s(rho, n_qubits, normalize=True) == pytest.approx(0.0, abs=1e-9)
    print("Smoke check passed.")


# --------------------------------------------------------------------- #
# Finding D: mean qg_Z as the T1-aware companion to qg_S
# --------------------------------------------------------------------- #
def _t1_sweep(n_qubits, seed, gammas):
    qc = quantum_volume(n_qubits, depth=n_qubits, seed=seed)
    return [
        t1_aware_profile(
            layered_noisy_density_matrix(qc, n_qubits, amplitude_damping_error(g)), n_qubits
        )
        for g in gammas
    ]


def test_finding_d_pair_separates_points_qg_s_alone_confuses():
    (s0, b0), (s4, b4) = _t1_sweep(4, 0, [0.0, 0.4])
    assert s0 == pytest.approx(0.9193, abs=1e-4)
    assert s4 == pytest.approx(0.9176, abs=1e-4)
    assert abs(s0 - s4) < 0.002          # qg_S alone: nearly indistinguishable
    assert b0 == pytest.approx(-0.0289, abs=1e-4)
    assert b4 == pytest.approx(0.3099, abs=1e-4)
    assert b4 - b0 > 0.3                 # mean qg_Z: clearly different


def test_finding_d_mean_qg_z_rises_monotonically_on_reference_circuit():
    gammas = np.linspace(0.0, 1.0, 21)
    biases = [b for _, b in _t1_sweep(4, 0, gammas)]
    assert np.all(np.diff(biases) > 0)
    assert biases[-1] == pytest.approx(1.0, abs=1e-9)


def test_finding_d_counterexample_mean_qg_z_is_not_always_monotonic():
    gammas = np.linspace(0.0, 1.0, 21)
    biases = [b for _, b in _t1_sweep(3, 1, gammas)]
    assert np.diff(biases).min() < -0.01    # dips before rising
    assert biases[-1] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("lam", [0.1, 0.4, 0.7, 1.0])
def test_finding_d_dephasing_keeps_mean_qg_z_near_zero(lam):
    n_qubits = 4
    qc = quantum_volume(n_qubits, depth=n_qubits, seed=0)
    rho = layered_noisy_density_matrix(qc, n_qubits, phase_damping_error(lam))
    assert abs(mean_qg_z(rho, n_qubits)) < 0.03
