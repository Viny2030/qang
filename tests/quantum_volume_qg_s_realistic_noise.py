"""
qg_S under REALISTIC, gate-level noise channels: amplitude damping (T1)
and phase damping (T2), applied via Qiskit Aer's density-matrix
simulation to the same family of Quantum Volume circuits used throughout
this codebase -- as opposed to the toy global-depolarizing-on-the-
output-distribution model used everywhere else so far
(examples/quantum_volume_qg_s.py, quantum_volume_qg_s_finite_shots.py,
quantum_volume_qg_s_vs_xeb.py).

Noise is injected in two different places, because WHERE it acts turns
out to matter as much as WHAT it is:

  * "readout-only": a single-qubit error applied once to every qubit,
    right before the final measurement (after the whole circuit has
    already run).
  * "layered" / mid-circuit: the SAME per-qubit error, but applied after
    EVERY 2-qubit gate throughout the circuit (Quantum Volume's model
    circuit is built entirely from n_qubits/2 * depth 2-qubit "unitary"
    blocks) -- much closer to how decoherence actually accumulates on
    real hardware, where a qubit decays throughout the whole computation,
    not just while waiting to be read out.

Three findings, all on the same 4-qubit, depth-4 Quantum Volume circuit
(seed=0) used throughout this codebase:

  Finding A (an exact, provable blind spot): qg_S is EXACTLY invariant
  under phase damping (dephasing) applied only at readout -- to
  floating-point precision, for every damping strength tested, including
  lambda = 1.0 (total dephasing). This is not a numerical coincidence:
  phase damping is diagonal-preserving -- it only kills off-diagonal
  coherences -- so it cannot change the Z-basis measurement-outcome
  distribution qg_S is computed from, no matter how strong it is. This is
  the same kind of structural blind spot already documented for
  graph-state entanglement in qang.circuits' module docstring, now shown
  for an entire class of NOISE rather than a class of states.

  Finding B (the blind spot does not survive mid-circuit): the exact same
  phase-damping channel, applied throughout the circuit instead of only
  at readout, is NOT invisible to qg_S at all -- it drives qg_S up
  monotonically, similarly to depolarizing noise. Intermediate coherence
  loss is converted into ordinary population randomization by the
  entangling gates that come after it, so qg_S recovers its sensitivity
  as soon as the dephasing happens anywhere but the very end of the
  circuit -- exactly where real T2 decay actually occurs.

  Finding C (a genuine caution for qg_S as a benchmark, extending
  examples/quantum_volume_qg_s.py's story): under mid-circuit AMPLITUDE
  damping (T1-type relaxation, whose fixed point is the deterministic
  ground state |00...0>, not the maximally mixed state), qg_S is NOT
  monotonic in the noise strength. It rises at first, exactly like
  depolarizing or dephasing noise, but then falls all the way back to 0
  as the damping strength approaches 1 -- because near-total T1 decay
  collapses the circuit onto a single, zero-entropy outcome. A device
  dominated by amplitude damping could therefore show a LOWER qg_S at a
  more broken operating point than at a less broken one: qg_S's
  monotonic "more noise -> higher qg_S" behavior, relied on throughout
  examples/quantum_volume_qg_s.py, is a property of the noise channel's
  fixed point (maximally mixed), not a universal property of qg_S itself.

  Finding D (the practical fix for Finding C): pair qg_S with mean qg_Z
  (qang.multiqubit.mean_qg_z), the register-averaged <sigma_z>. Unital
  noise (depolarizing, dephasing) drives mean qg_Z towards 0; amplitude
  damping drives it towards +1, its |00...0> fixed point. On the
  reference circuit, gamma = 0.0 and gamma = 0.4 give almost the same
  qg_S (0.9193 vs 0.9176) but clearly different mean qg_Z (-0.029 vs
  +0.310), so the pair (qg_S, mean qg_Z) separates operating points qg_S
  alone confuses. mean qg_Z rises monotonically with gamma on this
  circuit, but NOT on every circuit (n=3, seed=1 dips before rising --
  pinned in the tests), so this is an observed regularity, not a theorem.

WARNING: if T1 relaxation may be significant on your device, do not
report qg_S alone -- always report it together with mean qg_Z (see
t1_aware_profile below).
"""

import numpy as np
from qiskit.circuit.library import quantum_volume
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, phase_damping_error

from qang.multiqubit import joint_qg_s, mean_qg_z


def _density_matrix(qc) -> np.ndarray:
    """Exact (infinite-shot) density matrix of a circuit, via Aer's
    density-matrix simulation method -- isolates the noise channel's
    effect from any finite-shot sampling noise (see
    examples/quantum_volume_qg_s_finite_shots.py for that separate
    concern)."""
    qc = qc.copy()
    qc.save_density_matrix()
    sim = AerSimulator(method="density_matrix")
    result = sim.run(qc).result()
    return np.asarray(result.data(0)["density_matrix"])


def readout_only_noisy_density_matrix(qc, n_qubits: int, error_1q) -> np.ndarray:
    """Apply a single-qubit quantum error to every qubit once, right
    before the final measurement -- noise that only affects readout."""
    qc_noisy = qc.copy()
    for q in range(n_qubits):
        qc_noisy.append(error_1q.to_instruction(), [q])
    return _density_matrix(qc_noisy)


def layered_noisy_density_matrix(qc, n_qubits: int, error_1q, gate_name: str = "unitary") -> np.ndarray:
    """Apply the same single-qubit quantum error after every gate named
    gate_name (Quantum Volume's model circuit is built entirely from
    2-qubit 'unitary' blocks), via a NoiseModel -- noise that accumulates
    throughout the circuit, not just at readout."""
    error_2q = error_1q.tensor(error_1q)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_2q, [gate_name])
    qc_noisy = qc.copy()
    qc_noisy.save_density_matrix()
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    result = sim.run(qc_noisy).result()
    return np.asarray(result.data(0)["density_matrix"])


def t1_aware_profile(rho: np.ndarray, n_qubits: int):
    """(qg_S, mean qg_Z) for a density matrix: the T1-aware pair of
    Finding D. qg_S alone cannot tell "a little noise" from "a lot of
    amplitude damping"; mean qg_Z (near 0 for unital noise, towards +1
    for amplitude damping) resolves that ambiguity."""
    return joint_qg_s(rho, n_qubits, normalize=True), mean_qg_z(rho, n_qubits)


if __name__ == "__main__":
    n_qubits = 4
    qc = quantum_volume(n_qubits, depth=n_qubits, seed=0)
    qg_s_ideal = joint_qg_s(_density_matrix(qc), n_qubits, normalize=True)
    print(f"Ideal (noiseless) qg_S = {qg_s_ideal:.4f}")
    print()

    print("Finding A: qg_S is exactly invariant under READOUT-ONLY dephasing.")
    for lam in [0.0, 0.3, 0.7, 1.0]:
        rho = readout_only_noisy_density_matrix(qc, n_qubits, phase_damping_error(lam))
        qs = joint_qg_s(rho, n_qubits, normalize=True)
        print(f"  lambda={lam:.1f}  qg_S={qs:.6f}")
    print()

    print("Finding B: the SAME dephasing channel, applied MID-CIRCUIT instead, is not invisible.")
    for lam in [0.0, 0.1, 0.4, 0.7, 1.0]:
        rho = layered_noisy_density_matrix(qc, n_qubits, phase_damping_error(lam))
        qs = joint_qg_s(rho, n_qubits, normalize=True)
        print(f"  lambda={lam:.1f}  qg_S={qs:.4f}")
    print()

    print("Finding C: mid-circuit AMPLITUDE damping makes qg_S non-monotonic in the noise strength.")
    for gamma in [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]:
        rho = layered_noisy_density_matrix(qc, n_qubits, amplitude_damping_error(gamma))
        qs = joint_qg_s(rho, n_qubits, normalize=True)
        print(f"  gamma={gamma:.2f}  qg_S={qs:.4f}")
    print()

    print("Finding D: the pair (qg_S, mean qg_Z) separates what qg_S alone confuses.")
    for gamma in [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]:
        rho = layered_noisy_density_matrix(qc, n_qubits, amplitude_damping_error(gamma))
        qs, bias = t1_aware_profile(rho, n_qubits)
        print(f"  gamma={gamma:.2f}  qg_S={qs:.4f}  mean_qg_Z={bias:+.4f}")
