"""
Quantum Volume, reformulated in terms of qg_S: the first connection
between the qang framework and this widely used industry benchmarking
protocol (Cross et al., 2019, "Validating quantum computers using
randomized model circuits", Phys. Rev. A 100, 032328).

Quantum Volume works by generating a random "model circuit", computing
its ideal output distribution, marking the outcomes above the median
probability as "heavy", and then checking what fraction of a noisy
device's shots land in that heavy set (the "heavy output probability",
HOP). A device passes if HOP exceeds 2/3 with statistical confidence.

qg_S (Section 2.2 of the paper; qang.multiqubit.joint_qg_s for the
multi-qubit generalization) is the normalized Shannon entropy of a
measurement-outcome distribution. As a circuit's output is corrupted by
noise, its distribution flattens towards uniform: qg_S rises towards its
maximum (1.0), while HOP falls towards its noise floor (0.5, uniform
guessing over a heavy/non-heavy split). This file demonstrates that the
two move together: qg_S is a much cheaper, distribution-level noise
proxy for exactly the same degradation that heavy output probability is
designed to detect, without needing to know the heavy set (which itself
requires the ideal-circuit simulation) at all.

Noise model: for simplicity and to isolate the pure-entropy relationship
from any specific gate-error model, noise is modeled as global
depolarization on the *output distribution* itself: p_noisy(x) =
(1 - p) * p_ideal(x) + p / 2^n. This is the same model that motivates
the standard `2/3` HOP threshold's noise floor of 0.5 at p=1.
"""

import numpy as np
from qiskit.circuit.library import quantum_volume
from qiskit.quantum_info import Statevector

from qang.multiqubit import joint_qg_s


def heavy_set_from_ideal(probs: np.ndarray) -> list:
    """Outcomes with probability strictly above the median (Cross et al. 2019)."""
    median = np.median(probs)
    return [i for i, p in enumerate(probs) if p > median]


def heavy_output_probability(probs: np.ndarray, heavy_set: list) -> float:
    return float(sum(probs[i] for i in heavy_set))


def mix_with_uniform(probs: np.ndarray, p_noise: float, dim: int) -> np.ndarray:
    """Global depolarizing noise model on the output distribution."""
    return (1.0 - p_noise) * probs + p_noise * (1.0 / dim)


def qg_s_of_distribution(probs: np.ndarray, n_qubits: int) -> float:
    rho = np.diag(probs.astype(complex))
    return joint_qg_s(rho, n_qubits=n_qubits, normalize=True)


def run_noise_sweep(n_qubits: int, seed: int, noise_levels: np.ndarray):
    """For one random QV circuit, sweep noise and return (p, HOP, qg_S) triples."""
    qc = quantum_volume(n_qubits, depth=n_qubits, seed=seed)
    probs_ideal = Statevector.from_instruction(qc).probabilities()
    heavy_set = heavy_set_from_ideal(probs_ideal)
    dim = 2**n_qubits

    rows = []
    for p_noise in noise_levels:
        probs_noisy = mix_with_uniform(probs_ideal, p_noise, dim)
        hop = heavy_output_probability(probs_noisy, heavy_set)
        qg_s = qg_s_of_distribution(probs_noisy, n_qubits)
        rows.append((p_noise, hop, qg_s))
    return rows


if __name__ == "__main__":
    n_qubits = 4
    n_circuits = 10
    noise_levels = np.linspace(0.0, 1.0, 11)

    print(f"Quantum Volume ({n_qubits} qubits) heavy output probability vs qg_S,")
    print(f"across {n_circuits} random circuits x {len(noise_levels)} noise levels.")
    print()

    print("Ideal (noiseless) values per circuit:")
    all_hop, all_qgs = [], []
    for seed in range(n_circuits):
        rows = run_noise_sweep(n_qubits, seed, noise_levels)
        p0, hop0, qgs0 = rows[0]
        passed = "PASS" if hop0 > 2.0 / 3.0 else "fail"
        print(f"  seed={seed}: HOP_ideal={hop0:.4f} ({passed}, threshold 2/3)  qg_S_ideal={qgs0:.4f}")
        for _, hop, qgs in rows:
            all_hop.append(hop)
            all_qgs.append(qgs)

    corr = float(np.corrcoef(all_hop, all_qgs)[0, 1])
    print()
    print(f"Pearson correlation (heavy output probability, qg_S) across all "
          f"{len(all_hop)} (circuit, noise level) points: {corr:.4f}")
    print()
    print("Single-circuit trace (seed=0), full noise sweep:")
    for p_noise, hop, qgs in run_noise_sweep(n_qubits, 0, noise_levels):
        print(f"  p_noise={p_noise:.2f}  HOP={hop:.4f}  qg_S={qgs:.4f}")
