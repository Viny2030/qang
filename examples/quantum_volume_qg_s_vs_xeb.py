"""
qg_S vs linear XEB (cross-entropy benchmarking): a direct comparison
between qang's entropy-based benchmark and cross-entropy benchmarking
(XEB), the metric used in Google's quantum supremacy experiments
(Arute et al., 2019, "Quantum supremacy using a programmable
superconducting processor", Nature 574, 505) and widely adopted since.

examples/quantum_volume_qg_s.py already compared qg_S against Heavy
Output Probability, via exact distributions. This file compares qg_S
against a different, and arguably more standard, cross-entropy-based
benchmark instead: LINEAR XEB.

Linear XEB estimates a circuit's fidelity from M measured bitstrings
x_1..x_M (sampled from the real, possibly noisy device) together with
the EXACT ideal probabilities p_ideal(x) -- computed once, in
simulation, the same requirement qg_S's finite-shot estimator has (see
examples/quantum_volume_qg_s_finite_shots.py):

    F_XEB = (2^n / M) * sum_i p_ideal(x_i)  -  1

Noise model: the same global-depolarizing mix used throughout this
codebase, p_noisy(x) = (1 - p) * p_ideal(x) + p / 2^n (see
examples/quantum_volume_qg_s.py's mix_with_uniform).

Two findings, both against the same family of Quantum Volume circuits
used elsewhere in this codebase (n_qubits-qubit, depth = n_qubits):

  Finding A (an honest asymmetry, exact/infinite-shot level): linear
  XEB's absolute scale is NOT calibration-free. Its exact expected value
  at zero added noise (p=0) is A - 1, where
  A = 2^n * sum_x p_ideal(x)^2 is the circuit's own "participation
  ratio" -- not 1, unless A happens to equal 2, the value expected only
  in the idealized limit of a Haar-random ("Porter-Thomas") output
  distribution. For the very same 3-6 qubit Quantum Volume circuits used
  elsewhere in this codebase (examples/quantum_volume_qg_s.py,
  examples/quantum_volume_qg_s_finite_shots.py), we measure A ranging
  from about 1.23 to 2.33 -- nowhere near a universal constant, so a raw
  XEB value cannot be read off as "the fidelity" without first computing
  this circuit-specific constant (which itself needs the same exact
  simulation qg_S already uses). qg_S carries no such caveat: it is
  always exactly the Shannon entropy of whatever the true output
  distribution actually is, with no assumption about that distribution's
  shape.

  Finding B (finite-shot statistics, same methodology as
  quantum_volume_qg_s_finite_shots.py's Miller-Madow study): unlike
  qg_S's plug-in Shannon-entropy estimator -- shown in that file to be
  negatively biased at finite shot count, needing the Miller-Madow
  correction -- linear XEB's finite-shot estimator is unbiased at every
  shot count tested, including as few as 10 shots (bias is always within
  5 standard errors of zero). This is not a numerical coincidence: XEB
  is a sample mean of a fixed, bounded per-shot quantity
  (2^n * p_ideal(x_i)), and a sample mean is always an exactly unbiased
  estimator of its expectation, however few samples are drawn. Shannon
  entropy, by contrast, is a strictly concave (nonlinear) functional of
  the underlying distribution -- exactly why its plug-in estimator is
  biased (a direct consequence of Jensen's inequality), while a linear
  benchmark like XEB structurally cannot be.
"""

import numpy as np
from qiskit.circuit.library import quantum_volume
from qiskit.quantum_info import Statevector

from qang.multiqubit import joint_qg_s


def mix_with_uniform(probs: np.ndarray, p_noise: float, dim: int) -> np.ndarray:
    """Global depolarizing noise model on the output distribution -- the
    same model used in examples/quantum_volume_qg_s.py, duplicated here
    (2 lines) so this file stays runnable standalone."""
    return (1.0 - p_noise) * probs + p_noise * (1.0 / dim)


def sample_counts(probs: np.ndarray, n_shots: int, rng: np.random.Generator) -> dict:
    """A single simulated experiment: n_shots i.i.d. draws from probs,
    returned as an outcome -> count dict -- the same helper used in
    examples/quantum_volume_qg_s_finite_shots.py, duplicated here so this
    file stays runnable standalone."""
    sample = rng.multinomial(n_shots, probs)
    return {i: int(c) for i, c in enumerate(sample) if c > 0}


def participation_ratio(probs_ideal: np.ndarray) -> float:
    """2^n * sum_x p_ideal(x)^2 -- exactly 2 in the idealized Porter-Thomas
    limit of a Haar-random state; the circuit-specific constant that sets
    linear XEB's absolute scale (see this module's docstring, Finding A)."""
    dim = len(probs_ideal)
    return float(dim * np.sum(probs_ideal ** 2))


def linear_xeb_fidelity_exact(probs_ideal: np.ndarray, probs_actual: np.ndarray) -> float:
    """Exact (infinite-shot) linear XEB value: 2^n * E_{x~probs_actual}[p_ideal(x)] - 1."""
    dim = len(probs_ideal)
    return float(dim * np.sum(probs_actual * probs_ideal) - 1.0)


def linear_xeb_fidelity_from_counts(counts: dict, probs_ideal: np.ndarray) -> float:
    """Finite-shot linear XEB estimate from observed counts (outcome ->
    count) together with the exact ideal probabilities -- exactly the
    information a real XEB experiment uses."""
    dim = len(probs_ideal)
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("counts must contain at least one observed shot.")
    mean_ideal_prob = sum(c * probs_ideal[outcome] for outcome, c in counts.items()) / total
    return float(dim * mean_ideal_prob - 1.0)


def mean_xeb_bias_at_shot_budget(
    probs_ideal: np.ndarray, probs_actual: np.ndarray, n_shots: int, n_trials: int, rng: np.random.Generator
):
    """Average, over n_trials independent simulated experiments at a fixed
    shot budget, the finite-shot linear XEB estimate's bias and standard
    deviation relative to the exact value -- the same protocol used for
    qg_S's plug-in/Miller-Madow bias in
    examples/quantum_volume_qg_s_finite_shots.py."""
    exact = linear_xeb_fidelity_exact(probs_ideal, probs_actual)
    estimates = []
    for _ in range(n_trials):
        counts = sample_counts(probs_actual, n_shots, rng)
        estimates.append(linear_xeb_fidelity_from_counts(counts, probs_ideal))
    return exact, float(np.mean(estimates)) - exact, float(np.std(estimates))


if __name__ == "__main__":
    print("Finding A: linear XEB's noiseless-limit value is A - 1, not 1, where")
    print("A = 2^n * sum(p_ideal^2) is the circuit's own participation ratio.")
    for n_qubits in [3, 4, 5, 6]:
        qc = quantum_volume(n_qubits, depth=n_qubits, seed=0)
        probs_ideal = Statevector.from_instruction(qc).probabilities()
        a = participation_ratio(probs_ideal)
        xeb_at_zero_noise = linear_xeb_fidelity_exact(probs_ideal, probs_ideal)
        print(f"  n_qubits={n_qubits}  A={a:.4f}  XEB(p_noise=0)={xeb_at_zero_noise:.4f}  (A - 1 = {a - 1.0:.4f})")
        qg_s_ideal = joint_qg_s(np.diag(probs_ideal.astype(complex)), n_qubits=n_qubits, normalize=True)
        print(f"    for comparison, qg_S(p_noise=0)={qg_s_ideal:.4f} (no calibration needed)")
    print()

    print("Finding B: finite-shot linear XEB is unbiased even at tiny shot counts,")
    print("unlike qg_S's plug-in Shannon-entropy estimator (see quantum_volume_qg_s_finite_shots.py).")
    n_qubits = 4
    qc = quantum_volume(n_qubits, depth=n_qubits, seed=0)
    probs_ideal = Statevector.from_instruction(qc).probabilities()
    dim = 2 ** n_qubits
    rng = np.random.default_rng(123)
    for p_noise in [0.0, 0.3, 0.7]:
        probs_actual = mix_with_uniform(probs_ideal, p_noise, dim)
        for n_shots in [10, 50, 500, 5000]:
            exact, bias, std = mean_xeb_bias_at_shot_budget(
                probs_ideal, probs_actual, n_shots, n_trials=2000, rng=rng
            )
            se = std / np.sqrt(2000)
            print(
                f"  p_noise={p_noise:.1f}  n_shots={n_shots:5d}  exact={exact:+.4f}  "
                f"bias={bias:+.5f}  (SE of mean={se:.5f})"
            )
