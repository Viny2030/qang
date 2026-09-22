"""
Finite-shot qg_S on Quantum Volume circuits: how badly the plug-in
Shannon entropy estimator is biased at realistic shot budgets, and how
much the Miller-Madow correction (qang.multiqubit) closes that bias.

examples/quantum_volume_qg_s.py already showed that qg_S is a cheap,
distribution-level proxy for heavy output probability -- but that file
computes qg_S from the EXACT ideal or noise-mixed probability
distribution, which is only available in simulation. On real hardware (or
any shot-based simulator) you only ever observe a finite number of
measurement counts, and the plug-in (maximum-likelihood) Shannon entropy
estimator built directly from those counts is well known to
systematically UNDERESTIMATE the true entropy -- worse when the shot
budget is small relative to the number of possible outcomes, 2^n_qubits.
That is exactly the regime a benchmarker is in as a circuit grows: more
qubits means more possible outcomes to resolve, at a shot budget that
does not usually grow to match.

Sampling here is done directly from each circuit's exact ideal
probabilities via a multinomial draw (numpy only, no gate-level noise
simulator needed): this isolates the finite-SAMPLING bias itself from any
question of device noise, which examples/quantum_volume_qg_s.py already
covers separately.

Two findings, both against an actual randomly generated Quantum Volume
circuit's exact ideal distribution as ground truth (never a synthetic toy
distribution):

  Finding A (bias shrinks with shot budget, as expected, but Miller-Madow
  is consistently and substantially better at every budget tested): on a
  4-qubit QV circuit, sweeping the shot budget from 50 to 20,000, the
  plug-in estimator's mean bias shrinks from about -0.054 to -0.0002 (in
  the normalized qg_S units of examples/quantum_volume_qg_s.py, i.e.
  qg_S in [0, 1]), while Miller-Madow's mean bias is smaller at every
  single budget tested, by roughly 5x at the low end.

  Finding B (the bias gets worse with qubit count at a FIXED shot budget,
  and Miller-Madow's advantage grows correspondingly): at a fixed budget
  of 1,000 shots, the plug-in bias roughly quadruples in magnitude from
  3 qubits (8 outcomes) to 6 qubits (64 outcomes), while Miller-Madow's
  bias stays much smaller and does not show the same growth -- exactly
  the "more qubits, same shot budget" trade-off a real benchmarking run
  faces.
"""

import numpy as np
from qiskit.circuit.library import quantum_volume
from qiskit.quantum_info import Statevector

from qang.multiqubit import joint_qg_s, joint_qg_s_from_counts


def sample_counts(probs: np.ndarray, n_shots: int, rng: np.random.Generator) -> dict:
    """A single simulated experiment: n_shots i.i.d. draws from probs,
    returned as an outcome -> count dict (only observed outcomes appear,
    exactly like a real device's counts dict)."""
    sample = rng.multinomial(n_shots, probs)
    return {i: int(c) for i, c in enumerate(sample) if c > 0}


def mean_bias_at_shot_budget(
    probs: np.ndarray, n_qubits: int, n_shots: int, n_trials: int, rng: np.random.Generator
):
    """Average, over n_trials independent simulated experiments at a
    fixed shot budget, the plug-in and Miller-Madow qg_S estimates'
    bias relative to the exact (infinite-shot) qg_S of the same
    distribution."""
    exact = joint_qg_s(np.diag(probs.astype(complex)), n_qubits, normalize=True)
    plugins, mms = [], []
    for _ in range(n_trials):
        counts = sample_counts(probs, n_shots, rng)
        plugins.append(joint_qg_s_from_counts(counts, n_qubits, normalize=True, bias_correction="none"))
        mms.append(joint_qg_s_from_counts(counts, n_qubits, normalize=True, bias_correction="miller_madow"))
    return exact, float(np.mean(plugins)) - exact, float(np.mean(mms)) - exact


if __name__ == "__main__":
    n_trials = 500

    print("Finding A: shot-budget sweep on a single 4-qubit QV circuit.")
    n_qubits = 4
    qc = quantum_volume(n_qubits, depth=n_qubits, seed=0)
    probs = Statevector.from_instruction(qc).probabilities()
    rng = np.random.default_rng(123)
    for n_shots in [50, 100, 500, 1000, 5000, 20000]:
        exact, bias_plugin, bias_mm = mean_bias_at_shot_budget(probs, n_qubits, n_shots, n_trials, rng)
        print(
            f"  n_shots={n_shots:6d}  exact qg_S={exact:.4f}  "
            f"plug-in bias={bias_plugin:+.5f}  Miller-Madow bias={bias_mm:+.5f}"
        )
    print()

    print("Finding B: fixed 1,000-shot budget, varying qubit count.")
    n_shots_fixed = 1000
    for nq in [3, 4, 5, 6]:
        qc = quantum_volume(nq, depth=nq, seed=0)
        probs_nq = Statevector.from_instruction(qc).probabilities()
        rng2 = np.random.default_rng(7)
        exact, bias_plugin, bias_mm = mean_bias_at_shot_budget(probs_nq, nq, n_shots_fixed, n_trials, rng2)
        print(
            f"  n_qubits={nq} (2^n={2 ** nq:4d} outcomes)  exact qg_S={exact:.4f}  "
            f"plug-in bias={bias_plugin:+.5f}  Miller-Madow bias={bias_mm:+.5f}"
        )
