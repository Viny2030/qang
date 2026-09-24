"""
Which kind of noise is it? qg features vs the standard benchmarks.

A classification task on 4-qubit, depth-4 Quantum Volume circuits (24
different circuits, seeds 0-23). Each circuit runs under one of three
gate-level noise channels applied after every two-qubit block (as in
examples/quantum_volume_qg_s_realistic_noise.py): amplitude damping (T1),
phase damping (dephasing) or depolarizing, at 10 strengths. From N
measured shots we ask one question: is the noise dissipative (T1) or
unital (dephasing / depolarizing)?

Each feature is turned into a classifier by the simplest possible rule, a
single threshold chosen on 16 training circuits and scored on the 8
held-out circuits, so the comparison measures the feature, not a model:

  * mean qg_Z (qang.multiqubit.mean_qg_z_from_counts): needs only counts.
  * qg_S (Miller-Madow): needs only counts.
  * linear XEB and heavy output probability (HOP): need the ideal output
    distribution, i.e. a classical simulation of the circuit.

Finding A (strong noise, 2-50% per gate). mean qg_Z alone identifies T1
on unseen circuits clearly better than XEB, HOP or qg_S at every shot
budget: 0.88-0.91 vs 0.70-0.76 (0.67 is the accuracy of always answering
"unital"), and it already works at N = 100 shots. XEB and HOP mostly
measure how much noise there is, not which kind. By strength (N = 1000):
mean qg_Z is right 0.98 of the time above 25% per gate, 0.90 between 10%
and 25%, and 0.75 below 10%.

Finding B (realistic weak noise, 0.5-5% per gate). Nothing works:
every feature is at chance (0.67), even with 100,000 shots. The reason
is not shot noise but the circuits themselves: the ideal mean qg_Z of a
random 4-qubit QV circuit already varies from circuit to circuit (std
0.14 over these 24), more than the T1 shift it is supposed to detect.

Practical consequence: passive detection of T1 on arbitrary payload
circuits only works when the damping is strong. To detect realistic T1
use a designed probe whose ideal mean qg_Z is known, such as the idle
delay sweep of examples/nisq_hardware_validation.py, where mean qg_Z
rises monotonically with the delay on device-calibrated noise.

"""

import os
import sys

import numpy as np
from qiskit.circuit.library import quantum_volume
from qiskit.quantum_info import Statevector
from qiskit_aer.noise import amplitude_damping_error, depolarizing_error, phase_damping_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qang.multiqubit import joint_qg_s_from_counts, mean_qg_z, mean_qg_z_from_counts  # noqa: E402
from quantum_volume_qg_s_realistic_noise import layered_noisy_density_matrix  # noqa: E402

N_QUBITS = 4
DIM = 2**N_QUBITS
N_CIRCUITS = 24
TRAIN_SEEDS = range(16)
NOISE = {
    "T1": amplitude_damping_error,
    "dephasing": phase_damping_error,
    "depolarizing": lambda p: depolarizing_error(p, 1),
}
STRONG = np.linspace(0.02, 0.5, 10)
WEAK = np.linspace(0.005, 0.05, 10)
FEATURES = ("mean_qg_z", "qg_s", "xeb", "hop")


def build_dataset(strengths):
    """Exact noisy output distributions: list of
    (circuit seed, noise type, strength, ideal probs, noisy probs)."""
    rows = []
    for seed in range(N_CIRCUITS):
        qc = quantum_volume(N_QUBITS, depth=N_QUBITS, seed=seed)
        p_ideal = np.abs(Statevector(qc).data) ** 2
        for name, err in NOISE.items():
            for s in strengths:
                rho = layered_noisy_density_matrix(qc, N_QUBITS, err(float(s)))
                p = np.clip(np.real(np.diag(rho)), 0.0, None)
                rows.append((seed, name, float(s), p_ideal, p / p.sum()))
    return rows


def features_from_counts(counts, p_ideal):
    n = counts.sum()
    ph = counts / n
    cdict = {i: int(c) for i, c in enumerate(counts) if c > 0}
    heavy = p_ideal > np.median(p_ideal)
    return {
        "mean_qg_z": mean_qg_z_from_counts(cdict, N_QUBITS),
        "qg_s": joint_qg_s_from_counts(cdict, N_QUBITS),
        "xeb": float(DIM * np.sum(ph * p_ideal) - 1.0),
        "hop": float(ph[heavy].sum()),
    }


def sample_features(dataset, n_shots, repeats=3, seed=0):
    """Sampled feature table: arrays seeds, labels, strengths, and one
    array per feature."""
    rng = np.random.default_rng(seed)
    seeds, labels, strengths = [], [], []
    feats = {f: [] for f in FEATURES}
    for _ in range(repeats):
        for circ, name, s, p_ideal, p in dataset:
            f = features_from_counts(rng.multinomial(n_shots, p), p_ideal)
            seeds.append(circ)
            labels.append(name)
            strengths.append(s)
            for k in FEATURES:
                feats[k].append(f[k])
    return np.array(seeds), np.array(labels), np.array(strengths), {k: np.array(v) for k, v in feats.items()}


def threshold_classifier(x_train, y_train):
    """Best single threshold (either direction) for a boolean label."""
    candidates = np.unique(np.quantile(x_train, np.linspace(0, 1, 401)))
    best = (-1.0, 0.0, 1)
    for t in candidates:
        for sign in (1, -1):
            acc = np.mean((sign * (x_train - t) > 0) == y_train)
            if acc > best[0]:
                best = (acc, t, sign)
    _, t, sign = best
    return lambda x: sign * (x - t) > 0


def t1_detection_accuracy(dataset, n_shots, feature, repeats=3, seed=0, strength_range=None):
    """Held-out accuracy of 'T1 vs unital' from one feature."""
    seeds, labels, strengths, feats = sample_features(dataset, n_shots, repeats, seed)
    y = labels == "T1"
    train = np.isin(seeds, list(TRAIN_SEEDS))
    test = ~train
    if strength_range is not None:
        lo, hi = strength_range
        test = test & (strengths >= lo) & (strengths < hi)
    clf = threshold_classifier(feats[feature][train], y[train])
    return float(np.mean(clf(feats[feature][test]) == y[test]))


def ideal_mean_qg_z_spread():
    vals = [mean_qg_z(Statevector(quantum_volume(N_QUBITS, depth=N_QUBITS, seed=s)).data, N_QUBITS)
            for s in range(N_CIRCUITS)]
    return float(np.std(vals)), float(min(vals)), float(max(vals))


if __name__ == "__main__":
    strong, weak = build_dataset(STRONG), build_dataset(WEAK)
    print("T1 vs unital, held-out circuits (0.67 = always answer 'unital')\n")
    for label, data, shots in [("strong 2-50%", strong, (100, 1000, 10000)),
                               ("weak 0.5-5%", weak, (1000, 10000, 100000))]:
        print(f"{label} per gate")
        for n in shots:
            accs = "  ".join(f"{f}: {t1_detection_accuracy(data, n, f):.2f}" for f in FEATURES)
            print(f"  N={n:>6}  {accs}")
    print("\nstrong noise, N=1000, mean qg_Z by strength:")
    for lo, hi in [(0.0, 0.1), (0.1, 0.25), (0.25, 0.51)]:
        print(f"  [{lo:.2f}, {hi:.2f}): {t1_detection_accuracy(strong, 1000, 'mean_qg_z', strength_range=(lo, hi)):.2f}")
    sd, lo, hi = ideal_mean_qg_z_spread()
    print(f"\nideal mean qg_Z across the 24 circuits: std {sd:.3f}, range [{lo:+.3f}, {hi:+.3f}]")
