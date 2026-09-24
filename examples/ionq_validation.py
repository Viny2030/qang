"""
qg metrics on IonQ trapped-ion devices (noisy cloud simulator or QPU).

Trapped-ion qubits have T1 of seconds, so the idle-delay T1 probe used for
IBM (examples/nisq_hardware_validation.py) does not apply. What IonQ tests
instead is a cross-platform prediction from RESEARCH_NOTES §10 and §18:

  Experiment 1 (noise benchmark). The 4-qubit, depth-4 Quantum Volume
  circuit (seed 0) used throughout this repository. If the device noise
  is mostly unital (depolarizing / dephasing), as expected for trapped
  ions, then qg_S rises and XEB / HOP fall, while the register's mean
  qg_Z stays close to its ideal value (-0.029), unlike a T1-dominated
  device, where it is pulled towards +1.

  Experiment 2 (cutting cost / qg of a coupling). For RZZ(theta) acting
  on |+>|+>, <X_0> = cos(theta) = qg exactly (qang.knitting). Measured on
  the device, the X_0 expectation reads off the qg of the coupling -- and
  therefore its cutting cost gamma = 1 + 2 sqrt(1 - qg^2) -- with a
  single-qubit measurement.

Modes:
  --mode local      Qiskit Aer, noiseless (no account; for testing).
  --mode ionq_sim   IonQ cloud simulator with a device noise model
                    (--noise aria-1 by default). Free.
  --mode ionq_qpu   IonQ hardware (--device qpu.aria-1 ...). COSTS MONEY:
                    the script prints the circuit sizes and asks for
                    --yes-i-accept-qpu-cost before submitting.

The API key is read from the IONQ_API_KEY environment variable or from a
file named .ionq_key in the repository root (listed in .gitignore). It is
never printed.

Requires: pip install qiskit qiskit-ionq   (qiskit-aer for --mode local)
"""

import argparse
import math
import os
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import quantum_volume
from qiskit.quantum_info import Statevector

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qang.knitting import pauli_rotation_cut_gamma  # noqa: E402
from qang.multiqubit import (  # noqa: E402
    joint_qg_s,
    joint_qg_s_from_counts,
    mean_qg_z,
    mean_qg_z_from_counts,
)

N_QUBITS = 4
RZZ_THETAS = [0.0, math.pi / 4, math.pi / 2, 2 * math.pi / 3]


def read_api_key():
    key = os.environ.get("IONQ_API_KEY")
    if key:
        return key.strip()
    path = os.path.join(ROOT, ".ionq_key")
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as fh:
            return fh.read().strip()
    raise RuntimeError("No IonQ key: set IONQ_API_KEY or create .ionq_key in the repository root.")


def get_backend(mode, device="qpu.aria-1"):
    if mode == "local":
        from qiskit_aer import AerSimulator

        return AerSimulator()
    from qiskit_ionq import IonQProvider

    provider = IonQProvider(token=read_api_key())
    if mode == "ionq_sim":
        return provider.get_backend("simulator")
    if mode == "ionq_qpu":
        return provider.get_backend(device)
    raise ValueError(f"unknown mode {mode!r}")


def run_counts(backend, circuits, shots, noise=None):
    """Counts (dict bitstring -> count) for each circuit, in order."""
    tc = transpile(circuits, backend=backend, optimization_level=1)
    kwargs = {"shots": shots}
    if noise is not None:
        kwargs["noise_model"] = noise
    counts = []
    for qc in tc:  # one job per circuit: works with every provider version
        job = backend.run(qc, **kwargs)
        counts.append(job.result().get_counts())
    return counts


def _to_index(counts):
    out = {}
    for key, c in counts.items():
        k = key.replace(" ", "")
        idx = int(k, 16) if k.startswith("0x") else int(k, 2)
        out[idx] = out.get(idx, 0) + int(c)
    return out


def _mean_qg_z_stderr(counts, total):
    """Standard error of mean qg_Z from the per-shot register average of Z
    (accounts for correlations between qubits)."""
    vals = np.array([1.0 - 2.0 * bin(i).count("1") / N_QUBITS for i in counts])
    w = np.array([counts[i] for i in counts], dtype=float)
    mean = np.sum(w * vals) / total
    var = np.sum(w * (vals - mean) ** 2) / (total - 1)
    return float(math.sqrt(var / total))


def qv_circuits():
    qv = quantum_volume(N_QUBITS, depth=N_QUBITS, seed=0).decompose()
    measured = qv.copy()
    measured.measure_all()
    return qv, measured


def experiment_noise_benchmark(backend, shots=1000, noise=None):
    qv, measured = qv_circuits()
    sv = Statevector.from_instruction(qv)
    p_ideal = np.abs(sv.data) ** 2
    dim = len(p_ideal)
    heavy = p_ideal > np.median(p_ideal)
    ideal = {
        "qg_s": joint_qg_s(sv.data, N_QUBITS),
        "mean_qg_z": mean_qg_z(sv.data, N_QUBITS),
        "hop": float(p_ideal[heavy].sum()),
        "xeb": float(dim * np.sum(p_ideal**2) - 1.0),
    }
    counts = _to_index(run_counts(backend, [measured], shots, noise)[0])
    total = sum(counts.values())
    measured_row = {
        "qg_s": joint_qg_s_from_counts(counts, N_QUBITS),
        "mean_qg_z": mean_qg_z_from_counts(counts, N_QUBITS),
        "hop": sum(c for i, c in counts.items() if heavy[i]) / total,
        "xeb": dim * sum(c * p_ideal[i] for i, c in counts.items()) / total - 1.0,
        "shots": total,
        "mean_qg_z_stderr": _mean_qg_z_stderr(counts, total),
    }
    return ideal, measured_row


def rzz_x0_circuit(theta):
    qc = QuantumCircuit(2, 1)
    qc.h(0)
    qc.h(1)
    qc.rzz(theta, 0, 1)
    qc.h(0)  # measure X on qubit 0
    qc.measure(0, 0)
    return qc


def experiment_rzz_qg(backend, shots=1000, noise=None, thetas=RZZ_THETAS):
    counts = run_counts(backend, [rzz_x0_circuit(t) for t in thetas], shots, noise)
    rows = []
    for theta, c in zip(thetas, counts):
        idx = _to_index(c)
        total = sum(idx.values())
        x0 = (idx.get(0, 0) - idx.get(1, 0)) / total
        qg = max(-1.0, min(1.0, x0))
        rows.append({
            "theta": theta,
            "qg_ideal": math.cos(theta),
            "x0_measured": x0,
            "gamma_ideal": pauli_rotation_cut_gamma(math.cos(theta)),
            "gamma_from_measurement": pauli_rotation_cut_gamma(qg),
        })
    return rows


def experiment_h2_chemistry(backend, shots=4000, noise=None):
    """H2 (Jordan-Wigner, 4 qubits) energy at the exact VQE point, raw,
    with readout mitigation, and with the qg electron-number filter
    (examples/chemistry_qg_symmetry_witness.py). On trapped ions T1 is
    negligible, so the witness mean qg_Z should stay near its ideal 0
    and the filter should matter less than on the IBM T1 model."""
    import chemistry_qg_symmetry_witness as chem

    t = chem.optimal_angle()
    circuits = [chem._measure_circuit(t, "ZZZZ", 0)] + [chem._measure_circuit(t, l, 0) for l in chem.XY_TERMS]
    circuits += chem._calibration_circuits()
    counts = run_counts(backend, circuits, shots, noise)
    P = []
    for c in counts:
        idx = _to_index(c)
        p = np.zeros(16)
        for i, n in idx.items():
            p[i] += n
        P.append(p / p.sum())
    inv = chem._readout_inverse(P[-2], P[-1])
    p_z, p_xy = P[0], {l: P[1 + k] for k, l in enumerate(chem.XY_TERMS)}
    p_z_ro, p_xy_ro = inv @ p_z, {l: inv @ p for l, p in p_xy.items()}
    w = np.array([bin(i).count("1") for i in range(16)])
    return {
        "mean_qg_z": float(np.sum(p_z * (1 - 2 * w / 4))),
        "kept_fraction": float(np.sum(p_z * (w == 2))),
        "hf_error": chem.HF_ENERGY - chem.FCI_ENERGY,
        "raw": chem._energy(p_z, p_xy) - chem.FCI_ENERGY,
        "readout": chem._energy(p_z_ro, p_xy_ro) - chem.FCI_ENERGY,
        "readout_qg_filter": chem._energy(chem.qg_filter(p_z_ro), p_xy_ro) - chem.FCI_ENERGY,
    }


def circuit_sizes():
    _, measured = qv_circuits()
    qv_t = transpile(measured, basis_gates=["rx", "ry", "rz", "cx"], optimization_level=1)
    rzz_t = transpile(rzz_x0_circuit(0.5), basis_gates=["rx", "ry", "rz", "cx"], optimization_level=1)
    two_q = lambda qc: sum(1 for i in qc.data if i.operation.num_qubits == 2)
    one_q = lambda qc: sum(1 for i in qc.data if i.operation.num_qubits == 1 and i.operation.name != "measure")
    return {"qv": (one_q(qv_t), two_q(qv_t)), "rzz": (one_q(rzz_t), two_q(rzz_t))}


def main(argv=None):
    ap = argparse.ArgumentParser(description="qg metrics on IonQ")
    ap.add_argument("--mode", choices=["local", "ionq_sim", "ionq_qpu"], default="local")
    ap.add_argument("--noise", default="aria-1", help="IonQ simulator noise model (ionq_sim only)")
    ap.add_argument("--device", default="qpu.aria-1", help="IonQ QPU name (ionq_qpu only)")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--yes-i-accept-qpu-cost", action="store_true")
    ap.add_argument("--only", choices=["all", "qv", "rzz", "h2"], default="all")
    args = ap.parse_args(argv)

    sizes = circuit_sizes()
    if args.mode == "ionq_qpu" and not args.yes_i_accept_qpu_cost:
        print("QPU run NOT submitted. Circuits (1-qubit, 2-qubit gates):")
        print(f"  QV 4 qubits: {sizes['qv']} x {args.shots} shots (1 circuit)")
        print(f"  RZZ probe:   {sizes['rzz']} x {args.shots} shots ({len(RZZ_THETAS)} circuits)")
        print("Check the price in the IonQ console, then rerun with --yes-i-accept-qpu-cost.")
        return
    noise = args.noise if args.mode == "ionq_sim" else None
    backend = get_backend(args.mode, args.device)
    label = f"{args.mode}" + (f" (noise model {noise})" if noise else "")
    print(f"Backend: {label}")

    if args.only in ("all", "qv"):
        ideal, meas = experiment_noise_benchmark(backend, args.shots, noise)
        print("\nExperiment 1: 4-qubit QV (seed 0)")
        print(f"  {'':10s} {'qg_S':>7} {'mean_qg_Z':>10} {'HOP':>7} {'XEB':>8}")
        print(f"  {'ideal':10s} {ideal['qg_s']:7.4f} {ideal['mean_qg_z']:+10.4f} {ideal['hop']:7.4f} {ideal['xeb']:+8.4f}")
        print(f"  {'measured':10s} {meas['qg_s']:7.4f} {meas['mean_qg_z']:+10.4f} {meas['hop']:7.4f} {meas['xeb']:+8.4f}")
        print(f"  shift in mean qg_Z: {meas['mean_qg_z'] - ideal['mean_qg_z']:+.4f}  "
              f"(standard error {meas['mean_qg_z_stderr']:.4f})")
    if args.only in ("all", "rzz"):
        print("\nExperiment 2: RZZ(theta) on |+>|+>, <X_0> should equal qg = cos(theta)")
        print(f"  {'theta':>6} {'qg ideal':>9} {'<X_0> meas':>11} {'gamma ideal':>12} {'gamma from meas':>16}")
        for r in experiment_rzz_qg(backend, args.shots, noise):
            print(f"  {r['theta']:6.3f} {r['qg_ideal']:+9.4f} {r['x0_measured']:+11.4f} "
                  f"{r['gamma_ideal']:12.4f} {r['gamma_from_measurement']:16.4f}")
    if args.only in ("all", "h2"):
        r = experiment_h2_chemistry(backend, args.shots, noise)
        print("\nExperiment 3: H2 energy error vs FCI (mHa)")
        print(f"  witness mean qg_Z {r['mean_qg_z']:+.4f} (ideal 0), shots kept by the filter {r['kept_fraction']:.2f}")
        print(f"  classical HF {1e3 * r['hf_error']:.1f} | raw {1e3 * r['raw']:.1f} | +readout {1e3 * r['readout']:.1f}"
              f" | +readout+qg filter {1e3 * r['readout_qg_filter']:.1f}")

if __name__ == "__main__":
    main()
