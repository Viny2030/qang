"""
The radius deficit on device noise models, with and without qang (§100)

§98 showed, under simulated T1, that the filtered radius deficit 1 - qg_Z^2
of a weight-conserving circuit equals the noiseless one-tangle of each qubit,
and that the raw readout reports false entanglement on product states. A
device adds what §98 did not have: gate errors, readout errors, crosstalk,
and time that the compiled RBS gates spend outside the weight-1 sector, where
a decay can be rotated back in and pass the filter (§68, §85). So the filter
is not exact here; this study measures how much of the §98 result survives.

Circuits (5 qubits, the §85 model E: X on qubit 0, a 4-RBS loader, 15
trained RBS gates; qang.qml.WeightQNN weight 1):
  trained   the 30 iris test inputs of §85; truth: the noiseless deficit
            1 - qg_Z^2 of every qubit (pure state, so the one-tangle).
  echo      X on qubit j (j = 0..4), the 15 trained RBS gates and their
            inverse: a product basis state again, truth deficit 0, with every
            gate of the forward and backward circuit noisy.
Each circuit 1000 shots, read twice from the same shots: with qang (only
weight-1 outcomes) and without qang (all outcomes). Estimator per qubit:
1 - (N q^2 - 1)/(N - 1) (qang.statistics.qg2_unbiased), N the shots used.
Backends: IBM fake brisbane, sherbrooke, torino (Aer); IonQ cloud simulator
with the aria-1 and forte-1 noise models (free).

Pre-registered predictions (committed before any noisy run; code checked
in --mode local, noiseless, only):
  R1  trained circuits: the mean absolute error of the deficit (over inputs
      and qubits) is lower with qang than without, on every backend.
  R2  trained circuits: with qang that error is at most 0.10 on every
      backend.
  R3  echo circuits: on the initially excited qubit the false deficit with
      qang is less than half of the false deficit without qang, on every
      backend.
  R4  echo circuits: without qang the false deficit of the excited qubit is
      above 0.2 on every backend (decay alone produces apparent
      entanglement).

Modes: --mode local (noiseless Aer), fake (the three IBM fake backends),
ionq_sim (--noise aria-1 / forte-1; resumable with --jobs). Uses the
installed library (pip install "qang>=0.6.4") and the §85 script's model and
circuits. Needs qiskit, qiskit-aer, qiskit-ibm-runtime (fake), qiskit-ionq
(IonQ) and scikit-learn.

Findings:

Trained circuits (30 iris inputs, true mean deficit 0.37) and echo circuits
(5 excited-qubit product states), 1000 shots each:

  backend          deficit error, trained       kept    false deficit of the       2q gates
                   with qang / without           shots   excited qubit (echo),      trained /
                                                         with qang / without        echo
  fake_brisbane    0.079 / 0.257   (3.2x)       0.69    0.66 / 0.77                62 / 108
  fake_sherbrooke  0.055 / 0.221   (4.0x)       0.72    0.49 / 0.67                62 / 108
  fake_torino      0.051 / 0.197   (3.8x)       0.74    0.37 / 0.56                62 / 108
  IonQ aria-1      0.074 / 0.245   (3.3x)       0.74    0.48 / 0.62                38 / 60
  IonQ forte-1     0.082 / 0.272   (3.3x)       0.70    0.54 / 0.71                38 / 60

  * R1, R2, R4 pass on all five backends; R3 fails on all five.
  * R1, R2: on the trained circuits the filtered deficit errs by 0.05-0.08
    (true values around 0.37), 3.2-4.0x less than the raw one. The §98
    result survives device noise as an error reduction, not as exactness.
  * R4: without qang every backend reports a false deficit of 0.56-0.77 on
    the excited qubit of a product state.
  * R3 FAILS: the filter removes only 15-33% of that false deficit (0.37-0.66
    remain). The filter discards shots that left the weight-1 sector; the
    remaining error moves the excitation inside the sector (gate errors,
    decays inside the compiled RBS gates rotated back into weight 1, §85),
    10-21% of the kept shots on the excited qubit. Near a pole the deficit
    amplifies that misplacement: 1 - qg_Z^2 = 4 eps (1 - eps) for a fraction
    eps of misplaced shots, so 10% of them already read as 0.36.
  Verdict. On device noise models the filtered radius deficit is a good
  estimate where the deficit is large (entangled qubits: 3-4x less error),
  but it is not a reliable test of "no entanglement": in-sector errors, which
  the filter cannot see, make product states look entangled. A deficit near
  0 needs an error model or an echo calibration like the one used here.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_hardware_qg as H  # noqa: E402
from qang.sectors import filter_distribution  # noqa: E402

N = H.N
FAKE = ("fake_brisbane", "fake_sherbrooke", "fake_torino")


def echo_circuit(model, j):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate

    q = lambda i: N - 1 - i  # noqa: E731
    qc = QuantumCircuit(N)
    qc.x(q(j))
    th = model.params_[: model.n_theta]
    gates = []
    k = 0
    for _ in range(model.layers):
        for pairs in model.sublayers:
            for a, b in pairs:
                gates.append((th[k], a, b))
                k += 1
    for t, a, b in gates:
        qc.append(UnitaryGate(H.rbs_matrix(t), label="RBS"), [q(a), q(b)])
    qc.barrier()
    for t, a, b in reversed(gates):
        qc.append(UnitaryGate(H.rbs_matrix(-t), label="RBS"), [q(a), q(b)])
    qc.measure_all()
    return qc


def deficits(p, zsign, shots):
    """Per-qubit 1 - unbiased qg^2 from a probability vector over 2^N outcomes
    (frequencies of ``shots`` shots), with and without the weight-1 filter."""
    out = {}
    for label, dist, n in (("without qang", p, shots),
                           ("with qang", filter_distribution(p, N, 1)[0], filter_distribution(p, N, 1)[1] * shots)):
        q = dist @ zsign
        n = max(int(round(n)), 2)
        out[label] = 1.0 - (n * q * q - 1.0) / (n - 1.0)
    out["kept"] = filter_distribution(p, N, 1)[1]
    return out


def evaluate(model, Xte, probs_trained, probs_echo, shots):
    truth = 1.0 - (model.probs(model.params_[: model.n_theta], model.encode(Xte)) @ model.zsign) ** 2
    err = {"with qang": [], "without qang": []}
    kept = []
    for p, t in zip(probs_trained, truth):
        d = deficits(p, model.zsign, shots)
        kept.append(d["kept"])
        for k in err:
            err[k].append(np.abs(d[k] - t))
    echo = {"with qang": [], "without qang": []}
    for j, p in enumerate(probs_echo):
        d = deficits(p, model.zsign, shots)
        for k in echo:
            echo[k].append(float(d[k][j]))
    return {
        "trained: deficit error with qang": float(np.mean(err["with qang"])),
        "trained: deficit error without qang": float(np.mean(err["without qang"])),
        "trained: mean true deficit": float(np.mean(truth)),
        "kept fraction": float(np.mean(kept)),
        "echo: false deficit of the excited qubit, with qang": echo["with qang"],
        "echo: false deficit of the excited qubit, without qang": echo["without qang"],
    }


def verdict(r):
    w, wo = r["echo: false deficit of the excited qubit, with qang"], r["echo: false deficit of the excited qubit, without qang"]
    return {
        "R1": r["trained: deficit error with qang"] < r["trained: deficit error without qang"],
        "R2": r["trained: deficit error with qang"] <= 0.10,
        "R3": float(np.mean(w)) < 0.5 * float(np.mean(wo)),
        "R4": float(np.mean(wo)) > 0.2,
    }


def run_backend(mode, name, noise, shots, jobs):
    model, Xte, _ = H.trained_model()
    H.check_circuits(model, Xte)
    circuits = [H.build_circuit(model, x) for x in Xte] + [echo_circuit(model, j) for j in range(N)]
    backend = H.get_backend(mode, name)
    counts, tc = H.run_counts(backend, mode, circuits, shots, noise, jobs_path=jobs)
    probs = [H.counts_to_probs(c) for c in counts]
    r = evaluate(model, Xte, probs[: len(Xte)], probs[len(Xte):], shots)
    two_q = [sum(1 for i in c.data if i.operation.num_qubits == 2) for c in tc]
    r["two-qubit gates: trained / echo"] = [float(np.mean(two_q[: len(Xte)])), float(np.mean(two_q[len(Xte):]))]
    r["backend"] = name if mode == "fake" else (f"ionq simulator {noise}" if mode == "ionq_sim" else mode)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "fake", "ionq_sim"], default="fake")
    ap.add_argument("--noise", default="aria-1")
    ap.add_argument("--shots", type=int, default=H.SHOTS)
    ap.add_argument("--jobs", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    names = FAKE if args.mode == "fake" else [None]
    results = []
    for name in names:
        r = run_backend(args.mode, name, args.noise, args.shots, args.jobs)
        v = verdict(r)
        r["verdict"] = v
        results.append(r)
        print(json.dumps(r, indent=1))
        print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()), flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
    return results


if __name__ == "__main__":
    main()
