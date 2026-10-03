"""
The qg-filtered QNN on hardware: IBM (fake or real) and IonQ (simulator or
QPU), with and without qang (§85)

§75-§82 were exact simulations. This script takes the weight-1 model E
(qang.qml.WeightQNN, 5 qubits, readout over all qg_Z), trains it on a
simulator without noise, compiles it to a Qiskit circuit and runs the test
set on a backend. Every input is read twice from the same shots: with qang
(only the weight-1 outcomes, qang.sectors.filter_distribution) and without
it (all outcomes).

Circuit: X on qubit 0, a cascade of 4 RBS gates loads the unary amplitudes
v = (x, 1)/|(x, 1)| (signed, exactly), then the 15 trained RBS gates.
RBS(t) on (a, b) is the real Givens rotation |10> -> cos t |10> + sin t |01>,
compiled by the transpiler. The compiled circuit is checked against
WeightQNN.probs (noiseless statevector) before anything is submitted.

What a device adds beyond §75-§82: gate (depolarizing-like) errors, readout
errors, crosstalk, and the time an RBS decomposition spends outside the
weight-1 sector (§68), where a decay can be rotated back into the sector and
pass the filter. So the filter is not expected to be exact here.

Modes:
  --mode local      Qiskit Aer, noiseless sampling (no account).
  --mode fake       calibration-based IBM fake backend (fake_brisbane by
                    default; qiskit-ibm-runtime, local Aer simulation).
  --mode ibm        a real IBM device through a saved QiskitRuntimeService
                    account (open plan: monthly free minutes). Needs
                    --yes-i-run-on-hardware.
  --mode ionq_sim   IonQ cloud simulator with a device noise model (free).
  --mode ionq_qpu   IonQ hardware. COSTS MONEY: prints the circuit sizes and
                    needs --yes-i-accept-qpu-cost.
Credentials are never read from or written to the code: IBM uses the
account saved with QiskitRuntimeService.save_account(...); IonQ reads
IONQ_API_KEY or the git-ignored .ionq_key. Nothing is printed.

Pre-registered predictions (committed before any backend run; evaluated
for every backend that is run, each run reported):
  K1  accuracy with qang >= accuracy without qang (same model, same shots).
  K2  accuracy with qang within 5 points of the noiseless accuracy.
  K3  the qg_Z error, mean over inputs and qubits of
      |qg_Z measured - qg_Z ideal|, is smaller with qang than without.
Setting: iris (versicolor vs virginica), split seed 8500, 30 test inputs,
1000 shots each, model trained with seed 8502 (120 epochs).

Uses the installed library (pip install "qang>=0.6.0"). Needs scikit-learn
and qiskit (+ qiskit-aer, qiskit-ibm-runtime for fake/ibm, qiskit-ionq for
IonQ).

Findings (python examples/qnn_hardware_qg.py --mode fake):

FINDINGS_PLACEHOLDER
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402
from qang.sectors import filter_distribution  # noqa: E402

N = 5
SPLIT_SEED = 8500
TRAIN_SEED = 8502
SHOTS = 1000


# --------------------------------------------------------------------- #
# model and circuits
# --------------------------------------------------------------------- #
def trained_model(epochs=120):
    Xa, ya, use_pca = Q.load("iris", np.random.default_rng(SPLIT_SEED))
    Xtr, Xte, ytr, yte = Q.split_prepare(Xa, ya, use_pca, SPLIT_SEED)
    m = WeightQNN(N, 1).fit(Xtr, ytr, epochs, seed=TRAIN_SEED)
    return m, Xte, yte


def rbs_matrix(t):
    """4x4 RBS for the qubit list [a, b] in Qiskit's little-endian order
    (index = bit of a + 2 * bit of b): |a=1, b=0> -> cos t |10> + sin t |01>."""
    c, s = math.cos(t), math.sin(t)
    U = np.eye(4)
    # Qiskit index = bit0 (first qubit in the list) + 2 bit1. List = [a, b]:
    # |a=1,b=0> -> index 1, |a=0,b=1> -> index 2
    U[1, 1], U[2, 2], U[2, 1], U[1, 2] = c, c, s, -s
    return U


def loader_angles(v):
    """Cascade RBS(0,1), RBS(1,2), ..., RBS(n-2,n-1) from |e_0> to sum v_i |e_i>."""
    v = np.asarray(v, float)
    n = len(v)
    ang = []
    for k in range(n - 2):
        ang.append(math.atan2(np.linalg.norm(v[k + 1:]), v[k]))
    ang.append(math.atan2(v[n - 1], v[n - 2]))
    # the cascade multiplies by sin of every earlier angle; norms are non-negative,
    # so the signs of v_k (k < n-2) sit in the cos factors and the last pair in atan2
    return ang


def build_circuit(model, x, measure=True):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate

    q = lambda i: N - 1 - i  # noqa: E731  (qang qubit i = Qiskit qubit n-1-i)
    qc = QuantumCircuit(N)
    v = np.concatenate([x, [1.0]])
    v = v / np.linalg.norm(v)
    qc.x(q(0))
    for k, t in enumerate(loader_angles(v)):
        qc.append(UnitaryGate(rbs_matrix(t), label="RBS"), [q(k), q(k + 1)])
    th = model.params_[: model.n_theta]
    k = 0
    for _ in range(model.layers):
        for pairs in model.sublayers:
            for a, b in pairs:
                qc.append(UnitaryGate(rbs_matrix(th[k]), label="RBS"), [q(a), q(b)])
                k += 1
    if measure:
        qc.measure_all()
    return qc


def check_circuits(model, X, tol=1e-9):
    from qiskit.quantum_info import Statevector

    ref = model.probs(model.params_[: model.n_theta], model.encode(X))
    err = 0.0
    for x, p in zip(X, ref):
        sv = Statevector(build_circuit(model, x, measure=False)).probabilities()
        err = max(err, float(np.abs(sv - p).max()))
    if err > tol:
        raise RuntimeError(f"compiled circuit differs from WeightQNN.probs by {err:.2e}")
    return err


# --------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------- #
def read_ionq_key():
    k = os.environ.get("IONQ_API_KEY")
    if k:
        return k.strip()
    path = os.path.join(ROOT, ".ionq_key")
    if os.path.exists(path):
        return open(path).read().strip()
    raise RuntimeError("No IonQ key: set IONQ_API_KEY or create .ionq_key in the repository root.")


def get_backend(mode, name=None):
    if mode == "local":
        from qiskit_aer import AerSimulator

        return AerSimulator()
    if mode == "fake":
        from qiskit_ibm_runtime import fake_provider

        name = name or "fake_brisbane"
        cls = getattr(fake_provider, "Fake" + name.split("_", 1)[1].capitalize())
        return cls()
    if mode == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        return service.backend(name) if name else service.least_busy(operational=True, simulator=False)
    if mode in ("ionq_sim", "ionq_qpu"):
        from qiskit_ionq import IonQProvider

        provider = IonQProvider(read_ionq_key())
        return provider.get_backend("simulator" if mode == "ionq_sim" else (name or "qpu.aria-1"))
    raise ValueError(mode)


def run_counts(backend, mode, circuits, shots, noise="aria-1", seed=11):
    from qiskit import transpile

    if mode in ("local", "fake"):
        from qiskit_aer import AerSimulator

        sim = AerSimulator.from_backend(backend) if mode == "fake" else backend
        tc = transpile(circuits, backend=sim, optimization_level=1, seed_transpiler=seed)
        res = sim.run(tc, shots=shots, seed_simulator=seed).result()
        return [res.get_counts(i) for i in range(len(tc))], tc
    tc = transpile(circuits, backend=backend, optimization_level=1, seed_transpiler=seed)
    if mode == "ibm":
        from qiskit_ibm_runtime import SamplerV2

        job = SamplerV2(mode=backend).run(tc, shots=shots)
        print("IBM job id:", job.job_id(), flush=True)
        res = job.result()
        return [r.data.meas.get_counts() for r in res], tc
    kw = {"shots": shots}
    if mode == "ionq_sim":
        kw["noise_model"] = noise
    out = []
    for c in tc:
        job = backend.run(c, **kw)
        print("IonQ job id:", job.job_id(), flush=True)
        out.append(job.result().get_counts())
    return out, tc


def counts_to_probs(counts):
    p = np.zeros(2**N)
    tot = sum(counts.values())
    for bits, c in counts.items():
        p[int(bits.replace(" ", ""), 2)] += c / tot
    return p


# --------------------------------------------------------------------- #
def evaluate(model, X, y, probs_list):
    th = model.params_
    ideal = model.qg_z(model.probs(th[: model.n_theta], model.encode(X)), qang=False)
    out = {}
    for label, q in (("with qang", True), ("without qang", False)):
        feats = np.array([model.qg_z(p, qang=q)[0] for p in probs_list])
        z = feats @ th[model.n_theta:model.n_theta + model.n_head] + th[-1]
        out[label] = {"accuracy": float(np.mean((z > 0).astype(int) == y)),
                      "qg_Z error": float(np.mean(np.abs(feats - ideal)))}
    out["kept fraction"] = float(np.mean([filter_distribution(p, N, 1)[1] for p in probs_list]))
    out["noiseless accuracy"] = model.score(X, y)
    out["difference"] = out["with qang"]["accuracy"] - out["without qang"]["accuracy"]
    return out


def verdict(r):
    return {
        "K1": r["with qang"]["accuracy"] >= r["without qang"]["accuracy"],
        "K2": r["with qang"]["accuracy"] >= r["noiseless accuracy"] - 0.05,
        "K3": r["with qang"]["qg_Z error"] < r["without qang"]["qg_Z error"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="qg-filtered QNN on hardware")
    ap.add_argument("--mode", choices=["local", "fake", "ibm", "ionq_sim", "ionq_qpu"], default="fake")
    ap.add_argument("--backend", default=None, help="fake_brisbane / IBM device name / IonQ QPU name")
    ap.add_argument("--noise", default="aria-1", help="IonQ simulator noise model")
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--yes-i-run-on-hardware", action="store_true")
    ap.add_argument("--yes-i-accept-qpu-cost", action="store_true")
    ap.add_argument("--out", default=None, help="write the result as JSON")
    args = ap.parse_args(argv)

    model, Xte, yte = trained_model(args.epochs)
    err = check_circuits(model, Xte)
    circuits = [build_circuit(model, x) for x in Xte]
    print(f"{len(circuits)} circuits, {args.shots} shots each; compiled circuits match WeightQNN.probs to {err:.1e}")
    if args.mode == "ibm" and not args.yes_i_run_on_hardware:
        print("Real IBM device: rerun with --yes-i-run-on-hardware (uses your account's QPU minutes).")
        return None
    backend = get_backend(args.mode, args.backend)
    if args.mode == "ionq_qpu" and not args.yes_i_accept_qpu_cost:
        from qiskit import transpile

        tc = transpile(circuits, backend=backend, optimization_level=1)
        two_q = sum(sum(1 for i in c.data if i.operation.num_qubits == 2) for c in tc)
        print(f"IonQ QPU: {len(tc)} circuits x {args.shots} shots, {two_q} two-qubit gates in total.")
        print("Check the price in the IonQ console, then rerun with --yes-i-accept-qpu-cost.")
        return None
    counts, tc = run_counts(backend, args.mode, circuits, args.shots, args.noise)
    probs = [counts_to_probs(c) for c in counts]
    r = evaluate(model, Xte, yte, probs)
    r["backend"] = getattr(backend, "name", str(backend))
    r["mode"] = args.mode
    r["two-qubit gates per circuit"] = float(np.mean([sum(1 for i in c.data if i.operation.num_qubits == 2) for c in tc]))
    v = verdict(r)
    print(json.dumps(r, indent=1))
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    if args.out:
        json.dump({"result": r, "verdict": v}, open(args.out, "w"), indent=1)
    return r, v


if __name__ == "__main__":
    main()
