"""
Echo calibration at 8 qubits, weight 2: a 28 x 28 transfer matrix (§115)

Section 107 calibrated the in-sector errors of a 5-qubit weight-2 circuit
(10 sector states) with 10 echo circuits. Here the same pipeline runs on 8
qubits at weight 2: 28 sector states, 28 echo circuits (X on two qubits, the
fixed RBS block of WeightQNN(8, 2) with random angles, its inverse). Inputs:
X on qubits 0 and 1, an 8-RBS loader whose angles change per input, then the
fixed block. 30 inputs at 1000 shots; echoes at 4000 shots each. Readouts
from the same shots: without qang (raw), qang (filter to weight 2) and
qang + echo (filter, then NNLS on M^1/2, qang.sectors.unmix_sector). Errors
are the mean |qg_Z - exact| over the 8 qubits and |qg_ZZ - exact| over the
28 pairs.

Backends: the IBM fake backends brisbane, sherbrooke and torino (Aer with
their calibrated noise), and the IonQ cloud simulator with the aria-1 and
forte-1 noise models in the native gate set (section 103).

Pre-registered predictions (committed before any noisy run; code checked in
noiseless --mode local only):
  X1  qg_Z error with qang + echo < with qang, on every backend.
  X2  qg_ZZ error with qang + echo < with qang, on every backend.
  X3  the echo removes at least 30% of the filtered qg_Z error on average.
  X4  qg_Z error with qang < without qang, on every backend.
  X5  that average reduction is smaller than at 5 qubits (40%, section 107):
      28 columns estimated from shots and deeper circuits leave less to
      recover.

python examples/qg_echo_8q_qg.py --mode fake --out fake.json
python examples/qg_echo_8q_qg.py --mode ionq_sim --noise aria-1 --jobs jobs_aria.json

Findings (30 inputs at 1000 shots, 28 echoes at 4000 shots, per backend):

  backend           qg_Z error (raw / qang / qang + echo)   qg_ZZ error (raw / qang / qang + echo)   echo removes
  fake_brisbane     0.185 / 0.081 / 0.055                   0.180 / 0.091 / 0.062                    32% / 32%
  fake_sherbrooke   0.162 / 0.060 / 0.041                   0.156 / 0.069 / 0.049                    31% / 29%
  fake_torino       0.123 / 0.046 / 0.030                   0.127 / 0.053 / 0.037                    34% / 31%
  IonQ aria-1       0.145 / 0.061 / 0.037                   0.146 / 0.070 / 0.045                    38% / 35%
  IonQ forte-1      0.167 / 0.065 / 0.047                   0.161 / 0.075 / 0.054                    27% / 29%

  * X1-X5 all pass.
  * The echo removes 32% of the qg_Z error left by the filter on average
    (27-38%) and 31% of the qg_ZZ error (29-35%); filter + echo is 3.4-4.1x
    below the raw readout.
  * The 8-qubit sector is much dirtier than the 5-qubit one: the echo
    diagonal is 0.50-0.80 (only half to four fifths of the kept shots of an
    echo come back to the prepared state), and the recovered share falls
    from 40% (section 107) to 32% (X5), the forte-1 model being the lowest
    (27%).
  Verdict. The echo calibration scales to a 28 x 28 transfer matrix: with 28
  extra circuits it still removes about a third of the error the filter
  leaves, on every backend, but a smaller share than at 5 qubits.
"""

import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_hardware_qg as H  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402
from qang.sectors import echo_transfer_matrix, unmix_sector  # noqa: E402

N, K = 8, 2
INPUTS, SHOTS, SHOTS_CAL = 30, 1000, 4000
LOADER = [(1, 2), (0, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7), (1, 4)]
PAIRS = list(itertools.combinations(range(N), 2))
SECTOR = [(a, b) for a, b in PAIRS]  # excitation pairs (qang qubit labels)
READOUTS = ("without qang", "qang", "qang + echo")


def q(i):
    return N - 1 - i  # qang qubit i = Qiskit qubit n-1-i


def block_gates(seed=115):
    m = WeightQNN(N, K)
    th = np.random.default_rng(seed).uniform(-math.pi, math.pi, m.n_theta)
    out, k = [], 0
    for _ in range(m.layers):
        for pairs in m.sublayers:
            for a, b in pairs:
                out.append((th[k], a, b))
                k += 1
    return out


def rbs(qc, t, a, b):
    from qiskit.circuit.library import UnitaryGate

    qc.append(UnitaryGate(H.rbs_matrix(t), label="RBS"), [q(a), q(b)])


def input_circuit(loader_angles, gates, measure=True):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(N)
    qc.x(q(0))
    qc.x(q(1))
    for t, (a, b) in zip(loader_angles, LOADER):
        rbs(qc, t, a, b)
    for t, a, b in gates:
        rbs(qc, t, a, b)
    if measure:
        qc.measure_all()
    return qc


def echo_circuit(pair, gates):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(N)
    for i in pair:
        qc.x(q(i))
    for t, a, b in gates:
        rbs(qc, t, a, b)
    qc.barrier()
    for t, a, b in reversed(gates):
        rbs(qc, -t, a, b)
    qc.measure_all()
    return qc


def state_index(pair):
    return sum(1 << (N - 1 - i) for i in pair)


def counts_to_probs(counts):
    p = np.zeros(2**N)
    tot = sum(counts.values())
    for bits, c in counts.items():
        p[int(bits.replace(" ", ""), 2)] += c / tot
    return p


def zsign():
    idx = np.arange(2**N)
    bits = np.array([(idx >> (N - 1 - i)) & 1 for i in range(N)]).T
    return 1.0 - 2.0 * bits


def features(p):
    z = p @ ZS
    zz = np.array([p @ (ZS[:, a] * ZS[:, b]) for a, b in PAIRS])
    return z, zz


ZS = zsign()


def get_counts(mode, name, noise, circuits, shots, jobs_path):
    from qiskit import transpile

    if mode in ("local", "fake"):
        from qiskit_aer import AerSimulator

        backend = H.get_backend(mode, name)
        sim = AerSimulator.from_backend(backend) if mode == "fake" else backend
        tc = transpile(circuits, backend=sim, optimization_level=1, seed_transpiler=115)
        return [sim.run(c, shots=s, seed_simulator=115 + i).result().get_counts() for i, (c, s) in enumerate(zip(tc, shots))]
    from qiskit_ionq import IonQProvider

    backend = IonQProvider(H.read_ionq_key()).get_backend("simulator", gateset="native")
    tc = transpile(circuits, backend=backend, optimization_level=1, seed_transpiler=115)
    ids = json.load(open(jobs_path)) if jobs_path and os.path.exists(jobs_path) else []
    for k in range(len(ids), len(tc)):
        ids.append(backend.run(tc[k], shots=shots[k], noise_model=noise).job_id())
        if jobs_path:
            json.dump(ids, open(jobs_path, "w"))
    cache = jobs_path + ".counts" if jobs_path else None
    out = json.load(open(cache)) if cache and os.path.exists(cache) else []
    for jid, s in list(zip(ids, shots))[len(out):]:
        for attempt in range(20):
            try:
                probs = backend.retrieve_job(jid).get_probabilities()
                out.append({k: int(round(v * s)) for k, v in probs.items()})
                break
            except Exception:
                if attempt == 19:
                    raise
                time.sleep(15)
        if cache:
            json.dump(out, open(cache, "w"))
    return out


def run_backend(mode, name, noise, jobs_path):
    from qiskit.quantum_info import Statevector

    gates = block_gates()
    loaders = np.random.default_rng(1150).uniform(-math.pi, math.pi, (INPUTS, len(LOADER)))
    circuits = [input_circuit(la, gates) for la in loaders] + [echo_circuit(pr, gates) for pr in SECTOR]
    truth = [Statevector(input_circuit(la, gates, measure=False)).probabilities() for la in loaders]
    shots = [SHOTS] * INPUTS + [SHOTS_CAL] * len(SECTOR)
    counts = get_counts(mode, name, noise, circuits, shots, jobs_path)
    probs = [counts_to_probs(c) for c in counts]
    M = echo_transfer_matrix(probs[INPUTS:], N, K, prepared=[state_index(pr) for pr in SECTOR])
    err = {k: {"qg_Z": [], "qg_ZZ": []} for k in READOUTS}
    for p, t in zip(probs[:INPUTS], truth):
        tz, tzz = features(t)
        from qang.sectors import filter_distribution

        for k, d in (("without qang", p), ("qang", filter_distribution(p, N, K)[0]),
                     ("qang + echo", unmix_sector(p, M, N, K, power=0.5))):
            z, zz = features(d)
            err[k]["qg_Z"].append(np.mean(np.abs(z - tz)))
            err[k]["qg_ZZ"].append(np.mean(np.abs(zz - tzz)))
    r = {f"{m} error, {k}": float(np.mean(v[m])) for k, v in err.items() for m in ("qg_Z", "qg_ZZ")}
    r["echo diagonal"] = [float(x) for x in np.diag(M)]
    r["backend"] = name if mode == "fake" else (f"ionq simulator {noise} (native)" if mode == "ionq_sim" else mode)
    return r


REDUCTION_5Q = 0.40  # mean qg_Z error reduction of the echo at 5 qubits (section 107)


def reduction(r, m="qg_Z"):
    return 1 - r[f"{m} error, qang + echo"] / r[f"{m} error, qang"]


def verdict(results):
    g = lambda r, m, k: r[f"{m} error, {k}"]  # noqa: E731
    red = float(np.mean([reduction(r) for r in results]))
    return {
        "X1": all(g(r, "qg_Z", "qang + echo") < g(r, "qg_Z", "qang") for r in results),
        "X2": all(g(r, "qg_ZZ", "qang + echo") < g(r, "qg_ZZ", "qang") for r in results),
        "X3": red >= 0.30,
        "X4": all(g(r, "qg_Z", "qang") < g(r, "qg_Z", "without qang") for r in results),
        "X5": red < REDUCTION_5Q,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "fake", "ionq_sim"], default="fake")
    ap.add_argument("--noise", default="aria-1")
    ap.add_argument("--jobs", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    names = ("fake_brisbane", "fake_sherbrooke", "fake_torino") if args.mode == "fake" else [None]
    rs = []
    for n in names:
        r = run_backend(args.mode, n, args.noise, args.jobs)
        rs.append(r)
        print(json.dumps(r), flush=True)
    if args.out:
        json.dump(rs, open(args.out, "w"), indent=1)
    if args.mode != "local":
        for r in rs:
            print(r["backend"], {m: round(reduction(r, m), 3) for m in ("qg_Z", "qg_ZZ")})
    return rs


if __name__ == "__main__":
    main()
