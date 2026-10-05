"""
Echo calibration in the weight-2 sector, with and without qang (§107)

§105-§106 calibrated the in-sector errors of weight-1 circuits with 5 echo
circuits. qang.sectors.echo_transfer_matrix and unmix_sector work for any
weight; this study tests weight 2 on five qubits, where the sector has 10
states, so the transfer matrix is 10 x 10 (10 echo circuits) and errors can
move either excitation.

Circuits (5 qubits): X on qubits 0 and 1, an input loader of 4 RBS gates on
the pairs (1,2), (0,3), (2,4), (3,4) with angles that change per input, then
a fixed block of 15 RBS gates (the qang.qml sublayer pattern, 3 layers,
angles drawn once, seed 107). 30 inputs (loader angles drawn with seed 1070),
1000 shots each. Calibration: for each of the 10 weight-2 basis states, the
state, the fixed block and its inverse (an echo), 4000 shots. Truth: the
noiseless output of each input.

Readouts of the outcome distribution:
  without qang   all shots
  qang           weight-2 shots (the filter)
  qang + echo    filtered distribution corrected with M^(1/2)
Figures of merit, mean absolute error against the noiseless values over
inputs: the five qg_Z and the ten qg_ZZ correlations (the weight-2 readout of
§81 and §90).

Backends: IBM fake brisbane, sherbrooke, torino (Aer) and the IonQ simulator
with the aria-1 and forte-1 noise models in the native gate set (§103).

Pre-registered predictions (committed before any noisy run; code checked in
noiseless --mode local only):
  W1  the qg_Z error is lower with qang + echo than with qang alone, on each
      of the five backends.
  W2  the qg_ZZ error is lower with qang + echo than with qang alone, on each
      backend.
  W3  qang + echo cuts the qang qg_Z error by at least 30%, averaged over the
      five backends.
  W4  the qg_Z error is lower with qang than without, on each backend.

python examples/qg_echo_weight2_qg.py --mode fake
python examples/qg_echo_weight2_qg.py --mode ionq_sim --noise aria-1 --jobs jobs.json

Findings:

FINDINGS_PLACEHOLDER
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

N, K = 5, 2
INPUTS, SHOTS, SHOTS_CAL = 30, 1000, 4000
LOADER = [(1, 2), (0, 3), (2, 4), (3, 4)]
PAIRS = list(itertools.combinations(range(N), 2))
SECTOR = [(a, b) for a, b in PAIRS]  # excitation pairs (qang qubit labels)
READOUTS = ("without qang", "qang", "qang + echo")


def q(i):
    return N - 1 - i  # qang qubit i = Qiskit qubit n-1-i


def block_gates(seed=107):
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
        tc = transpile(circuits, backend=sim, optimization_level=1, seed_transpiler=107)
        return [sim.run(c, shots=s, seed_simulator=107 + i).result().get_counts() for i, (c, s) in enumerate(zip(tc, shots))]
    from qiskit_ionq import IonQProvider

    backend = IonQProvider(H.read_ionq_key()).get_backend("simulator", gateset="native")
    tc = transpile(circuits, backend=backend, optimization_level=1, seed_transpiler=107)
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
    loaders = np.random.default_rng(1070).uniform(-math.pi, math.pi, (INPUTS, len(LOADER)))
    circuits = [input_circuit(la, gates) for la in loaders] + [echo_circuit(pr, gates) for pr in SECTOR]
    truth = [Statevector(input_circuit(la, gates, measure=False)).probabilities() for la in loaders]
    shots = [SHOTS] * INPUTS + [SHOTS_CAL] * len(SECTOR)
    counts = get_counts(mode, name, noise, circuits, shots, jobs_path)
    probs = [H.counts_to_probs(c) for c in counts]
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


def verdict(results):
    g = lambda r, m, k: r[f"{m} error, {k}"]  # noqa: E731
    return {
        "W1": all(g(r, "qg_Z", "qang + echo") < g(r, "qg_Z", "qang") for r in results),
        "W2": all(g(r, "qg_ZZ", "qang + echo") < g(r, "qg_ZZ", "qang") for r in results),
        "W3": np.mean([1 - g(r, "qg_Z", "qang + echo") / g(r, "qg_Z", "qang") for r in results]) >= 0.30,
        "W4": all(g(r, "qg_Z", "qang") < g(r, "qg_Z", "without qang") for r in results),
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
    return rs


if __name__ == "__main__":
    main()
