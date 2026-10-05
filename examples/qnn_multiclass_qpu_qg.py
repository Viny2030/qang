"""
Multiclass QNN with filter and echo on IonQ hardware: the run package (§116)

Prepares, and only on explicit approval submits, the §113 experiment on an
IonQ QPU: MultiClassQNN (5 qubits, 5-class digits, head readout, trained
noiselessly), N_TEST test inputs at SHOTS shots and 5 echo circuits (one per
class qubit) at SHOTS_CAL shots. Every input is read from the same shots
without qang (raw), with qang (filter to weight 1) and with qang + echo
(filter, then NNLS on M^1/2). Circuits are sent in IonQ's native gate set,
so the compiler cannot remove the echo's identity (section 103).

Modes:
  --mode plan        (default) build and transpile the circuits, check them
                     against the model, print circuits, shots and gate
                     counts. Submits nothing.
  --mode rehearsal   the same circuits on the IonQ cloud simulator with a
                     device noise model (--noise forte-1 by default). Free.
  --mode qpu         IonQ hardware (--backend qpu.forte-1 by default).
                     COSTS MONEY: refused unless --yes-i-accept-qpu-cost is
                     given. Jobs are saved to --jobs, so a rerun resumes
                     submission and retrieval; while jobs are queued the
                     script reports how many are done and stops.

python examples/qnn_multiclass_qpu_qg.py --mode plan
python examples/qnn_multiclass_qpu_qg.py --mode rehearsal --jobs jobs_rehearsal.json
python examples/qnn_multiclass_qpu_qg.py --mode qpu --jobs jobs_qpu.json --yes-i-accept-qpu-cost

Pre-registered predictions for the QPU run (committed before any hardware
submission; judged on the first complete QPU run):
  H1  decisions that differ from the noiseless model: fewer with qang than
      without qang.
  H2  decisions that differ: no more with qang + echo than with qang.
  H3  accuracy with qang >= accuracy without qang.
  H4  the echo diagonal has a mean below 0.95 (in-sector errors exist on the
      device, as on its noise model).

Plan (qpu.forte-1, native gate set): 55 circuits, 60000 shots, 2200
two-qubit gates in total (about 37 per input circuit, 73 per echo).

Rehearsal (IonQ cloud simulator, forte-1 noise model, native; not the
hardware run, so H1-H4 are not judged on it): noiseless accuracy 0.88 on the
50 inputs; without qang 0.84 (8 decisions differ from the noiseless model),
with qang 0.88 (4), with qang + echo 0.86 (1); kept fraction 0.68; echo
diagonal 0.79-0.84.

QPU run: pending (needs credits and an explicit cost approval).
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qg_radius_hardware_qg as R  # noqa: E402
import qnn_hardware_qg as H  # noqa: E402
import qnn_multiclass_echo_qg as E  # noqa: E402
from qang.sectors import echo_transfer_matrix  # noqa: E402

N = E.N
N_TEST, SHOTS, SHOTS_CAL = 50, 1000, 2000
READOUT = "head"
ONE = [1 << (N - 1 - q) for q in range(N)]


def package(n_test=N_TEST):
    m, X, y = E.model_and_data(READOUT)
    X, y = X[:n_test], y[:n_test]
    H.check_circuits(m, X)
    circuits = [H.build_circuit(m, x) for x in X] + [R.echo_circuit(m, j) for j in range(N)]
    shots = [SHOTS] * len(X) + [SHOTS_CAL] * N
    return m, X, y, circuits, shots


def ionq_backend(mode, name, key):
    from qiskit_ionq import IonQProvider

    provider = IonQProvider(key)
    return provider.get_backend("simulator" if mode == "rehearsal" else name, gateset="native")


def transpiled(backend, circuits):
    from qiskit import transpile

    return transpile(circuits, backend=backend, optimization_level=1, seed_transpiler=116)


def gate_counts(tc):
    one = sum(sum(1 for i in c.data if i.operation.num_qubits == 1) for c in tc)
    two = sum(sum(1 for i in c.data if i.operation.num_qubits == 2) for c in tc)
    return one, two


def submit_and_collect(backend, tc, shots, jobs_path, noise=None):
    """Submit the missing jobs, collect the finished ones. Returns the counts,
    or None while some jobs are still queued (rerun later to resume)."""
    ids = json.load(open(jobs_path)) if jobs_path and os.path.exists(jobs_path) else []
    for k in range(len(ids), len(tc)):
        kw = {"noise_model": noise} if noise else {}
        ids.append(backend.run(tc[k], shots=shots[k], **kw).job_id())
        if jobs_path:
            json.dump(ids, open(jobs_path, "w"))
    cache = jobs_path + ".counts" if jobs_path else None
    out = json.load(open(cache)) if cache and os.path.exists(cache) else []
    for jid, s in list(zip(ids, shots))[len(out):]:
        job = backend.retrieve_job(jid)
        if job.status().name != "DONE":
            print(f"{len(out)} of {len(ids)} jobs retrieved; job {jid} is {job.status().name}. Rerun later to resume.")
            return None
        probs = job.get_probabilities()
        out.append({k: int(round(v * s)) for k, v in probs.items()})
        if cache:
            json.dump(out, open(cache, "w"))
    return out


def evaluate(m, X, y, counts):
    probs = [H.counts_to_probs(c) for c in counts]
    M = echo_transfer_matrix(probs[len(X):], N, 1, prepared=ONE)
    noiseless = np.argmax(m.logits(m.params_, X), axis=1)
    lg = E.class_logits(m, np.array(probs[: len(X)]), M)
    r = {"noiseless accuracy": float(np.mean(noiseless == y)), "inputs": len(X)}
    for k, v in lg.items():
        pred = np.argmax(v, axis=1)
        r[k] = {"accuracy": float(np.mean(pred == y)), "differ": int(np.sum(pred != noiseless))}
    kept = [float(sum(p[i] for i in ONE)) for p in probs[: len(X)]]
    r["kept fraction"] = float(np.mean(kept))
    r["echo diagonal"] = [float(x) for x in np.diag(M)]
    return r


def verdict(r):
    return {
        "H1": r["qang"]["differ"] < r["without qang"]["differ"],
        "H2": r["qang + echo"]["differ"] <= r["qang"]["differ"],
        "H3": r["qang"]["accuracy"] >= r["without qang"]["accuracy"],
        "H4": float(np.mean(r["echo diagonal"])) < 0.95,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plan", "rehearsal", "qpu"], default="plan")
    ap.add_argument("--backend", default="qpu.forte-1")
    ap.add_argument("--noise", default="forte-1")
    ap.add_argument("--inputs", type=int, default=N_TEST)
    ap.add_argument("--jobs", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--yes-i-accept-qpu-cost", action="store_true")
    args = ap.parse_args(argv)

    m, X, y, circuits, shots = package(args.inputs)
    print(f"{len(circuits)} circuits ({len(X)} inputs x {SHOTS} shots + {N} echoes x {SHOTS_CAL} shots), "
          f"{sum(shots)} shots in total; compiled circuits match the model")
    if args.mode == "plan":
        try:
            backend = ionq_backend("qpu", args.backend, H.read_ionq_key())
            tc = transpiled(backend, circuits)
            one, two = gate_counts(tc)
            print(f"IonQ native gate set ({args.backend}): {one} single-qubit and {two} two-qubit gates over all "
                  f"circuits; per input circuit {two // len(circuits)} two-qubit gates on average")
        except Exception as exc:  # no key or no qiskit-ionq: the package itself is still checked
            print(f"(gate counts need qiskit-ionq and an IonQ key: {exc})")
        print("Nothing submitted. Check the price in the IonQ console before any --mode qpu run.")
        return None
    if args.mode == "qpu" and not args.yes_i_accept_qpu_cost:
        print("IonQ QPU costs money: rerun with --yes-i-accept-qpu-cost after checking the price. Nothing submitted.")
        return None
    backend = ionq_backend(args.mode, args.backend, H.read_ionq_key())
    tc = transpiled(backend, circuits)
    counts = submit_and_collect(backend, tc, shots, args.jobs, noise=args.noise if args.mode == "rehearsal" else None)
    if counts is None:
        return None
    r = evaluate(m, X, y, counts)
    r["backend"] = f"ionq simulator {args.noise} (native)" if args.mode == "rehearsal" else f"{args.backend} (native)"
    print(json.dumps(r))
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in verdict(r).items()))
    if args.out:
        json.dump(r, open(args.out, "w"), indent=1)
    return r


if __name__ == "__main__":
    main()
