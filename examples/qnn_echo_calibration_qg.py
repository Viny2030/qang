"""
Does the echo calibration change the classifier? Four datasets, five device
noise models, with and without qang (§106)

§105 showed that five echo circuits measure the in-sector errors the qg
filter leaves, and that inverting their transfer matrix
(qang.sectors.echo_transfer_matrix, unmix_sector, power 1/2) removes 38-47%
of the remaining error of the radius deficit. This study asks whether that
reaches the classifier. The model is the §91 one (qang.qml.WeightQNN, 5
qubits, weight 1, qg_Z readout, trained without noise; iris, breast cancer,
wine and digits, 189 test inputs, split seed 9100, training seed 9102), run
on the three IBM fake backends (Aer) and on the IonQ simulator with the
aria-1 and forte-1 noise models in the native gate set (§103). Per dataset,
5 echo circuits (excited qubit j, the 15 trained RBS gates, their inverse)
at 4000 shots calibrate the transfer matrix; each test input is read from
the same 1000 shots three ways:
  without qang   qg_Z from all shots
  qang           qg_Z from the weight-1 shots (the filter)
  qang + echo    the filtered distribution corrected with M^(1/2)
The decision value is the trained linear head applied to the five qg_Z.

Because accuracy moves by whole inputs and the §85/§91 margins were wide,
the predictions use the decision value as well as the decisions.

Pre-registered predictions (committed before any noisy run; code checked in
noiseless --mode local only):
  L1  the mean absolute error of the decision value (against the noiseless
      model) is lower with qang + echo than with qang, on each of the five
      backends.
  L2  the number of decisions that agree with the noiseless model is at least
      as large with qang + echo as with qang, on each backend.
  L3  pooled accuracy with qang + echo is at least the accuracy with qang
      minus 0.5 points, on each backend (the correction does no harm).
  L4  pooled accuracy with qang is at least the accuracy without qang, on each
      backend (the §91 direction holds on all five).

python examples/qnn_echo_calibration_qg.py --mode fake
python examples/qnn_echo_calibration_qg.py --mode ionq_sim --noise aria-1 --jobs-dir jobs_aria

Findings:

FINDINGS_PLACEHOLDER
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qg_radius_hardware_qg as R  # noqa: E402
import qnn_classifier_qg as Q  # noqa: E402
import qnn_hardware_qg as H  # noqa: E402
import qnn_ionq_datasets_qg as D  # noqa: E402
from qang.sectors import echo_transfer_matrix, unmix_sector  # noqa: E402

N = H.N
SHOTS, SHOTS_CAL = 1000, 4000
ONE = [1 << (N - 1 - q) for q in range(N)]
READOUTS = ("without qang", "qang", "qang + echo")


def get_counts(mode, name, noise, circuits, shots, jobs_path):
    from qiskit import transpile

    if mode in ("local", "fake"):
        from qiskit_aer import AerSimulator

        backend = H.get_backend(mode, name)
        sim = AerSimulator.from_backend(backend) if mode == "fake" else backend
        tc = transpile(circuits, backend=sim, optimization_level=1, seed_transpiler=106)
        return [sim.run(c, shots=s, seed_simulator=106 + i).result().get_counts() for i, (c, s) in enumerate(zip(tc, shots))]
    from qiskit_ionq import IonQProvider

    backend = IonQProvider(H.read_ionq_key()).get_backend("simulator", gateset="native")
    tc = transpile(circuits, backend=backend, optimization_level=1, seed_transpiler=106)
    ids = json.load(open(jobs_path)) if jobs_path and os.path.exists(jobs_path) else []
    for k in range(len(ids), len(tc)):
        ids.append(backend.run(tc[k], shots=shots[k], noise_model=noise).job_id())
        if jobs_path:
            json.dump(ids, open(jobs_path, "w"))
    out = []
    for jid, s in zip(ids, shots):
        for attempt in range(20):
            try:
                probs = backend.retrieve_job(jid).get_probabilities()
                out.append({k: int(round(v * s)) for k, v in probs.items()})
                break
            except Exception:
                if attempt == 19:
                    raise
                time.sleep(15)
    return out


def decision_values(model, probs, M):
    th = model.params_
    w, b = th[model.n_theta:model.n_theta + model.n_head], th[-1]
    out = {}
    for label in READOUTS:
        if label == "without qang":
            z = np.array([model.qg_z(p, qang=False)[0] for p in probs])
        elif label == "qang":
            z = np.array([model.qg_z(p, qang=True)[0] for p in probs])
        else:
            z = np.array([model.qg_z(unmix_sector(p, M, N, 1, power=0.5), qang=False)[0] for p in probs])
        out[label] = z @ w + b
    return out


def run_dataset(mode, name, noise, dataset, jobs_path):
    model, Xte, yte = D.trained(dataset)
    H.check_circuits(model, Xte)
    circuits = [H.build_circuit(model, x) for x in Xte] + [R.echo_circuit(model, j) for j in range(N)]
    shots = [SHOTS] * len(Xte) + [SHOTS_CAL] * N
    counts = get_counts(mode, name, noise, circuits, shots, jobs_path)
    probs = [H.counts_to_probs(c) for c in counts]
    M = echo_transfer_matrix(probs[len(Xte):], N, 1, prepared=ONE)
    th = model.params_
    ideal = model.qg_z(model.probs(th[: model.n_theta], model.encode(Xte)), qang=False)
    d0 = ideal @ th[model.n_theta:model.n_theta + model.n_head] + th[-1]
    dv = decision_values(model, probs[: len(Xte)], M)
    r = {"inputs": len(Xte), "noiseless correct": int(np.sum((d0 > 0).astype(int) == yte))}
    for k, v in dv.items():
        r[k] = {"correct": int(np.sum((v > 0).astype(int) == yte)), "agree": int(np.sum((v > 0) == (d0 > 0))),
                "decision error sum": float(np.sum(np.abs(v - d0)))}
    return r


def run_backend(mode, name, noise, jobs_dir):
    per = {}
    for ds in Q.DATASETS:
        jobs = os.path.join(jobs_dir, f"{ds}.json") if jobs_dir else None
        if jobs_dir:
            os.makedirs(jobs_dir, exist_ok=True)
        per[ds] = run_dataset(mode, name, noise, ds, jobs)
    n = sum(r["inputs"] for r in per.values())
    pooled = {"inputs": n, "noiseless accuracy": sum(r["noiseless correct"] for r in per.values()) / n}
    for k in READOUTS:
        pooled[k] = {"accuracy": sum(r[k]["correct"] for r in per.values()) / n,
                     "agree": sum(r[k]["agree"] for r in per.values()),
                     "decision error": sum(r[k]["decision error sum"] for r in per.values()) / n}
    label = name if mode == "fake" else (f"ionq simulator {noise} (native)" if mode == "ionq_sim" else mode)
    return {"backend": label, "pooled": pooled, "per dataset": per}


def verdict(results):
    P = [r["pooled"] for r in results]
    return {
        "L1": all(p["qang + echo"]["decision error"] < p["qang"]["decision error"] for p in P),
        "L2": all(p["qang + echo"]["agree"] >= p["qang"]["agree"] for p in P),
        "L3": all(p["qang + echo"]["accuracy"] >= p["qang"]["accuracy"] - 0.005 for p in P),
        "L4": all(p["qang"]["accuracy"] >= p["without qang"]["accuracy"] for p in P),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "fake", "ionq_sim"], default="fake")
    ap.add_argument("--noise", default="aria-1")
    ap.add_argument("--jobs-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    names = ("fake_brisbane", "fake_sherbrooke", "fake_torino") if args.mode == "fake" else [None]
    rs = []
    for n in names:
        r = run_backend(args.mode, n, args.noise, args.jobs_dir)
        rs.append(r)
        p = r["pooled"]
        print(r["backend"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items() if not isinstance(v, dict)},
              {k: {a: round(b, 4) for a, b in p[k].items()} for k in READOUTS}, flush=True)
    if args.out:
        json.dump(rs, open(args.out, "w"), indent=1)
    return rs


if __name__ == "__main__":
    main()
