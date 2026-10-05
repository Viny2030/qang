"""
Multiclass QNN on device noise models: filter and echo calibration (§113)

In the multiclass QNN of §110 (qang.qml.MultiClassQNN, weight 1) each qubit
is a class, so the errors the qg filter cannot see, the ones that move the
excitation to another qubit (§105), are confusions between classes. Five echo
circuits measure them as a 5 x 5 transfer matrix, and inverting M^(1/2) on the
filtered distribution (qang.sectors.unmix_sector) corrects them, as readout
mitigation does for bit flips.

Model: MultiClassQNN(5 qubits, 5 classes), digits 0-4 (4 PCA features scaled
to [-1, 1], the §110 preparation), trained without noise (120 epochs, seed
1130), two readouts: "qubit" (class = qubit) and "head" (linear softmax on the
five qg_Z). Test: 100 test inputs drawn with seed 1131, 1000 shots each;
calibration: 5 echo circuits at 4000 shots per model. Readouts without qang,
with qang (filter), and with qang + echo. Backends: IBM fake brisbane,
sherbrooke, torino (Aer) and the IonQ simulator with the aria-1 and forte-1
noise models in the native gate set (§103).

Pre-registered predictions (committed before any noisy run; code checked in
noiseless --mode local only):
  E1  qubit readout: accuracy with qang + echo >= accuracy with qang, on each
      of the five backends.
  E2  qubit readout: decisions that agree with the noiseless model, at least
      as many with qang + echo as with qang, on each backend.
  E3  qubit readout, pooled over the five backends: decisions differing from
      the noiseless model, without qang > qang > qang + echo (strictly).
  E4  head readout: accuracy with qang >= accuracy without qang, on each
      backend.

python examples/qnn_multiclass_echo_qg.py --mode fake
python examples/qnn_multiclass_echo_qg.py --mode ionq_sim --noise aria-1 --jobs-dir jobs_aria

Findings:

FINDINGS_PLACEHOLDER
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qg_radius_hardware_qg as R  # noqa: E402
import qnn_echo_calibration_qg as C  # noqa: E402
import qnn_hardware_qg as H  # noqa: E402
import qnn_multiclass_qg as MC  # noqa: E402
from qang.qml import MultiClassQNN  # noqa: E402
from qang.sectors import echo_transfer_matrix, filter_distribution, unmix_sector  # noqa: E402

N, CLASSES, N_TEST = 5, 5, 100
READOUTS = ("without qang", "qang", "qang + echo")


def model_and_data(readout, epochs=120):
    X, y = MC.load("digits")
    Xtr, Xte, ytr, yte = MC.prepare(X, y, 1130)
    pick = np.random.default_rng(1131).choice(len(Xte), N_TEST, replace=False)
    m = MultiClassQNN(N, CLASSES, readout=readout).fit(Xtr, ytr, epochs, seed=1130)
    return m, Xte[pick], yte[pick]


def class_logits(m, probs, M):
    head = m.params_[m.n_theta:]
    out = {}
    for label in READOUTS:
        if label == "without qang":
            d = probs
        elif label == "qang":
            d = np.array([filter_distribution(p, N, 1)[0] for p in probs])
        else:
            d = np.array([unmix_sector(p, M, N, 1, power=0.5) for p in probs])
        out[label] = m._logits(m.qg_z(d, qang=False), head)
    return out


def run_readout(mode, name, noise, readout, jobs_path):
    m, X, y = model_and_data(readout)
    H.check_circuits(m, X)
    circuits = [H.build_circuit(m, x) for x in X] + [R.echo_circuit(m, j) for j in range(N)]
    shots = [C.SHOTS] * len(X) + [C.SHOTS_CAL] * N
    counts = C.get_counts(mode, name, noise, circuits, shots, jobs_path)
    probs = [H.counts_to_probs(c) for c in counts]
    M = echo_transfer_matrix(probs[len(X):], N, 1, prepared=C.ONE)
    noiseless = np.argmax(m.logits(m.params_, X), axis=1)
    lg = class_logits(m, np.array(probs[: len(X)]), M)
    r = {"noiseless accuracy": float(np.mean(noiseless == y))}
    for k, v in lg.items():
        pred = np.argmax(v, axis=1)
        r[k] = {"accuracy": float(np.mean(pred == y)), "differ": int(np.sum(pred != noiseless))}
    r["echo diagonal"] = [float(x) for x in np.diag(M)]
    return r


def run_backend(mode, name, noise, jobs_dir):
    out = {"backend": name if mode == "fake" else (f"ionq simulator {noise} (native)" if mode == "ionq_sim" else mode)}
    for readout in ("qubit", "head"):
        jobs = os.path.join(jobs_dir, f"{readout}.json") if jobs_dir else None
        if jobs_dir:
            os.makedirs(jobs_dir, exist_ok=True)
        out[readout] = run_readout(mode, name, noise, readout, jobs)
    return out


def verdict(results):
    q = [r["qubit"] for r in results]
    tot = {k: sum(x[k]["differ"] for x in q) for k in READOUTS}
    return {
        "E1": all(x["qang + echo"]["accuracy"] >= x["qang"]["accuracy"] for x in q),
        "E2": all(x["qang + echo"]["differ"] <= x["qang"]["differ"] for x in q),
        "E3": tot["without qang"] > tot["qang"] > tot["qang + echo"],
        "E4": all(r["head"]["qang"]["accuracy"] >= r["head"]["without qang"]["accuracy"] for r in results),
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
        print(json.dumps(r), flush=True)
    if args.out:
        json.dump(rs, open(args.out, "w"), indent=1)
    return rs


if __name__ == "__main__":
    main()
