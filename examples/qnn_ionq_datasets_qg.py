"""
The qg-filtered QNN on the IonQ simulator with the Forte-1 noise model, on
all four datasets, with and without qang (§91)

§85 ran one dataset (iris, 30 inputs), where the model's margins were so wide
that neither readout changed a decision. This study uses the same pipeline
(examples/qnn_hardware_qg.py: model E from qang.qml trained without noise,
compiled to Qiskit, checked against WeightQNN.probs, each input read from the
same shots with and without the filter) on iris, breast cancer, wine and
digits: 189 test inputs, 1000 shots each, on the free IonQ cloud simulator
with the Forte-1 noise model (the QPU that the hardware run will use).

Pre-registered predictions (committed before the run):
  N1  pooled over the 189 inputs, accuracy with qang >= accuracy without qang.
  N2  pooled accuracy with qang within 3 points of the pooled noiseless
      accuracy.
  N3  on every dataset the qg_Z error without qang is at least twice the
      error with qang.
Setting: split seed 9100, training seed 9102, 120 epochs.

Needs the IonQ key (IONQ_API_KEY or the git-ignored .ionq_key; never
printed), qiskit, qiskit-ionq and scikit-learn. Job ids are saved per
dataset (--jobs-dir) so an interrupted run resumes without resubmitting.

Findings (python examples/qnn_ionq_datasets_qg.py --jobs-dir <dir>):

IonQ cloud simulator, Forte-1 noise model, 1000 shots per input, the
measured frequencies returned by the API (see the note below):

  dataset  inputs  accuracy qang / without (noiseless)   qg_Z error qang / without   kept
  iris     30      0.933 / 0.900 (0.967)                 0.064 / 0.176               0.70
  cancer   60      0.983 / 0.983 (0.983)                 0.065 / 0.182               0.70
  wine     39      0.974 / 0.949 (0.974)                 0.065 / 0.191               0.70
  digits   60      0.967 / 0.967 (0.983)                 0.060 / 0.171               0.72
  pooled   189     0.968 / 0.958 (0.979)

  * N1, N2, N3 pass.
  * Pooled over 189 inputs, qang is 1.1 points ahead (two inputs: one in
    iris, one in wine) and 1.1 points below the noiseless accuracy.
  * The qg_Z error is 2.7-2.9x smaller with qang on every dataset.
  * Note on the IonQ results. The API returns the measured frequencies of
    each job; qiskit-ionq's get_counts resamples them at random on every
    call, so two retrievals of the same job gave different counts. During
    this run the retrieval was changed to use the frequencies directly
    (examples/qnn_hardware_qg.py, ionq_counts); the numbers above are
    deterministic. Re-read the same way, the §85 IonQ runs give qg_Z errors
    0.052 / 0.172 (aria-1, kept 0.74) and 0.069 / 0.207 (forte-1, kept
    0.70), against 0.056 / 0.176 and 0.072 / 0.206 reported before.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
import qnn_hardware_qg as HW  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

SPLIT_SEED = 9100
TRAIN_SEED = 9102


def trained(name, epochs=120):
    Xa, ya, use_pca = Q.load(name, np.random.default_rng(SPLIT_SEED))
    Xtr, Xte, ytr, yte = Q.split_prepare(Xa, ya, use_pca, SPLIT_SEED)
    return WeightQNN(5, 1).fit(Xtr, ytr, epochs, seed=TRAIN_SEED), Xte, yte


def verdict(per, pooled):
    return {
        "N1": pooled["with qang"] >= pooled["without qang"],
        "N2": pooled["with qang"] >= pooled["noiseless"] - 0.03,
        "N3": all(r["without qang"]["qg_Z error"] >= 2 * r["with qang"]["qg_Z error"] for r in per.values()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "ionq_sim"], default="ionq_sim")
    ap.add_argument("--noise", default="forte-1")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--jobs-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    backend = HW.get_backend(args.mode)
    per, n_tot, acc = {}, 0, {"with qang": 0.0, "without qang": 0.0, "noiseless": 0.0}
    for name in Q.DATASETS:
        model, Xte, yte = trained(name, args.epochs)
        HW.check_circuits(model, Xte)
        circuits = [HW.build_circuit(model, x) for x in Xte]
        jobs = os.path.join(args.jobs_dir, f"{name}.json") if args.jobs_dir else None
        counts, _ = HW.run_counts(backend, args.mode, circuits, args.shots, args.noise, jobs_path=jobs)
        r = HW.evaluate(model, Xte, yte, [HW.counts_to_probs(c) for c in counts])
        r["inputs"] = len(Xte)
        per[name] = r
        n = len(Xte)
        n_tot += n
        acc["with qang"] += n * r["with qang"]["accuracy"]
        acc["without qang"] += n * r["without qang"]["accuracy"]
        acc["noiseless"] += n * r["noiseless accuracy"]
        print(f"{name}: {n} inputs | accuracy qang {r['with qang']['accuracy']:.3f} / without "
              f"{r['without qang']['accuracy']:.3f} (noiseless {r['noiseless accuracy']:.3f}) | qg_Z error qang "
              f"{r['with qang']['qg_Z error']:.3f} / without {r['without qang']['qg_Z error']:.3f} | kept "
              f"{r['kept fraction']:.2f}", flush=True)
    pooled = {k: v / n_tot for k, v in acc.items()}
    v = verdict(per, pooled)
    print("pooled:", {k: round(x, 4) for k, x in pooled.items()}, "inputs", n_tot)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    if args.out:
        json.dump({"per_dataset": per, "pooled": pooled, "verdict": v}, open(args.out, "w"), indent=1)
    return per, pooled, v


if __name__ == "__main__":
    main()
