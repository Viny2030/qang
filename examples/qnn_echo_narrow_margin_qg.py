"""
Echo calibration where decisions are fragile: narrow-margin inputs (§108)

In §106 the echo calibration halved the decision-value error left by the qg
filter, but on four datasets with wide margins only a handful of decisions
could change (15 / 7 / 3 over 945 readings). This study chooses inputs where
noise does change decisions: 120 test inputs selected, before any noisy run,
as those whose noiseless decision value is closest to zero.

Model: qang.qml.WeightQNN (5 qubits, weight 1, qg_Z readout), trained without
noise (120 epochs, seed 1082) on 300 synthetic inputs in [-1, 1]^4 labelled
by the rule x0 + 0.6 x1 - 0.4 x2 + 0.2 x3 > 0 (seed 1080). Test inputs: of
3000 candidates (seed 1081), 120 with |noiseless decision value| in the band
[0.4, 1.6], evenly spaced in rank. The band was fixed after a noiseless check
(the only run before these predictions): with 1000 shots the shot noise
alone moves the decision value by about 0.2, so smaller margins are flipped
by shot noise whatever the readout, while the noise-induced errors of §106
(0.3-1.7) can flip inputs in this band. Same circuits, shots and readouts as §106: 1000 shots per input; 5
echo circuits at 4000 shots; readouts without qang, qang (filter), qang +
echo (qang.sectors.unmix_sector, M^(1/2)). Backends: IBM fake brisbane,
sherbrooke, torino (Aer) and the IonQ simulator with the aria-1 and forte-1
noise models in the native gate set.

Primary figure of merit: decisions that differ from the noiseless model (the
labels are only a rule; what noise does is flip the model's own decisions).

Pre-registered predictions (committed before any noisy run; code checked in
noiseless --mode local only):
  M1  decisions differing from the noiseless model: fewer with qang + echo
      than with qang, on each of the five backends.
  M2  pooled over the five backends: without qang > qang > qang + echo
      (strictly).
  M3  the decision-value error is lower with qang + echo than with qang, on
      each backend.
  M4  pooled over the five backends, accuracy against the labels with
      qang + echo is at least the accuracy with qang.

python examples/qnn_echo_narrow_margin_qg.py --mode fake
python examples/qnn_echo_narrow_margin_qg.py --mode ionq_sim --noise aria-1 --jobs jobs.json

Findings:

120 inputs with noiseless margins 0.4-1.6 (noiseless accuracy 0.892 against
the rule labels), 1000 shots; 5 echo calibrations at 4000 shots:

  backend           decisions differing from    decision-value error       accuracy
                    noiseless (raw/qang/echo)   raw / qang / echo          raw / qang / echo
  fake_brisbane     10 / 5 / 6                  0.46 / 0.309 / 0.306       0.808 / 0.850 / 0.842
  fake_sherbrooke   11 / 1 / 2                  0.43 / 0.267 / 0.272       0.800 / 0.900 / 0.892
  fake_torino        6 / 4 / 3                  0.37 / 0.258 / 0.261       0.842 / 0.892 / 0.900
  IonQ aria-1       10 / 3 / 4                  0.52 / 0.353 / 0.350       0.808 / 0.883 / 0.875
  IonQ forte-1      10 / 4 / 4                  0.47 / 0.309 / 0.317       0.808 / 0.858 / 0.858
  pooled            47 / 17 / 19                                           0.813 / 0.877 / 0.873

  * M1-M4 all FAIL. The filter does what it did before: it cuts the flipped
    decisions from 47 to 17 (600 readings) and raises the accuracy by 6.4
    points. The echo calibration adds nothing here: 19 flips against 17,
    decision error within 0.01 of the filter's, accuracy 0.3 points lower.
  * Why, and why it differs from §106. With 1000 shots the decision value of
    this model already scatters by about 0.22 from shot noise alone (the
    noiseless check); with the filter the error is 0.26-0.35, so little
    in-sector bias is left for the calibration to remove, while inverting the
    transfer matrix adds variance. In §106 the trained heads had larger
    weights and the filtered error was 0.47-0.87, mostly bias, which the echo
    halved.
  Verdict. The echo calibration helps when the in-sector bias is large
  compared with the shot noise of the decision value, and not otherwise; it
  never helped here and cost little. A practical check before using it:
  compare the filtered decision error with its shot-noise level (for example
  from a noiseless simulation with the same shots).
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
from qang.qml import WeightQNN  # noqa: E402
from qang.sectors import echo_transfer_matrix  # noqa: E402

N = H.N
W_RULE = np.array([1.0, 0.6, -0.4, 0.2])
N_TRAIN, N_CAND, N_TEST = 300, 3000, 120
BAND = (0.4, 1.6)


def label(X):
    return (X @ W_RULE > 0).astype(int)


def model_and_inputs(epochs=120):
    Xtr = np.random.default_rng(1080).uniform(-1, 1, (N_TRAIN, 4))
    m = WeightQNN(N, 1).fit(Xtr, label(Xtr), epochs, seed=1082)
    Xc = np.random.default_rng(1081).uniform(-1, 1, (N_CAND, 4))
    th = m.params_
    d0 = m.qg_z(m.probs(th[: m.n_theta], m.encode(Xc)), qang=False) @ th[m.n_theta:m.n_theta + m.n_head] + th[-1]
    order = np.argsort(np.abs(d0))
    band = order[(np.abs(d0[order]) >= BAND[0]) & (np.abs(d0[order]) <= BAND[1])]
    pick = band[np.linspace(0, len(band) - 1, N_TEST).round().astype(int)]
    return m, Xc[pick], label(Xc[pick]), d0[pick]


def run_backend(mode, name, noise, jobs_path, epochs=120):
    model, X, y, d0 = model_and_inputs(epochs)
    H.check_circuits(model, X)
    circuits = [H.build_circuit(model, x) for x in X] + [R.echo_circuit(model, j) for j in range(N)]
    shots = [C.SHOTS] * len(X) + [C.SHOTS_CAL] * N
    counts = C.get_counts(mode, name, noise, circuits, shots, jobs_path)
    probs = [H.counts_to_probs(c) for c in counts]
    M = echo_transfer_matrix(probs[len(X):], N, 1, prepared=C.ONE)
    dv = C.decision_values(model, probs[: len(X)], M)
    r = {"inputs": len(X), "noiseless accuracy": float(np.mean((d0 > 0).astype(int) == y)),
         "median |noiseless decision value|": float(np.median(np.abs(d0)))}
    for k, v in dv.items():
        r[k] = {"accuracy": float(np.mean((v > 0).astype(int) == y)), "differ": int(np.sum((v > 0) != (d0 > 0))),
                "decision error": float(np.mean(np.abs(v - d0)))}
    r["backend"] = name if mode == "fake" else (f"ionq simulator {noise} (native)" if mode == "ionq_sim" else mode)
    return r


def verdict(results):
    tot = {k: sum(r[k]["differ"] for r in results) for k in C.READOUTS}
    n = sum(r["inputs"] for r in results)
    acc = {k: sum(r[k]["accuracy"] * r["inputs"] for r in results) / n for k in C.READOUTS}
    return {
        "M1": all(r["qang + echo"]["differ"] < r["qang"]["differ"] for r in results),
        "M2": tot["without qang"] > tot["qang"] > tot["qang + echo"],
        "M3": all(r["qang + echo"]["decision error"] < r["qang"]["decision error"] for r in results),
        "M4": acc["qang + echo"] >= acc["qang"],
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
