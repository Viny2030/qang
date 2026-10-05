"""
The whole pipeline on device noise models: calibrated training with the
filter, then the filter and the echo at run time (§119)

Sections 117 and 118 corrected unequal T1 and dephasing by training with qang
under the calibrated noise, and section 106 corrected in-sector errors with
echo circuits, each in isolation. Device noise models have all of them at
once, plus gate errors. Here the two corrections are combined, using only
what a user would have: the device's published calibration.

Calibration: transpile once to fix the physical qubits (the same layout for
every circuit), read T1 and T2 of those qubits from the backend target and
the circuit duration, and convert to per-sublayer noise for qang's
simulator: t = duration / 13 (4 loader + 9 trained sublayers),
gamma_q = 1 - exp(-t / T1_q), dephasing p = mean over qubits of
(1 - exp(-t / T_phi_q)) / 2 with 1 / T_phi = 1 / T2 - 1 / (2 T1) (>= 0).

Models (weight 1, 5 qubits; iris, cancer, wine, digits as in section 106,
189 test inputs):
  A  trained noiselessly (section 106)
  C  trained with qang under the calibrated noise
  D  trained without qang under the calibrated noise
Run on the IBM fake backends brisbane, sherbrooke and torino (Aer with their
noise models), 1000 shots per input, 5 echo circuits per model at 4000
shots. Readouts: A raw, A qang, A qang + echo, C qang, C qang + echo, D raw.
Accuracy over the 189 inputs, pooled per backend. (IonQ's simulator exposes
no per-qubit T1/T2, so it is not used here.)

Pre-registered predictions (committed before any noisy run; code checked in
noiseless --mode local only):
  K1  pooled over the three backends, C with qang + echo is at least as
      accurate as A with qang + echo.
  K2  C with qang is at least as accurate as A with qang, on every backend.
  K3  pooled over the three backends, C with qang + echo is within 1 point
      of A's noiseless accuracy.
  K4  pooled over the three backends, C with qang + echo is more accurate
      than D (raw readout).

python examples/qnn_full_pipeline_qg.py --mode local     # noiseless check
python examples/qnn_full_pipeline_qg.py --mode fake --out fake.json

Findings:

FINDINGS_PLACEHOLDER
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qg_radius_hardware_qg as R  # noqa: E402
import qnn_classifier_qg as Q  # noqa: E402
import qnn_hardware_qg as H  # noqa: E402
import qnn_ionq_datasets_qg as DS  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402
from qang.sectors import echo_transfer_matrix, unmix_sector  # noqa: E402

N = H.N
SHOTS, SHOTS_CAL, EPOCHS = 1000, 4000, 120
SUBLAYERS = 13
ONE = [1 << (N - 1 - q) for q in range(N)]
BACKENDS = ("fake_brisbane", "fake_sherbrooke", "fake_torino")


def data(name):
    Xa, ya, use_pca = Q.load(name, np.random.default_rng(DS.SPLIT_SEED))
    return Q.split_prepare(Xa, ya, use_pca, DS.SPLIT_SEED)


def calibration(backend, circuit):
    """Fixed layout and per-sublayer qang noise (gamma per qang qubit, mean dephasing)."""
    from qiskit import transpile

    tc = transpile(circuit, backend=backend, optimization_level=1, seed_transpiler=119)
    layout = list(tc.layout.final_index_layout())
    t = tc.estimate_duration(backend.target, unit="s") / SUBLAYERS
    gam, deph = np.zeros(N), []
    for v, phys in enumerate(layout):  # Qiskit virtual qubit v = qang qubit N-1-v
        qp = backend.target.qubit_properties[phys]
        gam[N - 1 - v] = 1 - math.exp(-t / qp.t1)
        rate = max(1 / qp.t2 - 1 / (2 * qp.t1), 0.0)
        deph.append((1 - math.exp(-t * rate)) / 2)
    return layout, {"gamma": gam, "dephasing": float(np.mean(deph)), "sublayer time": t}


def counts_for(mode, backend, circuits, shots, layout):
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator.from_backend(backend) if mode == "fake" else AerSimulator()
    tc = transpile(circuits, backend=sim, optimization_level=1, seed_transpiler=119,
                   initial_layout=layout if mode == "fake" else None)
    return [sim.run(c, shots=s, seed_simulator=1190 + i).result().get_counts() for i, (c, s) in enumerate(zip(tc, shots))]


def readouts(model, probs, M):
    w, b = model.params_[model.n_theta:model.n_theta + model.n_head], model.params_[-1]
    raw = np.array([model.qg_z(p, qang=False)[0] for p in probs]) @ w + b
    filt = np.array([model.qg_z(p, qang=True)[0] for p in probs]) @ w + b
    echo = np.array([model.qg_z(unmix_sector(p, M, N, 1, power=0.5), qang=False)[0] for p in probs]) @ w + b
    return {"raw": raw, "qang": filt, "qang + echo": echo}


def run_backend(mode, name, epochs=EPOCHS, datasets=Q.DATASETS):
    backend = H.get_backend("fake", name or "fake_brisbane")
    out = {"backend": name if mode == "fake" else "local", "per dataset": {}}
    cal_done = None
    for ds in datasets:
        Xtr, Xte, ytr, yte = data(ds)
        A = WeightQNN(N, 1).fit(Xtr, ytr, epochs, seed=DS.TRAIN_SEED)
        if cal_done is None:
            layout, cal = calibration(backend, H.build_circuit(A, Xte[0]))
            cal_done = (layout, cal)
            out["calibration"] = {"layout": layout, "gamma": cal["gamma"].tolist(), "dephasing": cal["dephasing"],
                                  "sublayer time": cal["sublayer time"]}
        layout, cal = cal_done
        C = WeightQNN(N, 1).fit(Xtr, ytr, epochs, gamma=cal["gamma"], dephasing=cal["dephasing"], qang=True, seed=DS.TRAIN_SEED)
        Dm = WeightQNN(N, 1).fit(Xtr, ytr, epochs, gamma=cal["gamma"], dephasing=cal["dephasing"], qang=False, seed=DS.TRAIN_SEED)
        circuits, shots = [], []
        for m in (A, C):
            circuits += [H.build_circuit(m, x) for x in Xte] + [R.echo_circuit(m, j) for j in range(N)]
            shots += [SHOTS] * len(Xte) + [SHOTS_CAL] * N
        circuits += [H.build_circuit(Dm, x) for x in Xte]
        shots += [SHOTS] * len(Xte)
        probs = [H.counts_to_probs(c) for c in counts_for(mode, backend, circuits, shots, layout)]
        n = len(Xte)
        r = {"inputs": n, "noiseless correct": int(np.sum(A.predict(Xte) == yte))}
        for tag, m, off in (("A", A, 0), ("C", C, n + N)):
            M = echo_transfer_matrix(probs[off + n: off + n + N], N, 1, prepared=ONE)
            for k, v in readouts(m, probs[off: off + n], M).items():
                r[f"{tag} {k}"] = int(np.sum((v > 0).astype(int) == yte))
        dv = readouts(Dm, probs[2 * (n + N):], np.eye(N))["raw"]
        r["D raw"] = int(np.sum((dv > 0).astype(int) == yte))
        out["per dataset"][ds] = r
    tot = sum(r["inputs"] for r in out["per dataset"].values())
    keys = [k for k in next(iter(out["per dataset"].values())) if k != "inputs"]
    out["pooled"] = {k: sum(r[k] for r in out["per dataset"].values()) / tot for k in keys}
    out["pooled"]["inputs"] = tot
    return out


KEYS = ("noiseless correct", "A raw", "A qang", "A qang + echo", "C qang", "C qang + echo", "D raw")


def verdict(results):
    P = [r["pooled"] for r in results]
    mean = lambda k: float(np.mean([p[k] for p in P]))  # noqa: E731
    return {
        "K1": mean("C qang + echo") >= mean("A qang + echo"),
        "K2": all(p["C qang"] >= p["A qang"] for p in P),
        "K3": mean("noiseless correct") - mean("C qang + echo") <= 0.01,
        "K4": mean("C qang + echo") > mean("D raw"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "fake"], default="fake")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    names = BACKENDS if args.mode == "fake" else [None]
    rs = []
    for nm in names:
        r = run_backend(args.mode, nm)
        rs.append(r)
        print(r["backend"], {k: round(r["pooled"][k], 4) for k in KEYS}, "calibration", r["calibration"], flush=True)
    if args.mode == "fake":
        print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in verdict(rs).items()))
    if args.out:
        json.dump(rs, open(args.out, "w"), indent=1)
    return rs


if __name__ == "__main__":
    main()
