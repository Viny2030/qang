"""
Readout error in qang's QNN simulator: does it close the kept-fraction gap? (§129)

§124 put the published two-qubit gate errors into qang's simulator, and its
prediction of the kept (weight-1) fraction of the noiselessly trained model A
was still too high on every IBM fake backend: 0.703 / 0.774 / 0.845 against
0.671 / 0.697 / 0.724 measured with Aer and the device noise model (R4
failed). The device noise model also has readout error, which moves shots out
of the weight-1 sector at measurement; the simulator did not. Here each
qubit's published asymmetric readout error (prob_meas1_prep0,
prob_meas0_prep1 of the physical qubit in the fixed layout) is applied to the
simulator's output distribution, and the prediction is compared with the
device values recorded in §124 (examples/data/qnn_gate_noise_fake.json).
No new device simulation is run.

Same models, layout and calibration as §124 (weight 1, iris, cancer, wine,
digits; 189 test inputs; brisbane, sherbrooke, torino).

Sanity check (not a prediction): without readout error the code reproduces
§124's predicted kept fractions to 1e-9.

Pre-registered predictions (committed before the run):
  K1  with readout error the predicted kept fraction of A is within 0.05 of
      the device value on every backend (R4 of §124, retried).
  K2  the gap |device - predicted| shrinks by at least half on every backend.
  K3  the filtered accuracy of A predicted with readout error is within 2
      inputs (of 189) of the device value on every backend.

python examples/qnn_readout_error_qg.py

Findings:

FINDINGS_PLACEHOLDER
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
import qnn_full_pipeline_qg as P  # noqa: E402
import qnn_gate_noise_qg as G  # noqa: E402
import qnn_hardware_qg as H  # noqa: E402
import qnn_ionq_datasets_qg as DS  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

N = G.N
ONE = G.ONE
DEVICE = os.path.join(HERE, "data", "qnn_gate_noise_fake.json")


def readout_errors(backend, layout):
    """[(e01, e10)] per Qiskit virtual qubit v (bit v from the right of the index), from the physical qubit
    layout[v]: e01 = P(read 1 | prepared 0), e10 = P(read 0 | prepared 1)."""
    props = backend.properties()
    return [(props.qubit_property(q, "prob_meas1_prep0")[0], props.qubit_property(q, "prob_meas0_prep1")[0])
            for q in layout]


def apply_readout(probs, errors):
    """Independent per-qubit confusion on a batch of 2^N distributions (rows)."""
    p = np.asarray(probs, float).reshape((-1,) + (2,) * N)
    for v, (e01, e10) in enumerate(errors):
        axis = N - v  # bit v from the right = tensor axis N - v (axis 0 is the batch)
        M = np.array([[1 - e01, e10], [e01, 1 - e10]])
        p = np.moveaxis(np.tensordot(M, np.moveaxis(p, axis, 0), axes=(1, 0)), 0, axis)
    return p.reshape(len(probs), 2 ** N)


def run_backend(name, datasets=Q.DATASETS):
    backend = H.get_backend("fake", name)
    layout = gcal = noise = ro = None
    kept0, kept1, acc1, n_tot = [], [], 0, 0
    for ds in datasets:
        Xtr, Xte, ytr, yte = P.data(ds)
        A = WeightQNN(N, 1).fit(Xtr, ytr, G.EPOCHS, seed=DS.TRAIN_SEED)
        if layout is None:
            layout, cal = P.calibration(backend, H.build_circuit(A, Xte[0]))
            gcal = G.gate_calibration(backend, H.build_circuit(A, Xte[0]), layout)
            noise = {"gamma": cal["gamma"], "dephasing": cal["dephasing"]}
            ro = readout_errors(backend, layout)
        sim = G.gate_model(gcal["depolarizing"])
        sim.params_ = A.params_
        pred = sim.probs(A.params_[: A.n_theta], A.encode(Xte), **noise)
        pred_ro = apply_readout(pred, ro)
        kept0.append(float(np.mean([sum(p[i] for i in ONE) for p in pred])))
        kept1.append(float(np.mean([sum(p[i] for i in ONE) for p in pred_ro])))
        F = np.array([WeightQNN.qg_z(A, p, qang=True)[0] for p in pred_ro])
        d = F @ A.params_[A.n_theta:A.n_theta + A.n_head] + A.params_[-1]
        acc1 += int(np.sum((d > 0).astype(int) == yte))
        n_tot += len(yte)
    return {"backend": name, "readout errors": ro, "kept predicted (no readout)": float(np.mean(kept0)),
            "kept predicted (readout)": float(np.mean(kept1)), "A qang predicted (readout)": acc1, "inputs": n_tot}


def device_values():
    out = {}
    for r in json.load(open(DEVICE, encoding="utf-8")):
        out[r["backend"]] = {"kept device": r["kept"]["kept device"], "kept predicted 124": r["kept"]["kept predicted"],
                             "A qang device": round(r["pooled"]["A qang"] * r["pooled"]["inputs"])}
    return out


def verdict(rows, dev):
    k1 = all(abs(dev[r["backend"]]["kept device"] - r["kept predicted (readout)"]) <= 0.05 for r in rows)
    k2 = all(abs(dev[r["backend"]]["kept device"] - r["kept predicted (readout)"])
             <= 0.5 * abs(dev[r["backend"]]["kept device"] - dev[r["backend"]]["kept predicted 124"]) for r in rows)
    k3 = all(abs(dev[r["backend"]]["A qang device"] - r["A qang predicted (readout)"]) <= 2 for r in rows)
    return {"K1": k1, "K2": k2, "K3": k3}


def main():
    dev = device_values()
    rows = []
    for name in P.BACKENDS:
        r = run_backend(name)
        d = dev[name]
        print(json.dumps({"backend": name, "kept device": round(d["kept device"], 4),
                          "kept predicted 124": round(d["kept predicted 124"], 4),
                          "reproduced without readout": round(r["kept predicted (no readout)"], 4),
                          "kept predicted with readout": round(r["kept predicted (readout)"], 4),
                          "A qang device / predicted": [d["A qang device"], r["A qang predicted (readout)"]],
                          "mean readout error": round(float(np.mean([sum(e) / 2 for e in r["readout errors"]])), 4)}),
              flush=True)
        rows.append(r)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in verdict(rows, dev).items()))
    return rows


if __name__ == "__main__":
    main()
