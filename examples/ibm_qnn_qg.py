"""
qang's weight-conserving QNN on IBM devices through IBM Runtime, one job per
data set (§139)

qang.ibm.qnn_template writes the weight-1 WeightQNN (amplitude loader plus
the trained RBS layers) as ONE parameterized circuit of native XX+YY gates;
qang.ibm.run_qnn transpiles it once, sends every test input of a data set as a
single Sampler pub, and reads the decision value raw, with the qg filter, and
with the filter after calibrated readout correction. The template reproduces
WeightQNN.probs exactly (tested).

Set-up (fixed before the run): model A of §124 (weight 1, trained
noiselessly; iris, cancer, wine, digits; 189 test inputs); FakeBrisbane,
FakeSherbrooke, FakeTorino through IBM Runtime's Sampler in local mode, 1000
shots per input (as §124). Two layouts per backend and data set: the
transpiler's default (optimization_level 1, seed_transpiler 137) and
qang.ibm.best_chain. Noiseless: 185 of 189 per backend, 555 of 567 pooled.
§124 (one circuit per input, Aer with the device noise model) gave A with the
filter 182 / 184 / 185 = 551 pooled.

Pre-registered predictions (committed before the run):
  Q1  pooled over the three backends, best_chain + filter + readout
      correction is at least as accurate as the default layout + filter.
  Q2  pooled, best_chain + filter + readout correction gets at least 552 of
      567 (within 3 inputs of noiseless).
  Q3  pooled, the default layout + filter is within 3 inputs of §124's 551
      (the one-job template agrees with the per-input circuits).

python examples/ibm_qnn_qg.py

Findings:

FINDINGS_PLACEHOLDER
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import qnn_classifier_qg as Q  # noqa: E402
import qnn_full_pipeline_qg as P  # noqa: E402
import qnn_gate_noise_qg as G  # noqa: E402
import qnn_ionq_datasets_qg as DS  # noqa: E402
from qang import ibm  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

BACKENDS = ("FakeBrisbane", "FakeSherbrooke", "FakeTorino")
SHOTS = 1000
OUT = os.path.join(HERE, "data", "ibm_qnn_qg.json")


def run_backend(name, datasets=Q.DATASETS, shots=SHOTS):
    import qiskit_ibm_runtime.fake_provider as fp

    b = getattr(fp, name)()
    acc = {"noiseless": 0, "default raw": 0, "default qang": 0, "default qang+readout": 0,
           "best raw": 0, "best qang": 0, "best qang+readout": 0}
    kept = {"default": [], "best": []}
    layouts = {}
    total = 0
    for ds in datasets:
        Xtr, Xte, ytr, yte = P.data(ds)
        A = WeightQNN(G.N, 1).fit(Xtr, ytr, G.EPOCHS, seed=DS.TRAIN_SEED)
        acc["noiseless"] += int(np.sum(A.predict(Xte) == yte))
        total += len(yte)
        for tag, lay in (("default", None), ("best", "best")):
            r = ibm.run_qnn(A, Xte, b, layout=lay, shots=shots)
            layouts[f"{ds} {tag}"] = r["layout"]
            kept[tag] += r["kept_raw"]
            for k, key in (("raw", "raw"), ("qang", "qang"), ("qang+readout", "qang_readout")):
                acc[f"{tag} {k}"] += int(np.sum((r["decision"][key] > 0).astype(int) == yte))
    return {"backend": name, "inputs": total, "correct": acc,
            "kept": {k: float(np.mean(v)) for k, v in kept.items()}, "layouts": layouts}


def verdict(rows):
    tot = {k: sum(r["correct"][k] for r in rows) for k in rows[0]["correct"]}
    return tot, {"Q1": tot["best qang+readout"] >= tot["default qang"],
                 "Q2": tot["best qang+readout"] >= 552,
                 "Q3": abs(tot["default qang"] - 551) <= 3}


def main():
    rows = []
    for name in BACKENDS:
        r = run_backend(name)
        print(json.dumps({"backend": name, **r["correct"], "kept": {k: round(v, 3) for k, v in r["kept"].items()}}),
              flush=True)
        rows.append(r)
    tot, v = verdict(rows)
    print(json.dumps({"pooled": tot}))
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "pooled": tot, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
