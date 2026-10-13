"""
Choosing the qubits for qang's ring-coupled QNN by scoring the circuit that
runs (§141)

§139 found that best_chain kept fewer shots than the transpiler's default on
two backends: the QNN's last sublayer couples qubits 4 and 0 (a ring), and the
chain was scored on a circuit with bound angles rather than the parameterized
template that ran. qang.ibm.best_layout scores every candidate (the default
layout and 100 chains of the coupling map) by transpiling the circuit that will
run, routing included, and keeps the highest predicted kept fraction
(kept_mixed_floor), so it is never worse than the default by the prediction.

Design data (computed before the run, prediction only): best_layout keeps the
default on FakeSherbrooke and FakeTorino (no chain is predicted better for the
ring) and picks the chain [104, 111, 122, 123, 124] on FakeBrisbane
(predicted 0.597 vs 0.572).

Set-up: as §139 (model A, four datasets, 189 test inputs, 1000 shots, IBM
Runtime local Sampler, seed_transpiler 137), layouts "default" and "best"
(qang.ibm.run_qnn(layout="best")).

Pre-registered predictions (committed before the run):
  R1  on every backend the measured kept fraction with best_layout is at least
      the default's minus 0.005.
  R2  on FakeBrisbane best_layout keeps at least 0.01 more of the shots than
      the default.
  R3  pooled over the three backends, best_layout + filter + readout
      correction gets at least 551 of 567 (§139's best).

python examples/ibm_qnn_layout_qg.py

Findings:

                  kept default / best    correct (of 189) default: raw qang qang+ro / best: raw qang qang+ro
  FakeBrisbane    0.639 / 0.661          183 184 184 / 181 184 184
  FakeSherbrooke  0.721 / 0.722          183 182 183 / 182 182 182
  FakeTorino      0.757 / 0.755          184 184 184 / 182 183 184
  pooled best + filter + readout: 550 of 567

  R1 PASS: best_layout never keeps fewer shots than the default (by more
     than 0.005); on sherbrooke and torino it is the default layout.
  R2 PASS: on brisbane it keeps 2.2 points more shots (0.661 vs 0.639), as
     predicted (+2.5).
  R3 FAIL: 550, one short of 551.
  * best_layout fixes §139's problem: scoring the circuit that runs, routing
    included, it no longer picks layouts worse than the default.
  * Same-layout runs differ by up to 2 inputs (torino raw: 184 vs 182 with
    the identical layout): accuracy differences of 1-2 inputs between these
    pipelines are shot noise; the kept fraction is the sharper measure.
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
OUT = os.path.join(HERE, "data", "ibm_qnn_layout_qg.json")


def run_backend(name, shots=1000):
    import qiskit_ibm_runtime.fake_provider as fp

    b = getattr(fp, name)()
    acc = {f"{t} {k}": 0 for t in ("default", "best") for k in ("raw", "qang", "qang+readout")}
    kept = {"default": [], "best": []}
    layouts = {}
    for ds in Q.DATASETS:
        Xtr, Xte, ytr, yte = P.data(ds)
        A = WeightQNN(G.N, 1).fit(Xtr, ytr, G.EPOCHS, seed=DS.TRAIN_SEED)
        for tag, lay in (("default", None), ("best", "best")):
            r = ibm.run_qnn(A, Xte, b, layout=lay, shots=shots)
            layouts[f"{ds} {tag}"] = r["layout"]
            kept[tag] += r["kept_raw"]
            for k, key in (("raw", "raw"), ("qang", "qang"), ("qang+readout", "qang_readout")):
                acc[f"{tag} {k}"] += int(np.sum((r["decision"][key] > 0).astype(int) == yte))
    return {"backend": name, "correct": acc, "kept": {k: float(np.mean(v)) for k, v in kept.items()},
            "layouts": layouts}


def verdict(rows):
    by = {r["backend"]: r for r in rows}
    pooled = sum(r["correct"]["best qang+readout"] for r in rows)
    return {"R1": all(r["kept"]["best"] >= r["kept"]["default"] - 0.005 for r in rows),
            "R2": by["FakeBrisbane"]["kept"]["best"] - by["FakeBrisbane"]["kept"]["default"] >= 0.01,
            "R3": pooled >= 551}, pooled


def main():
    rows = []
    for name in BACKENDS:
        r = run_backend(name)
        print(json.dumps({"backend": name, **r["correct"], "kept": {k: round(v, 4) for k, v in r["kept"].items()},
                          "best layout": r["layouts"]["iris best"]}), flush=True)
        rows.append(r)
    v, pooled = verdict(rows)
    print(json.dumps({"pooled best qang+readout": pooled}))
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
