"""
Predicting the qg filter's kept fraction from IBM calibration, and choosing
the qubits with it (§138)

qang.ibm.predict_kept_fraction turns a device's published calibration (gate
errors, T1, T2, gate durations, readout errors) into the fraction of shots a
weight-preserving circuit will keep under the qg filter, before anything is
run. qang.ibm.best_chain uses it to pick the chain of physical qubits with the
highest predicted kept fraction.

Set-up (fixed before the run): the 5-qubit weight-1 circuits of §137 (seeds
0-2, measured); FakeBrisbane, FakeSherbrooke, FakeTorino through IBM
Runtime's Sampler in local mode, 4000 shots per circuit. Per backend, ten
layouts: the transpiler's default (optimization_level 1, seed_transpiler
137), eight random chains (qang.ibm.linear_chains, seed 138, the first eight)
and best_chain over 100 chains (seed 0). Measured kept fraction = mean over the
three circuits of the share of weight-1 shots.

Pre-registered predictions (committed before the run):
  L1  on every backend, the mean |predicted - measured| kept fraction over
      the ten layouts is at most 0.05.
  L2  on every backend, the Spearman rank correlation between predicted and
      measured kept fraction over the ten layouts is at least 0.7.
  L3  on every backend, best_chain's measured kept fraction is at least the
      default layout's minus 0.01.

python examples/ibm_layout_qg.py

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

import ibm_runtime_qg as E  # noqa: E402
from qang import ibm  # noqa: E402
from qang import interop as IO  # noqa: E402

SEEDS = (0, 1, 2)
SHOTS = 4000
OUT = os.path.join(HERE, "data", "ibm_layout_qg.json")


def measure_layout(backend, layout, seeds=SEEDS, shots=SHOTS):
    from qiskit import transpile

    pred, kept = [], []
    for s in seeds:
        qm = E.circuit(s)
        qm.measure_all()
        kw = {"initial_layout": layout} if layout is not None else {}
        t = transpile(qm, backend, optimization_level=1, seed_transpiler=137, **kw)
        pred.append(ibm.predict_kept_fraction(t, backend, 1)["kept"])
        res = E.sampler(backend).run([t], shots=shots).result()[0]
        c = IO._from_strings(IO._bitarray(res, None).get_counts(), qubit0_last=True)
        kept.append(IO.weight_filter(c, 1)[1])
    return float(np.mean(pred)), float(np.mean(kept))


def run_backend(name):
    import qiskit_ibm_runtime.fake_provider as fp
    from scipy.stats import spearmanr

    b = getattr(fp, name)()
    layouts = {"default": None}
    for i, ch in enumerate(ibm.linear_chains(b, E.N, limit=8, seed=138)):
        layouts[f"random {i}"] = ch
    qm = E.circuit(0)
    qm.measure_all()
    layouts["best_chain"] = ibm.best_chain(qm, b, 1, limit=100, seed=0, seed_transpiler=137)[0]
    rows = {k: measure_layout(b, v) for k, v in layouts.items()}
    p = np.array([v[0] for v in rows.values()])
    m = np.array([v[1] for v in rows.values()])
    return {"backend": name, "layouts": {k: {"layout": layouts[k], "predicted": v[0], "measured": v[1]}
                                         for k, v in rows.items()},
            "mae": float(np.mean(np.abs(p - m))), "spearman": float(spearmanr(p, m)[0]),
            "best_minus_default": rows["best_chain"][1] - rows["default"][1]}


def verdict(rows):
    return {"L1": all(r["mae"] <= 0.05 for r in rows),
            "L2": all(r["spearman"] >= 0.7 for r in rows),
            "L3": all(r["best_minus_default"] >= -0.01 for r in rows)}


def main():
    rows = []
    for name in E.BACKENDS:
        r = run_backend(name)
        print(json.dumps({"backend": name, "mae": round(r["mae"], 4), "spearman": round(r["spearman"], 3),
                          "best - default": round(r["best_minus_default"], 4),
                          "pred/meas": {k: [round(v["predicted"], 3), round(v["measured"], 3)]
                                        for k, v in r["layouts"].items()}}), flush=True)
        rows.append(r)
    v = verdict(rows)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
