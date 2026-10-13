"""
Planning shots for a target qg precision on IBM devices (§143)

With little device time, the question before a run is how many shots give the
precision you need once the qg filter has discarded its share.
qang.ibm.plan_shots answers it from the calibration: the filter keeps a
fraction k (predict_kept_fraction, kept_mixed_floor), a filtered qg_Z from N k
shots has standard error sqrt((1 - qg^2)/(N k)), so N = max_i (1 - qg_i^2) /
(k se^2); the device time is about N (circuit duration + repetition delay).

Set-up (fixed before the run): circuits seeds 0 and 1 of §137 (5 qubits,
weight 1); FakeBrisbane, FakeSherbrooke, FakeTorino through IBM Runtime's
Sampler in local mode; transpiled with optimization_level 1, seed 137; target
standard error 0.02 with the ideal qg values; the planned N run 30 times
(independent repetitions in one job). Measured standard error = standard
deviation of each qubit's filtered qg_Z over the 30 repetitions.

Pre-registered predictions (committed before the run):
  P1  on every backend the worst qubit's measured standard error is at most
      0.025 (1.25 x the target).
  P2  on every backend the mean ratio measured / predicted standard error
      (per qubit, predicted from the planned N, the predicted k and the
      ideal qg) lies in [0.8, 1.25].

python examples/ibm_shot_plan_qg.py

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

TARGET = 0.02
REPEATS = 30
OUT = os.path.join(HERE, "data", "ibm_shot_plan_qg.json")


def run_backend(name, seeds=(0, 1)):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile

    b = getattr(fp, name)()
    out = []
    for s in seeds:
        qc = E.circuit(s)
        z = np.array(E.ideal(qc))
        qm = qc.copy()
        qm.measure_all()
        t = transpile(qm, b, optimization_level=1, seed_transpiler=137)
        plan = ibm.plan_shots(t, b, 1, TARGET, qg=z)
        res = E.sampler(b).run([t] * REPEATS, shots=plan["shots"]).result()
        vals = []
        for r in res:
            c = IO._from_strings(IO._bitarray(r, None).get_counts(), qubit0_last=True)
            vals.append([q for q, _ in IO.filtered_qg_z_profile(c, 1)])
        sd = np.std(np.array(vals), axis=0, ddof=1)
        pred = np.sqrt((1 - z ** 2) / (plan["shots"] * plan["kept"]))
        out.append({"seed": s, "plan": plan, "measured_se": sd.tolist(), "predicted_se": pred.tolist()})
    worst = max(max(o["measured_se"]) for o in out)
    ratio = float(np.mean([m / p for o in out for m, p in zip(o["measured_se"], o["predicted_se"])]))
    return {"backend": name, "circuits": out, "worst_se": float(worst), "mean_ratio": ratio}


def verdict(rows):
    return {"P1": all(r["worst_se"] <= 0.025 for r in rows),
            "P2": all(0.8 <= r["mean_ratio"] <= 1.25 for r in rows)}


def main():
    rows = []
    for name in E.BACKENDS:
        r = run_backend(name)
        print(json.dumps({"backend": name, "shots": [o["plan"]["shots"] for o in r["circuits"]],
                          "qpu_s": [round(o["plan"]["qpu_seconds"], 2) for o in r["circuits"]],
                          "worst_se": round(r["worst_se"], 4), "mean_ratio": round(r["mean_ratio"], 3)}), flush=True)
        rows.append(r)
    v = verdict(rows)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
