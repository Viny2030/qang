"""
Zero-noise extrapolation with the qg filter on IBM device noise (§142)

IBM Runtime's own mitigation (resilience levels, ZNE) has no effect in local
mode, so qang.ibm.zne_qg does ZNE itself: every two-qubit gate of the
transpiled circuit is folded (G -> G G^dagger G, noise scale 3), both scales run
on the Sampler, and qg_Z is extrapolated linearly to zero noise. Folding keeps a
weight-preserving circuit weight-preserving, so ZNE combines with the qg
filter and readout correction (qang.ibm.ibm_qg).

Set-up (fixed before the run): the ten 5-qubit weight-1 circuits of §137
(seeds 0-9); FakeBrisbane, FakeSherbrooke, FakeTorino through IBM Runtime's
Sampler in local mode; transpiled once (optimization_level 1, seed 137);
scales (1, 3), 8000 shots each; linear extrapolation. Error: mean
|qg_Z - ideal| over 5 qubits and 10 circuits, for raw, readout, filtered and
filtered_readout, each without and with ZNE.

Pre-registered predictions (committed before the run):
  Z1  on every backend ZNE lowers the error of filtered_readout.
  Z2  on every backend filtered_readout without ZNE beats raw with ZNE (the
      filter does more than ZNE alone).
  Z3  ZNE + filtered_readout is the best of the eight on at least two of the
      three backends.

python examples/ibm_zne_qg.py

Findings:

Mean |qg_Z - ideal| (10 circuits, 8000 shots per scale):

                    raw     readout   filtered   filtered_readout
  FakeBrisbane      0.106   0.075     0.025      0.024
    + ZNE           0.053   0.025     0.012      0.015
  FakeSherbrooke    0.064   0.037     0.016      0.014
    + ZNE           0.034   0.013     0.011      0.011
  FakeTorino        0.102   0.036     0.020      0.013
    + ZNE           0.085   0.022     0.015      0.014

  Z1 FAIL: ZNE lowers filtered_readout on brisbane (0.024 -> 0.015) and
     sherbrooke (0.014 -> 0.011), not on torino (0.013 -> 0.014).
  Z2 PASS: the filter (with readout correction) beats ZNE alone on every
     backend, by 2-7x.
  Z3 FAIL: the best combination is filter + ZNE (without readout
     correction) on brisbane and sherbrooke, filter + readout without ZNE on
     torino.
  * ZNE and the filter stack: filter + ZNE reaches 0.011-0.015 on every
     backend, the lowest errors of these sections.
  * Readout correction and ZNE overlap: folding scales the gate noise but not
    the readout, and once the filter has removed most readout damage the two
    corrections add variance rather than accuracy (torino, where readout
    errors are largest, is the exception where readout correction alone is
    best).
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

KINDS = ("raw", "readout", "filtered", "filtered_readout")
OUT = os.path.join(HERE, "data", "ibm_zne_qg.json")


def run_backend(name, seeds=range(10), shots=8000):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile

    b = getattr(fp, name)()
    err = {f"{k}{z}": [] for k in KINDS for z in ("", " + zne")}
    for s in seeds:
        qc = E.circuit(s)
        z = np.array(E.ideal(qc))
        qm = qc.copy()
        qm.measure_all()
        t = transpile(qm, b, optimization_level=1, seed_transpiler=137)
        r = ibm.zne_qg(E.sampler(b), t, b, 1, scales=(1, 3), shots=shots)
        for k in KINDS:
            err[k].append(float(np.mean(np.abs(np.array(r["values"][k][0]) - z))))
            err[f"{k} + zne"].append(float(np.mean(np.abs(np.array(r["zne"][k]) - z))))
    return {"backend": name, "error": {k: float(np.mean(v)) for k, v in err.items()}}


def verdict(rows):
    e = {r["backend"]: r["error"] for r in rows}
    return {"Z1": all(x["filtered_readout + zne"] < x["filtered_readout"] for x in e.values()),
            "Z2": all(x["filtered_readout"] < x["raw + zne"] for x in e.values()),
            "Z3": sum(x["filtered_readout + zne"] == min(x.values()) for x in e.values()) >= 2}


def main():
    rows = []
    for name in E.BACKENDS:
        r = run_backend(name)
        print(json.dumps({"backend": name, **{k: round(v, 4) for k, v in r["error"].items()}}), flush=True)
        rows.append(r)
    v = verdict(rows)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
