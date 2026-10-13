"""
The qg filter across the IBM fleet: 18 more device noise models (§144)

§137 measured the four qg read-outs of qang.ibm.ibm_qg (raw, readout-corrected,
filtered, filtered + readout) on three IBM fake backends. Here the same
protocol runs on every other fake backend of qiskit-ibm-runtime with at least
100 qubits, which §137 never saw: Aachen, Berlin, Boston, Brussels, Cusco,
Fez, Kawasaki, Kingston, Kyiv, Kyoto, Marrakesh, Miami, Nighthawk, Osaka,
Pittsburgh, Quebec, Strasbourg, WashingtonV2 (ECR, CZ and CX devices; 120 to
156 qubits).

Set-up (fixed before the run): circuits seeds 0-4 of §137 (5 qubits, weight
1); IBM Runtime's Sampler in local mode; optimization_level 1, seed 137; 4000
shots. Error: mean |qg_Z - ideal| over 5 qubits and 5 circuits.

Pre-registered predictions (committed before the run):
  F1  on every one of the 18 backends the filter at least halves the raw
      error.
  F2  filtered + readout is the lowest of the four on at least 15 of the 18.
  F3  the median over the 18 backends of raw error / (filtered + readout
      error) is at least 3.

python examples/ibm_fleet_qg.py

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

FLEET = ("FakeAachen", "FakeBerlin", "FakeBoston", "FakeBrussels", "FakeCusco", "FakeFez", "FakeKawasaki",
         "FakeKingston", "FakeKyiv", "FakeKyoto", "FakeMarrakesh", "FakeMiami", "FakeNighthawk", "FakeOsaka",
         "FakePittsburgh", "FakeQuebec", "FakeStrasbourg", "FakeWashingtonV2")
OUT = os.path.join(HERE, "data", "ibm_fleet_qg.json")


def main():
    rows = []
    for name in FLEET:
        try:
            r = E.run_backend(name, seeds=range(5), shots=4000)
        except Exception as e:  # a backend that cannot run is reported, not hidden
            r = {"backend": name, "error": None, "failure": f"{type(e).__name__}: {e}"[:300]}
        print(json.dumps({"backend": name, **({k: round(v, 4) for k, v in r["error"].items()} if r.get("error")
                                               else {"failure": r.get("failure")})}), flush=True)
        rows.append(r)
    ok = [r for r in rows if r.get("error")]
    e = [r["error"] for r in ok]
    ratio = [x["raw"] / x["filtered_readout"] for x in e]
    v = {"F1": len(ok) == len(FLEET) and all(x["filtered"] <= x["raw"] / 2 for x in e),
         "F2": sum(x["filtered_readout"] == min(x.values()) for x in e) >= 15,
         "F3": len(ok) == len(FLEET) and float(np.median(ratio)) >= 3}
    print(json.dumps({"backends run": len(ok), "median raw / filtered_readout": round(float(np.median(ratio)), 2),
                      "filtered_readout lowest on": sum(x["filtered_readout"] == min(x.values()) for x in e)}))
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok_ else 'FAIL'}" for k, ok_ in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
