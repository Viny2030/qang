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

Mean |qg_Z - ideal| (5 circuits, 4000 shots):

  backend          raw     readout   filtered   filtered+readout   raw / filt+ro
  FakeAachen       0.037   0.032     0.009      0.014              2.5
  FakeBerlin       0.034   0.023     0.012      0.012              2.9
  FakeBoston       0.038   0.039     0.010      0.010              3.8
  FakeBrussels     0.077   0.040     0.016      0.013              6.1
  FakeCusco        crashed (see below)
  FakeFez          0.037   0.035     0.012      0.012              3.1
  FakeKawasaki     0.062   0.045     0.014      0.014              4.4
  FakeKingston     0.076   0.036     0.009      0.015              5.0
  FakeKyiv         0.073   0.052     0.021      0.024              3.1
  FakeKyoto        0.647   0.649     0.269      0.264              2.4
  FakeMarrakesh    0.030   0.029     0.011      0.012              2.6
  FakeMiami        0.029   0.013     0.010      0.009              3.1
  FakeNighthawk    0.030   0.026     0.012      0.011              2.7
  FakeOsaka        0.094   0.064     0.018      0.020              4.8
  FakePittsburgh   0.023   0.021     0.008      0.008              2.8
  FakeQuebec       0.108   0.092     0.028      0.024              4.6
  FakeStrasbourg   0.071   0.027     0.012      0.013              5.3
  FakeWashingtonV2 0.184   0.147     0.048      0.042              4.4

  F1 FAIL: the run on FakeCusco crashed, so the prediction (every one of the
     18) cannot hold. On the 17 that ran, the filter halves the raw error on
     every one (by 2.4-8.9x).
  F2 FAIL: filtered + readout is the lowest on 10 of 17, not 15 of 18; where
     readout errors are small, readout correction adds variance and the
     filter alone is lowest (7 backends).
  F3 FAIL: the protocol required all 18; over the 17 that ran the median
     ratio is 3.12 (>= 3).
  * The crash was a qang bug: FakeCusco publishes physical qubit 0 as always
    reading 1 (P(1|0) = 1, P(0|1) = 0), the transpiler placed the circuit on
    it, and the readout inverse was singular. qang.ibm now leaves such qubits
    uncorrected and reports them (``uncorrectable_qubits``).
  * Exploratory (after scoring), FakeCusco with the fix: default layout (on
    the dead qubit) raw 0.202, filtered + readout 0.098; with
    qang.ibm.best_layout, which avoids it ([2..6]): raw 0.087, filtered +
    readout 0.021.
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
