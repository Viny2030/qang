"""
qg read-out on IBM devices through IBM Runtime: readout correction and the qg
filter, alone and together (§137)

qang.ibm reads a SamplerV2 result (IBM Runtime's client-side Sampler; here in
local mode on the fake backends, which run Aer with each device's noise model)
for a circuit transpiled to the device, and returns per-qubit qg_Z four ways:
  raw              the measured shots;
  readout          readout-corrected with the published per-qubit asymmetric
                   errors of the measured physical qubits (inverse of the
                   tensor-product confusion matrix);
  filtered         the qg filter (Hamming weight 1) on the raw shots;
  filtered_readout the qg filter on the readout-corrected distribution.

Circuits (fixed before the run): 10 random weight-1 circuits on 5 qubits, X on
qubit 0 then two layers of XX+YY (RBS-type) gates on neighbours with angles
uniform in [0.3, 2.8] (seeds 0-9); qang.ibm.WeightPreservationCheck confirms
each is weight-preserving with initial weight 1. Transpiled with
optimization_level 1, seed_transpiler 137; 8000 shots each; backends
FakeBrisbane, FakeSherbrooke, FakeTorino. Error: mean |qg_Z - ideal| over
the 5 qubits and 10 circuits.

Pre-registered predictions (committed before the run):
  M1  readout correction lowers the error of the raw read-out on every
      backend.
  M2  filtered_readout has the lowest error of the four on every backend.
  M3  the filter alone beats readout correction alone on every backend
      (gate errors dominate on these devices, §124/§129).

python examples/ibm_runtime_qg.py

Findings:

Mean |qg_Z - ideal| (5 qubits x 10 circuits, 8000 shots):

  backend          raw      readout   filtered   filtered + readout   kept raw / readout
  FakeBrisbane     0.104    0.074     0.023      0.022                0.73 / 0.82
  FakeSherbrooke   0.065    0.038     0.019      0.016                0.82 / 0.90
  FakeTorino       0.102    0.036     0.018      0.013                0.69 / 0.92

  M1 PASS: readout correction lowers the raw error on every backend (by
     29-64%).
  M2 PASS: filter + readout correction is the best on every backend.
  M3 PASS: the filter alone beats readout correction alone everywhere.
  * The two corrections are complementary: readout correction alone leaves
    2-3x the filtered error; the filter alone is close to the combination,
    which adds most on torino (0.018 -> 0.013), whose measured qubits have
    the largest readout errors.
  * After readout correction the weight-1 mass rises from 0.69-0.82 to
    0.82-0.92: about a third to two thirds of the shots the filter would
    discard were readout errors, not gate errors.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import ibm  # noqa: E402

N = 5
SEEDS = range(10)
SHOTS = 8000
BACKENDS = ("FakeBrisbane", "FakeSherbrooke", "FakeTorino")
KINDS = ("raw", "readout", "filtered", "filtered_readout")
OUT = os.path.join(HERE, "data", "ibm_runtime_qg.json")


def circuit(seed):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate

    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(N)
    qc.x(0)
    for _ in range(2):
        for i in range(N - 1):
            qc.append(XXPlusYYGate(rng.uniform(0.3, 2.8)), [i, i + 1])
    return qc


def ideal(qc):
    from qiskit.quantum_info import SparsePauliOp, Statevector

    sv = Statevector(qc)
    return [float(sv.expectation_value(SparsePauliOp("I" * (N - 1 - i) + "Z" + "I" * i)).real) for i in range(N)]


def sampler(backend):
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler
    except ImportError:  # pragma: no cover
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    return Sampler(mode=backend)


def run_backend(name, seeds=SEEDS, shots=SHOTS):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile
    from qiskit.transpiler import PassManager

    b = getattr(fp, name)()
    errs = {k: [] for k in KINDS}
    kept = {"raw": [], "readout": []}
    for s in seeds:
        qc = circuit(s)
        pm = PassManager([ibm.WeightPreservationCheck()])
        pm.run(qc)
        assert pm.property_set["qg_weight_preserving"] and pm.property_set["qg_initial_weight"] == 1
        z = ideal(qc)
        qm = qc.copy()
        qm.measure_all()
        t = transpile(qm, b, optimization_level=1, seed_transpiler=137)
        res = sampler(b).run([t], shots=shots).result()[0]
        out = ibm.ibm_qg(res, t, b, weight=1)
        for k in KINDS:
            errs[k].append(float(np.mean([abs(v - zi) for (v, _), zi in zip(out[k], z)])))
        kept["raw"].append(out["kept_raw"])
        kept["readout"].append(out["kept_readout"])
    return {"backend": name, "error": {k: float(np.mean(v)) for k, v in errs.items()},
            "kept": {k: float(np.mean(v)) for k, v in kept.items()}}


def verdict(rows):
    e = {r["backend"]: r["error"] for r in rows}
    return {"M1": all(x["readout"] < x["raw"] for x in e.values()),
            "M2": all(x["filtered_readout"] == min(x.values()) for x in e.values()),
            "M3": all(x["filtered"] < x["readout"] for x in e.values())}


def main():
    rows = []
    for name in BACKENDS:
        r = run_backend(name)
        print(json.dumps({"backend": name, **{k: {kk: round(vv, 4) for kk, vv in v.items()}
                                               for k, v in r.items() if k != "backend"}}), flush=True)
        rows.append(r)
    v = verdict(rows)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
