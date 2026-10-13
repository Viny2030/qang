"""
What the qg filter cannot see: coherent conditional-phase (ZZ) errors on IBM
device noise models (§145)

The device noise models used so far (§137-§144) are incoherent channels. Real
two-qubit gates also carry coherent errors; a common one is a miscalibrated
conditional phase, i.e. an extra exp(-i eps ZZ / 2) on the pair of every
two-qubit gate. That error conserves the Hamming weight, so the qg filter
cannot remove it: it changes the state inside the sector.

Set-up (fixed before the run): the ten 5-qubit weight-1 circuits of §137
(seeds 0-9), transpiled to FakeBrisbane, FakeSherbrooke and FakeTorino
(optimization_level 1, seed 137); Aer with the device noise model plus a
coherent RZZ(eps) error after every two-qubit gate on its pair,
eps in {0, 0.05, 0.10} rad; 8000 shots (seed 1450). Error: mean
|qg_Z - ideal| over 5 qubits and 10 circuits; kept = weight-1 share of the
raw shots.

Pre-registered predictions (committed before the run):
  X1  on every backend the kept fraction at eps = 0.10 is within 0.01 of
      eps = 0 (the error does not move shots out of the sector).
  X2  on every backend the filtered error at eps = 0.10 is at least 1.5x
      that at eps = 0.
  X3  on every backend the filter's gain (raw / filtered error) is smaller at
      eps = 0.10 than at eps = 0.

python examples/ibm_coherent_zz_qg.py

Findings:

Mean |qg_Z - ideal| raw / filtered, and kept fraction (10 circuits, 8000 shots):

  backend          eps = 0                 eps = 0.05              eps = 0.10
  FakeBrisbane     0.103 / 0.024, 0.731    0.104 / 0.024, 0.729    0.107 / 0.025, 0.722
  FakeSherbrooke   0.064 / 0.016, 0.819    0.065 / 0.016, 0.816    0.068 / 0.017, 0.804
  FakeTorino       0.101 / 0.018, 0.690    0.107 / 0.019, 0.685    0.124 / 0.030, 0.671

  X1 FAIL: the kept fraction falls by 0.009 / 0.015 / 0.020 at eps = 0.10:
     the error does move shots out of the sector.
  X2 FAIL: the filtered error grows by 1.04x / 1.04x on the ECR devices and
     1.62x on torino (CZ); the prediction needed 1.5x on every backend.
  X3 FAIL: the filter's gain falls on brisbane (4.31 -> 4.30) and torino
     (5.53 -> 4.20) but rises on sherbrooke (4.01 -> 4.09).
  * The premise was wrong in the same way as §133: the conditional phase is
    added to the NATIVE gate (ECR or CZ), and inside the native
    decomposition of an XX+YY gate the qubits are not in the computational
    frame, so the error is not weight-preserving for the logical circuit. Part
    of it leaves the sector (kept falls) and the filter removes that part.
  * Its effect depends on the native gate: on the ECR devices a 0.1 rad
    conditional phase barely changes the filtered read-out; on torino (CZ) it
    raises the filtered error by 60%. A ZZ error on the logical pair itself
    (e.g. always-on ZZ between neighbours idling in the computational frame)
    would stay invisible to the filter; it was not simulated here.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import ibm_runtime_qg as E  # noqa: E402
from qang import interop as IO  # noqa: E402

EPS = (0.0, 0.05, 0.10)
SHOTS = 8000
OUT = os.path.join(HERE, "data", "ibm_coherent_zz_qg.json")


def noise_model(backend, eps):
    from qiskit.circuit.library import RZZGate
    from qiskit_aer.noise import NoiseModel, coherent_unitary_error

    nm = NoiseModel.from_backend(backend)
    if eps == 0:
        return nm
    err = coherent_unitary_error(RZZGate(eps).to_matrix())
    name = next(g for g in ("ecr", "cz", "cx") if g in backend.target.operation_names)
    for qs in backend.target[name]:
        if qs is not None and len(qs) == 2:
            nm.add_quantum_error(err, name, list(qs), warnings=False)
    return nm


def run_backend(name, seeds=range(10), eps_list=EPS, shots=SHOTS):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    b = getattr(fp, name)()
    circs, ideals = [], []
    for s in seeds:
        qc = E.circuit(s)
        ideals.append(np.array(E.ideal(qc)))
        qm = qc.copy()
        qm.measure_all()
        circs.append(transpile(qm, b, optimization_level=1, seed_transpiler=137))
    out = {}
    for eps in eps_list:
        sim = AerSimulator(noise_model=noise_model(b, eps))
        res = sim.run(circs, shots=shots, seed_simulator=1450).result()
        raw, filt, kept = [], [], []
        for i, z in enumerate(ideals):
            c = IO.canonical_counts(res.get_counts(i), sdk="qiskit")
            raw.append(np.mean([abs(q - zi) for (q, _), zi in zip(IO.qg_z_profile(c), z)]))
            filt.append(np.mean([abs(q - zi) for (q, _), zi in zip(IO.filtered_qg_z_profile(c, 1), z)]))
            kept.append(IO.weight_filter(c, 1)[1])
        out[str(eps)] = {"raw": float(np.mean(raw)), "filtered": float(np.mean(filt)), "kept": float(np.mean(kept))}
    return {"backend": name, "by_eps": out}


def verdict(rows):
    lo, hi = str(EPS[0]), str(EPS[-1])
    return {"X1": all(abs(r["by_eps"][hi]["kept"] - r["by_eps"][lo]["kept"]) <= 0.01 for r in rows),
            "X2": all(r["by_eps"][hi]["filtered"] >= 1.5 * r["by_eps"][lo]["filtered"] for r in rows),
            "X3": all(r["by_eps"][hi]["raw"] / r["by_eps"][hi]["filtered"]
                      < r["by_eps"][lo]["raw"] / r["by_eps"][lo]["filtered"] for r in rows)}


def main():
    rows = []
    for name in E.BACKENDS:
        r = run_backend(name)
        print(json.dumps({"backend": name, **{e: {k: round(v, 4) for k, v in d.items()} for e, d in r["by_eps"].items()}}),
              flush=True)
        rows.append(r)
    v = verdict(rows)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
