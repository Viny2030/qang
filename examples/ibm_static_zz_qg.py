"""
The filter's blind spot: always-on ZZ crosstalk between coupled qubits (§150)

§145 put a coherent ZZ error on the pair of every two-qubit gate and found it
partly filtered, because there it acts inside the native decomposition of the
logical gate, where the qubits are not in the computational frame. Always-on
ZZ is different: every pair of coupled qubits picks up exp(-i theta ZZ / 2)
all the time, idle or not, with theta = 2 pi nu t (nu: the residual ZZ rate of
the coupler). Between logical gates the qubits are in the computational
frame, where ZZ only changes phases inside the weight sector: the filter
cannot see it. How much of the damage passes the filter?

Set-up (fixed before the run): the ten 5-qubit weight-1 circuits of §137
(seeds 0-9), transpiled to FakeBrisbane, FakeSherbrooke, FakeTorino
(optimization_level 1, seed 137); qang.ibm.add_static_zz inserts after every
layer of the transpiled circuit, on every coupling-map edge between used
qubits, exp(-i theta ZZ/2) with theta = 2 pi nu x (layer duration from the
device calibration); nu in {0, 5, 15} kHz; Aer with the device noise model
(the ZZ terms themselves noiseless), 8000 shots, seed 1500. Error: mean
|qg_Z - ideal| over 5 qubits and 10 circuits. Structural (before the run):
the circuits last about 9-10 us on the two ECR devices and 2.1 us on
FakeTorino, so at 15 kHz each edge accumulates about 0.9 rad there and
0.2 rad on FakeTorino.

Pre-registered predictions (committed before the run):
  Z1  the filtered error at 15 kHz is above the filtered error at 0 kHz on
      all 3 backends.
  Z2  the filter removes less than half of the added error: filtered(15) -
      filtered(0) > 0.5 x (raw(15) - raw(0)) on all 3 backends.
  Z3  the kept fraction changes by less than 0.01 between 0 and 15 kHz on all
      3 backends.
  Z4  the increase of the filtered error from 0 to 15 kHz is smallest on
      FakeTorino (shortest circuits).

python examples/ibm_static_zz_qg.py
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

NUS = (0.0, 5e3, 15e3)
SHOTS = 8000
BACKENDS = ("FakeBrisbane", "FakeSherbrooke", "FakeTorino")
OUT = os.path.join(HERE, "data", "ibm_static_zz_qg.json")


def run_backend(name, seeds=range(10), nus=NUS, shots=SHOTS):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    b = getattr(fp, name)()
    circs, ideals = [], []
    for s in seeds:
        qc = E.circuit(s)
        ideals.append(np.array(E.ideal(qc)))
        qm = qc.copy()
        qm.measure_all()
        circs.append(transpile(qm, b, optimization_level=1, seed_transpiler=137))
    sim = AerSimulator(noise_model=NoiseModel.from_backend(b))
    out = {}
    for nu in nus:
        cs = [ibm.add_static_zz(t, b, nu) for t in circs]
        res = sim.run(cs, shots=shots, seed_simulator=1500).result()
        raw, filt, kept = [], [], []
        for i, z in enumerate(ideals):
            c = IO.canonical_counts(res.get_counts(i), sdk="qiskit")
            raw.append(np.mean([abs(q - zi) for (q, _), zi in zip(IO.qg_z_profile(c), z)]))
            filt.append(np.mean([abs(q - zi) for (q, _), zi in zip(IO.filtered_qg_z_profile(c, 1), z)]))
            kept.append(IO.weight_filter(c, 1)[1])
        out[str(int(nu))] = {"raw": float(np.mean(raw)), "filtered": float(np.mean(filt)), "kept": float(np.mean(kept))}
    return {"backend": name, "by_nu": out}


def verdict(rows):
    d = {r["backend"]: r["by_nu"] for r in rows}
    lo, hi = str(int(NUS[0])), str(int(NUS[-1]))
    inc = {n: x[hi]["filtered"] - x[lo]["filtered"] for n, x in d.items()}
    return {"Z1": all(x[hi]["filtered"] > x[lo]["filtered"] for x in d.values()),
            "Z2": all(x[hi]["filtered"] - x[lo]["filtered"] > 0.5 * (x[hi]["raw"] - x[lo]["raw"]) for x in d.values()),
            "Z3": all(abs(x[hi]["kept"] - x[lo]["kept"]) < 0.01 for x in d.values()),
            "Z4": min(inc, key=inc.get) == "FakeTorino"}


def main():
    rows = []
    for name in BACKENDS:
        r = run_backend(name)
        print(json.dumps({"backend": name, **{k: {kk: round(vv, 4) for kk, vv in v.items()}
                                              for k, v in r["by_nu"].items()}}), flush=True)
        rows.append(r)
    v = verdict(rows)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
