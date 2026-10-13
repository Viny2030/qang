"""
Hamiltonian simulation and a quantum walk on the XY chain, with the qg filter (§147)

H = J sum_k (X_k X_{k+1} + Y_k Y_{k+1}) / 2 + sum_k h_k Z_k / 2 conserves the
number of excitations, and so does every gate of its Trotter circuit (RZ and
XX+YY), so the qg filter applies. With one excitation it is a tight-binding
model: a continuous-time quantum walk with hopping J and on-site energies
-h_k.

Part A (Trotter steps). More first-order Trotter steps r lower the Trotter
error but make the circuit deeper, so on a noisy device the error has a
minimum at some r*. Does the filter move r* to deeper circuits?
n = 5, J = 1, h_k uniform in [-1, 1] (seed 147), T = 3, the excitation starts
on qubit 0; r in (1, 2, 3, 4, 6, 8, 12). Error: mean_k |P_k - P_k^exact(T)|,
P_k = P(qubit k excited), exact = no Trotter error. Noiseless Trotter error
(statevector, computed before committing): 0.381, 0.363, 0.221, 0.140, 0.078,
0.054, 0.033 for r = 1 ... 12.

Part B (quantum walk). h = 0, n = 7, the walker starts on the middle qubit
(3); times t = 0.4, 0.8, 1.2, 1.6, Trotter step 0.2 (2, 4, 6, 8 steps).
sigma^2(t) = sum_k P_k (k - 3)^2 with P normalized to sum 1. Exact (no Trotter
error): 0.320, 1.278, 2.832, 4.759: ballistic, sigma^2 ~ 2 J^2 t^2 until the
front reaches the ends. A classical continuous-time random walk with the same
rate gives 0.791, 1.513, 2.113, 2.585 (diffusive). The exponent is the
least-squares slope of log sigma^2 against log t: about 1.9 quantum, about
0.85 classical. With the Trotter steps (statevector): 0.327, 1.291, 2.832,
4.737, exponent 1.94. A fully mixed weight-1 state has sigma^2 = 4.

Common set-up: IBM Runtime's Sampler in local mode, optimization_level 1, seed
137, 4000 shots, the 21 device noise models of §146.

Pre-registered predictions (committed before the run):
  H1  on every backend that runs (at least 20 of 21 run), the best error over
      r with filter + readout is below the best raw error over r.
  H2  r*(filter + readout) >= r*(raw) on at least 18 of the backends that
      run.
  H3  r*(filter + readout) > r*(raw) on at least 11 of them.
  W1  the filter + readout exponent is at least 1.5 (ballistic) on at least
      18 of the backends that run.
  W2  the raw exponent is below the filter + readout exponent on at least 18.
  W3  filter + readout sigma^2 is within 20% of the exact Trotterized value
      at all four times on at least 15.

python examples/ibm_trotter_walk_qg.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import ibm  # noqa: E402

SHOTS, SEED = 4000, 147
NA, T, RS = 5, 3.0, (1, 2, 3, 4, 6, 8, 12)
NB, MID, TIMES, DT = 7, 3, (0.4, 0.8, 1.2, 1.6), 0.2
BACKENDS = ("FakeBrisbane", "FakeSherbrooke", "FakeTorino",
            "FakeAachen", "FakeBerlin", "FakeBoston", "FakeBrussels", "FakeCusco", "FakeFez", "FakeKawasaki",
            "FakeKingston", "FakeKyiv", "FakeKyoto", "FakeMarrakesh", "FakeMiami", "FakeNighthawk", "FakeOsaka",
            "FakePittsburgh", "FakeQuebec", "FakeStrasbourg", "FakeWashingtonV2")
KINDS = ("raw", "readout", "filtered", "filtered_readout")
OUT = os.path.join(HERE, "data", "ibm_trotter_walk_qg.json")


def fields():
    return np.random.default_rng(SEED).uniform(-1, 1, NA)


def circuits():
    h = fields()
    a = [ibm.xy_trotter(NA, 1.0, h, T, r, initial=(0,)) for r in RS]
    b = [ibm.xy_trotter(NB, 1.0, None, t, int(round(t / DT)), initial=(MID,)) for t in TIMES]
    return a, b, ibm.xy_exact_occupations(NA, 1.0, h, T, 0)


def sigma2(p):
    p = np.clip(np.asarray(p, float), 0, None)
    p = p / p.sum()
    return float(p @ (np.arange(len(p)) - MID) ** 2)


def exponent(s2):
    return float(np.polyfit(np.log(TIMES), np.log(np.maximum(s2, 1e-6)), 1)[0])


def theory():
    from qiskit.quantum_info import Statevector

    a, b, exact = circuits()
    trot_err = []
    for qc in a:
        sv = Statevector(qc.remove_final_measurements(inplace=False)).data
        trot_err.append(float(np.abs(np.array([abs(sv[1 << k]) ** 2 for k in range(NA)]) - exact).mean()))
    s2 = []
    for qc in b:
        sv = Statevector(qc.remove_final_measurements(inplace=False)).data
        s2.append(sigma2([abs(sv[1 << k]) ** 2 for k in range(NB)]))
    return trot_err, s2


def sampler(backend):
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler
    except ImportError:  # pragma: no cover
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    return Sampler(mode=backend)


def run_backend(name, s2_trot):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile

    a, b, exact = circuits()
    bk = getattr(fp, name)()
    ts = [transpile(qc, bk, optimization_level=1, seed_transpiler=137) for qc in a + b]
    res = sampler(bk).run(ts, shots=SHOTS).result()
    outs = [ibm.ibm_qg(r, t, bk, weight=1) for r, t in zip(res, ts)]
    oa, ob = outs[:len(a)], outs[len(a):]
    err = {k: [float(np.abs(ibm.occupations(o, k) - exact).mean()) for o in oa] for k in KINDS}
    rstar = {k: RS[int(np.argmin(v))] for k, v in err.items()}
    s2 = {k: [sigma2(ibm.occupations(o, k)) for o in ob] for k in KINDS}
    return {"backend": name, "trotter_error": err, "r_star": rstar,
            "best_error": {k: float(min(v)) for k, v in err.items()},
            "sigma2": s2, "exponent": {k: exponent(v) for k, v in s2.items()},
            "sigma2_within_20pct": {k: bool(all(abs(x - y) <= 0.2 * y for x, y in zip(v, s2_trot)))
                                    for k, v in s2.items()},
            "kept_raw_A": [o["kept_raw"] for o in oa], "kept_raw_B": [o["kept_raw"] for o in ob],
            "two_qubit_gates": [sum(1 for i in t.data if i.operation.num_qubits == 2) for t in ts]}


def main():
    trot_err, s2_trot = theory()
    print(json.dumps({"noiseless trotter error": [round(x, 4) for x in trot_err],
                      "sigma2 trotterized": [round(x, 3) for x in s2_trot],
                      "exponent trotterized": round(exponent(s2_trot), 3)}), flush=True)
    rows = []
    for name in BACKENDS:
        try:
            r = run_backend(name, s2_trot)
            print(json.dumps({"backend": name, "r*": r["r_star"],
                              "best": {k: round(v, 4) for k, v in r["best_error"].items()},
                              "exp": {k: round(v, 2) for k, v in r["exponent"].items()},
                              "w3": r["sigma2_within_20pct"]["filtered_readout"]}), flush=True)
        except Exception as e:  # a backend that cannot run is reported, not hidden
            r = {"backend": name, "failure": f"{type(e).__name__}: {e}"[:300]}
            print(json.dumps(r), flush=True)
        rows.append(r)
    ok = [r for r in rows if "r_star" in r]
    fr = "filtered_readout"
    v = {"H1": len(ok) >= 20 and all(r["best_error"][fr] < r["best_error"]["raw"] for r in ok),
         "H2": sum(r["r_star"][fr] >= r["r_star"]["raw"] for r in ok) >= 18,
         "H3": sum(r["r_star"][fr] > r["r_star"]["raw"] for r in ok) >= 11,
         "W1": sum(r["exponent"][fr] >= 1.5 for r in ok) >= 18,
         "W2": sum(r["exponent"]["raw"] < r["exponent"][fr] for r in ok) >= 18,
         "W3": sum(r["sigma2_within_20pct"][fr] for r in ok) >= 15}
    summary = {"backends run": len(ok),
               "median best error": {k: round(float(np.median([r["best_error"][k] for r in ok])), 4) for k in KINDS},
               "median r*": {k: float(np.median([r["r_star"][k] for r in ok])) for k in KINDS},
               "r* f+ro >= raw": sum(r["r_star"][fr] >= r["r_star"]["raw"] for r in ok),
               "r* f+ro > raw": sum(r["r_star"][fr] > r["r_star"]["raw"] for r in ok),
               "median exponent": {k: round(float(np.median([r["exponent"][k] for r in ok])), 2) for k in KINDS},
               "exponent >= 1.5": {k: sum(r["exponent"][k] >= 1.5 for r in ok) for k in KINDS},
               "sigma2 within 20%": {k: sum(r["sigma2_within_20pct"][k] for r in ok) for k in KINDS}}
    print(json.dumps(summary))
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok_ else 'FAIL'}" for k, ok_ in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"noiseless_trotter_error": trot_err, "sigma2_trotterized": s2_trot, "rows": rows,
                   "summary": summary, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
