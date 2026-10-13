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

Findings:

r* = the number of Trotter steps with the lowest error; best = that error;
exponent = slope of log sigma^2 against log t in part B; W3 = sigma^2 within
20% at all four times (filter + readout); kept = share of shots in the
weight-1 sector at r = 12 (96 native two-qubit gates):

  backend          r* raw  f+ro  best raw  f+ro   exp raw  f+ro  W3   kept
  FakeBrisbane          6     6     0.187  0.091    0.59  1.76  yes   0.35
  FakeSherbrooke        8     8     0.152  0.072    0.14  1.26  no    0.49
  FakeTorino            8    12     0.152  0.054    0.59  1.83  yes   0.50
  FakeAachen            8    12     0.094  0.043    1.13  1.90  yes   0.73
  FakeBerlin            8    12     0.101  0.046    1.12  1.88  yes   0.69
  FakeBoston            8    12     0.123  0.063    1.30  1.84  yes   0.61
  FakeBrussels          6     8     0.175  0.079    0.70  1.70  no    0.40
  FakeCusco             1     6     0.246  0.154    0.17  1.29  no    0.30
  FakeFez               8    12     0.129  0.067    0.87  1.90  yes   0.58
  FakeKawasaki          6     8     0.147  0.078    0.89  1.84  yes   0.49
  FakeKingston         12    12     0.099  0.036    0.91  1.96  yes   0.68
  FakeKyiv              6     8     0.159  0.084    0.98  1.84  yes   0.43
  FakeKyoto             6     6     0.378  0.185    0.00 -0.02  no    0.15
  FakeMarrakesh        12    12     0.097  0.039    1.31  1.88  yes   0.72
  FakeMiami             8    12     0.082  0.039    1.05  1.89  yes   0.77
  FakeNighthawk         8    12     0.095  0.046    1.31  1.96  yes   0.70
  FakeOsaka             6     8     0.176  0.091    0.85  1.72  yes   0.39
  FakePittsburgh        8    12     0.102  0.054    1.20  1.95  yes   0.71
  FakeQuebec            4     6     0.222  0.118    0.40  1.38  no    0.27
  FakeStrasbourg        6    12     0.131  0.056    0.83  1.89  yes   0.57
  FakeWashingtonV2      4     6     0.252  0.144    0.59  1.31  no    0.22

  H1 PASS: all 21 ran; the best filter + readout error is below the best
     raw error on every one (1.6-2.8x lower, median 2.0x).
  H2 PASS: r*(filter + readout) >= r*(raw) on 21 of 21.
  H3 PASS: strictly deeper on 16 of 21 (median r* 12 against 8).
  W1 FAIL: the filter + readout exponent is >= 1.5 on 16 of 21, not 18; it
     is 1.26-1.38 on the four noisiest devices after Kyoto (Sherbrooke,
     Cusco, Quebec, WashingtonV2) and about 0 on FakeKyoto, where the walk
     is lost.
  W2 PASS: the raw exponent is below the filter + readout one on 20 of 21
     (all but FakeKyoto, where both are about 0).
  W3 PASS: sigma^2 within 20% at all four times on 15 of 21 (raw: 0).
  * Without the filter the walk looks classical: the median raw exponent is
    0.87, the value of the classical random walk (0.85), because noise puts
    the walker everywhere at once from the start (raw sigma^2 at t = 0.4 is
    1.6 on FakeBrisbane, against 0.33 exact). With the filter the median is
    1.84 (noiseless 1.94): the ballistic spreading is visible.
  * On 11 devices r*(filter + readout) is 12, the deepest tested, so the
    true optimum may lie beyond it: H3 is a lower bound on the shift.
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
