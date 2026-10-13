"""
Interacting excitations: Trotter simulation of the XXZ chain with 1, 2 and 3 excitations and the qg filter (§148)

§147 simulated one excitation, which a classical computer handles as an n x n
matrix. Here the excitations interact: the XXZ chain
H = sum_k [J (X_k X_{k+1} + Y_k Y_{k+1}) / 2 + V Z_k Z_{k+1} / 4] + sum_k h_k Z_k / 2
conserves the weight in every sector, so the filter applies with 1, 2 or 3
excitations, and with two or more the dynamics is many-body (the ZZ term
binds neighbouring excitations). Each bond is one two-qubit gate
exp(-i dt [J (XX + YY)/2 + V ZZ/4]) (qang.ibm.xxz_trotter); second-order
(symmetric) Trotter is compared with first order.

The question: with more excitations the sector is larger (C(6, 3) = 20 of 64
states at half filling, against 6 with one excitation), so more of the noisy
shots land inside it and pass the filter. Does the filter's benefit shrink?

Set-up (fixed before the run): n = 6, J = 1, V = 2, h_k uniform in
[-0.5, 0.5] (seed 148), T = 2; the w excitations start on qubits 0..w-1
(w = 1, 2, 3); Trotter order 1 and 2; r in (1, 2, 3, 4, 6, 8) steps; 36
circuits per backend, 4000 shots, IBM Runtime's Sampler in local mode,
optimization_level 1, seed 137, the 21 device noise models of §146. Error:
mean_k |P_k - P_k^exact(T)|, P_k = P(qubit k excited), exact = full
Hilbert-space evolution; ibm_qg with weight = w.

Noiseless Trotter error (statevector, computed before committing), r = 1, 2,
3, 4, 6, 8:
  w = 1  order 1: 0.234 0.109 0.073 0.053 0.035 0.026
         order 2: 0.127 0.028 0.013 0.008 0.004 0.002
  w = 2  order 1: 0.233 0.145 0.066 0.040 0.024 0.017
         order 2: 0.173 0.069 0.028 0.015 0.006 0.003
  w = 3  order 1: 0.274 0.103 0.048 0.030 0.018 0.013
         order 2: 0.292 0.093 0.038 0.020 0.008 0.005

Pre-registered predictions (committed before the run):
  K1  in every sector, on every backend that runs (at least 20 of 21 run),
      the best filter + readout error (over r and order) is below the best
      raw error.
  K2  the filter's benefit shrinks with the number of excitations: the
      median over backends of best raw / best filter + readout error is
      larger for w = 1 than for w = 2, and larger for w = 2 than for w = 3.
  K3  for w = 2, second-order Trotter gives a lower best filter + readout
      error than first order on at least 15 of the backends that run.
  K4  with first-order Trotter, r*(filter + readout) >= r*(raw) on at least
      54 of the 63 (backend, sector) pairs.

python examples/ibm_xxz_many_body_qg.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import ibm  # noqa: E402

N, J, V, T, SHOTS, SEED = 6, 1.0, 2.0, 2.0, 4000, 148
WS, ORDERS, RS = (1, 2, 3), (1, 2), (1, 2, 3, 4, 6, 8)
BACKENDS = ("FakeBrisbane", "FakeSherbrooke", "FakeTorino",
            "FakeAachen", "FakeBerlin", "FakeBoston", "FakeBrussels", "FakeCusco", "FakeFez", "FakeKawasaki",
            "FakeKingston", "FakeKyiv", "FakeKyoto", "FakeMarrakesh", "FakeMiami", "FakeNighthawk", "FakeOsaka",
            "FakePittsburgh", "FakeQuebec", "FakeStrasbourg", "FakeWashingtonV2")
KINDS = ("raw", "readout", "filtered", "filtered_readout")
OUT = os.path.join(HERE, "data", "ibm_xxz_many_body_qg.json")


def fields():
    return np.random.default_rng(SEED).uniform(-0.5, 0.5, N)


def plan():
    h = fields()
    cases, exact = [], {}
    for w in WS:
        init = tuple(range(w))
        exact[w] = ibm.xxz_exact_occupations(N, J, V, h, T, init)
        for order in ORDERS:
            for r in RS:
                cases.append((w, order, r, ibm.xxz_trotter(N, J, V, h, T, r, init, order)))
    return cases, exact


def sampler(backend):
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler
    except ImportError:  # pragma: no cover
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    return Sampler(mode=backend)


def run_backend(name):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile

    cases, exact = plan()
    bk = getattr(fp, name)()
    ts = [transpile(qc, bk, optimization_level=1, seed_transpiler=137) for *_, qc in cases]
    res = sampler(bk).run(ts, shots=SHOTS).result()
    err = {}
    kept = {}
    for (w, order, r, _), rr, t in zip(cases, res, ts):
        o = ibm.ibm_qg(rr, t, bk, weight=w)
        key = f"w{w}_o{order}"
        err.setdefault(key, {k: [] for k in KINDS})
        for k in KINDS:
            err[key][k].append(float(np.abs(ibm.occupations(o, k) - exact[w]).mean()))
        kept.setdefault(key, []).append(o["kept_raw"])
    best = {w: {k: float(min(min(err[f"w{w}_o{o}"][k]) for o in ORDERS)) for k in KINDS} for w in WS}
    best_by_order = {f"w{w}_o{o}": {k: float(min(err[f"w{w}_o{o}"][k])) for k in KINDS} for w in WS for o in ORDERS}
    rstar = {f"w{w}_o{o}": {k: RS[int(np.argmin(err[f"w{w}_o{o}"][k]))] for k in KINDS} for w in WS for o in ORDERS}
    return {"backend": name, "error": err, "kept_raw": kept, "best": {str(w): v for w, v in best.items()},
            "best_by_order": best_by_order, "r_star": rstar,
            "two_qubit_gates": [sum(1 for i in t.data if i.operation.num_qubits == 2) for t in ts]}


def main():
    rows = []
    for name in BACKENDS:
        try:
            r = run_backend(name)
            print(json.dumps({"backend": name,
                              "best raw/f+ro": {w: [round(r["best"][str(w)]["raw"], 3),
                                                    round(r["best"][str(w)]["filtered_readout"], 3)] for w in WS},
                              "r* o1 raw/f+ro": {w: [r["r_star"][f"w{w}_o1"]["raw"],
                                                     r["r_star"][f"w{w}_o1"]["filtered_readout"]] for w in WS},
                              "w2 o1/o2 f+ro": [round(r["best_by_order"]["w2_o1"]["filtered_readout"], 3),
                                                round(r["best_by_order"]["w2_o2"]["filtered_readout"], 3)]}),
                  flush=True)
        except Exception as e:  # a backend that cannot run is reported, not hidden
            r = {"backend": name, "failure": f"{type(e).__name__}: {e}"[:300]}
            print(json.dumps(r), flush=True)
        rows.append(r)
    ok = [r for r in rows if "best" in r]
    fr = "filtered_readout"
    ratio = {w: float(np.median([r["best"][str(w)]["raw"] / r["best"][str(w)][fr] for r in ok])) for w in WS}
    k3 = sum(r["best_by_order"]["w2_o2"][fr] < r["best_by_order"]["w2_o1"][fr] for r in ok)
    k4 = sum(r["r_star"][f"w{w}_o1"][fr] >= r["r_star"][f"w{w}_o1"]["raw"] for r in ok for w in WS)
    v = {"K1": len(ok) >= 20 and all(r["best"][str(w)][fr] < r["best"][str(w)]["raw"] for r in ok for w in WS),
         "K2": ratio[1] > ratio[2] > ratio[3],
         "K3": k3 >= 15,
         "K4": k4 >= 54}
    summary = {"backends run": len(ok), "median ratio raw / f+ro": {w: round(x, 2) for w, x in ratio.items()},
               "median best": {w: {k: round(float(np.median([r["best"][str(w)][k] for r in ok])), 4) for k in KINDS}
                               for w in WS},
               "median kept (o1, r=8)": {w: round(float(np.median([r["kept_raw"][f"w{w}_o1"][-1] for r in ok])), 3)
                                         for w in WS},
               "w2 order 2 better": k3, "r* f+ro >= raw (o1)": k4,
               "median r* o1": {w: {k: float(np.median([r["r_star"][f"w{w}_o1"][k] for r in ok])) for k in ("raw", fr)}
                                for w in WS},
               "median r* o2": {w: {k: float(np.median([r["r_star"][f"w{w}_o2"][k] for r in ok])) for k in ("raw", fr)}
                                for w in WS}}
    print(json.dumps(summary))
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok_ else 'FAIL'}" for k, ok_ in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "summary": summary, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
