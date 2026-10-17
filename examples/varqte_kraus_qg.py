"""
Variational time evolution at fixed depth against Trotter, with the qg filter; channels as qg affine maps (§162)

Part K (identities, tests/test_channels_varqte.py, not predictions): every single-qubit channel is an affine map
r' = M r + c of the qg vector r = (qg_X, qg_Y, qg_Z) (qang.channels); amplitude damping, phase damping,
depolarizing and T1/T2 relaxation in closed form.

Part V: McLachlan variational real-time evolution (qang.varqte) with a weight-preserving ansatz of fixed depth
against second-order Trotter (qang.ibm.xxz_trotter), whose depth grows with the number of steps. The weight is
conserved by both, so the qg filter (weight 2) applies to both.

Checked during development, before the predictions (not a prediction): n = 5, two excitations, the same
Hamiltonian; with L = 3 layers the variational fidelity is 0.83 at t = 0.5 (the tangent space at the start is
too small); with L = 6 it is 0.9999 at t = 0.5 and 0.9956 at t = 1.

Set-up (fixed before the run): XXZ chain n = 5, J = 1, V = 2, fields uniform in [-0.5, 0.5] (seed 162),
excitations on qubits 0 and 1; times t in {0.5, 1.0, 1.5, 2.0}. Variational: L = 6 layers (54 parameters), RK4
with dt = 0.01 from angles ~ 1e-3 (seed 0). Trotter: order 2, r in {1, 2, 4, 8}; per backend and time, the best r
after the filter. FakeBrisbane, FakeTorino, FakeFez: transpile (optimization_level 1, seed 137),
AerSimulator.from_backend, 4000 shots (seed 162). Error: mean |occupation - exact| over the 5 qubits.

Pre-registered predictions (sha256 of this file recorded in examples/data/varqte_kraus_qg.prereg before the run):
  V1  noiseless variational fidelity >= 0.98 at all 4 times.
  V2  the filter lowers the variational error at all 12 (time, backend).
  V3  at t = 2 the filtered variational error is below the best filtered Trotter error on at least 2 of 3
      backends.
  V4  the filtered variational error varies by less than 0.03 across the 4 times on each backend (fixed depth).
  V5  the best filtered Trotter error is larger at t = 2 than at t = 0.5 on each backend.

python examples/varqte_kraus_qg.py

Findings (pre-registration sha256 60d87597...1756, recorded 2026-10-17 20:41 UTC; the run used that exact file,
then only this block was added; data/varqte_kraus_qg_run.log). Noiseless variational fidelity 0.99986, 0.9956,
0.967, 0.934 at t = 0.5, 1, 1.5, 2. Two-qubit gates: variational 48 at every t; Trotter (r = 1, 2, 4, 8) 18,
36, 72, 144.

  filtered error (mean |occupation - exact|), variational / best Trotter (its r):
               t = 0.5          1.0              1.5              2.0
  FakeBrisbane 0.052 / 0.018(1) 0.030 / 0.018(2) 0.030 / 0.045(2) 0.036 / 0.076(4)
  FakeTorino   0.034 / 0.016(2) 0.022 / 0.014(2) 0.012 / 0.035(4) 0.030 / 0.042(4)
  FakeFez      0.017 / 0.003(2) 0.008 / 0.011(2) 0.005 / 0.022(4) 0.022 / 0.032(4)

  V1 FAIL: the six-layer ansatz loses fidelity at late times (0.967 and 0.934).
  V2 PASS: the filter lowers the variational error everywhere (1.5-8.9x).
  V3 PASS: at t = 2 the variational circuit beats the best Trotter circuit on all 3 backends.
  V4 PASS: the variational error varies by 0.018-0.022 across the times on each backend.
  V5 PASS: the best Trotter error grows 2.6-11x from t = 0.5 to t = 2.
  * The crossover is near t = 1-1.5: before it a shallow Trotter circuit (18-36 gates) wins; after it Trotter
    needs 72 gates and the fixed-depth circuit wins, despite its own variational error.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import ibm  # noqa: E402
from qang import varqte as VQ  # noqa: E402

N, J, V, LAYERS, INIT = 5, 1.0, 2.0, 6, (0, 1)
TIMES = (0.5, 1.0, 1.5, 2.0)
RS = (1, 2, 4, 8)
SHOTS, SEED = 4000, 162
BACKENDS = ("FakeBrisbane", "FakeTorino", "FakeFez")
OUT = os.path.join(HERE, "data", "varqte_kraus_qg.json")


def main():
    from qiskit.quantum_info import Statevector
    from scipy.linalg import expm

    h = np.random.default_rng(SEED).uniform(-0.5, 0.5, N)
    H = VQ.xxz_matrix(N, J, V, h)
    thetas = VQ.evolve(H, LAYERS, TIMES)
    psi0 = Statevector(VQ.ansatz(np.zeros(VQ.n_params(LAYERS)), LAYERS, INIT)).data
    exact = {t: ibm.xxz_exact_occupations(N, J, V, h, t, INIT) for t in TIMES}
    fid = {t: float(abs(np.vdot(expm(-1j * H * t) @ psi0, Statevector(VQ.ansatz(thetas[t], LAYERS, INIT)).data)) ** 2)
           for t in TIMES}
    print(json.dumps({"noiseless fidelity": {t: round(f, 5) for t, f in fid.items()}}), flush=True)

    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    jobs = [(t, "var", 0, VQ.ansatz(thetas[t], LAYERS, INIT, measure=True)) for t in TIMES]
    jobs += [(t, "trot", r, ibm.xxz_trotter(N, J, V, h, t, r, INIT, order=2)) for t in TIMES for r in RS]
    rows = []
    for name in BACKENDS:
        b = getattr(fp, name)()
        ts = transpile([j[-1] for j in jobs], b, optimization_level=1, seed_transpiler=137)
        res = AerSimulator.from_backend(b).run(ts, shots=SHOTS, seed_simulator=SEED).result()
        row = {"backend": name, "gates": {}, "var": {}, "trot": {}}
        for n_, ((t, kind, r, _), tq) in enumerate(zip(jobs, ts)):
            c = res.get_counts(n_)
            e = {k: float(np.abs(VQ.occupations_from_counts(c, N, w) - exact[t]).mean())
                 for k, w in (("raw", None), ("filtered", 2))}
            g = sum(1 for x in tq.data if x.operation.num_qubits == 2)
            if kind == "var":
                row["var"][str(t)] = e
                row["gates"]["var"] = g
            else:
                row["trot"].setdefault(str(t), {})[str(r)] = {**e, "gates": g}
        row["best_trot"] = {t: min(v["filtered"] for v in d.values()) for t, d in row["trot"].items()}
        rows.append(row)
        print(json.dumps({"backend": name, "var 2q": row["gates"]["var"],
                          "var raw/filt": {t: [round(x["raw"], 4), round(x["filtered"], 4)]
                                           for t, x in row["var"].items()},
                          "best trotter filt": {t: round(x, 4) for t, x in row["best_trot"].items()},
                          "trotter 2q": {r: row["trot"]["2.0"][r]["gates"] for r in map(str, RS)}}), flush=True)
    v = {"V1": all(f >= 0.98 for f in fid.values()),
         "V2": all(r["var"][str(t)]["filtered"] < r["var"][str(t)]["raw"] for r in rows for t in TIMES),
         "V3": sum(r["var"]["2.0"]["filtered"] < r["best_trot"]["2.0"] for r in rows) >= 2,
         "V4": all(max(x["filtered"] for x in r["var"].values()) - min(x["filtered"] for x in r["var"].values()) < 0.03
                   for r in rows),
         "V5": all(r["best_trot"]["2.0"] > r["best_trot"]["0.5"] for r in rows)}
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"fidelity": fid, "thetas": {t: th.tolist() for t, th in thetas.items()}, "rows": rows,
                   "verdict": v}, fh, indent=1)
    return v


if __name__ == "__main__":
    main()
