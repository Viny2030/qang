"""
Teleportation through a quantum passive optical network, discrete variables, in qg terms (§157)

Yaghoobianzadeh & Salehi (Academia Quantum 2026, 10.20935/AcadQuant8460) study continuous-variable
teleportation over a QPON (fibre + passive 1:N splitter) and find (i) an optimum FINITE squeezing for the
non-assisted protocol, since loss and splitting punish excess squeezing, and (ii) that the assisted protocol,
where the other users measure and help, is better and pushes the optimum squeezing to infinity at short
distance. qang.qpon is the discrete-variable version: the resource cos(alpha)|00> + sin(alpha)|11>, Bob's
photon through fibre (T) and splitter (1/N) as beam splitters (RBS gates), standard teleportation. Closed
forms (qang.qpon): tan alpha* = sqrt(eta) / kappa, F* = (2 + eta / kappa) / 3, eta = T / N; kappa = 1
non-assisted, kappa = eta + 1 - T assisted, where the other users herald that the photon went to them and
those shots are discarded (a sector filter on their register). The optimal resource has
qg_Z = (kappa^2 - eta) / (kappa^2 + eta) on each qubit.

Checked during development, before the predictions (not a prediction): the circuit reproduces the closed
forms of the fidelity and the kept fraction to 3e-9 (Statevector; 5 links, 5 angles, both protocols). At
L = 0, N = 8 the maximally entangled resource gives F = 0.639 < 2/3, the optimal one 0.708.

Set-up (fixed before the run): links (L km, N) in {(0, 4), (0, 8), (20, 4), (20, 8)}, 0.2 dB/km; alpha in
{0.1, 0.2, ..., 1.4} and pi/4 (15 angles); the six cardinal inputs; FakeBrisbane (ECR), FakeTorino (CZ),
FakeFez (CZ), optimization_level 1, seed_transpiler 137, 4000 shots per circuit, sampler of the fake
backend (device noise model), no readout correction. F = mean over the six inputs of P(q2 = 0) (assisted:
among the shots with q4 = 0). "Best" = the grid angle with the highest measured F.

Pre-registered predictions (sha256 of this file recorded in examples/data/qpon_teleport_qg.prereg before
the run):
  Q1  non-assisted: the best grid angle is within 0.1 rad of alpha* for all 4 links on all 3 backends.
  Q2  non-assisted, L = 0, N = 8: F at the best grid angle exceeds 2/3 and F at alpha = pi/4 does not,
      on all 3 backends.
  Q3  the assisted best F exceeds the non-assisted best F for all 4 links on all 3 backends.
  Q4  assisted, L = 0 (N = 4 and 8): the best grid angle is above pi/4 on all 3 backends.
  Q5  |best F - F*| < 0.05 for both protocols, all links and backends (24 pairs).
  Q6  assisted: the kept fraction at alpha* rounded to the grid is within 0.03 of
      cos^2 alpha + kappa sin^2 alpha for all 4 links on all 3 backends.

python examples/qpon_teleport_qg.py
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import qpon as Q  # noqa: E402

LINKS = ((0, 4), (0, 8), (20, 4), (20, 8))
ALPHAS = tuple(round(0.1 * k, 1) for k in range(1, 15)) + (math.pi / 4,)
BACKENDS = ("FakeBrisbane", "FakeTorino", "FakeFez")
SHOTS = 4000
OUT = os.path.join(HERE, "data", "qpon_teleport_qg.json")
PI4 = float(f"{math.pi / 4:.4f}")  # the grid keys are rounded to 4 decimals


def sampler(backend):
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler
    except ImportError:  # pragma: no cover
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    return Sampler(mode=backend)


def run_backend(name):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile

    b = getattr(fp, name)()
    plan = [(L, N, a, k) for L, N in LINKS for a in ALPHAS for k in range(6)]
    qcs = [Q.teleport_circuit(a, Q.fibre_transmission(L), N, k) for L, N, a, k in plan]
    ts = transpile(qcs, b, optimization_level=1, seed_transpiler=137)
    res = sampler(b).run(ts, shots=SHOTS).result()
    counts = [r.data.meas.get_counts() for r in res]
    out = {"backend": name, "two_qubit_gates": int(np.mean([sum(1 for i in t.data if i.operation.num_qubits == 2)
                                                            for t in ts])), "links": {}}
    i = 0
    for L, N in LINKS:
        row = {}
        for a in ALPHAS:
            cs = counts[i:i + 6]
            i += 6
            row[f"{a:.4f}"] = {"non": Q.fidelity_from_counts(cs, False), "assisted": Q.fidelity_from_counts(cs, True)}
        out["links"][f"{L},{N}"] = row
    return out


def summarize(r):
    s = {}
    for L, N in LINKS:
        row = r["links"][f"{L},{N}"]
        d = {}
        for prot, assisted in (("non", False), ("assisted", True)):
            eta, kap = Q.link(L, N, assisted)
            F = {float(a): v[prot]["fidelity"] for a, v in row.items()}
            best = max(F, key=F.get)
            a_star = Q.optimal_angle(eta, kap)
            grid = min((float(a) for a in row if float(a) != PI4), key=lambda a: abs(a - a_star))
            d[prot] = {"best_alpha": best, "best_F": F[best], "alpha_star": a_star,
                       "F_star": Q.optimal_fidelity(eta, kap),
                       "F_pi4": F[PI4], "F_pi4_exact": Q.average_fidelity(math.pi / 4, eta, kap),
                       "kept_at_grid_star": row[f"{grid:.4f}"][prot]["kept"],
                       "kept_exact": Q.heralding_probability(grid, kap)}
        s[f"{L},{N}"] = d
    return s


def verdict(sums):
    v = {k: True for k in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")}
    for s in sums.values():
        for (L, N) in LINKS:
            d = s[f"{L},{N}"]
            v["Q1"] &= abs(d["non"]["best_alpha"] - d["non"]["alpha_star"]) <= 0.1 + 1e-9
            v["Q3"] &= d["assisted"]["best_F"] > d["non"]["best_F"]
            v["Q5"] &= all(abs(d[p]["best_F"] - d[p]["F_star"]) < 0.05 for p in ("non", "assisted"))
            v["Q6"] &= abs(d["assisted"]["kept_at_grid_star"] - d["assisted"]["kept_exact"]) <= 0.03
            if L == 0:
                v["Q4"] &= d["assisted"]["best_alpha"] > PI4 + 1e-3
            if (L, N) == (0, 8):
                v["Q2"] &= d["non"]["best_F"] > 2 / 3 and not d["non"]["F_pi4"] > 2 / 3
    return v


def main():
    rows, sums = [], {}
    for name in BACKENDS:
        r = run_backend(name)
        s = summarize(r)
        sums[name] = s
        rows.append(r)
        for link, d in s.items():
            print(json.dumps({"backend": name, "link (L,N)": link, "2q": r["two_qubit_gates"],
                              **{p: {"best a": round(d[p]["best_alpha"], 3), "a*": round(d[p]["alpha_star"], 3),
                                     "best F": round(d[p]["best_F"], 4), "F*": round(d[p]["F_star"], 4),
                                     "F(pi/4)": round(d[p]["F_pi4"], 4),
                                     "kept": round(d[p]["kept_at_grid_star"], 3)} for p in ("non", "assisted")}}),
                  flush=True)
    v = verdict(sums)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "summary": sums, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
