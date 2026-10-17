"""
Teleportation in a QPON, part two: the fidelity from three qg correlators, and dual rail with the qg filter (§158)

§157 found the discrete-variable optimum resource of a downlink QPON and read the assisted protocol (the other
users herald a photon that went to them) as a sector filter; the filter does not touch the device noise, since
the teleportation circuit does not conserve the weight. Two questions follow.

A. The average teleportation fidelity is a linear qg read-out of the resource alone:
   F = (3 + qg_XX - qg_YY + qg_ZZ) / 6 (qang.qpon.fidelity_from_correlators). Three circuits without the Bell
   measurement predict the fidelity, and the difference to the teleported fidelity is the noise added by the
   Bell measurement and the corrections.
B. Dual rail: one photon in two modes per qubit (qang.qpon.dual_rail_teleport_circuit, 8 qubits). Every gate
   conserves the number of photons, every loss leaves Bob's pair with weight 0, and the qg filter (weight 1 on
   each pair) heralds all of it, fibre included. Noiseless: filtered F = 1 at alpha = pi/4 for any distance and
   N, kept = eta = T / N; raw (Bob guesses when no photon arrives) F = (1 + eta) / 2, below the single-rail
   optimum (2 + eta) / 3. The price is the circuit: 64 two-qubit gates on FakeTorino against 11.

Checked during development, before the predictions (not a prediction): the correlator read-out reproduces the
closed form to 1.5e-9 and the dual-rail circuit its three closed forms to 2.3e-8 (Statevector, tests/test_qpon.py).

Set-up (fixed before the run): the links of §157, (L km, N) in {(0, 4), (0, 8), (20, 4), (20, 8)}, 0.2 dB/km.
  Part A, single rail: for each link and each protocol (non-assisted, assisted) alpha = alpha* of that protocol;
    the resource circuit in the X, Y and Z bases (F_corr; assisted: shots with q4 = 0) and the teleportation of
    the six cardinal inputs (F_tele).
  Part B, dual rail: alpha = pi/4, eta = T / N, six inputs; filtered and raw F, kept fraction.
  FakeBrisbane (ECR), FakeTorino (CZ), FakeFez (CZ), optimization_level 1, seed_transpiler 137, 10000 shots per
  circuit, sampler of the fake backend (device noise model), no readout correction. 96 circuits per backend.

Pre-registered predictions (sha256 of this file recorded in examples/data/qpon_teleport_dual_rail_qg.prereg
before the run):
  A1  |F_corr - F*| < 0.03 for all 24 (link, protocol, backend).
  A2  F_corr > F_tele for all 24.
  A3  the gap F_corr - F_tele is larger for the assisted protocol than for the non-assisted one on all
      12 (link, backend) (noise pulls a fidelity towards 1/2 in proportion to its distance from 1/2).
  B1  the filtered dual-rail F varies by less than 0.05 across the 4 links on each backend.
  B2  kept / eta varies by less than a factor 1.3 across the 4 links on each backend.
  B3  at L = 20 km (N = 4 and 8) the filtered dual-rail F exceeds the single-rail F_tele of both protocols on
      all 3 backends.
  B4  the raw dual-rail F is below the single-rail non-assisted F_tele for all 12 (link, backend).
  B5  the filtered dual-rail F exceeds 2/3 for all 12.

python examples/qpon_teleport_dual_rail_qg.py

Findings (pre-registration sha256 3f11d66d...4ef2, recorded 2026-10-17 14:09 UTC; the run used that exact file,
then only this block was added; data/qpon_teleport_dual_rail_qg_run.log). Two-qubit gates: resource 4-5,
single-rail teleportation 10-11, dual rail 65-66.

  link        F* non / assisted   F_corr / F_tele non   assisted        dual rail filtered / raw, kept (eta)
  FakeBrisbane 0,4   0.750 / 1.000    0.725 / 0.729   0.932 / 0.924   0.786 / 0.572, 0.227 (0.250)
               0,8   0.708 / 1.000    0.693 / 0.693   0.897 / 0.886   0.720 / 0.536, 0.149 (0.125)
               20,4  0.700 / 0.714    0.685 / 0.682   0.701 / 0.697   0.709 / 0.531, 0.133 (0.100)
               20,8  0.683 / 0.692    0.665 / 0.670   0.679 / 0.676   0.650 / 0.516, 0.099 (0.050)
  FakeTorino   0,4                    0.741 / 0.727   0.963 / 0.938   0.847 / 0.595, 0.242
               0,8                    0.694 / 0.691   0.941 / 0.911   0.790 / 0.552, 0.158
               20,4                   0.687 / 0.679   0.698 / 0.696   0.783 / 0.545, 0.139
               20,8                   0.672 / 0.666   0.686 / 0.678   0.727 / 0.528, 0.106
  FakeFez      0,4                    0.739 / 0.735   0.973 / 0.959   0.869 / 0.594, 0.245
               0,8                    0.701 / 0.697   0.959 / 0.932   0.814 / 0.548, 0.147
               20,4                   0.688 / 0.689   0.707 / 0.706   0.791 / 0.538, 0.127
               20,8                   0.676 / 0.674   0.687 / 0.686   0.710 / 0.519, 0.086

  A1 FAIL in 5 of 24: F_corr is 0.027-0.103 below F* for the assisted protocol at L = 0 (and on Brisbane at
     L = 20 km). The three correlator circuits have only 4-5 two-qubit gates, but they read two data qubits and
     the herald, so they carry most of the noise the teleportation sees.
  A2 FAIL in 3 of 24: the non-assisted gaps are -0.006 to +0.013, within twice their shot error.
  A3 FAIL in 2 of 12 (Brisbane and Torino at 20,4); the gap is larger for the assisted protocol in 10.
  B1 FAIL: the filtered dual-rail F falls with the distance, 0.79 -> 0.65 (Brisbane), 0.85 -> 0.73 (Torino),
     0.87 -> 0.71 (Fez).
  B2 FAIL: kept / eta grows from 0.9-1.0 at (0, 4) to 1.7-2.1 at (20, 8).
  B3 FAIL on one of 6: Brisbane at (20, 8); dual rail wins the other 5.
  B4 PASS: the raw dual-rail F (0.516-0.595) is below the single-rail non-assisted one everywhere.
  B5 FAIL on one of 12: Brisbane at (20, 8), 0.650.
  * B1, B2 and B5 have one cause: false heralds. When the photon is lost, a device error can still leave Bob's pair
    with weight 1; (kept - eta) / (1 - eta) is 3-6% at L = 20 km. Those shots carry F of about 1/2, and as eta
    falls they become a larger share of the kept ones. It is the dark-count floor of a real heralded link,
    produced here by gate and readout errors.
  * Post hoc (not pre-registered): at L = 20 km the filtered dual rail beats the NOISELESS single-rail optimum F* on
    Torino and Fez (0.783 and 0.791 against 0.714 at N = 4; 0.727 and 0.710 against 0.692 at N = 8), with six times
    the two-qubit gates. Heralding every loss is worth more than those gates cost, until false heralds take over.
  * Post hoc: |F_corr - F_tele| <= 0.029 in all 24: three short circuits predict the teleported fidelity to 0.03,
    but they do not separate the Bell measurement's noise from the resource's.
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
PROTOCOLS = (("non", False), ("assisted", True))
BACKENDS = ("FakeBrisbane", "FakeTorino", "FakeFez")
SHOTS = 10000
OUT = os.path.join(HERE, "data", "qpon_teleport_dual_rail_qg.json")


def sampler(backend):
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler
    except ImportError:  # pragma: no cover
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    return Sampler(mode=backend)


def plan():
    jobs = []  # (part, link, protocol, kind, index, circuit)
    for L, N in LINKS:
        T = Q.fibre_transmission(L)
        for prot, assisted in PROTOCOLS:
            a = Q.optimal_angle(*Q.link(L, N, assisted))
            for b in "XYZ":
                jobs.append(("A", (L, N), prot, "corr", b, Q.resource_circuit(a, T, N, b)))
            for k in range(6):
                jobs.append(("A", (L, N), prot, "tele", k, Q.teleport_circuit(a, T, N, k)))
        for k in range(6):
            jobs.append(("B", (L, N), None, "dual", k, Q.dual_rail_teleport_circuit(math.pi / 4, T / N, k)))
    return jobs


def run_backend(name):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile

    b = getattr(fp, name)()
    jobs = plan()
    ts = transpile([j[-1] for j in jobs], b, optimization_level=1, seed_transpiler=137)
    res = sampler(b).run(ts, shots=SHOTS).result()
    counts = [r.data.meas.get_counts() for r in res]
    gates = {}
    for j, t in zip(jobs, ts):
        gates.setdefault(j[3], []).append(sum(1 for i in t.data if i.operation.num_qubits == 2))
    out = {"backend": name, "two_qubit_gates": {k: float(np.mean(v)) for k, v in gates.items()}, "links": {}}
    for L, N in LINKS:
        key = f"{L},{N}"
        row = {}
        for prot, assisted in PROTOCOLS:
            eta, kap = Q.link(L, N, assisted)
            sel = [c for j, c in zip(jobs, counts) if j[1] == (L, N) and j[2] == prot]
            corr = {b: Q.correlator_from_counts(c, assisted) for b, c in zip("XYZ", sel[:3])}
            tele = Q.fidelity_from_counts(sel[3:], assisted)
            row[prot] = {"F_star": Q.optimal_fidelity(eta, kap),
                         "correlators": {b: v[0] for b, v in corr.items()},
                         "F_corr": Q.fidelity_from_correlators(corr["X"][0], corr["Y"][0], corr["Z"][0]),
                         "F_tele": tele["fidelity"], "F_tele_se": tele["se"], "kept_tele": tele["kept"]}
        dual = Q.dual_rail_fidelity_from_counts([c for j, c in zip(jobs, counts) if j[1] == (L, N) and j[3] == "dual"])
        eta = Q.fibre_transmission(L) / N
        row["dual"] = {**dual, "eta": eta, "kept_over_eta": dual["kept"] / eta,
                       "exact": Q.dual_rail_fidelities(math.pi / 4, eta)}
        out["links"][key] = row
    return out


def verdict(rows):
    v = {k: True for k in ("A1", "A2", "A3", "B1", "B2", "B3", "B4", "B5")}
    for r in rows:
        d = r["links"]
        filt = [d[f"{L},{N}"]["dual"]["filtered"] for L, N in LINKS]
        ratio = [d[f"{L},{N}"]["dual"]["kept_over_eta"] for L, N in LINKS]
        v["B1"] &= max(filt) - min(filt) < 0.05
        v["B2"] &= max(ratio) / min(ratio) < 1.3
        for L, N in LINKS:
            x = d[f"{L},{N}"]
            for prot, _ in PROTOCOLS:
                v["A1"] &= abs(x[prot]["F_corr"] - x[prot]["F_star"]) < 0.03
                v["A2"] &= x[prot]["F_corr"] > x[prot]["F_tele"]
            gap = {p: x[p]["F_corr"] - x[p]["F_tele"] for p, _ in PROTOCOLS}
            v["A3"] &= gap["assisted"] > gap["non"]
            if L == 20:
                v["B3"] &= x["dual"]["filtered"] > max(x["non"]["F_tele"], x["assisted"]["F_tele"])
            v["B4"] &= x["dual"]["raw"] < x["non"]["F_tele"]
            v["B5"] &= x["dual"]["filtered"] > 2 / 3
    return v


def main(backends=BACKENDS):
    rows = []
    for name in backends:
        r = run_backend(name)
        rows.append(r)
        print(json.dumps({"backend": name, "2q": r["two_qubit_gates"]}), flush=True)
        for key, x in r["links"].items():
            print(json.dumps({"backend": name, "link (L,N)": key,
                              **{p: {"F*": round(x[p]["F_star"], 4), "F_corr": round(x[p]["F_corr"], 4),
                                     "F_tele": round(x[p]["F_tele"], 4)} for p, _ in PROTOCOLS},
                              "dual": {"filtered": round(x["dual"]["filtered"], 4), "raw": round(x["dual"]["raw"], 4),
                                       "kept": round(x["dual"]["kept"], 4), "eta": round(x["dual"]["eta"], 4)}}),
                  flush=True)
    v = verdict(rows)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
