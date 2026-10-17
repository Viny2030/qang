"""
What oscillation circuits can and cannot see: mass ordering, Majorana phases, a sterile neutrino (§163)

Three open questions about neutrinos, each mapped to what a weight-preserving oscillation circuit reads:
* Mass ordering (is dm31 > 0?): in matter the sign of A dm31 decides whether nu or anti-nu mu -> e is enhanced, so
  the ordering is visible (qang.neutrino.NUFIT_IO, matter_trotter_circuit).
* Majorana phases: U_M = U diag(1, e^{i a21/2}, e^{i a31/2}) is diagonal in the mass basis, commutes with the
  propagation phases and cancels in U_M D U_M^dagger. No oscillation probability depends on it, in vacuum or
  matter (majorana_invisible: 4e-16). Neutrinoless double beta decay does: m_bb = |sum U_ei^2 m_i| (m_bb).
* A sterile neutrino (3+1): a fourth flavour that does not interact, one more qubit in the weight-1 sector
  (sterile_circuit, 4 qubits, RBS and phase gates only).

Checked during development, before the predictions (not a prediction): Majorana phases change no probability
(4e-16, vacuum and matter) and majorana_circuit equals the vacuum probabilities to 2e-16; sterile_circuit equals
sterile_probabilities to 2e-16; m_bb with m_lightest = 0 is 1.4-3.7 meV (NO) and 19-48 meV (IO) over the
Majorana phases; at 1300 km, 2.6 GeV, delta_CP = -pi/2 in matter P(mu -> e) is 0.087 / 0.021 (nu / anti-nu) for NO
and 0.041 / 0.058 for IO.

Set-up (fixed before the run): FakeBrisbane, FakeTorino, FakeFez; transpile (optimization_level 1, seed 137),
AerSimulator.from_backend, 8000 shots (seed 163); raw and filtered (weight-1 shots) probabilities.
  Part O: L = 1300 km, E = 2.6 GeV, delta_CP = -pi/2, rho = 2.85 g/cm^3, Y_e = 0.5, initial nu_mu,
    matter_trotter_circuit with r = 2, for NO and IO, nu and anti-nu (4 circuits). Exact reference: the
    noiseless r = 2 circuit.
  Part M (no prediction): majorana_circuit with (a21, a31) = (0, 0) and (pi/2, 3 pi/4), mu start, vacuum,
    1300 km, 2.6 GeV: the transpiled circuits are compared.
  Part S: 3+1 with hypothetical theta14 = theta24 = 0.15, theta34 = 0, dm41 = 1 eV^2; L = 0.5 km,
    E in {0.2, 0.3, 0.4, 0.6, 0.9, 1.4} GeV, initial nu_mu; P(mu -> mu) against sterile_probabilities.

Pre-registered predictions (sha256 of this file recorded in
examples/data/neutrino_ordering_majorana_sterile_qg.prereg before the run):
  O1  filtered: P_NO(nu) > P_IO(nu) and P_IO(anti-nu) > P_NO(anti-nu) on all 3 backends.
  O2  filtered: P(nu) - P(anti-nu) is positive for NO and negative for IO on all 3 backends.
  O3  filtered values within 0.015 of the noiseless r = 2 values in all 12 (circuit, backend).
  S1  filtered |P(mu -> mu) error| below raw at all 18 (E, backend).
  S2  median raw / filtered error ratio over the 18 >= 2.
  S3  the filtered P(mu -> mu) is lowest at E = 0.4 GeV (the exact minimum of the grid) on all 3 backends.

python examples/neutrino_ordering_majorana_sterile_qg.py

Findings (pre-registration sha256 0d32687f...3b60, recorded 2026-10-17 20:44 UTC; the run used that exact file,
then only this block was added; data/neutrino_ordering_majorana_sterile_qg_run.log):

  Part O, filtered P(mu -> e) (noiseless r = 2: NO 0.081 / 0.024, IO 0.045 / 0.053 for nu / anti-nu):
    FakeBrisbane  NO 0.105 / 0.057   IO 0.070 / 0.081
    FakeTorino    NO 0.092 / 0.042   IO 0.058 / 0.070
    FakeFez       NO 0.091 / 0.040   IO 0.055 / 0.068
  Part M: on all 3 backends the two transpiled Majorana circuits are identical: the optimizer merges the
    Majorana phase gates with their inverses and deletes them.
  Part S, P(mu -> mu) raw / filtered at the exact minimum E = 0.4 GeV (exact 0.915): Brisbane 0.678 / 0.879,
    Torino 0.749 / 0.890, Fez 0.792 / 0.890; median raw / filtered error ratio 5.5.

  O1 PASS, O2 PASS: the ordering is read correctly, with the right sign of the nu / anti-nu asymmetry, everywhere.
  O3 FAIL: the filtered values sit 0.010-0.033 above the noiseless ones (noise inside the sector moves
     probability toward the other flavours); the bias is common, so NO - IO for nu (0.034-0.036, exact 0.037)
     survives.
  S1 PASS, S2 PASS (5.5x), S3 PASS.
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import neutrino as NU  # noqa: E402

BACKENDS = ("FakeBrisbane", "FakeTorino", "FakeFez")
SHOTS, SEED = 8000, 163
L_O, E_O, D_CP, RHO, R_O = 1300.0, 2.6, -math.pi / 2, 2.85, 2
L_S, ES = 0.5, (0.2, 0.3, 0.4, 0.6, 0.9, 1.4)
STER = {"theta14": 0.15, "theta24": 0.15, "theta34": 0.0}
OUT = os.path.join(HERE, "data", "neutrino_ordering_majorana_sterile_qg.json")


def probs(counts, n, filtered):
    tot, p = 0, np.zeros(n)
    for k, c in counts.items():
        b = k.replace(" ", "")
        if b.count("1") != 1:
            if filtered:
                continue
            tot += c
            continue
        tot += c
        p[n - 1 - b.index("1")] += c
    return p / tot


def main():
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit_aer import AerSimulator

    o_jobs = [(o, anti, NU.matter_trotter_circuit("mu", L_O, E_O, D_CP, RHO, R_O, antineutrino=anti, params=prm))
              for o, prm in (("NO", NU.NUFIT_NO), ("IO", NU.NUFIT_IO)) for anti in (False, True)]
    o_exact = {(o, a): Statevector(qc.remove_final_measurements(inplace=False)).probabilities()[1 << 0]
               for o, a, qc in o_jobs}
    s_jobs = [(E, NU.sterile_circuit("mu", L_S, E, **STER)) for E in ES]
    s_exact = {E: NU.sterile_probabilities(L_S, E, **STER)[1, 1] for E in ES}
    m_qcs = [NU.majorana_circuit("mu", L_O, E_O, D_CP, a, b) for a, b in ((0, 0), (math.pi / 2, 3 * math.pi / 4))]
    print(json.dumps({"O exact (r=2)": {f"{o} {'anti' if a else 'nu'}": round(v, 4) for (o, a), v in o_exact.items()},
                      "S exact P(mu->mu)": {E: round(v, 4) for E, v in s_exact.items()}}), flush=True)
    rows = []
    for name in BACKENDS:
        b = getattr(fp, name)()
        qcs = [j[-1] for j in o_jobs] + [j[-1] for j in s_jobs]
        ts = transpile(qcs, b, optimization_level=1, seed_transpiler=137)
        res = AerSimulator.from_backend(b).run(ts, shots=SHOTS, seed_simulator=SEED).result()
        r = {"backend": name, "O": {}, "S": {}}
        for i, (o, a, _) in enumerate(o_jobs):
            c = res.get_counts(i)
            r["O"][f"{o},{int(a)}"] = {k: float(probs(c, 3, f)[0]) for k, f in (("raw", False), ("filtered", True))}
        for i, (E, _) in enumerate(s_jobs):
            c = res.get_counts(len(o_jobs) + i)
            r["S"][str(E)] = {k: float(probs(c, 4, f)[1]) for k, f in (("raw", False), ("filtered", True))}
        tm = transpile(m_qcs, b, optimization_level=1, seed_transpiler=137)
        r["M_identical_after_transpile"] = bool(tm[0] == tm[1])
        r["M_ops"] = [dict(t.count_ops()) for t in tm]
        rows.append(r)
        print(json.dumps({"backend": name, "O filtered": {k: round(v["filtered"], 4) for k, v in r["O"].items()},
                          "S raw/filt": {E: [round(v["raw"], 4), round(v["filtered"], 4)] for E, v in r["S"].items()},
                          "Majorana circuits identical after transpile": r["M_identical_after_transpile"]}),
              flush=True)
    f = lambda r, o, a: r["O"][f"{o},{a}"]["filtered"]  # noqa: E731
    err = {(r["backend"], E): {k: abs(r["S"][str(E)][k] - s_exact[E]) for k in ("raw", "filtered")}
           for r in rows for E in ES}
    v = {"O1": all(f(r, "NO", 0) > f(r, "IO", 0) and f(r, "IO", 1) > f(r, "NO", 1) for r in rows),
         "O2": all(f(r, "NO", 0) - f(r, "NO", 1) > 0 > f(r, "IO", 0) - f(r, "IO", 1) for r in rows),
         "O3": all(abs(f(r, o, a) - o_exact[(o, bool(a))]) < 0.015 for r in rows for o in ("NO", "IO") for a in (0, 1)),
         "S1": all(e["filtered"] < e["raw"] for e in err.values()),
         "S2": float(np.median([e["raw"] / e["filtered"] for e in err.values()])) >= 2,
         "S3": all(min(ES, key=lambda E: r["S"][str(E)]["filtered"]) == 0.4 for r in rows)}
    print(json.dumps({"S median ratio": round(float(np.median([e["raw"] / e["filtered"] for e in err.values()])), 2)}))
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"O_exact": {f"{o},{int(a)}": x for (o, a), x in o_exact.items()},
                   "S_exact": {str(E): x for E, x in s_exact.items()}, "rows": rows,
                   "S_errors": {f"{b},{E}": e for (b, E), e in err.items()}, "verdict": v}, fh, indent=1)
    return v


if __name__ == "__main__":
    main()
