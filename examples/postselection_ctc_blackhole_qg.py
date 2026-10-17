"""
Time loops and black holes as teleportation with postselection, read in qg (§164)

Two information-theoretic models (qang.postselection), not statements about spacetime:
* P-CTC (Lloyd et al. 2011): a qubit sent to its own past is half of a Bell pair projected back on the Bell
  state. If it is rotated by RX(phi) the postselection succeeds with probability cos^2(phi/2), whose qang angle
  is phi; the grandfather paradox (phi = pi) has probability 0. If a system S controls the rotation, the
  paradoxical branch is suppressed: postselected qg_Z of S = (1 - c^2) / (1 + c^2), c = cos(phi/2).
* Hayden-Preskill with the Yoshida-Kitaev decoder: a message qubit falls into a 3-qubit black hole; with a
  scrambling Clifford, postselecting one radiated qubit and its partner on a Bell state succeeds with
  probability 1/4 and recovers the message's entanglement with fidelity 1; without scrambling, 1 and 1/4.

Checked during development, before the predictions (not a prediction): the P-CTC success and the controlled qg_Z
equal the formulas at phi in {0, pi/3, pi/2, 2 pi/3, pi} to 1e-15; find_scrambler(seed 164) returns a Clifford
(6 h, 4 cx, 3 s, 2 z, 2 sdg, 1 y) that spreads every single-qubit Pauli over all 3 qubits; the noiseless decoder
gives P = 0.25, F = 1 with it and P = 1, F = 0.25 with the identity.

Set-up (fixed before the run): FakeBrisbane, FakeTorino, FakeFez; transpile (optimization_level 1, seed 137),
AerSimulator.from_backend, 8000 shots (seed 164); no readout correction.
  Part C: pctc_circuit at phi in {0, pi/3, pi/2, 2 pi/3, pi}, uncontrolled and controlled.
  Part H: yk_decoder_circuit with the scrambler and with the identity.

Pre-registered predictions (sha256 of this file recorded in examples/data/postselection_ctc_blackhole_qg.prereg
before the run):
  C1  uncontrolled: |success - cos^2(phi/2)| < 0.05 at all 15 (phi, backend).
  C2  uncontrolled, phi = pi (the paradox): success < 0.05 on all 3 backends.
  C3  controlled: |postselected qg_Z(S) - (1 - c^2)/(1 + c^2)| < 0.08 at all 15.
  H1  the decoded fidelity with the scrambler exceeds that with the identity by more than 0.3 on all 3.
  H2  with the scrambler, |success - 0.25| < 0.05 on all 3.
  H3  with the scrambler, the decoded fidelity exceeds 0.5 on all 3.

python examples/postselection_ctc_blackhole_qg.py

Findings (pre-registration sha256 e9de2195...ee4f, recorded 2026-10-17 20:56 UTC; the run used that exact file,
then only this block was added; data/postselection_ctc_blackhole_qg_run.log):

  P-CTC success, phi = 0, pi/3, pi/2, 2pi/3, pi (exact 1, 0.75, 0.5, 0.25, 0):
    FakeBrisbane 0.950 0.719 0.486 0.253 0.019 | FakeTorino 0.931 0.708 0.491 0.270 0.056
    FakeFez      0.983 0.741 0.497 0.254 0.009
  controlled, postselected qg_Z(S) (exact 0, 0.143, 0.333, 0.600, 1):
    FakeBrisbane 0.003 0.126 0.325 0.507 0.903 | FakeTorino -0.004 0.080 0.221 0.348 0.600
    FakeFez      0.005 0.138 0.336 0.539 0.955
  Decoder (90 two-qubit gates with the scrambler): success / fidelity, scrambler vs identity:
    FakeBrisbane 0.260 / 0.547 vs 0.986 / 0.248 | FakeTorino 0.248 / 0.578 vs 0.989 / 0.250
    FakeFez      0.246 / 0.742 vs 0.995 / 0.250

  C1 FAIL on 2 of 15: phi = 0 on Brisbane (0.950, off by 0.050) and Torino (0.931).
  C2 FAIL on Torino (0.056); Brisbane 0.019 and Fez 0.009 suppress the paradox to the noise floor.
  C3 FAIL: Fez is within 0.061 everywhere, Brisbane misses at pi (0.903), and Torino's controlled rotation loses
     the suppression (0.600 at pi).
  H1 FAIL on Brisbane by 0.001 (gap 0.299); Torino 0.328, Fez 0.492.
  H2 PASS: success 0.246-0.260 with the scrambler.
  H3 PASS: the decoded fidelity is 0.547-0.742, above 1/2 on all 3.
  * The qang angle of the P-CTC success, arccos(2P - 1), reads back the rotation phi; device noise pulls P toward
    1/2 at both ends, as it pulls any fidelity.
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import postselection as PS  # noqa: E402

PHIS = (0.0, math.pi / 3, math.pi / 2, 2 * math.pi / 3, math.pi)
BACKENDS = ("FakeBrisbane", "FakeTorino", "FakeFez")
SHOTS, SEED = 8000, 164
OUT = os.path.join(HERE, "data", "postselection_ctc_blackhole_qg.json")


def main():
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    scr = PS.find_scrambler()
    jobs = [("C", phi, ctrl, PS.pctc_circuit(phi, ctrl)) for phi in PHIS for ctrl in (False, True)]
    jobs += [("H", name, None, PS.yk_decoder_circuit(u)) for name, u in (("scrambler", scr),
                                                                         ("identity", QuantumCircuit(3)))]
    rows = []
    for bname in BACKENDS:
        b = getattr(fp, bname)()
        ts = transpile([j[-1] for j in jobs], b, optimization_level=1, seed_transpiler=137)
        res = AerSimulator.from_backend(b).run(ts, shots=SHOTS, seed_simulator=SEED).result()
        r = {"backend": bname, "C": {}, "H": {}, "H_2q": {}}
        for i, (part, key, ctrl, _) in enumerate(jobs):
            c = res.get_counts(i)
            if part == "C":
                r["C"][f"{key:.4f},{int(ctrl)}"] = PS.pctc_from_counts(c)
            else:
                r["H"][key] = PS.yk_from_counts(c)
                r["H_2q"][key] = sum(1 for x in ts[i].data if x.operation.num_qubits == 2)
        rows.append(r)
        print(json.dumps({"backend": bname,
                          "C uncontrolled success": [round(r["C"][f"{p:.4f},0"]["success"], 4) for p in PHIS],
                          "C controlled qg_Z(S)": [round(r["C"][f"{p:.4f},1"]["qg_z_S"], 4) for p in PHIS],
                          "H": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in r["H"].items()},
                          "H 2q": r["H_2q"]}), flush=True)
    v = {"C1": all(abs(r["C"][f"{p:.4f},0"]["success"] - PS.pctc_success(p)) < 0.05 for r in rows for p in PHIS),
         "C2": all(r["C"][f"{math.pi:.4f},0"]["success"] < 0.05 for r in rows),
         "C3": all(abs(r["C"][f"{p:.4f},1"]["qg_z_S"] - PS.pctc_controlled_qg_z(p)) < 0.08 for r in rows for p in PHIS),
         "H1": all(r["H"]["scrambler"]["fidelity"] - r["H"]["identity"]["fidelity"] > 0.3 for r in rows),
         "H2": all(abs(r["H"]["scrambler"]["success"] - 0.25) < 0.05 for r in rows),
         "H3": all(r["H"]["scrambler"]["fidelity"] > 0.5 for r in rows)}
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "verdict": v}, fh, indent=1)
    return v


if __name__ == "__main__":
    main()
