"""
Neutrino flavour oscillations in qg terms (§151)

Two flavours on one qubit: qg_Z = cos^2(2 theta) + sin^2(2 theta) cos(Delta), the
spherical law of cosines on the precession cone; at the oscillation maximum the
qang angle is exactly 4 theta, so theta = arccos(qg_Z) / 4 with the
theta-independent shot-noise std 1 / (4 sqrt(N + 1/2)) of §125.

Three flavours on three qubits in the weight-1 sector (qang.neutrino): PMNS as
three RBS rotations and two phase gates, propagation as phase gates; every
gate conserves the weight, so the qg filter applies.

Set-up (fixed before the run):
  Part A: theta in {0.1, 0.3, 0.5, 0.7} rad, Delta = pi, N = 1000 shots,
    2000 binomial repetitions (seed 151), estimator arccos(qg_Z) / 4.
  Part B: vacuum, NuFIT-like normal ordering (qang.neutrino.NUFIT_NO),
    delta_CP = -pi/2, L = 1300 km (a DUNE-like baseline; matter effects
    ignored), E in {0.8, 1.2, 1.6, 2.0, 2.6, 3.2, 4.0, 5.0} GeV, initial nu_mu,
    read all three flavours; FakeBrisbane (ECR), FakeTorino (CZ), FakeFez (CZ),
    optimization_level 1, seed_transpiler 137, 4000 shots, sampler of the
    fake backend (device noise model). Error: mean |P_hat - P| over the
    3 flavours and 8 energies, for raw, readout, filtered, filtered_readout.
  Part C: A_CP = P(mu -> e) - P(anti-mu -> anti-e) at E = 2.6 GeV (first
    maximum 2.63 GeV), exact 0.0250; 20000 shots per circuit, same backends.

Checked during development, before the predictions (not a prediction): the
one- and three-qubit circuits reproduce the analytic vacuum probabilities to
1e-15 (nu and anti-nu, several delta_CP and energies).

Pre-registered predictions (sha256 of this file recorded in
examples/data/neutrino_oscillation_qg.prereg before the run):
  N1  Part A: the std of theta_hat is within 10% of 1 / (4 sqrt(N + 1/2)) and
      |mean - theta| < 0.005 for all four theta.
  N2  Part B: filtered_readout error below raw on all 3 backends, and the
      median ratio raw / filtered_readout at least 2.
  N3  Part B: the filtered_readout probabilities of each energy sum to 1
      within 0.01 (they sum to 1 by construction after the filter; the
      readout-corrected ones need not) and P(mu -> mu) has its minimum at
      E = 2.6 GeV on all 3 backends.
  N4  Part C: the filtered_readout A_CP is positive and within 0.015 of 0.0250
      on all 3 backends.

python examples/neutrino_oscillation_qg.py

Findings (pre-registration sha256 5a594bf4...8d57, recorded 2026-10-16
14:17 UTC; the run used that exact file, then only this block was added):

Part A, theta_hat = arccos(qg_Z) / 4 at Delta = pi, N = 1000, 2000 reps
(std theory 0.0079):

  theta  0.1      0.3      0.5      0.7
  mean   0.09985  0.29991  0.50003  0.70060
  std    0.00793  0.00772  0.00784  0.00805

Part B, mean |P_hat - P| (3 flavours x 8 energies, 4000 shots), and Part C,
A_CP at 2.6 GeV (exact 0.0250, 20000 shots):

  backend       raw     readout  filtered  filt+ro   kept   A_CP raw / filt+ro
  FakeBrisbane  0.0480  0.0377   0.0141    0.0125    0.863  0.0242 / 0.0256
  FakeTorino    0.0343  0.0283   0.0103    0.0100    0.914  0.0174 / 0.0218
  FakeFez       0.0252  0.0238   0.0086    0.0085    0.938  0.0203 / 0.0219

  N1 PASS: std within 3% of 1 / (4 sqrt(N + 1/2)) at every theta; bias below
     0.001.
  N2 PASS: filter + readout below raw on all 3 (ratio 3.8, 3.4, 3.0; median
     3.4). Most of the gain is the filter: readout correction alone gives
     1.1-1.3x.
  N3 PASS: the minimum of P(mu -> mu) is at 2.6 GeV on all 3. The sum check
     is close to a tautology (it holds by construction up to the readout
     correction) and should not count as evidence.
  N4 PASS: A_CP 0.0256, 0.0218, 0.0219 (exact 0.0250). The raw estimate is
     also positive (0.017-0.024): A_CP is a difference of two probabilities
     with nearly the same circuit, so most of the common shrinkage cancels,
     as with the fidelity classifier of §149; the filter halves the error on
     the two CZ devices (0.008 -> 0.003) and on brisbane changes it little.
  * The circuits are 3 qubits and 18 two-qubit gates after transpilation:
    the kept share is 86-94%, higher than in §146-§149, so the filter costs
    little here.
  * Limits: vacuum only (DUNE-like baselines have sizeable matter effects),
    fixed NuFIT-like parameters, simulated device noise; the oscillation is a
    textbook two- and three-level problem, so this tests the read-out, not
    neutrino physics.
"""

import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import ibm  # noqa: E402
from qang import neutrino as nu  # noqa: E402

SEED = 151
THETAS, N_A, REPS = (0.1, 0.3, 0.5, 0.7), 1000, 2000
L, DCP, ENERGIES, SHOTS_B = 1300.0, -math.pi / 2, (0.8, 1.2, 1.6, 2.0, 2.6, 3.2, 4.0, 5.0), 4000
E_C, SHOTS_C = 2.6, 20000
BACKENDS = ("FakeBrisbane", "FakeTorino", "FakeFez")
KINDS = ("raw", "readout", "filtered", "filtered_readout")
OUT = os.path.join(HERE, "data", "neutrino_oscillation_qg.json")
PREREG = os.path.join(HERE, "data", "neutrino_oscillation_qg.prereg")


def part_a():
    rng = np.random.default_rng(SEED)
    rows = []
    for th in THETAS:
        p1 = nu.two_flavour_probability(th, math.pi)
        k1 = rng.binomial(N_A, p1, REPS)
        est = np.array([nu.mixing_angle_from_maximum(1 - 2 * k / N_A) for k in k1])
        rows.append({"theta": th, "mean": float(est.mean()), "std": float(est.std(ddof=1)),
                     "std_theory": nu.mixing_angle_std(N_A)})
    return rows


def sampler(backend):
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler
    except ImportError:  # pragma: no cover
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    return Sampler(mode=backend)


def probs(out, kind):
    return ibm.occupations(out, kind)


def run_backend(name):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile

    b = getattr(fp, name)()
    smp = sampler(b)
    qcs = [nu.three_flavour_circuit("mu", L, E, DCP) for E in ENERGIES]
    ts = [transpile(qc, b, optimization_level=1, seed_transpiler=137) for qc in qcs]
    res = smp.run(ts, shots=SHOTS_B).result()
    outs = [ibm.ibm_qg(r, t, b, weight=1) for r, t in zip(res, ts)]
    exact = np.array([nu.three_flavour_probabilities(L, E, DCP)[1] for E in ENERGIES])
    est = {k: np.array([probs(o, k) for o in outs]) for k in KINDS}
    err = {k: float(np.abs(v - exact).mean()) for k, v in est.items()}
    sums = est["filtered_readout"].sum(axis=1)
    surv_min = ENERGIES[int(np.argmin(est["filtered_readout"][:, 1]))]
    qc_c = [nu.three_flavour_circuit("mu", L, E_C, DCP, anti) for anti in (False, True)]
    tc = [transpile(qc, b, optimization_level=1, seed_transpiler=137) for qc in qc_c]
    rc = smp.run(tc, shots=SHOTS_C).result()
    oc = [ibm.ibm_qg(r, t, b, weight=1) for r, t in zip(rc, tc)]
    acp = {k: float(probs(oc[0], k)[0] - probs(oc[1], k)[0]) for k in KINDS}
    return {"backend": name, "error": err, "estimates": {k: v.tolist() for k, v in est.items()},
            "sums_filtered_readout": sums.tolist(), "survival_min_E": surv_min, "acp": acp,
            "kept_raw": [o["kept_raw"] for o in outs],
            "two_qubit_gates": [sum(1 for i in t.data if i.operation.num_qubits == 2) for t in ts]}


def main():
    with open(__file__, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    print(json.dumps({"sha256": digest}), flush=True)
    a = part_a()
    for r in a:
        print(json.dumps({k: round(v, 5) for k, v in r.items()}), flush=True)
    acp_exact = nu.cp_asymmetry(L, E_C, DCP)
    print(json.dumps({"A_CP exact": round(acp_exact, 5), "E first max": round(nu.first_maximum_energy(L), 3)}))
    rows = []
    for name in BACKENDS:
        try:
            r = run_backend(name)
            print(json.dumps({"backend": name, "err": {k: round(v, 4) for k, v in r["error"].items()},
                              "acp": {k: round(v, 4) for k, v in r["acp"].items()},
                              "survival min at": r["survival_min_E"],
                              "kept": round(float(np.mean(r["kept_raw"])), 3),
                              "2q gates": int(np.mean(r["two_qubit_gates"]))}), flush=True)
        except Exception as e:  # a backend that cannot run is reported, not hidden
            r = {"backend": name, "failure": f"{type(e).__name__}: {e}"[:300]}
            print(json.dumps(r), flush=True)
        rows.append(r)
    ok = [r for r in rows if "error" in r]
    fr = "filtered_readout"
    ratio = [r["error"]["raw"] / r["error"][fr] for r in ok]
    v = {"N1": all(abs(r["std"] / r["std_theory"] - 1) <= 0.10 and abs(r["mean"] - r["theta"]) < 0.005 for r in a),
         "N2": len(ok) == 3 and all(r["error"][fr] < r["error"]["raw"] for r in ok) and float(np.median(ratio)) >= 2,
         "N3": len(ok) == 3 and all(max(abs(s - 1) for s in r["sums_filtered_readout"]) <= 0.01
                                    and r["survival_min_E"] == 2.6 for r in ok),
         "N4": len(ok) == 3 and all(r["acp"][fr] > 0 and abs(r["acp"][fr] - acp_exact) <= 0.015 for r in ok)}
    print(json.dumps({"median raw / filtered_readout": round(float(np.median(ratio)), 2) if ratio else None}))
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok_ else 'FAIL'}" for k, ok_ in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"sha256": digest, "part_a": a, "acp_exact": acp_exact, "rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    if "--prereg" in sys.argv:
        with open(__file__, "rb") as fh:
            d = hashlib.sha256(fh.read()).hexdigest()
        from datetime import datetime, timezone

        with open(PREREG, "w", encoding="utf-8") as fh:
            fh.write(f"{d}  examples/neutrino_oscillation_qg.py  {datetime.now(timezone.utc).isoformat()}\n")
        print(d)
    else:
        main()
