"""
Neutrinos through the Earth: constant-density matter (MSW) as a weight-preserving Trotter circuit (§152)

§151 left out matter. Electrons in the Earth add a potential A = 2 sqrt(2) G_F N_e E
to the nu_e entry of the flavour Hamiltonian (the MSW effect; -A for
antineutrinos), which on a 1300 km baseline is large and mimics CP
violation. qang.neutrino.matter_trotter_circuit alternates the exact vacuum
step of §151 (RBS and phase gates) with a phase gate on the nu_e qubit
(second-order, Strang). Every gate conserves the weight, so the qg filter
applies; this is the set-up of §147 (more steps: less Trotter error, more
noise) with a physical Hamiltonian.

Set-up (fixed before the run): NUFIT_NO parameters, L = 1300 km,
rho = 2.85 g/cm^3, Y_e = 0.5, initial nu_mu, E in {0.8, 1.2, 1.6, 2.0, 2.6,
3.2, 4.0, 5.0} GeV, delta_CP = -pi/2, steps r in {1, 2, 3, 4, 6}. Error: mean
|P_hat - P_exact| over the 3 flavours and 8 energies (exact: expm of the
flavour Hamiltonian). Noisy runs: FakeBrisbane, FakeTorino, FakeFez,
optimization_level 1, seed_transpiler 137, 4000 shots. Part C: A_CP at
E = 2.6 GeV with delta_CP = 0 (pure matter effect), r = 2, 20000 shots.

Checked during development, before the predictions (not a prediction): r = 64
agrees with the exact matter probabilities to 7e-6, and rho = 0 reproduces
the vacuum of §151. Nothing at r <= 6 was computed.

Pre-registered predictions (sha256 of this file recorded in
examples/data/neutrino_matter_qg.prereg before the run):
  M1  noiseless Trotter error below 0.005 at r = 2 and below 0.001 at r = 4.
  M2  exact: at delta_CP = 0 the matter-induced A_CP at 2.6 GeV is positive
      and above 0.01.
  M3  noisy: the best (over r) filtered_readout error is below the best raw
      error on all 3 backends, median ratio at least 2.
  M4  noisy: the r with the smallest filtered_readout error is >= the r with
      the smallest raw error on all 3 backends.
  M5  noisy: the filtered_readout A_CP of Part C is positive and within 0.015
      of the exact value on all 3 backends.

python examples/neutrino_matter_qg.py

Findings (pre-registration sha256 1f1ededb...7d91, recorded 2026-10-16; the
run used that exact file, then only this block was added):

Noiseless Trotter error, r = 1, 2, 3, 4, 6: 0.0123, 0.0052, 0.0020, 0.0011,
0.0005. A_CP at delta_CP = 0, 2.6 GeV: exact 0.0422, Trotter r = 2 0.0333.

  backend       best raw  best f+ro  r* raw/f+ro  kept   A_CP raw / f+ro (delta_CP = 0)
  FakeBrisbane  0.0463    0.0165     1 / 1        0.777  0.0284 / 0.0348
  FakeTorino    0.0274    0.0094     1 / 1        0.839  0.0313 / 0.0354
  FakeFez       0.0229    0.0138     1 / 1        0.866  0.0281 / 0.0291

  M1 FAIL, narrowly: 0.0052 at r = 2 (needed < 0.005) and 0.0011 at r = 4
     (needed < 0.001); the error falls as about r^-2 as expected, the
     constants were set slightly too tight.
  M2 PASS: matter alone gives A_CP = 0.042 at delta_CP = 0, larger than the
     vacuum CP signal of §151 at delta_CP = -pi/2 (0.025). On this baseline
     the matter effect has to be separated before CP violation can be read.
  M3 PASS: the filter lowers the best error 2.8x, 2.9x and 1.7x (median 2.8).
  M4 PASS, but vacuously: r* = 1 for every kind on every backend. At r = 1
     the Trotter error (0.012) is already below the noise, so the trade-off
     of §147 never appears and the filter does not move r* deeper here.
  M5 PASS: A_CP 0.035, 0.035, 0.029 against 0.042, all positive. Most of
     the remaining gap is Trotter error at r = 2 (0.033), not noise: the
     filtered values lie within 0.004 of the r = 2 value on all 3, the raw
     ones 0.002-0.005 below it.
  * The matter-induced asymmetry is recovered with the filter on every
    backend, and the circuits remain short (18-36 two-qubit gates at r = 1-2);
    the deeper circuits (r = 6: 88-108 gates) only add noise at these
    parameters.
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

L, RHO, YE, DCP = 1300.0, 2.85, 0.5, -math.pi / 2
ENERGIES, RS, SHOTS = (0.8, 1.2, 1.6, 2.0, 2.6, 3.2, 4.0, 5.0), (1, 2, 3, 4, 6), 4000
E_C, R_C, SHOTS_C = 2.6, 2, 20000
BACKENDS = ("FakeBrisbane", "FakeTorino", "FakeFez")
KINDS = ("raw", "readout", "filtered", "filtered_readout")
OUT = os.path.join(HERE, "data", "neutrino_matter_qg.json")
PREREG = os.path.join(HERE, "data", "neutrino_matter_qg.prereg")


def exact():
    return np.array([nu.matter_probabilities(L, E, DCP, RHO, YE)[1] for E in ENERGIES])


def noiseless(ex):
    from qiskit.quantum_info import Statevector

    out = {}
    for r in RS:
        P = [np.abs(Statevector(nu.matter_trotter_circuit("mu", L, E, DCP, RHO, r, YE, measure=False)).data[[1, 2, 4]])
             ** 2 for E in ENERGIES]
        out[r] = float(np.abs(np.array(P) - ex).mean())
    return out


def sampler(backend):
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler
    except ImportError:  # pragma: no cover
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    return Sampler(mode=backend)


def run_backend(name, ex):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile

    b = getattr(fp, name)()
    smp = sampler(b)
    cases = [(r, E) for r in RS for E in ENERGIES]
    ts = [transpile(nu.matter_trotter_circuit("mu", L, E, DCP, RHO, r, YE), b, optimization_level=1,
                    seed_transpiler=137) for r, E in cases]
    res = smp.run(ts, shots=SHOTS).result()
    outs = [ibm.ibm_qg(x, t, b, weight=1) for x, t in zip(res, ts)]
    err = {k: [] for k in KINDS}
    for i, r in enumerate(RS):
        block = outs[i * len(ENERGIES):(i + 1) * len(ENERGIES)]
        for k in KINDS:
            err[k].append(float(np.abs(np.array([ibm.occupations(o, k) for o in block]) - ex).mean()))
    rstar = {k: RS[int(np.argmin(v))] for k, v in err.items()}
    tc = [transpile(nu.matter_trotter_circuit("mu", L, E_C, 0.0, RHO, R_C, YE, anti), b, optimization_level=1,
                    seed_transpiler=137) for anti in (False, True)]
    rc = smp.run(tc, shots=SHOTS_C).result()
    oc = [ibm.ibm_qg(x, t, b, weight=1) for x, t in zip(rc, tc)]
    acp = {k: float(ibm.occupations(oc[0], k)[0] - ibm.occupations(oc[1], k)[0]) for k in KINDS}
    return {"backend": name, "error": err, "r_star": rstar, "best": {k: float(min(v)) for k, v in err.items()},
            "acp": acp, "kept_raw": [o["kept_raw"] for o in outs],
            "two_qubit_gates": {r: int(np.mean([sum(1 for g in t.data if g.operation.num_qubits == 2)
                                                for t in ts[i * len(ENERGIES):(i + 1) * len(ENERGIES)]]))
                                for i, r in enumerate(RS)}}


def main():
    with open(__file__, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    print(json.dumps({"sha256": digest}), flush=True)
    ex = exact()
    nl = noiseless(ex)
    acp_exact = nu.cp_asymmetry(L, E_C, 0.0, rho=RHO, ye=YE)
    acp_trot = float(np.abs(__import__("qiskit.quantum_info", fromlist=["Statevector"]).Statevector(
        nu.matter_trotter_circuit("mu", L, E_C, 0.0, RHO, R_C, YE, measure=False)).data[1]) ** 2 - np.abs(
        __import__("qiskit.quantum_info", fromlist=["Statevector"]).Statevector(
            nu.matter_trotter_circuit("mu", L, E_C, 0.0, RHO, R_C, YE, True, measure=False)).data[1]) ** 2)
    print(json.dumps({"noiseless trotter error": {r: round(v, 5) for r, v in nl.items()},
                      "A_CP(delta=0) exact": round(acp_exact, 5), "A_CP(delta=0) trotter r=2": round(acp_trot, 5)}),
          flush=True)
    rows = []
    for name in BACKENDS:
        try:
            r = run_backend(name, ex)
            print(json.dumps({"backend": name, "best": {k: round(v, 4) for k, v in r["best"].items()},
                              "r*": r["r_star"], "acp": {k: round(v, 4) for k, v in r["acp"].items()},
                              "kept": round(float(np.mean(r["kept_raw"])), 3), "2q": r["two_qubit_gates"]}), flush=True)
        except Exception as e:  # a backend that cannot run is reported, not hidden
            r = {"backend": name, "failure": f"{type(e).__name__}: {e}"[:300]}
            print(json.dumps(r), flush=True)
        rows.append(r)
    ok = [r for r in rows if "best" in r]
    fr = "filtered_readout"
    ratio = [r["best"]["raw"] / r["best"][fr] for r in ok]
    v = {"M1": nl[2] < 0.005 and nl[4] < 0.001,
         "M2": acp_exact > 0.01,
         "M3": len(ok) == 3 and all(r["best"][fr] < r["best"]["raw"] for r in ok) and float(np.median(ratio)) >= 2,
         "M4": len(ok) == 3 and all(r["r_star"][fr] >= r["r_star"]["raw"] for r in ok),
         "M5": len(ok) == 3 and all(r["acp"][fr] > 0 and abs(r["acp"][fr] - acp_exact) <= 0.015 for r in ok)}
    print(json.dumps({"median best raw / filtered_readout": round(float(np.median(ratio)), 2) if ratio else None}))
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok_ else 'FAIL'}" for k, ok_ in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"sha256": digest, "noiseless": nl, "acp_exact": acp_exact, "acp_trotter_r2": acp_trot,
                   "rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    if "--prereg" in sys.argv:
        with open(__file__, "rb") as fh:
            d = hashlib.sha256(fh.read()).hexdigest()
        from datetime import datetime, timezone

        with open(PREREG, "w", encoding="utf-8") as fh:
            fh.write(f"{d}  examples/neutrino_matter_qg.py  {datetime.now(timezone.utc).isoformat()}\n")
        print(d)
    else:
        main()
