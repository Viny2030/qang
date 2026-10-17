"""
Solar neutrinos: the adiabatic MSW conversion as a weight-preserving circuit (§156)

A nu_e born at the centre of the Sun meets an electron density that falls
from about (rho Y_e) = 100 g/cm^3 to zero. Above a few MeV the density at the
core exceeds the MSW resonance, the neutrino follows the heavier matter
eigenstate adiabatically and leaves the Sun as nu_2: P_ee falls from about
0.54 (vacuum averaging, low energy) to about 0.30 (sin^2 theta12, 8B
neutrinos). On the way to the Earth the mass states lose their coherence, so
what arrives is a mixture of nu_1 and nu_2, and

    P_ee = c13^4 (cos^2 theta12 p_1 + sin^2 theta12 p_2) + s13^4,

linear in the mass-basis populations, i.e. in qg_Z of the nu_2 qubit.

qang.neutrino.solar_circuit (two effective flavours, nu_3 decoupled; checked
against three flavours to 0.001 during development): the density is cut into
r constant-density slabs, uniform in the matter angle theta_m; each slab is
one exact 2x2 propagator, written as RZ RBS RZ on two qubits in the weight-1
sector (qubit 0 = nu_e, qubit 1 = nu_x), and a final RBS rotates to the
vacuum mass basis (qubit 1 = nu_2). Every gate conserves the weight, so the
qg filter applies. Two controls: the one-qubit encoding (each slab one
single-qubit rotation; no filter possible, no two-qubit gate), and the
compiled circuit: since only p_2 is observed, the whole Sun can be replaced by
the one RBS that loads p_2 (computed classically).

Set-up (fixed before the run): NUFIT_NO parameters, exponential profile with
scale height h = 100 km from x = 0 to X = 800 km (a scaled Sun: the real one,
h ~ 6e4 km, is far more adiabatic, with the same adiabatic answer); E in
{0.5, 1, 2, 4, 6, 10, 15} MeV; r in {4, 8, 16}. Noisy runs: FakeBrisbane,
FakeTorino, FakeFez, optimization_level 1, seed_transpiler 137, 4000 shots.
Error: mean over the 7 energies of |P_ee_hat - P_ee_ref|. The reference for
the noisy runs is the noiseless circuit itself (the lesson of §154), except
in S6.

Computed during development, before the predictions (not predictions):
  exact P_ee (20000 slabs) 0.536, 0.517, 0.478, 0.405, 0.353, 0.305, 0.304;
  the adiabatic formula differs by at most 0.010 (at 10 MeV h = 100 km is
  slightly non-adiabatic). Noiseless slab error against exact, mean (max):
  r = 4 0.019 (0.034), r = 8 0.009 (0.016), r = 16 0.003 (0.008). MSW drop
  P_ee(0.5) - P_ee(10 MeV): exact 0.231, slab r = 8 0.242. Two-qubit gates
  after transpiling (all 3 backends): r = 4, 8, 16 -> 11, 19, 35; compiled 3.
  Nothing noisy was run.

Pre-registered predictions (sha256 of this file recorded in
examples/data/neutrino_solar_qg.prereg before the run):
  S1  unary slab circuit, r = 8: the filtered_readout error is below the raw
      error on all 3 backends, median ratio raw / filtered_readout >= 2.
  S2  unary slab circuit, r = 8, filtered_readout: P_ee(0.5 MeV) -
      P_ee(10 MeV) > 0.15 on all 3 backends (the MSW drop is seen).
  S3  compiled unary circuit: filtered_readout error below 0.01 on all 3.
  S4  compiled unary circuit, raw read-out: error below the filtered_readout
      error of the r = 8 slab circuit on all 3 (depth costs more than the
      filter recovers).
  S5  one-qubit encoding, r = 8 (the transpiler fuses it into one gate),
      readout-corrected: error below 0.01 on all 3.
  S6  against the exact P_ee, the r in {4, 8, 16} with the smallest
      filtered_readout error is >= the r with the smallest raw error, on all
      3 backends.

python examples/neutrino_solar_qg.py

Findings (pre-registration sha256 7414231d...20ad, recorded 2026-10-17; the
run used that exact file, then only this block was added):

Mean |P_ee_hat - P_ee(noiseless circuit)| over 7 energies, raw / filtered +
readout (kept fraction of the raw shots):

  backend       slab r = 4       slab r = 8       slab r = 16      compiled         one qubit raw / ro
  FakeBrisbane  0.018 / 0.009    0.025 / 0.011    0.038 / 0.021    0.008 / 0.002    0.004 / 0.002
  FakeTorino    0.011 / 0.004    0.018 / 0.008    0.027 / 0.014    0.005 / 0.002    0.031 / 0.003
  FakeFez       0.016 / 0.008    0.027 / 0.015    0.042 / 0.026    0.006 / 0.003    0.004 / 0.002
  kept (r = 8): 0.86, 0.77, 0.85; compiled 0.93, 0.81, 0.96. Two-qubit gates:
  11, 19, 35 (slabs), 3 (compiled).

  S1 PASS, narrowly: the filter lowers the r = 8 error 2.2x, 2.3x and 1.9x
     (median 2.19, needed 2).
  S2 PASS: filtered MSW drop P_ee(0.5) - P_ee(10 MeV) = 0.218, 0.218, 0.202
     (noiseless circuit 0.242, exact 0.231).
  S3 PASS: the compiled circuit with filter + readout is within 0.002-0.003.
  S4 PASS: the compiled circuit even without the filter (0.005-0.008) beats
     the filtered r = 8 slab circuit (0.008-0.015).
  S5 PASS: one qubit, readout-corrected, 0.002-0.003. On FakeTorino its raw
     error is 0.031: that qubit has a large readout error, which the
     correction removes.
  S6 PASS: against the exact P_ee the best r is 8 for every read-out on
     Brisbane and Torino (equal, so this holds trivially there) and moves
     from 4 (raw) to 8 (filtered) on Fez.
  * The filter does what it did in §147 and §152: about 2x on a
    weight-preserving circuit, more depth bought with it on one backend.
  * But the decisive fact is physical, not about the filter: the phases are
    lost between the Sun and the Earth, so the observable is one population
    and the whole Sun is one rotation. Any slab circuit (more slabs: less
    discretization error, more noise) loses to the compiled one, and for two
    flavours a single qubit with readout correction is as good as anything.
    A quantum computer adds nothing to the adiabatic answer itself; what it
    could add is a case where coherence survives (day-night regeneration in
    the Earth, or collective effects, §154).
"""

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import ibm  # noqa: E402
from qang import neutrino as nu  # noqa: E402

H, X = 100.0, 800.0
ENERGIES = (0.0005, 0.001, 0.002, 0.004, 0.006, 0.010, 0.015)
RS, R0, SHOTS = (4, 8, 16), 8, 4000
BACKENDS = ("FakeBrisbane", "FakeTorino", "FakeFez")
KINDS = ("raw", "readout", "filtered", "filtered_readout")
OUT = os.path.join(HERE, "data", "neutrino_solar_qg.json")
PREREG = os.path.join(HERE, "data", "neutrino_solar_qg.prereg")


def pee(p2s):
    return np.array([nu.solar_pee_from_p2(float(p)) for p in p2s])


def references():
    ex = pee([nu.solar_p2_exact(E, H, X) for E in ENERGIES])
    slab = {r: pee([nu.solar_p2(E, H, X, r) for E in ENERGIES]) for r in RS}
    return ex, slab


def sampler(backend):
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler
    except ImportError:  # pragma: no cover
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    return Sampler(mode=backend)


def run_backend(name, ex, slab):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile

    b = getattr(fp, name)()
    smp = sampler(b)
    tp = lambda qc: transpile(qc, b, optimization_level=1, seed_transpiler=137)  # noqa: E731
    cases = [("slab", r) for r in RS] + [("compiled", R0)]
    ts = [tp(nu.solar_circuit(E, H, X, r, compiled=(kind == "compiled"))) for kind, r in cases for E in ENERGIES]
    res = smp.run(ts, shots=SHOTS).result()
    outs = [ibm.ibm_qg(x, t, b, weight=1) for x, t in zip(res, ts)]
    est = {}
    for i, (kind, r) in enumerate(cases):
        block = outs[i * len(ENERGIES):(i + 1) * len(ENERGIES)]
        est[f"{kind}{r}"] = {k: pee([ibm.occupations(o, k)[1] for o in block]) for k in KINDS}
    tq = [tp(nu.solar_circuit(E, H, X, R0, encoding="qubit")) for E in ENERGIES]
    rq = smp.run(tq, shots=SHOTS).result()
    oq = [ibm.ibm_qg(x, t, b) for x, t in zip(rq, tq)]
    est["qubit8"] = {k: pee([ibm.occupations(o, k)[0] for o in oq]) for k in ("raw", "readout")}
    err = {key: {k: float(np.abs(v - (slab[R0] if key in ("compiled8", "qubit8") else slab[int(key[4:])])).mean())
                 for k, v in d.items()} for key, d in est.items()}
    err_exact = {key: {k: float(np.abs(v - ex).mean()) for k, v in d.items()} for key, d in est.items()}
    rstar = {k: RS[int(np.argmin([err_exact[f"slab{r}"][k] for r in RS]))] for k in KINDS}
    return {"backend": name, "estimate": {key: {k: v.tolist() for k, v in d.items()} for key, d in est.items()},
            "error": err, "error_exact": err_exact, "r_star_exact": rstar,
            "kept_raw": {f"{kind}{r}": float(np.mean([o["kept_raw"] for o in outs[i * len(ENERGIES):(i + 1) *
                                                                                  len(ENERGIES)]]))
                         for i, (kind, r) in enumerate(cases)},
            "two_qubit_gates": {f"{kind}{r}": int(np.mean([sum(1 for g in t.data if g.operation.num_qubits == 2)
                                                           for t in ts[i * len(ENERGIES):(i + 1) * len(ENERGIES)]]))
                                for i, (kind, r) in enumerate(cases)}}


def main():
    with open(__file__, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    print(json.dumps({"sha256": digest}), flush=True)
    ex, slab = references()
    print(json.dumps({"exact": np.round(ex, 4).tolist(),
                      "slab error vs exact": {r: round(float(np.abs(v - ex).mean()), 4) for r, v in slab.items()}}),
          flush=True)
    rows = []
    for name in BACKENDS:
        try:
            r = run_backend(name, ex, slab)
            print(json.dumps({"backend": name, "error": {key: {k: round(v, 4) for k, v in d.items()}
                                                         for key, d in r["error"].items()},
                              "r* (exact)": r["r_star_exact"], "kept": {k: round(v, 3) for k, v in r["kept_raw"].items()},
                              "2q": r["two_qubit_gates"]}), flush=True)
        except Exception as e:  # a backend that cannot run is reported, not hidden
            r = {"backend": name, "failure": f"{type(e).__name__}: {e}"[:300]}
            print(json.dumps(r), flush=True)
        rows.append(r)
    ok = [r for r in rows if "error" in r]
    fr = "filtered_readout"
    ratio = [r["error"]["slab8"]["raw"] / r["error"]["slab8"][fr] for r in ok]
    drop = [r["estimate"]["slab8"][fr][0] - r["estimate"]["slab8"][fr][5] for r in ok]
    v = {"S1": len(ok) == 3 and all(r["error"]["slab8"][fr] < r["error"]["slab8"]["raw"] for r in ok)
         and float(np.median(ratio)) >= 2,
         "S2": len(ok) == 3 and all(d > 0.15 for d in drop),
         "S3": len(ok) == 3 and all(r["error"]["compiled8"][fr] < 0.01 for r in ok),
         "S4": len(ok) == 3 and all(r["error"]["compiled8"]["raw"] < r["error"]["slab8"][fr] for r in ok),
         "S5": len(ok) == 3 and all(r["error"]["qubit8"]["readout"] < 0.01 for r in ok),
         "S6": len(ok) == 3 and all(r["r_star_exact"][fr] >= r["r_star_exact"]["raw"] for r in ok)}
    print(json.dumps({"median raw / filtered_readout (slab8)": round(float(np.median(ratio)), 2) if ratio else None,
                      "MSW drop (slab8, f+ro)": [round(float(d), 3) for d in drop]}))
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok_ else 'FAIL'}" for k, ok_ in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"sha256": digest, "exact": ex.tolist(), "slab": {r: s.tolist() for r, s in slab.items()},
                   "rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    if "--prereg" in sys.argv:
        with open(__file__, "rb") as fh:
            d = hashlib.sha256(fh.read()).hexdigest()
        from datetime import datetime, timezone

        with open(PREREG, "w", encoding="utf-8") as fh:
            fh.write(f"{d}  examples/neutrino_solar_qg.py  {datetime.now(timezone.utc).isoformat()}\n")
        print(d)
    else:
        main()
