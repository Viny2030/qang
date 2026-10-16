"""
Collective neutrino oscillations: the conserved number is the mass number, not the flavour number (§154)

§151 called dense neutrino gases "a natural next target for the weight filter"
because "the total flavour number is conserved". That holds for the
neutrino-neutrino term alone. With the vacuum term it is the total number of
nu_2 (the mass number) that is conserved:

    H_m = - sum_i omega_i Z_i / 2 + (mu / 2N) sum_{i<j} (X_i X_j + Y_i Y_j + Z_i Z_j)

in the mass basis (qang.neutrino.collective_mass_hamiltonian; single-angle
approximation, two flavours, one qubit per neutrino). So the filter applies to
read-outs in the mass basis, and a flavour initial state is a superposition of
mass sectors with known, conserved probabilities. For that case
qang.sectors.reweight_sectors rescales each mass sector to its known
probability instead of keeping one sector.

Set-up (fixed before the run): N = 4, omega = (0.25, 0.5, 0.75, 1.0),
mu = 4, theta = 0.195, t in {0.5, 1, 1.5, 2}, dt = 0.5 (steps = 2t), Trotter
circuit qang.neutrino.collective_trotter_circuit (exchange gate on all 6
pairs per step, order reversed every other step). Exact: expm of H_m.
Noisy runs: FakeBrisbane, FakeTorino, FakeFez, optimization_level 1,
seed_transpiler 137, 8000 shots. Error: mean |P_hat - P_exact| over the 4
neutrinos and the 4 times.
  Part A  mass-basis initial state |nu_1 nu_1 nu_2 nu_2> ("0011"), read-out
          P(nu_2) in the mass basis: weight 2 exactly; read-outs raw,
          readout, filtered, filtered_readout.
  Part B  flavour initial state (nu_e nu_e nu_x nu_x), read-out P(nu_2) in the
          mass basis: raw, readout, reweighted, reweighted_readout (sector
          probabilities 0.0013, 0.0670, 0.8633, 0.0670, 0.0013).
  Part C  the same flavour state read in the flavour basis, P(nu_x): raw and
          readout only, since no weight is conserved there.

python examples/neutrino_collective_qg.py --dev   (structural checks only)
python examples/neutrino_collective_qg.py

Checked during development, before the predictions (not predictions):
  * H_m conserves the mass weight exactly, and the total flavour number is
    not conserved (tests/test_neutrino.py); the Trotter circuit agrees with
    expm to 1e-5 at 300 steps (t = 3, an earlier parameter set) in all four
    basis combinations. Nothing at the chosen dt was computed.
  * Two-qubit gates after transpilation, Part A, t = 0.5 / 1 / 1.5 / 2:
    27 / 54 / 81 / 108 on all three backends. A first choice (mu = 2,
    t = 1..4) needed up to 216 and was dropped before any run.
  * Part B sector probabilities (conserved): 0.0013, 0.0670, 0.8633, 0.0670,
    0.0013 for 0..4 nu_2.

Pre-registered predictions (sha256 of this file recorded in
examples/data/neutrino_collective_qg.prereg before the run):
  C1  noiseless Trotter error of Part A below 0.02.
  C2  Part A: filtered_readout error below raw on all 3 backends, median
      raw / filtered_readout at least 2.
  C3  Part A: that gain is smaller at t = 2 than at t = 0.5 on at least 2 of
      3 backends (deeper circuits scramble inside the sector, which no
      filter sees).
  C4  Part B: reweighted_readout error below raw on all 3 backends, median
      gain at least 1.3.
  C5  Part B gains less than Part A: median gain B < median gain A (a known
      distribution over sectors constrains less than one sharp sector).
  C6  exact physics: the total flavour number sum_i P(nu_x) moves by more than
      0.05 from its initial value 2 at some t in the grid (so it is not the
      conserved quantity).
  C7  Part C, where nothing is conserved: its readout error is larger than
      the Part B reweighted_readout error on all 3 backends.

Findings (pre-registration sha256 6a3441ef...c5b7, recorded 2026-10-16; the
run used that exact file, then only this block was added):

All seven predictions FAIL. The reason is one design error, not seven
results: at dt = 0.5 the noiseless Trotter error is 0.095 (Part A; B 0.089,
C 0.090), LARGER than the whole noisy error against the exact evolution
(0.074-0.088). Scored against the exact evolution every read-out is mostly
Trotter error, and a read-out that removes noise moves the estimate towards
the Trotter value, which here is not closer to the exact one.

  backend       A raw / f+ro     B raw / rw+ro    C raw / ro       kept A (t = 0.5 .. 2)
  FakeBrisbane  0.0748 / 0.0756  0.0750 / 0.0771  0.0769 / 0.0786  0.79 0.66 0.58 0.52
  FakeTorino    0.0761 / 0.0844  0.0748 / 0.0803  0.0760 / 0.0782  0.87 0.80 0.76 0.71
  FakeFez       0.0799 / 0.0883  0.0740 / 0.0769  0.0758 / 0.0767  0.91 0.85 0.80 0.76

  C1 FAIL: Trotter error 0.095 (needed < 0.02); the exchange gates of the six
     pairs do not commute and mu dt / 2N = 0.25 is too coarse for 4 pairs
     per qubit.
  C2-C5, C7 FAIL as scored (median gain A 0.90, B 0.96); see the post-hoc
     scoring below for what the filter does to the noise.
  C6 FAIL: the total flavour number moves by only 0.016 (needed > 0.05). It
     is not conserved (tests/test_neutrino.py), but with theta = 0.195 and
     omega t <= 2 the change is small: the correction to §151 is right in
     principle and small at these parameters.

Post-hoc, NOT pre-registered (examples/neutrino_collective_qg_posthoc.py,
new samples): the same circuits scored against the noiseless Trotter
circuit, as §153 and the chemistry sections do:

  backend       A raw / f+ro     B raw / rw+ro    C raw / ro
  FakeBrisbane  0.0537 / 0.0246  0.0541 / 0.0339  0.0567 / 0.0548
  FakeTorino    0.0318 / 0.0113  0.0313 / 0.0191  0.0347 / 0.0309
  FakeFez       0.0275 / 0.0124  0.0268 / 0.0156  0.0266 / 0.0256

  The filter in the mass basis gains 2.2x, 2.8x, 2.2x; reweighting the known
  sector distribution 1.6x, 1.6x, 1.7x, less, as C5 expected; the flavour
  basis, with nothing conserved, gains nothing beyond readout correction.
  Both gains fall at t = 2 on all 3 backends (A: 1.5, 2.0, 1.2; B: 1.1, 0.8,
  0.9), as C3 expected. These agree with the intended predictions, but they
  were computed after seeing the failure and do not count as tests.
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
from qang.sectors import reweight_sectors  # noqa: E402

OMEGAS, MU, THETA = (0.25, 0.5, 0.75, 1.0), 4.0, 0.195
TIMES, DT, SHOTS = (0.5, 1.0, 1.5, 2.0), 0.5, 8000
N = len(OMEGAS)
BACKENDS = ("FakeBrisbane", "FakeTorino", "FakeFez")
PARTS = {"A": ("0011", "mass", "mass"), "B": ("eexx", "flavour", "mass"), "C": ("eexx", "flavour", "flavour")}
OUT = os.path.join(HERE, "data", "neutrino_collective_qg.json")
PREREG = os.path.join(HERE, "data", "neutrino_collective_qg.prereg")


def circuit(part, t, measure=True):
    ini, ib, mb = PARTS[part]
    return nu.collective_trotter_circuit(OMEGAS, MU, THETA, ini, t, int(round(t / DT)), ib, mb, measure=measure)


def exact(part):
    ini, ib, mb = PARTS[part]
    return np.array([nu.collective_exact(OMEGAS, MU, THETA, ini, t, ib, mb) for t in TIMES])


def noiseless(part):
    from qiskit.quantum_info import Statevector

    idx = np.arange(2 ** N)
    out = []
    for t in TIMES:
        p = Statevector(circuit(part, t, measure=False)).probabilities()
        out.append([p[(idx >> i) & 1 == 1].sum() for i in range(N)])
    return np.array(out)


def target(part):
    ini, ib, _ = PARTS[part]
    return nu.mass_sector_probabilities(ini, THETA, ib)


def readouts(counts, transpiled, backend, part):
    """P(qubit i = 1) per read-out. Distributions are indexed with qubit 0 as the most significant bit,
    the order of ibm.readout_inverse_distribution."""
    shots = sum(counts.values())
    p = np.zeros(2 ** N)
    for s, c in counts.items():
        p[int(s[::-1], 2)] += c / shots
    errs = ibm.readout_errors(backend, ibm.measured_physical_qubits(transpiled))
    q = ibm.readout_inverse_distribution(p, errs)
    idx = np.arange(2 ** N)
    occ = lambda d: np.array([d[(idx >> (N - 1 - i)) & 1 == 1].sum() / d.sum() for i in range(N)])  # noqa: E731
    out = {"raw": occ(p), "readout": occ(q)}
    if part in ("A", "B"):
        name = "filtered" if part == "A" else "reweighted"
        tg = target(part)
        r, kept = reweight_sectors(p, N, tg)
        rq, _ = reweight_sectors(np.clip(q, 0, None), N, tg)
        out[name], out[name + "_readout"], out["kept"] = occ(r), occ(rq), kept
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
    cases = [(part, t) for part in PARTS for t in TIMES]
    ts = [transpile(circuit(part, t), b, optimization_level=1, seed_transpiler=137) for part, t in cases]
    res = sampler(b).run(ts, shots=SHOTS).result()
    row = {"backend": name, "two_qubit_gates": {}, "error": {}, "error_by_time": {}, "kept": {}}
    for part in PARTS:
        ks = [i for i, (p_, _) in enumerate(cases) if p_ == part]
        outs = [readouts(res[i].data.meas.get_counts(), ts[i], b, part) for i in ks]
        kinds = [k for k in outs[0] if k != "kept"]
        row["two_qubit_gates"][part] = [sum(1 for x in ts[i].data if x.operation.num_qubits == 2) for i in ks]
        row["error_by_time"][part] = {k: [float(np.abs(o[k] - e).mean()) for o, e in zip(outs, ex[part])]
                                      for k in kinds}
        row["error"][part] = {k: float(np.mean(v)) for k, v in row["error_by_time"][part].items()}
        if "kept" in outs[0]:
            row["kept"][part] = [float(o["kept"]) for o in outs]
    return row


def development():
    """Structural checks only (no noisy run): sector probabilities and two-qubit gate counts."""
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile

    gates = {}
    for name in BACKENDS:
        b = getattr(fp, name)()
        gates[name] = {t: sum(1 for x in transpile(circuit("A", t), b, optimization_level=1, seed_transpiler=137)
                              .data if x.operation.num_qubits == 2) for t in TIMES}
    print(json.dumps({"sectors_B": [round(x, 4) for x in target("B")], "two_qubit_gates_A": gates}, indent=1))


def main():
    with open(__file__, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    print(json.dumps({"sha256": digest}), flush=True)
    ex = {p: exact(p) for p in PARTS}
    nl = {p: float(np.abs(noiseless(p) - ex[p]).mean()) for p in PARTS}
    flav = ex["C"].sum(axis=1)
    print(json.dumps({"noiseless trotter error": nl, "total P(nu_x) exact by time": flav.tolist()}), flush=True)
    rows = []
    for name in BACKENDS:
        try:
            r = run_backend(name, ex)
        except Exception as e:  # a backend that cannot run is reported, not hidden
            r = {"backend": name, "failure": f"{type(e).__name__}: {e}"[:300]}
        print(json.dumps({k: v for k, v in r.items() if k != "error_by_time"}), flush=True)
        rows.append(r)
    ok = [r for r in rows if "error" in r]
    full = len(ok) == 3
    gA = [r["error"]["A"]["raw"] / r["error"]["A"]["filtered_readout"] for r in ok]
    gB = [r["error"]["B"]["raw"] / r["error"]["B"]["reweighted_readout"] for r in ok]
    gA2 = [r["error_by_time"]["A"]["raw"][-1] / r["error_by_time"]["A"]["filtered_readout"][-1] for r in ok]
    gA0 = [r["error_by_time"]["A"]["raw"][0] / r["error_by_time"]["A"]["filtered_readout"][0] for r in ok]
    v = {"C1": nl["A"] < 0.02,
         "C2": full and all(r["error"]["A"]["filtered_readout"] < r["error"]["A"]["raw"] for r in ok)
         and float(np.median(gA)) >= 2,
         "C3": full and sum(a < b for a, b in zip(gA2, gA0)) >= 2,
         "C4": full and all(r["error"]["B"]["reweighted_readout"] < r["error"]["B"]["raw"] for r in ok)
         and float(np.median(gB)) >= 1.3,
         "C5": full and float(np.median(gB)) < float(np.median(gA)),
         "C6": bool(np.abs(flav - 2).max() > 0.05),
         "C7": full and all(r["error"]["C"]["readout"] > r["error"]["B"]["reweighted_readout"] for r in ok)}
    print(json.dumps({"median gain A": float(np.median(gA)) if gA else None,
                      "median gain B": float(np.median(gB)) if gB else None}))
    print("predictions: " + ", ".join(f"{k} {'PASS' if x else 'FAIL'}" for k, x in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"sha256": digest, "noiseless": nl, "exact": {p: e.tolist() for p, e in ex.items()},
                   "rows": rows, "verdict": v}, fh, indent=1, default=float)
    return rows, v


if __name__ == "__main__":
    if "--dev" in sys.argv:
        development()
    elif "--prereg" in sys.argv:
        with open(__file__, "rb") as fh:
            d = hashlib.sha256(fh.read()).hexdigest()
        from datetime import datetime, timezone

        with open(PREREG, "w", encoding="utf-8") as fh:
            fh.write(f"{d}  examples/neutrino_collective_qg.py  {datetime.now(timezone.utc).isoformat()}\n")
        print(d)
    else:
        main()
