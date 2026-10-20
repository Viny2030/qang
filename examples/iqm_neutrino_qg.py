"""
First real-hardware data point: the three-flavour neutrino circuit of §151 on IQM Resonance (Garnet)

Pre-registered. The author approved predictions I1-I4 on 2026-10-20; this file was frozen (sha256 in
examples/data/iqm_neutrino_qg.prereg) before any job was submitted to IQM.

Why this experiment: 3 qubits, 12 CZ after compilation to IQM's native gates (r, cz), all gates
weight-preserving, so the qg filter applies; in simulation (§151) the filter lowered the error 3.0-3.8x on
IBM noise models, and 2.2x on IQM's own fake backends (rehearsal below, 2026-10-20).

Free rehearsal, 2026-10-20 (IQMFakeApollo ~ Garnet, IQMFakeAphrodite ~ Emerald; 4000 shots, 8 energies,
optimization_level 3, seed_transpiler 137): mean |P_hat - P| raw / filtered, kept fraction:
  IQMFakeApollo     0.1263 / 0.0584 (2.16x), kept 0.688
  IQMFakeAphrodite  0.1161 / 0.0515 (2.25x), kept 0.708

Modes (the token is read from the environment variable IQM_TOKEN, never from a file in the repository):
  python examples/iqm_neutrino_qg.py                      free rehearsal on the fake backends
  python examples/iqm_neutrino_qg.py --probe --yes-i-accept-qpu-cost
                                                          1 circuit, 100 shots on Garnet, to measure the
                                                          real credit cost before anything else
  python examples/iqm_neutrino_qg.py --run --shots N --yes-i-accept-qpu-cost
                                                          the 8 circuits, N shots each (N fixed after the probe)
  python examples/iqm_neutrino_qg.py --prereg             freeze this file (sha256) before --run

Set-up: as §151 Part B — vacuum, NUFIT_NO, delta_CP = -pi/2, L = 1300 km, initial nu_mu,
E in {0.8, 1.2, 1.6, 2.0, 2.6, 3.2, 4.0, 5.0} GeV, read all three flavours. Error: mean |P_hat - P| over
3 flavours x 8 energies; raw = all shots, filtered = weight-1 shots only.

Pre-registered predictions (for --run; the probe is a cost measurement, not a test):
  I1  the filtered error is below the raw error.
  I2  raw / filtered is at least 1.5.
  I3  the filtered P(mu -> mu) has its minimum at E = 2.6 GeV (first oscillation maximum 2.63 GeV).
  I4  the kept fraction is at least 0.5 on average.

Findings: (after the run)
"""

import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import interop as IO  # noqa: E402
from qang import neutrino as nu  # noqa: E402

L, DCP = 1300.0, -math.pi / 2
ENERGIES = (0.8, 1.2, 1.6, 2.0, 2.6, 3.2, 4.0, 5.0)
SERVER = "https://resonance.iqm.tech/"
QC_NAME = "garnet"
DATA = os.path.join(HERE, "data")
PREREG = os.path.join(DATA, "iqm_neutrino_qg.prereg")


def circuits(energies=ENERGIES):
    return [nu.three_flavour_circuit("mu", L, E, DCP) for E in energies]


def exact(energies=ENERGIES):
    return np.array([nu.three_flavour_probabilities(L, E, DCP)[1] for E in energies])


def analyse(counts_list, energies=ENERGIES):
    ex = exact(energies)
    raw, fil, kept = [], [], []
    for counts in counts_list:
        c = IO.canonical_counts(counts, sdk="qiskit")
        raw.append([(1 - z) / 2 for z, _ in IO.qg_z_profile(c)])
        fil.append([(1 - z) / 2 for z, _ in IO.filtered_qg_z_profile(c, 1)])
        kept.append(IO.weight_filter(c, 1)[1])
    raw, fil = np.array(raw), np.array(fil)
    out = {"raw_error": float(np.abs(raw - ex).mean()), "filtered_error": float(np.abs(fil - ex).mean()),
           "kept": float(np.mean(kept)), "raw": raw.tolist(), "filtered": fil.tolist(), "exact": ex.tolist()}
    if len(energies) == len(ENERGIES):
        out["survival_min_E"] = energies[int(np.argmin(fil[:, 1]))]
    return out


def verdict(r):
    return {"I1": r["filtered_error"] < r["raw_error"],
            "I2": r["raw_error"] / r["filtered_error"] >= 1.5,
            "I3": r.get("survival_min_E") == 2.6,
            "I4": r["kept"] >= 0.5}


def rehearsal(shots=4000):
    from iqm.qiskit_iqm import IQMFakeAphrodite, IQMFakeApollo
    from qiskit import transpile

    for B in (IQMFakeApollo, IQMFakeAphrodite):
        b = B()
        ts = [transpile(qc, b, optimization_level=3, seed_transpiler=137) for qc in circuits()]
        res = b.run(ts, shots=shots).result()
        r = analyse([res.get_counts(i) for i in range(len(ts))])
        print(json.dumps({"backend": b.name, "raw": round(r["raw_error"], 4), "filtered": round(r["filtered_error"], 4),
                          "ratio": round(r["raw_error"] / r["filtered_error"], 2), "kept": round(r["kept"], 3),
                          "cz": ts[0].count_ops().get("cz"), "verdict (rehearsal only)": verdict(r)}), flush=True)


def real_backend():
    if not os.environ.get("IQM_TOKEN"):
        sys.exit("IQM_TOKEN is not set: export it in this shell only, never write it to a file in the repository.")
    from iqm.qiskit_iqm import IQMProvider

    return IQMProvider(SERVER, quantum_computer=QC_NAME).get_backend()


def submit(energies, shots, tag):
    from qiskit import transpile

    b = real_backend()
    ts = [transpile(qc, b, optimization_level=3, seed_transpiler=137) for qc in circuits(energies)]
    print(json.dumps({"backend": b.name, "circuits": len(ts), "shots": shots,
                      "cz": [t.count_ops().get("cz") for t in ts]}), flush=True)
    t0 = time.time()
    job = b.run(ts, shots=shots)
    print(json.dumps({"job_id": job.job_id()}), flush=True)
    res = job.result()
    r = analyse([res.get_counts(i) for i in range(len(ts))], energies)
    r.update({"job_id": job.job_id(), "backend": b.name, "shots": shots, "energies": list(energies),
              "wall_seconds": round(time.time() - t0, 1)})
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, f"iqm_neutrino_qg_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=1)
    return r


def main(argv):
    if "--prereg" in argv:
        with open(__file__, "rb") as fh:
            d = hashlib.sha256(fh.read()).hexdigest()
        from datetime import datetime, timezone

        os.makedirs(DATA, exist_ok=True)
        with open(PREREG, "w", encoding="utf-8") as fh:
            fh.write(f"{d}  examples/iqm_neutrino_qg.py  {datetime.now(timezone.utc).isoformat()}\n")
        print(d)
        return
    paid = "--probe" in argv or "--run" in argv
    if paid and "--yes-i-accept-qpu-cost" not in argv:
        sys.exit("This spends IQM credits. Re-run with --yes-i-accept-qpu-cost once the cost has been approved.")
    if "--probe" in argv:
        r = submit((2.6,), 100, "probe")
        print(json.dumps({k: r[k] for k in ("job_id", "raw_error", "filtered_error", "kept", "wall_seconds")}))
        print("Check the credits used for this job in the IQM Resonance dashboard before choosing --shots for --run.")
    elif "--run" in argv:
        if not os.path.exists(PREREG):
            sys.exit("Freeze the predictions first: python examples/iqm_neutrino_qg.py --prereg")
        shots = int(argv[argv.index("--shots") + 1]) if "--shots" in argv else sys.exit("--run needs --shots N")
        r = submit(ENERGIES, shots, "run")
        v = verdict(r)
        print(json.dumps({"raw": round(r["raw_error"], 4), "filtered": round(r["filtered_error"], 4),
                          "kept": round(r["kept"], 3), "survival min at": r["survival_min_E"]}))
        print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    else:
        rehearsal()


if __name__ == "__main__":
    main(sys.argv[1:])
