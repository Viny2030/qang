"""
H2 with the qg filter on IonQ Forte-1 hardware: submit, then fetch (§128)

The first real-device run of the repository. Same circuits as Experiment 3 of
examples/ionq_validation.py (§20, §21): H2 (Jordan-Wigner, 4 qubits) at the
exact VQE angle, one Z-basis circuit, four XY-basis circuits and two readout
calibration circuits, 7 circuits x 5000 shots. Forte-1 queues for days, so
submission and retrieval are separate steps; the job ids are stored in
examples/data/ionq_qpu_h2_jobs.json.

Cost (IonQ job-estimate API, read 2026-10-11): 25.79-26.66 USD per circuit,
about 181.50 USD for the 7 circuits, accepted by the author before
submission. Nothing is submitted without --yes-i-accept-qpu-cost.

Pre-registered predictions (committed before submission). The IonQ cloud
simulator with the forte-1 noise model gave, at 2000 shots (§20): witness
mean qg_Z +0.005, 97% of shots kept, energy error 33.1 mHa raw and 12.7 mHa
with readout mitigation and the qg filter (Hartree-Fock 20.3 mHa).
  H1  the witness mean qg_Z is within 0.03 of its ideal 0 (noise on trapped
      ions is close to unital, §10.4, §18).
  H2  at least 90% of the Z-basis shots are kept by the filter.
  H3  the qg filter lowers the energy error by at least 5 mHa against
      readout mitigation alone.
  H4  with the filter the energy error is below Hartree-Fock (20.3 mHa).

Usage (the API key is read from IONQ_API_KEY or .ionq_key; never printed):
    python examples/ionq_qpu_h2_qg.py submit --yes-i-accept-qpu-cost
    python examples/ionq_qpu_h2_qg.py status
    python examples/ionq_qpu_h2_qg.py fetch
    python examples/ionq_qpu_h2_qg.py rehearse      # IonQ simulator, forte-1 noise (free)

Findings:

FINDINGS_PLACEHOLDER
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

JOBS = os.path.join(HERE, "data", "ionq_qpu_h2_jobs.json")
DEVICE = "qpu.forte-1"
SHOTS = 5000


def circuits():
    import chemistry_qg_symmetry_witness as chem

    t = chem.optimal_angle()
    labels = ["ZZZZ"] + list(chem.XY_TERMS) + ["cal0", "cal1"]
    circs = [chem._measure_circuit(t, "ZZZZ", 0)] + [chem._measure_circuit(t, l, 0) for l in chem.XY_TERMS]
    circs += chem._calibration_circuits()
    return labels, circs


def backend(device=DEVICE):
    from ionq_validation import read_api_key
    from qiskit_ionq import IonQProvider

    return IonQProvider(token=read_api_key()).get_backend(device)


def analyse(count_dicts):
    """Energy errors (Ha) from the 7 count dictionaries, in the order of circuits()."""
    import chemistry_qg_symmetry_witness as chem
    from ionq_validation import _to_index

    P = []
    for c in count_dicts:
        p = np.zeros(16)
        for i, n in _to_index(c).items():
            p[i] += n
        P.append(p / p.sum())
    inv = chem._readout_inverse(P[-2], P[-1])
    p_z, p_xy = P[0], {l: P[1 + k] for k, l in enumerate(chem.XY_TERMS)}
    p_z_ro, p_xy_ro = inv @ p_z, {l: inv @ p for l, p in p_xy.items()}
    w = np.array([bin(i).count("1") for i in range(16)])
    return {
        "mean_qg_z": float(np.sum(p_z * (1 - 2 * w / 4))),
        "kept_fraction": float(np.sum(p_z * (w == 2))),
        "hf_error": chem.HF_ENERGY - chem.FCI_ENERGY,
        "raw": chem._energy(p_z, p_xy) - chem.FCI_ENERGY,
        "readout": chem._energy(p_z_ro, p_xy_ro) - chem.FCI_ENERGY,
        "readout_qg_filter": chem._energy(chem.qg_filter(p_z_ro), p_xy_ro) - chem.FCI_ENERGY,
    }


def verdict(r):
    return {
        "H1": abs(r["mean_qg_z"]) <= 0.03,
        "H2": r["kept_fraction"] >= 0.90,
        "H3": r["readout"] - r["readout_qg_filter"] >= 0.005,
        "H4": r["readout_qg_filter"] < r["hf_error"],
    }


def report(r):
    print(f"witness mean qg_Z {r['mean_qg_z']:+.4f} (ideal 0), kept {r['kept_fraction']:.3f}")
    print(f"energy error (mHa): HF {1e3 * r['hf_error']:.1f} | raw {1e3 * r['raw']:.1f} | "
          f"+readout {1e3 * r['readout']:.1f} | +readout+qg filter {1e3 * r['readout_qg_filter']:.1f}")
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in verdict(r).items()))


def submit(accept):
    from qiskit import transpile

    labels, circs = circuits()
    if not accept:
        print(f"NOT submitted: {len(circs)} circuits x {SHOTS} shots on {DEVICE} (about 181.50 USD). "
              "Rerun with --yes-i-accept-qpu-cost.")
        return None
    if os.path.exists(JOBS):
        raise SystemExit(f"{JOBS} exists: jobs were already submitted (use status / fetch).")
    b = backend()
    tc = transpile(circs, backend=b, optimization_level=1)
    ids = []
    for lab, qc in zip(labels, tc):
        job = b.run(qc, shots=SHOTS)
        ids.append({"label": lab, "job_id": job.job_id()})
        print(f"submitted {lab}: {job.job_id()}", flush=True)
    rec = {"device": DEVICE, "shots": SHOTS, "submitted": datetime.now(timezone.utc).isoformat(), "jobs": ids}
    with open(JOBS, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def _load():
    with open(JOBS, encoding="utf-8") as fh:
        return json.load(fh)


def status():
    rec, b = _load(), backend()
    for j in rec["jobs"]:
        print(f"{j['label']:6s} {j['job_id']}  {b.retrieve_job(j['job_id']).status()}")


def fetch():
    rec, b = _load(), backend()
    counts = []
    for j in rec["jobs"]:
        job = b.retrieve_job(j["job_id"])
        if not job.in_final_state():
            print(f"{j['label']} not finished yet: {job.status()}")
            return None
        counts.append(job.result().get_counts())
    rec["counts"] = counts
    r = analyse(counts)
    rec["result"] = r
    rec["verdict"] = verdict(r)
    with open(JOBS, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    report(r)
    return r


def rehearse(shots=2000, noise="forte-1"):
    from qiskit import transpile

    labels, circs = circuits()
    b = backend("simulator")
    tc = transpile(circs, backend=b, optimization_level=1)
    counts = [b.run(qc, shots=shots, noise_model=noise).result().get_counts() for qc in tc]
    r = analyse(counts)
    report(r)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["submit", "status", "fetch", "rehearse"])
    ap.add_argument("--yes-i-accept-qpu-cost", action="store_true")
    a = ap.parse_args(argv)
    if a.action == "submit":
        return submit(a.yes_i_accept_qpu_cost)
    return {"status": status, "fetch": fetch, "rehearse": rehearse}[a.action]()


if __name__ == "__main__":
    main()
