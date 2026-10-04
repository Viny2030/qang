"""
Why the IonQ noise models did not shrink the Bloch vector in §102 (§103)

In §102 the angle read from the three qg values (with qang) beat
arccos(qg_Z) (without qang) by 5-9x on the IBM noise models, but lost on the
IonQ simulator with the aria-1 and forte-1 noise models, where arccos(qg_Z)
erred only 0.01 rad, at its shot-noise level: the Bloch vector barely shrank.
The noise block was 8 pairs of CX on the same two qubits, separated by
barriers. Barriers are not sent to the IonQ API, so the hypothesis is that
IonQ's compiler cancelled the pairs (QIS gate set, compiled and optimized
server-side). Circuits submitted in IonQ's native gate set (GPI, GPI2, MS)
are run as given, so the same block compiled locally to 16 MS gates cannot be
removed.

Configurations (IonQ cloud simulator, aria-1 and forte-1 noise models; the
§102 angles, readouts and shots: 11 angles, with qang 1000 shots per basis,
without qang 3000 Z shots):
  qis-block     the §102 circuit, QIS gate set (cx), rerun
  qis-none      Ry(theta) only, no CX block, QIS gate set
  native-block  the §102 circuit transpiled locally to GPI/GPI2/MS (16 MS)
                and submitted with the native gate set

Pre-registered predictions (committed before the run; circuits checked
locally, nothing submitted):
  N1  QIS: the mean error of arccos(qg_Z) with the CX block is within
      0.01 rad of the error without it, on both noise models (the block has
      no effect, as if removed).
  N2  native: the mean error of arccos(qg_Z) exceeds 0.05 rad on both noise
      models (16 MS gates do shrink the vector).
  N3  native: the mean error is lower with qang than without, on both noise
      models (the §102 result for IBM extends to IonQ once the block runs).

python examples/qg_direction_ionq_native_qg.py --noise aria-1 --jobs jobs_aria.json
(resumable; needs qiskit-ionq and an IonQ key in IONQ_API_KEY or the
git-ignored .ionq_key, never printed; the simulator is free)

Findings:

FINDINGS_PLACEHOLDER
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qg_direction_radians_qg as A  # noqa: E402
import qnn_hardware_qg as H  # noqa: E402
from qang.statistics import direction_estimate  # noqa: E402

CONFIGS = ("qis-block", "qis-none", "native-block")


def circuits(config, theta):
    saved = A.PAIRS
    try:
        A.PAIRS = 0 if config == "qis-none" else saved
        return A.device_circuits(theta)
    finally:
        A.PAIRS = saved


def build_all(provider):
    from qiskit import transpile

    qis = provider.get_backend("simulator")
    nat = provider.get_backend("simulator", gateset="native")
    jobs = []  # (config, angle index, basis index, backend, circuit, shots)
    for config in CONFIGS:
        backend = nat if config == "native-block" else qis
        for i, th in enumerate(A.DEVICE_GRID):
            cs = circuits(config, th)
            tc = transpile(cs, backend=backend, optimization_level=1, seed_transpiler=103)
            for j, c in enumerate(tc):
                jobs.append((config, i, j, backend, c, A.SHOTS_Z if j == 3 else A.SHOTS_BASIS))
    return jobs


def run(noise, jobs_path):
    provider = __import__("qiskit_ionq").IonQProvider(H.read_ionq_key())
    jobs = build_all(provider)
    ids = json.load(open(jobs_path)) if jobs_path and os.path.exists(jobs_path) else []
    for k in range(len(ids), len(jobs)):
        _, _, _, backend, c, s = jobs[k]
        ids.append(backend.run(c, shots=s, noise_model=noise).job_id())
        if jobs_path:
            json.dump(ids, open(jobs_path, "w"))
    counts = []
    for (config, i, j, backend, c, s), jid in zip(jobs, ids):
        for attempt in range(20):
            try:
                probs = backend.retrieve_job(jid).get_probabilities()
                counts.append({k: int(round(v * s)) for k, v in probs.items()})
                break
            except Exception:
                if attempt == 19:
                    raise
                time.sleep(15)
    out = {}
    for config in CONFIGS:
        ew, eo = [], []
        for i, th in enumerate(A.DEVICE_GRID):
            cc = [counts[k] for k, jb in enumerate(jobs) if jb[0] == config and jb[1] == i]
            cx, cy, cz, cz3 = (A.k0(x) for x in cc)
            ew.append(abs(direction_estimate(cx, cy, cz)[0] - th))
            eo.append(abs(math.acos(2 * cz3[0] / cz3[1] - 1) - th))
        two_q = [sum(1 for d in jb[4].data if d.operation.name in ("cx", "ms")) for jb in jobs if jb[0] == config]
        out[config] = {"error with qang": float(np.mean(ew)), "error without qang": float(np.mean(eo)),
                       "errors with qang": ew, "errors without qang": eo, "two-qubit gates": float(np.mean(two_q))}
    return out


def verdict(results):
    """results: {noise model: run() output}."""
    return {
        "N1": all(abs(r["qis-block"]["error without qang"] - r["qis-none"]["error without qang"]) <= 0.01 for r in results.values()),
        "N2": all(r["native-block"]["error without qang"] > 0.05 for r in results.values()),
        "N3": all(r["native-block"]["error with qang"] < r["native-block"]["error without qang"] for r in results.values()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", default="aria-1")
    ap.add_argument("--jobs", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    r = run(args.noise, args.jobs)
    for config, v in r.items():
        print(f"{args.noise} {config:13s} error with qang {v['error with qang']:.4f}, without {v['error without qang']:.4f}, "
              f"2q gates {v['two-qubit gates']:.0f}")
    if args.out:
        json.dump(r, open(args.out, "w"), indent=1)
    return r


if __name__ == "__main__":
    main()
