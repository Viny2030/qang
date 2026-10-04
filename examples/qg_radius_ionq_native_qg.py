"""
The §100 radius study on IonQ, rerun in the native gate set (§104)

§103 found that IonQ's compiler removes identities from circuits sent in the
standard (QIS) gate set. The §100 echo circuits (an excited basis state, 15
RBS gates and their inverse) are identities, and their IonQ rows were run in
the QIS gate set, so they may understate the noise. This study reruns the
§100 IonQ part with every circuit transpiled locally to GPI/GPI2/MS and
submitted in the native gate set, which IonQ runs as given. Same model,
inputs, shots (1000) and estimators as §100, with and without qang.

Pre-registered predictions (committed before any submission; circuits only
transpiled locally):
  Q1  echo: the raw false deficit of the excited qubit is at least as large
      in the native gate set as in §100 (0.62 aria-1, 0.71 forte-1), on both
      noise models.
  Q2  trained circuits: the deficit error is lower with qang than without, on
      both noise models (R1 of §100 holds).
  Q3  echo: the filter still removes less than half of the false deficit
      (R3 of §100 fails again), on both noise models.
  Q4  trained circuits: the deficit error with qang differs from §100 (0.074
      aria-1, 0.082 forte-1) by less than 0.03 (RBS circuits are not
      identities, so the compiler could not remove much).

python examples/qg_radius_ionq_native_qg.py --noise aria-1 --jobs jobs.json
(resumable; IonQ key from IONQ_API_KEY or the git-ignored .ionq_key, never
printed; the simulator is free)

Findings:

FINDINGS_PLACEHOLDER
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qg_radius_hardware_qg as R  # noqa: E402
import qnn_hardware_qg as H  # noqa: E402

QIS_100 = {"aria-1": {"echo raw": 0.618, "trained qang": 0.074}, "forte-1": {"echo raw": 0.709, "trained qang": 0.082}}


def native_circuits(backend):
    from qiskit import transpile

    model, Xte, _ = H.trained_model()
    H.check_circuits(model, Xte)
    circs = [H.build_circuit(model, x) for x in Xte] + [R.echo_circuit(model, j) for j in range(H.N)]
    return model, Xte, transpile(circs, backend=backend, optimization_level=1, seed_transpiler=104)


def run(noise, jobs_path, shots=H.SHOTS):
    from qiskit_ionq import IonQProvider

    backend = IonQProvider(H.read_ionq_key()).get_backend("simulator", gateset="native")
    model, Xte, tc = native_circuits(backend)
    ids = json.load(open(jobs_path)) if jobs_path and os.path.exists(jobs_path) else []
    for k in range(len(ids), len(tc)):
        ids.append(backend.run(tc[k], shots=shots, noise_model=noise).job_id())
        if jobs_path:
            json.dump(ids, open(jobs_path, "w"))
    counts = []
    for jid in ids:
        for attempt in range(20):
            try:
                probs = backend.retrieve_job(jid).get_probabilities()
                counts.append({k: int(round(v * shots)) for k, v in probs.items()})
                break
            except Exception:
                if attempt == 19:
                    raise
                time.sleep(15)
    probs = [H.counts_to_probs(c) for c in counts]
    r = R.evaluate(model, Xte, probs[: len(Xte)], probs[len(Xte):], shots)
    ms = [sum(1 for d in c.data if d.operation.name == "ms") for c in tc]
    r["MS gates: trained / echo"] = [float(np.mean(ms[: len(Xte)])), float(np.mean(ms[len(Xte):]))]
    r["noise model"] = noise
    return r


def verdict(results):
    """results: {noise model: run() output}."""
    out = {}
    m = lambda v: float(np.mean(v))  # noqa: E731
    out["Q1"] = all(m(r["echo: false deficit of the excited qubit, without qang"]) >= QIS_100[n]["echo raw"] for n, r in results.items())
    out["Q2"] = all(r["trained: deficit error with qang"] < r["trained: deficit error without qang"] for r in results.values())
    out["Q3"] = all(m(r["echo: false deficit of the excited qubit, with qang"])
                    >= 0.5 * m(r["echo: false deficit of the excited qubit, without qang"]) for r in results.values())
    out["Q4"] = all(abs(r["trained: deficit error with qang"] - QIS_100[n]["trained qang"]) < 0.03 for n, r in results.items())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", default="aria-1")
    ap.add_argument("--jobs", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    r = run(args.noise, args.jobs)
    print(json.dumps(r, indent=1))
    if args.out:
        json.dump(r, open(args.out, "w"), indent=1)
    return r


if __name__ == "__main__":
    main()
