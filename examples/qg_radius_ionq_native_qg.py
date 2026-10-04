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

IonQ simulator, native gate set (38 MS gates per trained circuit, 60 per
echo), 1000 shots, against the §100 QIS-gate-set rows:

  noise model   deficit error, trained       kept    false deficit, echo     filter
                with / without qang                  with / without qang     removes
  aria-1        0.091 / 0.298   (3.3x)       0.66    0.61 / 0.74             17%
     §100 QIS   0.074 / 0.245   (3.3x)       0.74    0.48 / 0.62             22%
  forte-1       0.089 / 0.289   (3.2x)       0.69    0.59 / 0.71             17%
     §100 QIS   0.082 / 0.272   (3.3x)       0.70    0.54 / 0.71             23%

  * Q1-Q4 pass on both noise models.
  * Q1: the raw false deficit of the echo circuits is 0.74 natively against
    0.62 in §100 on aria-1 (the QIS run understated the noise) and the same
    (0.71) on forte-1.
  * Q2, Q4: on the trained circuits the filter keeps its 3.2-3.3x advantage,
    and the error with qang moves by less than 0.02 (RBS circuits are not
    identities).
  * Q3: the filter removes 17% of the false deficit, less than in §100.
  Verdict. The §100 conclusions hold in IonQ's native gate set: the filtered
  radius is 3x more accurate where the deficit is large, and it is not a
  test of "no entanglement". The §100 caveat is resolved: the QIS echo rows
  understated the noise on aria-1 by 0.12 and not at all on forte-1.
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
