"""
Calibrating the in-sector errors with echo circuits, with and without qang
(§105)

§100 and §104 found that the qg filter cuts the error of the radius deficit
3-4x on device noise models, but leaves the errors that move an excitation
inside the weight-1 sector: 10-21% of the kept shots of a product state land
on the wrong qubit. Those errors can be measured. An echo circuit (excited
basis state |e_j>, the 15 trained RBS gates and their inverse) should return
|e_j>; its filtered outcome distribution over the five excitation positions
is column j of a 5x5 transfer matrix M (in-sector "readout" matrix). Inverting
it on the filtered distribution of a trained circuit mitigates the in-sector
error, as readout mitigation does for bit flips. The echo has about twice
the gates of the forward circuit, so the matrix square root M^(1/2) is the
natural model of the forward circuit's in-sector error; M itself is the
over-correcting alternative.

Readouts of the weight-1 excitation probabilities p_q (then
qg_Z = 1 - 2 p_q and the deficit 1 - qg_Z^2):
  without qang   all shots
  qang           filtered shots
  qang + echo    filtered shots, then M^(1/2) inverted (non-negative least
                 squares on the simplex)
  qang + echo full   the same with M
Circuits: the §100 model (5 qubits, weight 1), 30 iris test inputs at 1000
shots; calibration: the 5 echo circuits at 4000 shots each. Backends: IBM
fake brisbane, sherbrooke, torino (Aer) and the IonQ simulator with the
aria-1 and forte-1 noise models in the native gate set (§103).

Pre-registered predictions (committed before any noisy run; code checked in
noiseless --mode local only):
  C1  the mean absolute error of the deficit on the trained circuits is lower
      with qang + echo than with qang alone, on each of the five backends.
  C2  qang + echo full has a larger deficit error than qang + echo, averaged
      over the five backends (over-correction).
  C3  the mean absolute error of qg_Z is lower with qang + echo than with qang
      alone, on each of the five backends.
  C4  qang + echo cuts the qang deficit error by at least a third, averaged
      over the five backends.

python examples/qg_sector_echo_mitigation_qg.py --mode fake
python examples/qg_sector_echo_mitigation_qg.py --mode ionq_sim --noise aria-1 --jobs jobs.json

Findings:

30 trained circuits (1000 shots) and 5 echo calibrations (4000 shots) per
backend; mean absolute error of the deficit 1 - qg_Z^2:

  backend           without   qang    qang + echo   qang + echo   echo diagonal
                    qang                (M^1/2)       full (M)
  fake_brisbane     0.264     0.082   0.043 (-47%)   0.053         0.77-0.84
  fake_sherbrooke   0.231     0.059   0.032 (-46%)   0.038         0.88-0.89
  fake_torino       0.197     0.050   0.031 (-38%)   0.034         0.90-0.92
  IonQ aria-1       0.301     0.090   0.049 (-46%)   0.050         0.80-0.82
  IonQ forte-1      0.285     0.089   0.047 (-47%)   0.039         0.82-0.83

  * C1-C4 pass.
  * C1, C4: the echo calibration cuts the error of the filtered deficit by
    38-47% (45% on average) on every backend; with the filter it is 6-7x
    below the raw readout. qg_Z itself (C3): 0.024-0.037 against 0.039-0.072
    with the filter alone and 0.14-0.22 without qang.
  * C2 passes on average (0.040 against 0.043), but not everywhere: on forte-1
    the full matrix M is better than its square root (0.039 against 0.047),
    so the forward circuit's in-sector error there is closer to the echo's
    than to half of it.
  * The echo diagonal says how much of the sector the filter cannot clean:
    8-23% of the kept shots of a product state land on another qubit.
  Verdict. In-sector errors are not invisible after all: five echo circuits
  measure them, and inverting their transfer matrix on the filtered
  distribution removes nearly half of what the filter leaves. Filter plus
  echo calibration is 6-7x more accurate than the raw readout on all five
  device noise models.
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

N = H.N
SHOTS, SHOTS_CAL = 1000, 4000
ONE = [1 << (N - 1 - q) for q in range(N)]  # basis index of an excitation on qang qubit q


def sector_probs(p):
    """Filtered weight-1 distribution over the excitation positions (length N)."""
    v = np.array([p[i] for i in ONE], float)
    return v / v.sum() if v.sum() > 0 else np.full(N, 1.0 / N)


def transfer_matrix(echo_probs):
    """Column j = filtered distribution of the echo prepared with qubit j excited."""
    return np.stack([sector_probs(p) for p in echo_probs], axis=1)


def matrix_sqrt(M):
    from scipy.linalg import sqrtm

    S = np.real(sqrtm(M))
    S = np.clip(S, 0.0, None)
    return S / S.sum(axis=0, keepdims=True)


def unmix(M, f):
    """Non-negative least squares for M x = f, then normalized (x on the simplex)."""
    from scipy.optimize import nnls

    x, _ = nnls(M, f)
    return x / x.sum() if x.sum() > 0 else np.full(N, 1.0 / N)


def readouts(p, M_half, M_full):
    """Excitation probabilities of the five qubits for the four readouts."""
    raw = np.array([sum(p[i] for i in range(2**N) if (i >> (N - 1 - q)) & 1) for q in range(N)])
    f = sector_probs(p)
    return {"without qang": raw, "qang": f, "qang + echo": unmix(M_half, f), "qang + echo full": unmix(M_full, f)}


def evaluate(model, Xte, probs_trained, probs_echo):
    truth_z = model.probs(model.params_[: model.n_theta], model.encode(Xte)) @ model.zsign
    M = transfer_matrix(probs_echo)
    Mh = matrix_sqrt(M)
    out = {k: {"deficit": [], "qg_Z": []} for k in ("without qang", "qang", "qang + echo", "qang + echo full")}
    for p, tz in zip(probs_trained, truth_z):
        for k, pq in readouts(p, Mh, M).items():
            z = 1 - 2 * pq
            out[k]["deficit"].append(np.mean(np.abs((1 - z**2) - (1 - tz**2))))
            out[k]["qg_Z"].append(np.mean(np.abs(z - tz)))
    r = {f"deficit error, {k}": float(np.mean(v["deficit"])) for k, v in out.items()}
    r.update({f"qg_Z error, {k}": float(np.mean(v["qg_Z"])) for k, v in out.items()})
    r["echo diagonal (fraction on the right qubit)"] = [float(x) for x in np.diag(M)]
    return r


def get_counts(mode, name, noise, circuits, shots, jobs_path):
    from qiskit import transpile

    if mode in ("local", "fake"):
        from qiskit_aer import AerSimulator

        backend = H.get_backend(mode, name)
        sim = AerSimulator.from_backend(backend) if mode == "fake" else backend
        tc = transpile(circuits, backend=sim, optimization_level=1, seed_transpiler=105)
        return [sim.run(c, shots=s, seed_simulator=105 + i).result().get_counts() for i, (c, s) in enumerate(zip(tc, shots))]
    from qiskit_ionq import IonQProvider

    backend = IonQProvider(H.read_ionq_key()).get_backend("simulator", gateset="native")
    tc = transpile(circuits, backend=backend, optimization_level=1, seed_transpiler=105)
    ids = json.load(open(jobs_path)) if jobs_path and os.path.exists(jobs_path) else []
    for k in range(len(ids), len(tc)):
        ids.append(backend.run(tc[k], shots=shots[k], noise_model=noise).job_id())
        if jobs_path:
            json.dump(ids, open(jobs_path, "w"))
    out = []
    for jid, s in zip(ids, shots):
        for attempt in range(20):
            try:
                probs = backend.retrieve_job(jid).get_probabilities()
                out.append({k: int(round(v * s)) for k, v in probs.items()})
                break
            except Exception:
                if attempt == 19:
                    raise
                time.sleep(15)
    return out


def run_backend(mode, name, noise, jobs_path):
    model, Xte, _ = H.trained_model()
    H.check_circuits(model, Xte)
    circuits = [H.build_circuit(model, x) for x in Xte] + [R.echo_circuit(model, j) for j in range(N)]
    shots = [SHOTS] * len(Xte) + [SHOTS_CAL] * N
    counts = get_counts(mode, name, noise, circuits, shots, jobs_path)
    probs = [H.counts_to_probs(c) for c in counts]
    r = evaluate(model, Xte, probs[: len(Xte)], probs[len(Xte):])
    r["backend"] = name if mode == "fake" else (f"ionq simulator {noise} (native)" if mode == "ionq_sim" else mode)
    return r


def verdict(results):
    d = lambda r, k: r[f"deficit error, {k}"]  # noqa: E731
    return {
        "C1": all(d(r, "qang + echo") < d(r, "qang") for r in results),
        "C2": np.mean([d(r, "qang + echo full") for r in results]) > np.mean([d(r, "qang + echo") for r in results]),
        "C3": all(r["qg_Z error, qang + echo"] < r["qg_Z error, qang"] for r in results),
        "C4": np.mean([1 - d(r, "qang + echo") / d(r, "qang") for r in results]) >= 1 / 3,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "fake", "ionq_sim"], default="fake")
    ap.add_argument("--noise", default="aria-1")
    ap.add_argument("--jobs", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    names = ("fake_brisbane", "fake_sherbrooke", "fake_torino") if args.mode == "fake" else [None]
    rs = [run_backend(args.mode, n, args.noise, args.jobs) for n in names]
    for r in rs:
        print(json.dumps(r, indent=1), flush=True)
    if args.out:
        json.dump(rs, open(args.out, "w"), indent=1)
    return rs


if __name__ == "__main__":
    main()
