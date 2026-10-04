"""
The polar angle in radians, separated from the radius, with and without
qang (§102)

qang reads a qubit through qg_Z = r cos(theta): the polar bias mixes the
direction of the Bloch vector (theta, in radians) with its length r. The
usual angle readout, theta = arccos(qg_Z) from Z-basis shots, therefore
moves towards pi/2 whenever noise shortens the vector. With the radius of
§97 the two can be separated:

    theta = atan2(sqrt(qg_X^2 + qg_Y^2), qg_Z) = arccos(qg_Z / r)

(qang.formulation.direction_from_qg; from counts,
qang.statistics.direction_estimate, with the equatorial part from unbiased
squares). Depolarizing noise shrinks r and leaves this theta unchanged;
dephasing shrinks only qg_X, qg_Y and moves it towards the poles, while
arccos(qg_Z) is then exact; amplitude damping moves both.

Readouts, same total shots S = 3000 per angle:
  with qang     theta from the three qg values (1000 shots in each of the
                X, Y, Z bases);
  without qang  theta = arccos(qg_Z) from 3000 Z-basis shots.
Truth: the prepared angle.

Part 1, simulation (single qubit, phi = 0.3, 31 angles in [0.1, pi - 0.1]):
channels depolarizing p = 0.1, 0.2, 0.3 (q -> (1 - p) q), dephasing with
coherence factor 0.9, 0.8, 0.7 (qg_X, qg_Y scaled), amplitude damping
gamma = 0.1, 0.2, 0.3; exact values and 200 repetitions of the shots.
Part 2, device noise models: Ry(theta) on a qubit, then 8 pairs of CX with
a second qubit (each pair is the identity, separated by barriers so the
transpiler keeps them), measured in X, Y, Z; 11 angles in [0.15, pi - 0.15];
IBM fake brisbane, sherbrooke, torino (Aer) and the IonQ simulator with the
aria-1 and forte-1 noise models.

Pre-registered predictions (committed before the run; code checked on a
noiseless 2-angle grid only):
  A1  depolarizing, exact: the qang angle equals the prepared one to 1e-12
      at every angle; without qang the mean absolute error over the grid
      exceeds 0.05 rad for every p.
  A2  dephasing, exact: without qang the angle is exact (1e-12); with qang
      the mean absolute error exceeds 0.05 rad at coherence factor 0.7.
  A3  amplitude damping, exact: the mean absolute error over the grid is
      lower with qang than without, for every gamma.
  A4  shots, depolarizing p = 0.2: the RMSE averaged over the grid is lower
      with qang than without.
  A5  shots, no noise: the RMSE averaged over the grid is lower without
      qang (it has three times the Z shots), the cost of reading the radius.
  A6  device noise models: the mean absolute error over the 11 angles is
      lower with qang than without on each of the five backends.

Uses the installed library (pip install "qang>=0.6.7"). Part 2 needs qiskit,
qiskit-aer, qiskit-ibm-runtime (fake) or qiskit-ionq (IonQ; key from
IONQ_API_KEY or the git-ignored .ionq_key, never printed).
python examples/qg_direction_radians_qg.py --part sim
python examples/qg_direction_radians_qg.py --part device --mode fake
python examples/qg_direction_radians_qg.py --part device --mode ionq_sim --noise aria-1 --jobs jobs.json

Findings:

Part 1, simulation (31 angles; exact = infinite shots; RMSE over 200
repetitions of 3000 shots):

  channel              mean abs error (rad), exact     RMSE (rad)
                       with qang     without qang      with / without
  depolarizing 0.1     0             0.122
  depolarizing 0.2     0             0.213             0.037 / 0.216
  depolarizing 0.3     0             0.294
  dephasing 0.9        0.035         0
  dephasing 0.8        0.073         0
  dephasing 0.7        0.116         0
  amp. damping 0.1     0.072         0.150
  amp. damping 0.2     0.159         0.270
  amp. damping 0.3     0.270         0.382
  no noise             0             0                 0.028 / 0.018

Part 2, device noise models (11 angles, Ry(theta) + 16 CX in identity
pairs, mean abs error in rad):

  backend          with qang   without qang
  fake_brisbane    0.053       0.267
  fake_sherbrooke  0.029       0.207
  fake_torino      0.041       0.374
  IonQ aria-1      0.026       0.012
  IonQ forte-1     0.019       0.009

  * A1-A5 pass; A6 fails (passes on the three IBM backends, fails on both
    IonQ noise models).
  * A1: under depolarizing noise the angle from the three qg values is exact
    at every angle; arccos(qg_Z) errs by 0.12-0.29 rad on average and up to
    0.70 rad near the poles. With shots (p = 0.2) the qang angle has 5.8x
    lower RMSE.
  * A2: dephasing is where qang loses: arccos(qg_Z) is exact and the qang
    angle is pulled towards the poles by 0.03-0.12 rad.
  * A3: under amplitude damping neither is exact; the qang angle errs 29-52%
    less on average and its worst case is 2.3-4.5x smaller.
  * A5: with no noise the cost of reading the radius shows: arccos(qg_Z) has
    1.6x lower RMSE (three times the Z shots).
  * A6: on the IBM noise models the Bloch vector shrinks almost uniformly
    (the error of arccos(qg_Z) grows symmetrically towards both poles, up to
    0.8 rad), and the qang angle is 5-9x more accurate. On the IonQ noise
    models arccos(qg_Z) errs by only 0.01 rad, at the level of its shot noise,
    so the vector barely shrank and the extra Z shots win (2.1-2.2x). Whether
    IonQ's compiler removed the identity CX pairs (barriers are not sent to
    the API) was not checked.
  Verdict. Reading the angle in radians from the three qg values, separated
  from the radius, removes the depolarizing bias exactly and roughly halves
  the amplitude-damping bias; it pays for it with three bases and loses when
  the noise is dephasing or too weak to shorten the vector. A practical rule
  follows: if the measured radius is close to 1, use arccos(qg_Z) with all
  shots in Z; if it is clearly below 1 and the noise is not pure dephasing,
  use the radius-separated angle.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

from qang.formulation import direction_from_qg
from qang.statistics import direction_estimate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PHI = 0.3
GRID = np.linspace(0.1, math.pi - 0.1, 31)
DEVICE_GRID = np.linspace(0.15, math.pi - 0.15, 11)
CHANNELS = [("depolarizing", p) for p in (0.1, 0.2, 0.3)] + [("dephasing", c) for c in (0.9, 0.8, 0.7)] + \
           [("amplitude damping", g) for g in (0.1, 0.2, 0.3)]
SHOTS_BASIS, SHOTS_Z = 1000, 3000
REPS = 200
PAIRS = 8


def bloch(theta, phi=PHI):
    return np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])


def channel(q, kind, x):
    q = np.array(q, float)
    if kind == "depolarizing":
        return (1 - x) * q
    if kind == "dephasing":
        return np.array([x * q[0], x * q[1], q[2]])
    if kind == "amplitude damping":
        s = math.sqrt(1 - x)
        return np.array([s * q[0], s * q[1], x + (1 - x) * q[2]])
    if kind == "none":
        return q
    raise ValueError(kind)


def angles_exact(q):
    return direction_from_qg(*q)[0], math.acos(max(-1.0, min(1.0, q[2])))


def angles_shots(q, rng):
    p = (1 + q) / 2
    kx, ky, kz = rng.binomial(SHOTS_BASIS, p)
    with_q = direction_estimate((kx, SHOTS_BASIS), (ky, SHOTS_BASIS), (kz, SHOTS_BASIS))[0]
    kz3 = rng.binomial(SHOTS_Z, p[2])
    without = math.acos(2 * kz3 / SHOTS_Z - 1)
    return with_q, without


def part_sim(reps=REPS, grid=GRID):
    rng = np.random.default_rng(102)
    rows = []
    for kind, x in CHANNELS + [("none", 0.0)]:
        ex_w, ex_o, rm_w, rm_o = [], [], [], []
        for th in grid:
            q = channel(bloch(th), kind, x)
            a, b = angles_exact(q)
            ex_w.append(abs(a - th))
            ex_o.append(abs(b - th))
            if (kind, x) in (("depolarizing", 0.2), ("none", 0.0)):
                est = np.array([angles_shots(q, rng) for _ in range(reps)])
                rm_w.append(math.sqrt(np.mean((est[:, 0] - th) ** 2)))
                rm_o.append(math.sqrt(np.mean((est[:, 1] - th) ** 2)))
        rows.append({"channel": kind, "strength": x, "exact error with qang": float(np.mean(ex_w)),
                     "exact max error with qang": float(np.max(ex_w)),
                     "exact error without qang": float(np.mean(ex_o)),
                     "exact max error without qang": float(np.max(ex_o)),
                     "RMSE with qang": float(np.mean(rm_w)) if rm_w else None,
                     "RMSE without qang": float(np.mean(rm_o)) if rm_o else None})
    return rows


def verdict_sim(rows):
    g = {(r["channel"], r["strength"]): r for r in rows}
    dep = [r for r in rows if r["channel"] == "depolarizing"]
    amp = [r for r in rows if r["channel"] == "amplitude damping"]
    deph = [r for r in rows if r["channel"] == "dephasing"]
    return {
        "A1": all(r["exact max error with qang"] < 1e-12 and r["exact error without qang"] > 0.05 for r in dep),
        "A2": all(r["exact max error without qang"] < 1e-12 for r in deph) and g[("dephasing", 0.7)]["exact error with qang"] > 0.05,
        "A3": all(r["exact error with qang"] < r["exact error without qang"] for r in amp),
        "A4": g[("depolarizing", 0.2)]["RMSE with qang"] < g[("depolarizing", 0.2)]["RMSE without qang"],
        "A5": g[("none", 0.0)]["RMSE without qang"] < g[("none", 0.0)]["RMSE with qang"],
    }


# --------------------------------------------------------------------- #
def device_circuits(theta):
    from qiskit import QuantumCircuit

    out = []
    for basis in ("X", "Y", "Z", "Z3"):
        qc = QuantumCircuit(2, 1)
        qc.ry(theta, 0)
        for _ in range(PAIRS):
            qc.barrier()
            qc.cx(0, 1)
            qc.barrier()
            qc.cx(0, 1)
        qc.barrier()
        if basis == "X":
            qc.h(0)
        elif basis == "Y":
            qc.sdg(0)
            qc.h(0)
        qc.measure(0, 0)
        out.append(qc)
    return out


def k0(counts):
    tot = sum(counts.values())
    zero = sum(v for k, v in counts.items() if k.replace(" ", "")[-1] == "0")
    return zero, tot


def part_device(mode, noise, jobs, grid=DEVICE_GRID):
    import qnn_hardware_qg as H
    from qiskit import transpile

    names = ("fake_brisbane", "fake_sherbrooke", "fake_torino") if mode == "fake" else [None]
    results = []
    for name in names:
        backend = H.get_backend(mode, name)
        circs, shots = [], []
        for th in grid:
            c = device_circuits(th)
            circs += c
            shots += [SHOTS_BASIS, SHOTS_BASIS, SHOTS_BASIS, SHOTS_Z]
        if mode in ("local", "fake"):
            from qiskit_aer import AerSimulator

            sim = AerSimulator.from_backend(backend) if mode == "fake" else backend
            tc = transpile(circs, backend=sim, optimization_level=1, seed_transpiler=102)
            counts = [sim.run(c, shots=s, seed_simulator=102 + i).result().get_counts() for i, (c, s) in enumerate(zip(tc, shots))]
        else:
            tc = transpile(circs, backend=backend, optimization_level=1, seed_transpiler=102)
            counts = ionq_counts(backend, tc, shots, noise, jobs)
        err_w, err_o = [], []
        for i, th in enumerate(grid):
            cx, cy, cz, cz3 = (k0(counts[4 * i + j]) for j in range(4))
            tw = direction_estimate(cx, cy, cz)[0]
            to = math.acos(2 * cz3[0] / cz3[1] - 1)
            err_w.append(abs(tw - th))
            err_o.append(abs(to - th))
        two_q = float(np.mean([sum(1 for d in c.data if d.operation.num_qubits == 2 and d.operation.name != "barrier") for c in tc]))
        label = name if mode == "fake" else (f"ionq simulator {noise}" if mode == "ionq_sim" else "local")
        results.append({"backend": label, "error with qang": float(np.mean(err_w)), "error without qang": float(np.mean(err_o)),
                        "errors with qang": err_w, "errors without qang": err_o, "two-qubit gates": two_q})
    return results


def ionq_counts(backend, tc, shots, noise, jobs_path):
    import time

    ids = json.load(open(jobs_path)) if jobs_path and os.path.exists(jobs_path) else []
    for i in range(len(ids), len(tc)):
        job = backend.run(tc[i], shots=shots[i], noise_model=noise)
        ids.append(job.job_id())
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["sim", "device"], default="sim")
    ap.add_argument("--mode", choices=["local", "fake", "ionq_sim"], default="fake")
    ap.add_argument("--noise", default="aria-1")
    ap.add_argument("--jobs", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.part == "sim":
        rows = part_sim()
        for r in rows:
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
        v = verdict_sim(rows)
        print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
        res = {"rows": rows, "verdict": v}
    else:
        rs = part_device(args.mode, args.noise, args.jobs)
        for r in rs:
            r["A6"] = r["error with qang"] < r["error without qang"]
            print(r["backend"], f"error with qang {r['error with qang']:.4f}, without {r['error without qang']:.4f}, "
                  f"2q gates {r['two-qubit gates']:.0f}, A6 {'PASS' if r['A6'] else 'FAIL'}")
        res = rs
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
