"""
IonQ noisy cloud simulator: 1D Hubbard dynamics (§29) and constrained
QAOA (§30) on trapped-ion noise models (aria-1, forte-1).

Why these two. On fake_brisbane (heavy-hex) both experiments were
dominated by routing: 56 ECR per Hubbard Trotter step, 134-497 ECR per
QAOA circuit, and routing created spin leak (§29) and drowned the
XY-mixer QAOA (§30). IonQ devices are all-to-all, so these runs test
the two predictions that follow:

  * Hubbard: with no routing, the spin leak should be small and the
    second (spin) filter should add little over the N filter, as for
    H2 on IonQ (§20);
  * QAOA: the qg filter should keep its advantage for the XY mixer
    (it doubled P(opt) on the generic all-to-all model of §30).

Modes: "local" (Aer with the generic all-to-all model of §29, for tests
and dry runs) and "ionq_sim" (IonQ's cloud simulator with
noise_model = aria-1 or forte-1; free, needs a key in IONQ_API_KEY or
the git-ignored .ionq_key). Nothing here submits to a QPU.

Results are appended to examples/data/ionq_sim_results.json, one entry
per (experiment, noise model, seed), so an interrupted run resumes.
Shots: 2000 per circuit (the IonQ noisy simulator rejects more).

Findings (recorded in examples/data/ionq_sim_results.json; Hubbard:
3 repetitions per noise model; QAOA: 5 graphs, one repetition):

  Hubbard, |error| of double occupancy / charge imbalance, time average
  over 1, 2, 4, 6 Trotter steps:
                 readout-mitigated   + N filter      + spin filters
    aria-1       0.022 / 0.029       0.016 / 0.021   0.013 / 0.020
    forte-1      0.029 / 0.042       0.021 / 0.023   0.017 / 0.019
    spin leak at 1, 2, 4, 6 steps: aria 0.1%, 1.1%, 3.5%, 8.3%;
    forte 0.1%, 1.8%, 6.8%, 11.5%. Both register witnesses stay within
    ±0.012 of 0 on average (single runs ±0.025: unital noise); the kept
    fraction falls to 0.53 / 0.45.

  * Without routing the spin leak is small at shallow depth (0.1% at one
    step, vs 8% on fake_brisbane), as predicted, but it grows with depth
    under the depolarizing-dominated trapped-ion noise, and at 4-6 steps
    the spin filters beat the N filter (double occupancy 0.032 -> 0.023
    on aria-1 and 0.044 -> 0.033 on forte-1 at 6 steps).
  * The filters cut the time-averaged error by 30-55% on both models, in
    line with the generic all-to-all model of §29.

  QAOA exactly K, P(opt) noisy -> noisy + qg filter (random feasible
  guess 0.096):
                 standard      qg budget     qg warm       XY mixer
    aria-1 p=1   0.047->0.102  0.062->0.114  0.163->0.311  0.155->0.276
    aria-1 p=2   0.080->0.178  0.066->0.143  0.114->0.263  0.174->0.363
    forte-1 p=1  0.046->0.106  0.053->0.098  0.146->0.299  0.120->0.256
    forte-1 p=2  0.074->0.168  0.062->0.146  0.093->0.235  0.111->0.293

  * The filter doubles the XY-mixer P(opt) on both models (1.8-2.6x),
    reproducing the generic all-to-all result of §30 (0.150 -> 0.278,
    0.177 -> 0.366). On fake_brisbane the same circuits collapsed to the
    random-guess level: here connectivity is what keeps the signal.
  * Penalty QAOA again ends at the random-feasible level after filtering;
    greedy (98.5% of optimum) still beats every variant (ratio <= 0.87).

These are vendor noise models run on IonQ's simulator, not hardware.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hubbard_trotter_qg_filters as HB  # noqa: E402
import qaoa_k_constraint_qg as QA  # noqa: E402
from ionq_validation import get_backend  # noqa: E402

RESULTS = os.path.join(HERE, "data", "ionq_sim_results.json")
ANGLES = os.path.join(HERE, "data", "qaoa_k_angles.json")
SHOTS = 2000
HUBBARD_STEPS = (1, 2, 4, 6)
QAOA_GRAPHS = range(5)


def _probs(counts, n):
    p = np.zeros(2**n)
    total = 0
    for key, c in counts.items():
        k = key.replace(" ", "")
        idx = int(k, 16) if k.startswith("0x") else int(k, 2)
        p[idx] += c
        total += c
    return p / total


def load_results(path=RESULTS):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_results(res, path=RESULTS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)


PENDING = os.path.join(HERE, "data", "ionq_sim_pending.json")


class Pending(Exception):
    """Jobs submitted but not finished; rerun the same command later."""


def _run(backend, circuits, noise, mode, key=None, wait_s=120):
    if mode == "local":
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        sim = AerSimulator(noise_model=HB.all_to_all_noise_model())
        tc = transpile(circuits, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=1)
        res = sim.run(tc, shots=SHOTS, seed_simulator=11).result()
        return [res.get_counts(i) for i in range(len(tc))]
    # IonQ: submit every circuit at once, remember the job ids, collect later
    import time

    from qiskit import transpile

    pending = load_results(PENDING)
    if key not in pending:
        tc = transpile(circuits, backend=backend, optimization_level=1)
        pending[key] = [backend.run(qc, shots=SHOTS, noise_model=noise).job_id() for qc in tc]
        save_results(pending, PENDING)
    jobs = [backend.retrieve_job(j) for j in pending[key]]
    t0 = time.time()
    while True:
        states = [j.status().name for j in jobs]
        if all(s == "DONE" for s in states):
            break
        if any(s in ("ERROR", "CANCELLED") for s in states):
            raise RuntimeError(f"IonQ job failed: {states}")
        if time.time() - t0 > wait_s:
            raise Pending(f"{states.count('DONE')}/{len(jobs)} jobs done; rerun the same command to collect")
        time.sleep(5)
    counts = [j.result().get_counts() for j in jobs]
    pending.pop(key)
    save_results(pending, PENDING)
    return counts


# --------------------------------------------------------------------- #
# Hubbard
# --------------------------------------------------------------------- #
def hubbard(backend, noise, mode, key=None):
    circuits = [HB.trotter_circuit(s) for s in HUBBARD_STEPS] + HB._calibration()
    counts = _run(backend, circuits, noise, mode, key)
    P = [_probs(c, HB.N_QUBITS) for c in counts]
    inv = HB.readout_inverse(P[-2], P[-1])
    rows = {}
    for j, steps in enumerate(HUBBARD_STEPS):
        ideal = HB.ideal_probabilities(steps)
        pm = inv @ P[j]
        qg_up, qg_down, kept_n, kept_s = HB.spin_witnesses(P[j])
        row = {"qg_up": qg_up, "qg_down": qg_down, "kept_n": kept_n, "kept_spin": kept_s,
               "spin_leak": 1 - kept_s / kept_n}
        for name, obs in HB.OBSERVABLES.items():
            row[f"{name}_ideal"] = obs(ideal)
            for m, pp in (("raw", pm), ("n_filter", HB.n_filter(pm)), ("spin_filter", HB.spin_filter(pm))):
                row[f"{name}_{m}"] = obs(pp)
        rows[str(steps)] = row
    return rows


# --------------------------------------------------------------------- #
# QAOA exactly K
# --------------------------------------------------------------------- #
def qaoa(backend, noise, mode, p=1, key=None):
    with open(ANGLES, encoding="utf-8") as fh:
        angles = json.load(fh)
    circuits, keys = [], []
    for g in QAOA_GRAPHS:
        edges = QA.random_graph(g)
        for v in QA.VARIANTS:
            circuits.append(QA.qaoa_circuit(edges, v, np.array(angles[f"p{p}_g{g}_{v}"])))
            keys.append((g, v))
    counts = _run(backend, circuits, noise, mode, key)
    rows = {}
    for (g, v), c in zip(keys, counts):
        edges = QA.random_graph(g)
        pn = _probs(c, QA.N_NODES)
        pi = QA.ideal_distribution(edges, v, np.array(angles[f"p{p}_g{g}_{v}"]))
        rows[f"g{g}_{v}"] = {"ideal": QA.sample_metrics(pi, edges, False), "noisy": QA.sample_metrics(pn, edges, False),
                             "noisy_f": QA.sample_metrics(pn, edges, True),
                             "random_feasible": QA.random_feasible_baseline(edges)}
    return rows


def summarize_qaoa(rows):
    out = {}
    for v in QA.VARIANTS:
        rs = [r for k, r in rows.items() if k.endswith("_" + v)]
        out[v] = {k: float(np.mean([r[k]["p_opt"] for r in rs])) for k in ("ideal", "noisy", "noisy_f")}
        out[v]["kept"] = float(np.mean([r["noisy"]["kept"] for r in rs]))
        out[v]["ratio_f"] = float(np.mean([r["noisy_f"]["ratio"] for r in rs]))
    out["random_feasible"] = float(np.mean([r["random_feasible"]["p_opt"] for k, r in rows.items() if k.endswith("_xy")]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("local", "ionq_sim"), default="local")
    ap.add_argument("--noise", choices=("aria-1", "forte-1"), default="aria-1")
    ap.add_argument("--only", choices=("hubbard", "qaoa1", "qaoa2"), required=True)
    ap.add_argument("--tag", default="run0", help="repetition label (IonQ simulator seeds are not controllable)")
    args = ap.parse_args(argv)
    key = f"{args.mode}|{args.noise if args.mode == 'ionq_sim' else 'generic'}|{args.only}|{args.tag}"
    res = load_results()
    if key in res:
        print("already done:", key)
        return res[key]
    backend = get_backend(args.mode) if args.mode == "ionq_sim" else None
    noise = args.noise if args.mode == "ionq_sim" else None
    try:
        if args.only == "hubbard":
            out = hubbard(backend, noise, args.mode, key)
        else:
            out = qaoa(backend, noise, args.mode, p=int(args.only[-1]), key=key)
    except Pending as exc:
        print("pending:", key, "-", exc)
        return None
    res[key] = out
    save_results(res)
    print("saved", key)
    return out


if __name__ == "__main__":
    main()
