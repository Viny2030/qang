"""
The milli-qang on real hardware: does reasoning in m-qg predict what the chip reads? (§168, IQM Resonance, Garnet)

§167 made the milli-qang the working scale of the book, but its content there is identities. This experiment asks
whether the m-qg toolkit predicts a real device: every prediction below is computed from part of the same job and
checked on the rest, in m-qg, before seeing it. Pre-registered as §165 was: this file is frozen (sha256 in
examples/data/iqm_milliqang_qg.prereg) and committed before any job is submitted.

Circuits, all on one physical qubit (QUBIT), compiled at optimization_level 0 with barriers so no gate is merged:
  poles      |0> (nothing) and |1> (one X): the readout errors e0 = P(1|0) and e1 = P(0|1)
  sweep      R_y(arccos(m/1000)) for m in {+707, +500, -500, -707} m-qg
  equator    R_y(pi/2), 10 identical copies
  chains     k X gates, k in {3, 5, 7, 9, 11, 15, 21, 27, 33, 41} (odd: the ideal reading is -1000 m-qg)
Each circuit gets SHOTS = 4000 shots. 26 circuits in one job.

Pre-registered predictions (for --run; the probe is a cost measurement, not a test):
  M1  readout in m-qg is affine. From the two poles alone, m_pred = (1 - e0 - e1) m + 1000 (e1 - e0) predicts the
      four sweep points and the pooled equator; at least 4 of these 5 lie within 3 sigma (binomial sigma of the
      point plus the propagated sigma of e0, e1).
  M2  shot noise is the whole story: the standard deviation of the 10 equator readings, divided by the binomial
      sigma_m = 2000 sqrt(p (1 - p) / SHOTS) at the pooled p, lies in [0.55, 1.45] (the 95% band for 9 degrees of
      freedom).
  M3  the gate error budget grows as the m-qg model says. With the readout undone (from the poles), the shortfall
      s(k) = 1000 + m_corr(k) of the X chains is fitted as a (k - 1) + b (k^2 - 1) on k = 3..9 and predicts
      k = 11, 15, 21, 27, 33, 41; at least 4 of these 6 lie within 3 sigma.
  M4  the accumulated gate error is visible in m-qg: s(41) exceeds 2 sigma.

Free rehearsal, 2026-10-23 (IQM's fake backends, 4000 shots, seed_simulator 168; a first version with 2000 shots
and chains up to k = 21 gave M4 FAIL on IQMFakeAphrodite, s(21) = 20 +- 12 m-qg, so the chains were lengthened and
the shots doubled before freezing):
  IQMFakeApollo     e0 0.056, e1 0.060; M1 z = 0.97 -0.66 -0.91 2.56 0.06; M2 ratio 0.83; M3 |z| <= 0.50;
                    s(41) = 66 +- 10 m-qg; M1-M4 pass
  IQMFakeAphrodite  e0 0.052, e1 0.053; M1 |z| <= 1.05; M2 ratio 1.40; M3 |z| <= 1.18; s(41) = 93 +- 10 m-qg;
                    M1-M4 pass
M2 is a 95% band, so it fails by chance one run in twenty even when shot noise is the whole story.

Modes (the token is read from the environment variable IQM_TOKEN, never from a file in the repository; run in the
environment that has iqm-client[qiskit], e.g. .venv-iqm):
  python examples/iqm_milliqang_qg.py                       free rehearsal on IQM's fake backends
  python examples/iqm_milliqang_qg.py --probe --yes-i-accept-qpu-cost      1 circuit, 100 shots, cost only
  python examples/iqm_milliqang_qg.py --prereg              freeze this file (sha256) before --run
  python examples/iqm_milliqang_qg.py --run --yes-i-accept-qpu-cost        the 26 circuits
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

from qang import probability as PB  # noqa: E402

SERVER = "https://resonance.iqm.tech/"
QC_NAME = "garnet"
QUBIT = 0
SHOTS = 4000
SWEEP = (707.0, 500.0, -500.0, -707.0)
EQUATOR_COPIES = 10
FIT_K = (3, 5, 7, 9)
TEST_K = (11, 15, 21, 27, 33, 41)
CHAINS = FIT_K + TEST_K
DATA = os.path.join(HERE, "data")
PREREG = os.path.join(DATA, "iqm_milliqang_qg.prereg")


def circuits():
    """(label, circuit) pairs, in submission order."""
    from qiskit import QuantumCircuit

    out = []

    def make(label, build):
        qc = QuantumCircuit(1, 1, name=label)
        build(qc)
        qc.barrier()
        qc.measure(0, 0)
        out.append((label, qc))

    make("pole0", lambda qc: None)
    make("pole1", lambda qc: qc.x(0))
    for m in SWEEP:
        make(f"sweep{m:+.0f}", lambda qc, m=m: qc.ry(PB.theta_from_milliqang(m), 0))
    for i in range(EQUATOR_COPIES):
        make(f"equator{i}", lambda qc: qc.ry(math.pi / 2, 0))
    for k in CHAINS:
        def chain(qc, k=k):
            for _ in range(k):
                qc.x(0)
                qc.barrier()
        make(f"chain{k}", chain)
    return out


def _p0(counts):
    n0 = counts.get("0", 0)
    n = sum(counts.values())
    return n0 / n, n


def analyse(counts_by_label):
    p = {lab: _p0(c) for lab, c in counts_by_label.items()}
    (p00, n0), (p01, n1) = p["pole0"], p["pole1"]
    e0, e1 = 1.0 - p00, p01
    se0 = math.sqrt(max(e0 * (1 - e0), 1.0 / n0) / n0)
    se1 = math.sqrt(max(e1 * (1 - e1), 1.0 / n1) / n1)
    contrast = 1.0 - e0 - e1

    def pred(mt):
        mp = contrast * mt + 1000.0 * (e1 - e0)
        sp = math.hypot((mt + 1000.0) * se0, (1000.0 - mt) * se1)
        return mp, sp

    # M1: sweep points and the pooled equator
    m1 = []
    for mt in SWEEP:
        pm, n = p[f"sweep{mt:+.0f}"]
        mp, sp = pred(mt)
        sig = math.hypot(2000.0 * math.sqrt(max(pm * (1 - pm), 1.0 / n) / n), sp)
        m1.append({"target": mt, "measured": PB.milliqang_from_p0(pm), "predicted": mp, "sigma": sig})
    eq = [p[f"equator{i}"] for i in range(EQUATOR_COPIES)]
    neq = sum(n for _, n in eq)
    peq = sum(pp * n for pp, n in eq) / neq
    mp, sp = pred(0.0)
    m1.append({"target": 0.0, "measured": PB.milliqang_from_p0(peq), "predicted": mp,
               "sigma": math.hypot(2000.0 * math.sqrt(peq * (1 - peq) / neq), sp)})
    for r in m1:
        r["z"] = (r["measured"] - r["predicted"]) / r["sigma"]
    m1_pass = sum(abs(r["z"]) <= 3 for r in m1) >= 4

    # M2: dispersion of the equator copies against the binomial sigma
    m_eq = np.array([PB.milliqang_from_p0(pp) for pp, _ in eq])
    sig_bin = 2000.0 * math.sqrt(peq * (1 - peq) / (neq / EQUATOR_COPIES))
    ratio = float(m_eq.std(ddof=1) / sig_bin)
    m2_pass = 0.55 <= ratio <= 1.45

    # M3, M4: readout-corrected shortfall of the X chains
    def shortfall(label):
        pm, n = p[label]
        pt = (pm - e1) / contrast
        s = 1000.0 + (2000.0 * pt - 1000.0)
        ss = 2000.0 * math.sqrt(max(pm * (1 - pm), 1.0 / n) / n) / contrast
        return s, ss

    s = {k: shortfall(f"chain{k}") for k in CHAINS}
    A = np.array([[k - 1.0, k * k - 1.0] for k in FIT_K])
    w = np.array([1.0 / s[k][1] for k in FIT_K])
    y = np.array([s[k][0] for k in FIT_K])
    coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    cov = np.linalg.inv((A * w[:, None]).T @ (A * w[:, None]))
    m3 = []
    for k in TEST_K:
        x = np.array([k - 1.0, k * k - 1.0])
        sf = float(math.sqrt(x @ cov @ x))
        sig = math.hypot(s[k][1], sf)
        m3.append({"k": k, "measured": s[k][0], "predicted": float(x @ coef), "sigma": sig,
                   "z": (s[k][0] - float(x @ coef)) / sig})
    m3_pass = sum(abs(r["z"]) <= 3 for r in m3) >= 4
    m4_pass = s[TEST_K[-1]][0] > 2 * s[TEST_K[-1]][1]

    return {"e0": e0, "e1": e1, "contrast": contrast, "M1": m1, "equator_m": m_eq.tolist(),
            "M2_ratio": ratio, "chains": {k: {"s": v[0], "sigma": v[1]} for k, v in s.items()},
            "fit_a_b": coef.tolist(), "M3": m3,
            "verdict": {"M1": m1_pass, "M2": m2_pass, "M3": m3_pass, "M4": m4_pass}}


def _transpile(pairs, backend):
    from qiskit import transpile

    return [transpile(qc, backend, initial_layout=[QUBIT], optimization_level=0) for _, qc in pairs]


def _summary(r):
    return {"e0": round(r["e0"], 4), "e1": round(r["e1"], 4),
            "M1 z": [round(x["z"], 2) for x in r["M1"]], "M2 ratio": round(r["M2_ratio"], 2),
            "M3 z": [round(x["z"], 2) for x in r["M3"]],
            "s(41) m-qg": round(r["chains"][TEST_K[-1]]["s"], 1), "s(41) sigma": round(r["chains"][TEST_K[-1]]["sigma"], 1),
            "verdict": r["verdict"]}


def rehearsal(shots=SHOTS):
    from iqm.qiskit_iqm import IQMFakeAphrodite, IQMFakeApollo

    pairs = circuits()
    for B in (IQMFakeApollo, IQMFakeAphrodite):
        b = B()
        ts = _transpile(pairs, b)
        res = b.run(ts, shots=shots, seed_simulator=168).result()
        r = analyse({lab: res.get_counts(i) for i, (lab, _) in enumerate(pairs)})
        out = {"backend": b.name, "circuits": len(ts), "r gates in chain41": ts[-1].count_ops().get("r")}
        out.update(_summary(r))
        print(json.dumps(out), flush=True)


def real_backend():
    if not os.environ.get("IQM_TOKEN"):
        sys.exit("IQM_TOKEN is not set: set it in this terminal only, never write it to a file in the repository.")
    from iqm.qiskit_iqm import IQMProvider

    return IQMProvider(SERVER, quantum_computer=QC_NAME).get_backend()


def submit(pairs, shots, tag):
    b = real_backend()
    ts = _transpile(pairs, b)
    print(json.dumps({"backend": b.name, "circuits": len(ts), "shots": shots, "qubit": QUBIT}), flush=True)
    t0 = time.time()
    job = b.run(ts, shots=shots)
    print(json.dumps({"job_id": job.job_id()}), flush=True)
    res = job.result()
    counts = {lab: res.get_counts(i) for i, (lab, _) in enumerate(pairs)}
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, f"iqm_milliqang_qg_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump({"job_id": job.job_id(), "backend": b.name, "qubit": QUBIT, "shots": shots,
                   "wall_seconds": round(time.time() - t0, 1), "counts": counts}, fh, indent=1)
    return counts


def main(argv):
    if "--prereg" in argv:
        with open(__file__, "rb") as fh:
            d = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()  # LF-normalized: verifiable from git
        from datetime import datetime, timezone

        os.makedirs(DATA, exist_ok=True)
        with open(PREREG, "w", encoding="utf-8") as fh:
            fh.write(f"{d}  examples/iqm_milliqang_qg.py  {datetime.now(timezone.utc).isoformat()}\n")
        print(d)
        return
    paid = "--probe" in argv or "--run" in argv
    if paid and "--yes-i-accept-qpu-cost" not in argv:
        sys.exit("This spends IQM credits. Re-run with --yes-i-accept-qpu-cost once the cost has been approved.")
    if "--probe" in argv:
        counts = submit([circuits()[0]], 100, "probe")
        print(json.dumps(counts))
        print("Check the credits used for this job in the IQM Resonance dashboard before --run.")
    elif "--run" in argv:
        if not os.path.exists(PREREG):
            sys.exit("Freeze the predictions first: python examples/iqm_milliqang_qg.py --prereg")
        counts = submit(circuits(), SHOTS, "run")
        r = analyse(counts)
        with open(os.path.join(DATA, "iqm_milliqang_qg_analysis.json"), "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1)
        print(json.dumps(_summary(r)))
        print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in r["verdict"].items()))
    else:
        rehearsal()


if __name__ == "__main__":
    main(sys.argv[1:])
