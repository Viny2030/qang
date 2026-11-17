"""
Replication of §168 on five qubits of IQM Garnet and on two days, with the chain model fixed in advance (§169)

§168 (qubit 0, one run) left three things to settle: whether the 16-55 m-qg error at intermediate R_y angles is a
property of the device or of qubit 0; whether the equator copies really scatter less than binomial shot noise
(M2 failed, ratio 0.50, chance 0.013) or that was a fluke; and whether the gate cost in m-qg is linear in the number
of gates everywhere. §168's M3 was badly designed (two parameters on four points); here the model is the one the
chip showed, a single slope per gate, fixed before any data. Pre-registered: this file is frozen (sha256 of its
LF-normalized bytes in examples/data/iqm_milliqang_replication_qg.prereg) and committed before any job.

Circuits, the 26 of §168 on each of QUBITS = (0, 4, 9, 13, 17), compiled at optimization_level 0 with barriers:
  poles      |0> and |1> (one X): e0 = P(1|0), e1 = P(0|1)
  sweep      R_y(arccos(m/1000)) for m in {+707, +500, -500, -707} m-qg
  equator    R_y(pi/2), 10 copies
  chains     k X gates, k in {3, 5, 7, 9, 11, 15, 21, 27, 33, 41}
130 circuits x 4000 shots in one job per day. Day 1 and day 2 are separate jobs at least 12 hours apart.

Pre-registered predictions (each run is judged on its own; R7 compares the two days):
  R1  the affine readout from the poles predicts the pooled equator within 3 sigma on at least 4 of 5 qubits.
  R2  the intermediate-angle error of §168 is general: on at least 3 of 5 qubits the mean |z| of the four sweep
      points exceeds 2.
  R3  the equator scatter is binomial when pooled: sqrt(sum over qubits of (n_copies - 1) s^2 / sigma_bin^2 / 45)
      lies in [0.79, 1.21] (the 95% band for 45 degrees of freedom). §168's 0.50 would be a fluke.
  R4  one slope per gate: c fitted on k = 3..9 as s(k) = c (k - 1) predicts k = 11..41 within 3 sigma on at least
      4 of 6 chains, on at least 4 of 5 qubits.
  R5  the gate cost is positive and budgetable: on every qubit the slope from all ten chains exceeds 3 sigma and
      lies in [0.5, 5] m-qg per X gate.
  R6  qubit 0 replicates §168: |c - 1.64| <= 3 sqrt(sigma_c^2 + 0.11^2).
  R7  (day 2 against day 1) the slope of each qubit agrees between days within 3 sigma on at least 4 of 5 qubits.

Modes (token in IQM_TOKEN only; run in the environment with iqm-client[qiskit], e.g. .venv-iqm):
  python examples/iqm_milliqang_replication_qg.py                     free rehearsal on IQM's fake backends
  python examples/iqm_milliqang_replication_qg.py --prereg            freeze this file before any run
  python examples/iqm_milliqang_replication_qg.py --run day1 --yes-i-accept-qpu-cost
  python examples/iqm_milliqang_replication_qg.py --run day2 --yes-i-accept-qpu-cost      (12 h or more later)
  python examples/iqm_milliqang_replication_qg.py --compare           R7 from the two saved runs
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
QUBITS = (0, 4, 9, 13, 17)
SHOTS = 4000
SWEEP = (707.0, 500.0, -500.0, -707.0)
EQUATOR_COPIES = 10
FIT_K = (3, 5, 7, 9)
TEST_K = (11, 15, 21, 27, 33, 41)
CHAINS = FIT_K + TEST_K
SLOPE_168, SIGMA_168 = 1.64, 0.11
DATA = os.path.join(HERE, "data")
PREREG = os.path.join(DATA, "iqm_milliqang_replication_qg.prereg")


def circuits():
    """(label, circuit) pairs for one qubit, in submission order (the layout is chosen at transpilation)."""
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
    n = sum(counts.values())
    return counts.get("0", 0) / n, n


def _binom_m(p, n):
    return 2000.0 * math.sqrt(max(p * (1 - p), 1.0 / n) / n)


def analyse_qubit(counts_by_label):
    p = {lab: _p0(c) for lab, c in counts_by_label.items()}
    (p00, n0), (p01, n1) = p["pole0"], p["pole1"]
    e0, e1 = 1.0 - p00, p01
    se0 = math.sqrt(max(e0 * (1 - e0), 1.0 / n0) / n0)
    se1 = math.sqrt(max(e1 * (1 - e1), 1.0 / n1) / n1)
    contrast = 1.0 - e0 - e1

    def z_at(mt, pm, n):
        mp = contrast * mt + 1000.0 * (e1 - e0)
        sig = math.hypot(_binom_m(pm, n), (mt + 1000.0) * se0, (1000.0 - mt) * se1)
        return (PB.milliqang_from_p0(pm) - mp) / sig

    sweep_z = [z_at(mt, *p[f"sweep{mt:+.0f}"]) for mt in SWEEP]
    eq = [p[f"equator{i}"] for i in range(EQUATOR_COPIES)]
    neq = sum(n for _, n in eq)
    peq = sum(pp * n for pp, n in eq) / neq
    equator_z = z_at(0.0, peq, neq)
    m_eq = np.array([PB.milliqang_from_p0(pp) for pp, _ in eq])
    sig_bin = _binom_m(peq, neq / EQUATOR_COPIES)
    chi2_eq = float((EQUATOR_COPIES - 1) * m_eq.var(ddof=1) / sig_bin ** 2)

    s, ss = {}, {}
    for k in CHAINS:
        pm, n = p[f"chain{k}"]
        s[k] = 1000.0 + (2000.0 * (pm - e1) / contrast - 1000.0)
        ss[k] = _binom_m(pm, n) / contrast

    def slope(ks):
        x = np.array([k - 1.0 for k in ks])
        w = np.array([1.0 / ss[k] ** 2 for k in ks])
        y = np.array([s[k] for k in ks])
        c = float((w * x * y).sum() / (w * x * x).sum())
        return c, float(1.0 / math.sqrt((w * x * x).sum()))

    c_fit, sc_fit = slope(FIT_K)
    chain_z = [(s[k] - c_fit * (k - 1)) / math.hypot(ss[k], sc_fit * (k - 1)) for k in TEST_K]
    c_all, sc_all = slope(CHAINS)
    return {"e0": e0, "e1": e1, "sweep_z": sweep_z, "equator_z": equator_z, "equator_m": m_eq.tolist(),
            "chi2_equator": chi2_eq, "ratio_equator": math.sqrt(chi2_eq / (EQUATOR_COPIES - 1)),
            "chains": {k: [s[k], ss[k]] for k in CHAINS}, "slope_fit": [c_fit, sc_fit], "chain_z": chain_z,
            "slope_all": [c_all, sc_all]}


def analyse(counts_by_qubit):
    q = {qb: analyse_qubit(c) for qb, c in counts_by_qubit.items()}
    dof = len(q) * (EQUATOR_COPIES - 1)
    pooled = math.sqrt(sum(r["chi2_equator"] for r in q.values()) / dof)
    r0 = q.get(0)
    verdict = {
        "R1": sum(abs(r["equator_z"]) <= 3 for r in q.values()) >= 4,
        "R2": sum(np.mean(np.abs(r["sweep_z"])) > 2 for r in q.values()) >= 3,
        "R3": 0.79 <= pooled <= 1.21,
        "R4": sum(sum(abs(z) <= 3 for z in r["chain_z"]) >= 4 for r in q.values()) >= 4,
        "R5": all(r["slope_all"][0] > 3 * r["slope_all"][1] and 0.5 <= r["slope_all"][0] <= 5 for r in q.values()),
        "R6": r0 is not None and abs(r0["slope_all"][0] - SLOPE_168) <= 3 * math.hypot(r0["slope_all"][1], SIGMA_168),
    }
    return {"qubits": q, "pooled_ratio": pooled, "verdict": verdict}


def _summary(r):
    rows = {qb: {"e0": round(x["e0"], 4), "e1": round(x["e1"], 4), "equator z": round(x["equator_z"], 2),
                 "sweep mean|z|": round(float(np.mean(np.abs(x["sweep_z"]))), 2),
                 "ratio": round(x["ratio_equator"], 2),
                 "slope": f"{x['slope_all'][0]:.2f} +- {x['slope_all'][1]:.2f}",
                 "chain |z| max": round(max(abs(z) for z in x["chain_z"]), 2)} for qb, x in r["qubits"].items()}
    return {"per qubit": rows, "pooled ratio": round(r["pooled_ratio"], 2), "verdict": r["verdict"]}


def _transpile_all(backend):
    from qiskit import transpile

    pairs = circuits()
    labels, ts = [], []
    for qb in QUBITS:
        for lab, qc in pairs:
            labels.append((qb, lab))
            ts.append(transpile(qc, backend, initial_layout=[qb], optimization_level=0))
    return labels, ts


def _split(labels, counts_list):
    out = {}
    for (qb, lab), c in zip(labels, counts_list):
        out.setdefault(qb, {})[lab] = c
    return out


def rehearsal(shots=SHOTS):
    from iqm.qiskit_iqm import IQMFakeAphrodite, IQMFakeApollo

    for B in (IQMFakeApollo, IQMFakeAphrodite):
        b = B()
        labels, ts = _transpile_all(b)
        res = b.run(ts, shots=shots, seed_simulator=169).result()
        r = analyse(_split(labels, [res.get_counts(i) for i in range(len(ts))]))
        out = {"backend": b.name, "circuits": len(ts)}
        out.update(_summary(r))
        print(json.dumps(out), flush=True)


def real_backend():
    if not os.environ.get("IQM_TOKEN"):
        sys.exit("IQM_TOKEN is not set: set it in this terminal only, never write it to a file in the repository.")
    from iqm.qiskit_iqm import IQMProvider

    return IQMProvider(SERVER, quantum_computer=QC_NAME).get_backend()


def run(tag):
    b = real_backend()
    labels, ts = _transpile_all(b)
    print(json.dumps({"backend": b.name, "circuits": len(ts), "shots": SHOTS, "qubits": QUBITS, "tag": tag}),
          flush=True)
    t0 = time.time()
    job = b.run(ts, shots=SHOTS)
    print(json.dumps({"job_id": job.job_id()}), flush=True)
    res = job.result()
    counts = [res.get_counts(i) for i in range(len(ts))]
    os.makedirs(DATA, exist_ok=True)
    raw = {"job_id": job.job_id(), "backend": b.name, "shots": SHOTS, "tag": tag,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wall_seconds": round(time.time() - t0, 1),
           "counts": [[qb, lab, c] for (qb, lab), c in zip(labels, counts)]}
    with open(os.path.join(DATA, f"iqm_milliqang_replication_qg_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=1)
    r = analyse(_split(labels, counts))
    with open(os.path.join(DATA, f"iqm_milliqang_replication_qg_{tag}_analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=1, default=float)
    return r


def compare():
    rs = []
    for tag in ("day1", "day2"):
        with open(os.path.join(DATA, f"iqm_milliqang_replication_qg_{tag}_analysis.json"), encoding="utf-8") as fh:
            rs.append(json.load(fh)["qubits"])
    agree = {}
    for qb in rs[0]:
        (c1, s1), (c2, s2) = rs[0][qb]["slope_all"], rs[1][qb]["slope_all"]
        agree[qb] = {"day1": round(c1, 2), "day2": round(c2, 2), "z": round((c2 - c1) / math.hypot(s1, s2), 2)}
    ok = sum(abs(a["z"]) <= 3 for a in agree.values()) >= 4
    print(json.dumps({"slopes": agree, "R7": "PASS" if ok else "FAIL"}))


def main(argv):
    if "--prereg" in argv:
        with open(__file__, "rb") as fh:
            d = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()  # LF-normalized: verifiable from git
        from datetime import datetime, timezone

        os.makedirs(DATA, exist_ok=True)
        with open(PREREG, "w", encoding="utf-8") as fh:
            fh.write(f"{d}  examples/iqm_milliqang_replication_qg.py  {datetime.now(timezone.utc).isoformat()}\n")
        print(d)
        return
    if "--compare" in argv:
        compare()
        return
    if "--run" in argv:
        if "--yes-i-accept-qpu-cost" not in argv:
            sys.exit("This spends IQM credits. Re-run with --yes-i-accept-qpu-cost once the cost has been approved.")
        if not os.path.exists(PREREG):
            sys.exit("Freeze the predictions first: python examples/iqm_milliqang_replication_qg.py --prereg")
        tag = argv[argv.index("--run") + 1]
        if tag not in ("day1", "day2"):
            sys.exit("--run needs day1 or day2")
        r = run(tag)
        print(json.dumps(_summary(r)))
        print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in r["verdict"].items()))
        return
    rehearsal()


if __name__ == "__main__":
    main(sys.argv[1:])
