"""
Hardware plan for the IonQ research-credit request: the exact circuits of
each track, their size in IonQ native gates (GPI/GPI2 = 1q, MS = 2q) after
transpiling for Forte, and a cost estimate from IonQ's job-estimate API
(GET /v0.4/jobs/estimate, read-only: nothing is submitted).

No variational loop runs on the QPU: every circuit has fixed parameters
optimised classically beforehand (H2 angle per bond length, QAOA angles in
examples/data/qaoa_k_angles.json), so the circuit count is final.

Usage:
    python examples/ionq_hardware_plan.py            # sizes only (no key needed)
    python examples/ionq_hardware_plan.py --estimate # + IonQ API estimate (key)
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BACKEND = "qpu.forte-1"
H2_BONDS = ("0.735", "1.5", "2.5")


def _h2_circuits(r_key):
    import chemistry_qg_symmetry_witness as CH
    from qiskit.quantum_info import SparsePauliOp

    with open(os.path.join(HERE, "data", "h2_dissociation_jw.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    key = min(data, key=lambda k: abs(float(k) - float(r_key)))
    h = SparsePauliOp.from_list(list(data[key]["terms"].items()))
    t = CH.optimal_angle(h)
    labels = ["ZZZZ"] + [l for l, _ in CH._terms(h) if any(ch in "XY" for ch in l)]
    return [CH._measure_circuit(t, l, 0) for l in labels], float(key)


def tracks():
    """name -> (list of circuits, shots per circuit, repetitions, debiasing)."""
    import chemistry_qg_symmetry_witness as CH
    import hubbard_trotter_qg_filters as HB
    import ionq_sim_qrng as QR
    import qaoa_k_constraint_qg as QA

    out = {}
    eq, _ = _h2_circuits("0.735")
    cal = CH._calibration_circuits()
    out["A1 H2 eq. + qg filter (decisive), 3 days"] = (eq + cal, 5000, 3, False)
    st = []
    for r in ("1.5", "2.5"):
        st += _h2_circuits(r)[0]
    out["A2 H2 1.5 and 2.5 A + qg filter"] = (st, 5000, 1, False)
    out["A3 H2 eq. with IonQ debiasing"] = (eq + cal, 2500, 1, True)
    out["B  H2 eq. native-MS ZNE, scales 3, 5"] = (("fold", eq), 2000, 1, False)
    with open(os.path.join(HERE, "data", "qaoa_k_angles.json"), encoding="utf-8") as fh:
        ang = json.load(fh)
    qa = [QA.qaoa_circuit(QA.random_graph(g), "xy", np.array(ang[f"p1_g{g}_xy"])) for g in range(5)]
    out["C  XY-QAOA exactly-K + filter, 5 graphs"] = (qa, 500, 1, False)
    out["D  Hubbard 8 q, 1-2-4 Trotter steps + cal"] = ([HB.trotter_circuit(s) for s in (1, 2, 4)] + HB._calibration(),
                                                      1000, 1, False)
    qr = [QR._circuit(t, st_) for t in (0.0, 0.08) for st_ in QR.SETTINGS] + QR._calibration()
    out["E  QRNG certification + readout cal."] = (qr, 2000, 1, False)
    return out


# Public resource estimator (ionq.com/programs/research-credits/resource-estimator,
# Forte, read 2026-09-26): minimum $25.79 per circuit, $168.20 with debiasing.
PUBLIC_ESTIMATE_USD = {
    "A1 H2 eq. + qg filter (decisive), 3 days": 3 * 181.67,
    "A2 H2 1.5 and 2.5 A + qg filter": 260.18,
    "A3 H2 eq. with IonQ debiasing": 1177.40,
    "B  H2 eq. native-MS ZNE, scales 3, 5": 385.60,
    "C  XY-QAOA exactly-K + filter, 5 graphs": 554.84,
    "D  Hubbard 8 q, 1-2-4 Trotter steps + cal": 271.25,
    "E  QRNG certification + readout cal.": 309.48,
}
CONTINGENCY_USD = 3997.00 - sum(PUBLIC_ESTIMATE_USD.values())  # one re-run of a failed track


def _count(qc, n_qubits):
    ops = qc.count_ops()
    g1 = sum(v for k, v in ops.items() if k not in ("measure", "barrier", "cx", "ms", "delay"))
    g2 = int(ops.get("cx", 0) + ops.get("ms", 0))
    return (n_qubits, int(g1), g2)


def qis_sizes(circuits):
    """(qubits, 1q, 2q) per circuit in a CX + single-qubit-rotation basis
    (what IonQ's compiler receives for a QIS-gate job)."""
    from qiskit import transpile

    out = []
    for c in circuits:
        if any(k in ("ms", "gpi", "gpi2") for k in c.count_ops()):  # already native (QRNG track)
            out.append(_count(c, c.num_qubits))
        else:
            t = transpile(c, basis_gates=["rx", "ry", "rz", "cx"], optimization_level=2, seed_transpiler=1)
            out.append(_count(t, c.num_qubits))
    return out


def all_sizes():
    import ionq_sim_zne_grover as Z
    from qiskit import transpile
    from qiskit_ionq import IonQProvider

    res = {}
    for name, (circs, shots, reps, deb) in tracks().items():
        if isinstance(circs, tuple):  # native-MS folded copies of the equilibrium H2 circuits
            nb = IonQProvider(token="none").get_backend("simulator", gateset="native")
            base = transpile(circs[1], backend=nb, optimization_level=1)
            sizes = [_count(Z.fold_ms(c, s), circs[1][0].num_qubits) for s in (3, 5) for c in base]
        else:
            sizes = qis_sizes(circs)
        res[name] = (sizes, shots, reps, deb)
    return res


def estimate(qubits, g1, g2, shots, mitigation=False):
    import requests

    from ionq_validation import read_api_key

    r = requests.get("https://api.ionq.co/v0.4/jobs/estimate",
                     headers={"Authorization": f"apiKey {read_api_key()}"},
                     params={"backend": BACKEND, "qubits": qubits, "shots": shots, "1q_gates": g1, "2q_gates": g2,
                             "error_mitigation": str(mitigation).lower()}, timeout=30)
    r.raise_for_status()
    d = r.json()
    return float(d["estimated_total_cost"]), d.get("estimated_unit"), d["rate_card"]["rates"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args(argv)
    sizes = all_sizes()
    total = 0.0
    for name, (rows, shots, reps, deb) in sizes.items():
        q = max(r[0] for r in rows)
        g2 = [r[2] for r in rows]
        g1 = [r[1] for r in rows]
        line = (f"{name:44s} {len(rows):2d} circuits x {reps}  {q:2d} qubits  2q {min(g2)}-{max(g2)}  "
                f"1q {min(g1)}-{max(g1)}  shots {shots}{'  debiased' if deb else ''}  "
                f"public ${PUBLIC_ESTIMATE_USD[name]:8.2f}")
        if args.estimate:
            cost = reps * sum(estimate(*r, shots, deb)[0] for r in rows)
            total += cost
            line += f"  API {cost:7.2f}"
        print(line)
    pub = sum(PUBLIC_ESTIMATE_USD.values())
    print(f"public estimator: tracks ${pub:.2f} + contingency ${CONTINGENCY_USD:.2f} = ${pub + CONTINGENCY_USD:.2f}")
    if args.estimate:
        _, unit, rates = estimate(4, 10, 2, 100)
        print(f"account API estimate: {total:.2f} {unit}; rate card {rates}")
    return sizes


if __name__ == "__main__":
    main()
