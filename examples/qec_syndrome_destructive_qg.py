"""
Leung-code syndromes without ancillas and without SWAPs (§93)

§88-§92 read T1 and dephasing from the Leung-code syndromes with three
ancillas. On heavy-hex devices that circuit needs SWAPs (the encoder and the
XXXX check need degree-3 and degree-4 vertices), so the data idle on qubits
that are not the ones whose calibration was compared, and the SWAPs add
errors (§92 diagnosis). With one round per shot the stabilizers can be read
destructively instead:

  encoder on a line: H d0; CX d0->d1; CX d1->d2; CX d2->d3
                     (|0000> + |1111>)/sqrt 2; X on d0, d1 gives |1_L>
  idle delay t on the four data qubits
  half the shots: measure the data in Z -> Z0Z1, Z2Z3 from bit parities
  half the shots: H on the data, measure  -> XXXX from the parity of all four

Four qubits on a path of the device (the path with the lowest sum of readout
and two-qubit errors), fixed by initial_layout, so no routing is needed and
the idle qubits are known. The floor (encoding and readout) and the idle
parameters are fitted jointly over all delays as in §92 (same model and
estimator), and compared with the calibration of the four qubits.

Pre-registered predictions (committed before the run; fake IBM backends
brisbane, sherbrooke and torino; delays 0, 10, 20, 40, 80, 160 us; 4000 shots
per delay, logical state and basis):
  M1  no routing: the final layout equals the initial layout on all three.
  M2  the joint-fit gamma is within 30% of the calibration gamma at
      t = 40, 80 and 160 us on all three.
  M3  the joint-fit p is within a factor 2 of the calibration p at the same
      delays on all three.
  M4  the fitted ZZ floor is below the §88 floor of the same backend
      (0.082 brisbane, 0.093 sherbrooke, 0.048 torino).

Modes: --mode fake (default) and --mode ibm (saved QiskitRuntimeService
account; needs --yes-i-run-on-hardware; 24 four-qubit circuits). Needs
qiskit, qiskit-aer, qiskit-ibm-runtime and scipy.

Findings (python examples/qec_syndrome_destructive_qg.py):

Fake backends (AerSimulator.from_backend), best 4-qubit path, joint fit:

  backend      path              ZZ floor   gamma 40/80/160 us,           p 40/80/160 us,
                                            calibration / syndromes       calibration / syndromes
  brisbane     112-126-125-124   0.030      0.155/0.286/0.489 /           0.131/0.193/0.265 /
                                            0.149/0.275/0.474             0.165/0.276/0.399
  sherbrooke   122-123-124-125   0.022      0.135/0.252/0.440 /           0.074/0.125/0.189 /
                                            0.132/0.246/0.432             0.087/0.159/0.267
  torino       99-92-80-81       0.028      0.194/0.349/0.574 /           0.152/0.227/0.311 /
                                            0.191/0.346/0.572             0.184/0.301/0.420

  * M1-M4 pass on all three: no routing; the damping of the four data qubits
    is read to 1-4% of the calibration (§88: 1-47%); the floor drops to
    2.2-3.0% (§88: 4.8-9.3%).
  * Dephasing reads 1.17-1.51x high against the arithmetic mean of the four
    qubits: within the factor 2, but biased. Resolved in §99: the XXXX
    syndrome reads the product mean, and against it the fit agrees to
    0.99-1.04 (examples/qec_syndrome_dephasing_target_qg.py).
  Real device: pending (--mode ibm --yes-i-run-on-hardware).
"""

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qec_syndrome_hardware_qg as S  # noqa: E402
import qec_syndrome_jointfit_qg as J  # noqa: E402

DELAYS_US = (0.0, 10.0, 20.0, 40.0, 80.0, 160.0)
CHECK_T = (40.0, 80.0, 160.0)
BACKENDS = ("fake_brisbane", "fake_sherbrooke", "fake_torino")
FLOOR_88 = {"fake_brisbane": 0.082, "fake_sherbrooke": 0.093, "fake_torino": 0.048}


def build(t_us, logical, basis):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4, 4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    if logical == 1:
        qc.x(0)
        qc.x(1)
    qc.barrier()
    if t_us > 0:
        for k in range(4):
            qc.delay(t_us, k, unit="us")
    qc.barrier()
    if basis == "X":
        qc.h(range(4))
    qc.measure(range(4), range(4))
    return qc


def best_path(backend):
    """4-qubit path minimizing summed readout and two-qubit gate errors."""
    props = backend.properties()
    cmap = backend.coupling_map
    edges = set(map(tuple, cmap.get_edges()))
    und = {(a, b) for a, b in edges} | {(b, a) for a, b in edges}
    nbr = {}
    for a, b in und:
        nbr.setdefault(a, set()).add(b)

    def edge_err(a, b):
        for name in ("ecr", "cz", "cx"):
            for e in ((a, b), (b, a)):
                try:
                    return props.gate_error(name, list(e))
                except Exception:
                    pass
        return 1.0

    best, best_cost = None, math.inf
    for q0 in nbr:
        stack = [[q0]]
        while stack:
            path = stack.pop()
            if len(path) == 4:
                cost = sum(props.readout_error(q) for q in path) + sum(edge_err(a, b) for a, b in zip(path, path[1:]))
                qp = [backend.qubit_properties(q) for q in path]
                if all(p.t1 and p.t2 for p in qp) and cost < best_cost:
                    best, best_cost = path, cost
                continue
            for n in nbr[path[-1]]:
                if n not in path:
                    stack.append(path + [n])
    return best


def flips(counts_z, counts_x):
    zz = xx = nz = nx = 0
    for key, m in counts_z.items():
        b = key.replace(" ", "")[::-1]  # b[i] = bit of qubit i
        zz += m * ((b[0] != b[1]) + (b[2] != b[3]))
        nz += m
    for key, m in counts_x.items():
        b = key.replace(" ", "")
        xx += m * (b.count("1") % 2)
        nx += m
    return zz, xx, nz, nx


def run(backend, mode, shots, path, seed=93):
    from qiskit import transpile

    circs, spec = [], []
    for t in DELAYS_US:
        for logical in (0, 1):
            for basis in ("Z", "X"):
                circs.append(build(t, logical, basis))
                spec.append((t, basis))
    if mode == "fake":
        from qiskit_aer import AerSimulator

        sim = AerSimulator.from_backend(backend)
        tc = transpile(circs, backend=sim, initial_layout=path, optimization_level=1, seed_transpiler=seed,
                       scheduling_method="alap")
        res = sim.run(tc, shots=shots, seed_simulator=seed).result()
        counts = [res.get_counts(i) for i in range(len(tc))]
    else:
        from qiskit_ibm_runtime import SamplerV2

        tc = transpile(circs, backend=backend, initial_layout=path, optimization_level=1, seed_transpiler=seed)
        job = SamplerV2(mode=backend).run(tc, shots=shots)
        print("IBM job id:", job.job_id(), flush=True)
        counts = [r.data.c.get_counts() for r in job.result()]
    no_routing = all(list(c.layout.final_index_layout()[:4]) == list(path) for c in tc)
    per_t = {}
    for (t, basis), c in zip(spec, counts):
        a = per_t.setdefault(t, {"Z": {}, "X": {}})
        for k, v in c.items():
            a[basis][k] = a[basis].get(k, 0) + v
    out = {}
    for t, d in per_t.items():
        zz, xx, nz, nx = flips(d["Z"], d["X"])
        out[t] = [zz, xx, nz]  # nz = nx by construction
        assert nz == nx
    return out, no_routing


def analyse(backend, per_t, path):
    fit = J.joint_fit(per_t)
    rows = []
    for t in CHECK_T:
        g_cal, p_cal = S.calibration(backend, path, t)
        rows.append({"t_us": t, "gamma calibration": g_cal, "gamma joint fit": 1 - math.exp(-t / fit["T1"]),
                     "p calibration": p_cal, "p joint fit": (1 - math.exp(-t / fit["T_phi"])) / 2})
    return fit, rows


def verdict(results):
    return {
        "M1": all(r["no routing"] for r in results.values()),
        "M2": all(abs(x["gamma joint fit"] / x["gamma calibration"] - 1) < 0.30 for r in results.values() for x in r["rows"]),
        "M3": all(0.5 <= x["p joint fit"] / max(x["p calibration"], 1e-12) <= 2.0 for r in results.values() for x in r["rows"]),
        "M4": all(r["fit"]["r0_ZZ"] < FLOOR_88.get(name, 1.0) for name, r in results.items()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fake", "ibm"], default="fake")
    ap.add_argument("--backend", default=None, help="IBM device (ibm mode)")
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--yes-i-run-on-hardware", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.mode == "ibm" and not args.yes_i_run_on_hardware:
        print(f"Real IBM device: {4 * len(DELAYS_US)} four-qubit circuits x {args.shots} shots. "
              "Rerun with --yes-i-run-on-hardware (uses your account's QPU minutes).")
        return None
    names = BACKENDS if args.mode == "fake" else [args.backend]
    results = {}
    for name in names:
        backend = S.get_backend(args.mode, name)
        path = best_path(backend)
        per_t, no_routing = run(backend, args.mode, args.shots, path)
        fit, rows = analyse(backend, per_t, path)
        key = name or getattr(backend, "name", "ibm")
        results[key] = {"path": [int(q) for q in path], "no routing": no_routing, "fit": fit, "rows": rows}
        print(key, "path", path, "no routing", no_routing, {k: round(v, 4) for k, v in fit.items()})
        for r in rows:
            print("  ", {k: round(v, 4) for k, v in r.items()}, flush=True)
    v = verdict(results) if args.mode == "fake" else {k: val for k, val in verdict(results).items() if k != "M4"}
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    if args.out:
        json.dump({"results": results, "verdict": v}, open(args.out, "w"), indent=1)
    return results, v


if __name__ == "__main__":
    main()
