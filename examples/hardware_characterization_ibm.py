"""
The §31 heralded qg characterization as circuits: IBM fake backends and real
IBM devices (§86)

§31 showed, with an exact single-qubit model, that a heralded sweep
(measure, apply I / X / Ry(pi/2), wait t, rotate back, measure again)
separates the thermal excited population of a qubit from its readout error,
which the standard suite cannot do, and returns T1, T2, the two readout
errors and qg_eq = 1 - 2 p_th (hence T_eff) from one joint fit. This script
builds the same 16 + 16 circuits in Qiskit with a mid-circuit measurement,
runs them on a backend and fits them with the §31 functions
(examples/hardware_characterization_qg.py: fit_standard, fit_qg). The
outcome +1 is the bit 0.

Modes:
  --mode fake  calibration-based IBM fake backend (fake_brisbane by default),
               simulated locally by Aer with the backend's noise model
               (thermal relaxation on delays and gates, asymmetric readout
               error, no thermal population).
  --mode ibm   a real IBM device through a saved QiskitRuntimeService
               account; needs --yes-i-run-on-hardware (32 circuits).
The qubit is --qubit, or the one with the lowest reported readout error.

Pre-registered predictions (committed before any backend run; each run is
reported). Fake backends (checks the pipeline end to end):
  C1  the qg fit recovers the backend's T1 and T2 for that qubit within 25%.
  C2  the qg fit recovers both readout errors (prob_meas1_prep0,
      prob_meas0_prep1) within 0.01 absolute.
  C3  qg_eq >= 0.98 (Aer has no thermal population: p_th <= 0.01).
Real device (evaluated when it is run):
  C1' the qg T1 is within 25% of the reported T1 (T2 is not compared: the
      sweep measures a Ramsey-type T2*, the device reports an echo T2).
  C4  the standard suite reports a larger e01 than the qg fit (it counts
      the thermal population as readout error, §31).

Uses the §31 module and the installed library. Needs qiskit, qiskit-aer,
qiskit-ibm-runtime and scipy. Credentials are never read from the code: the
IBM account is the one saved with QiskitRuntimeService.save_account(...).

Findings (python examples/hardware_characterization_ibm.py --mode fake):

FINDINGS_PLACEHOLDER
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hardware_characterization_qg as HC  # noqa: E402

SHOTS = 2000


def build(op, t_us, herald):
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    q = QuantumRegister(1, "q")
    c = ClassicalRegister(2 if herald else 1, "c")
    qc = QuantumCircuit(q, c)
    if herald:
        qc.measure(q[0], c[0])
    if op == "X":
        qc.x(q[0])
    elif op == "Y90":
        qc.ry(math.pi / 2, q[0])
    if t_us > 0:
        qc.delay(t_us, q[0], unit="us")
    if op == "Y90":
        qc.ry(-math.pi / 2, q[0])
    qc.measure(q[0], c[1] if herald else c[0])
    return qc


def circuits(protocol):
    spec = HC.qg_circuits() if protocol == "qg" else HC.standard_circuits()
    return spec, [build(op, t, protocol == "qg") for op, t in spec]


def to_array(counts, herald):
    """Qiskit counts -> the §31 order: (+1,+1), (+1,-1), (-1,+1), (-1,-1)
    with herald (key 'c1c0'), or (+1, -1) without."""
    if not herald:
        return np.array([counts.get("0", 0), counts.get("1", 0)])
    out = np.zeros(4, dtype=int)
    for key, n in counts.items():
        k = key.replace(" ", "")
        r2, r1 = k[0], k[1]  # c1 c0
        out[2 * (r1 == "1") + (r2 == "1")] += n
    return out


def get_backend(mode, name=None):
    if mode == "fake":
        from qiskit_ibm_runtime import fake_provider

        name = name or "fake_brisbane"
        return getattr(fake_provider, "Fake" + name.split("_", 1)[1].capitalize())()
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    return service.backend(name) if name else service.least_busy(operational=True, simulator=False)


def reported(backend, q):
    props = backend.properties()
    qp = backend.qubit_properties(q)
    e01 = props.qubit_property(q, "prob_meas1_prep0")[0]
    e10 = props.qubit_property(q, "prob_meas0_prep1")[0]
    return {"T1": qp.t1 * 1e6, "T2": qp.t2 * 1e6, "e01": float(e01), "e10": float(e10)}


def best_qubit(backend):
    props = backend.properties()
    errs = [(props.readout_error(q), q) for q in range(backend.num_qubits)
            if backend.qubit_properties(q).t1 and backend.qubit_properties(q).t2]
    return min(errs)[1]


def run(backend, mode, q, shots, seed=31):
    from qiskit import transpile

    out = {}
    for protocol in ("standard", "qg"):
        spec, circs = circuits(protocol)
        if mode == "fake":
            from qiskit_aer import AerSimulator

            sim = AerSimulator.from_backend(backend)
            tc = transpile(circs, backend=sim, initial_layout=[q], optimization_level=0, seed_transpiler=seed,
                           scheduling_method="alap")
            res = sim.run(tc, shots=shots, seed_simulator=seed).result()
            counts = [res.get_counts(i) for i in range(len(tc))]
        else:
            from qiskit_ibm_runtime import SamplerV2

            tc = transpile(circs, backend=backend, initial_layout=[q], optimization_level=0, seed_transpiler=seed)
            job = SamplerV2(mode=backend).run(tc, shots=shots)
            print(f"IBM job id ({protocol}):", job.job_id(), flush=True)
            counts = [r.data.c.get_counts() for r in job.result()]
        out[protocol] = [(op, t, to_array(c, protocol == "qg")) for (op, t), c in zip(spec, counts)]
    return out


def analyse(data):
    std = HC.fit_standard(data["standard"])
    qg = HC.fit_qg(data["qg"])
    for f in (std, qg):
        f["p_th"] = (1 - f["qg_eq"]) / 2
        f["T_eff_mK"] = 1e3 * float(HC.t_eff_from_qg(f["qg_eq"]))
    return {"standard": std, "qg": qg}


def verdict(fit, rep, mode):
    q = fit["qg"]
    rel = lambda a, b: abs(a - b) / b  # noqa: E731
    if mode == "fake":
        return {
            "C1": rel(q["T1"], rep["T1"]) < 0.25 and rel(q["T2"], rep["T2"]) < 0.25,
            "C2": abs(q["e01"] - rep["e01"]) < 0.01 and abs(q["e10"] - rep["e10"]) < 0.01,
            "C3": q["qg_eq"] >= 0.98,
        }
    return {"C1'": rel(q["T1"], rep["T1"]) < 0.25, "C4": fit["standard"]["e01"] > q["e01"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="heralded qg characterization on IBM backends")
    ap.add_argument("--mode", choices=["fake", "ibm"], default="fake")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--qubit", type=int, default=None)
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--yes-i-run-on-hardware", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.mode == "ibm" and not args.yes_i_run_on_hardware:
        print("Real IBM device: 32 single-qubit circuits x", args.shots,
              "shots. Rerun with --yes-i-run-on-hardware (uses your account's QPU minutes).")
        return None
    backend = get_backend(args.mode, args.backend)
    q = best_qubit(backend) if args.qubit is None else args.qubit
    rep = reported(backend, q)
    fit = analyse(run(backend, args.mode, q, args.shots))
    v = verdict(fit, rep, args.mode)
    result = {"backend": getattr(backend, "name", str(backend)), "qubit": q, "reported": rep, **fit}
    print(json.dumps(result, indent=1, default=float))
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    if args.out:
        json.dump({"result": result, "verdict": v}, open(args.out, "w"), indent=1, default=float)
    return result, v


if __name__ == "__main__":
    main()
