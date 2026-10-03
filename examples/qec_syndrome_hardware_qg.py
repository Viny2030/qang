"""
§33 on IBM backends: reading T1 and dephasing from the Leung-code syndromes
(§88)

§33 showed, in a code-capacity model, that the syndromes of the 4-qubit
Leung code are a free, continuous witness of the noise: every stabilizer
expectation is a power of the §27 single-qubit witness qg_X,

    <Z0Z1> = 1 - 2 gamma (1 - gamma),   <XXXX> = (1 - gamma)^2 (1 - 2p)^4,

so the ancilla flip rates invert in one line to the damping gamma and the
dephasing p of one idle period (estimate_from_leung). This script runs that
on circuits: 4 data qubits encoded in the Leung code (|0_L> = |0000> + |1111>,
|1_L> = |0011> + |1100>, half the shots each), an idle delay t, then one
round of syndrome extraction with 3 ancillas (Z0Z1, Z2Z3 and XXXX) measured
mid-circuit style (7 qubits, one round per shot).

Encoding and extraction add flips of their own. They are measured at t = 0
and removed by composing independent flips: r_idle = (r(t) - r(0)) /
(1 - 2 r(0)). The idle estimates are compared with the backend calibration of
the 4 data qubits: gamma_cal = mean (1 - exp(-t/T1)), p_cal = mean
(1 - exp(-t/T_phi))/2 with 1/T_phi = 1/T2 - 1/(2 T1).

Modes: --mode fake (calibration-based IBM fake backend, Aer) and --mode ibm
(saved QiskitRuntimeService account; needs --yes-i-run-on-hardware).

Pre-registered predictions (committed before any backend run; each run is
reported). Delays t = 0, 10, 20, 40, 80 us, 4000 shots per delay and
logical state:
  D1  the syndrome estimate of gamma is within 40% of gamma_cal at t = 40
      and 80 us.
  D2  the syndrome estimate of p is within a factor 2 of p_cal at t = 40 and
      80 us.
  D3  both flip rates (ZZ and XXXX) increase with t.
On a real device D2 compares with the reported (echo) T2, which can differ
from the dephasing seen during a free delay; it is reported, not tuned.

Uses the §33 estimators (examples/qec_syndrome_drift_tracking_qg.py). Needs
qiskit, qiskit-aer, qiskit-ibm-runtime and scipy.

Findings (python examples/qec_syndrome_hardware_qg.py --mode fake):

Fake IBM backends (Aer with the calibrated noise models), 4000 shots per
delay and logical state. Flip rates at t = 0 (the floor from encoding and
extraction), and gamma, p at t = 40 and 80 us, syndrome estimate against
calibration:

  backend          floor ZZ / XXXX   gamma 40 us     gamma 80 us     p 40 us        p 80 us        D1 D2 D3
  fake_brisbane    0.082 / 0.202     0.194 / 0.177   0.308 / 0.309   0.055 / 0.078  0.103 / 0.132  P  P  P
  fake_sherbrooke  0.093 / 0.180     0.067 / 0.127   0.162 / 0.237   0.017 / 0.024  0.044 / 0.047  F  P  F
  fake_torino      0.048 / 0.127     0.191 / 0.193   0.345 / 0.348   0.147 / 0.190  0.193 / 0.282  P  P  P

  * On brisbane and torino the Leung syndromes read the damping of the
    four data qubits to within 1-10% of the calibration and the
    dephasing at 0.68-0.78 of the calibration (D1-D3 pass).
  * On sherbrooke D1 and D3 fail. Its qubits have T1 ~ 300 us, so the
    damping in 40-80 us (gamma 0.13-0.24) is small against a floor of 9%
    ZZ flips from encoding and extraction; the ZZ rate at 10 us (0.075) is
    even below the t = 0 floor (0.093), and the floor-corrected gamma comes
    out 32-47% low. The floor correction assumes independent flips; with
    extraction errors that large it is biased.
  * The dephasing estimates sit below the calibration on all three (at
    0.68-0.94 of it): part of the XXXX signal is absorbed by the floor.
  * Pending: a real device (14 seven-qubit circuits x 4000 shots):
      python examples/qec_syndrome_hardware_qg.py --mode ibm --yes-i-run-on-hardware
    The circuits use only final measurements, but they measure idle
    damping during a delay; trapped ions (IonQ) have T1 of seconds and the
    IonQ simulator has no idle-noise model, so IonQ is not a meaningful
    target for this test.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from qec_syndrome_drift_tracking_qg import estimate_from_leung, leung_rates_closed_form  # noqa: E402

DELAYS_US = (0.0, 10.0, 20.0, 40.0, 80.0)
SHOTS = 4000


def build(t_us, logical):
    """Data qubits 0-3, ancillas 4 (Z0Z1), 5 (Z2Z3), 6 (XXXX); bits c0..c2."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    d = QuantumRegister(4, "d")
    a = QuantumRegister(3, "a")
    c = ClassicalRegister(3, "c")
    qc = QuantumCircuit(d, a, c)
    qc.h(d[0])
    for k in (1, 2, 3):
        qc.cx(d[0], d[k])
    if logical == 1:
        qc.x(d[0])
        qc.x(d[1])
    qc.barrier()
    if t_us > 0:
        for k in range(4):
            qc.delay(t_us, d[k], unit="us")
    qc.barrier()
    qc.cx(d[0], a[0])
    qc.cx(d[1], a[0])
    qc.cx(d[2], a[1])
    qc.cx(d[3], a[1])
    qc.h(a[2])
    for k in range(4):
        qc.cx(a[2], d[k])
    qc.h(a[2])
    qc.measure(a, c)
    return qc


def flips(counts):
    """(ZZ flips counted over both ZZ checks, XXXX flips, shots); key 'c2c1c0'."""
    zz = xx = n = 0
    for key, m in counts.items():
        k = key.replace(" ", "")
        c2, c1, c0 = k[0], k[1], k[2]
        zz += m * ((c0 == "1") + (c1 == "1"))
        xx += m * (c2 == "1")
        n += m
    return zz, xx, n


def get_backend(mode, name=None):
    if mode == "fake":
        from qiskit_ibm_runtime import fake_provider

        name = name or "fake_brisbane"
        return getattr(fake_provider, "Fake" + name.split("_", 1)[1].capitalize())()
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    return service.backend(name) if name else service.least_busy(operational=True, simulator=False, min_num_qubits=7)


def run(backend, mode, shots, seed=88):
    from qiskit import transpile

    circs, spec = [], []
    for t in DELAYS_US:
        for logical in (0, 1):
            circs.append(build(t, logical))
            spec.append((t, logical))
    if mode == "fake":
        from qiskit_aer import AerSimulator

        sim = AerSimulator.from_backend(backend)
        tc = transpile(circs, backend=sim, optimization_level=1, seed_transpiler=seed, scheduling_method="alap")
        res = sim.run(tc, shots=shots, seed_simulator=seed).result()
        counts = [res.get_counts(i) for i in range(len(tc))]
    else:
        from qiskit_ibm_runtime import SamplerV2

        tc = transpile(circs, backend=backend, optimization_level=1, seed_transpiler=seed)
        job = SamplerV2(mode=backend).run(tc, shots=shots)
        print("IBM job id:", job.job_id(), flush=True)
        counts = [r.data.c.get_counts() for r in job.result()]
    layout = tc[0].layout.final_index_layout()[:4]
    per_t = {}
    for (t, _), c in zip(spec, counts):
        zz, xx, n = flips(c)
        a = per_t.setdefault(t, [0, 0, 0])
        a[0] += zz
        a[1] += xx
        a[2] += n
    return per_t, layout


def calibration(backend, layout, t_us):
    g, p = [], []
    for q in layout:
        qp = backend.qubit_properties(q)
        T1, T2 = qp.t1 * 1e6, qp.t2 * 1e6
        g.append(1 - math.exp(-t_us / T1))
        inv = 1 / T2 - 1 / (2 * T1)
        p.append((1 - math.exp(-t_us * inv)) / 2 if inv > 0 else 0.0)
    return float(np.mean(g)), float(np.mean(p))


def analyse(per_t, backend, layout):
    zz0, xx0, n0 = per_t[0.0]
    r0z, r0x = zz0 / (2 * n0), xx0 / n0
    rows = []
    for t in DELAYS_US:
        zz, xx, n = per_t[t]
        rz, rx = zz / (2 * n), xx / n
        iz = max((rz - r0z) / (1 - 2 * r0z), 0.0)
        ix = max((rx - r0x) / (1 - 2 * r0x), 0.0)
        g_est, p_est = estimate_from_leung(iz * 2 * n, ix * n, n)
        g_cal, p_cal = calibration(backend, layout, t)
        rows.append({"t_us": t, "ZZ flip rate": rz, "XXXX flip rate": rx, "gamma syndrome": g_est,
                     "p syndrome": p_est, "gamma calibration": g_cal, "p calibration": p_cal,
                     "ZZ rate predicted (calibration)": leung_rates_closed_form(g_cal, p_cal)[0]})
    return rows


def verdict(rows):
    by = {r["t_us"]: r for r in rows}
    d1 = all(abs(by[t]["gamma syndrome"] / by[t]["gamma calibration"] - 1) < 0.4 for t in (40.0, 80.0))
    d2 = all(0.5 <= by[t]["p syndrome"] / max(by[t]["p calibration"], 1e-12) <= 2.0 for t in (40.0, 80.0))
    zz = [r["ZZ flip rate"] for r in rows]
    xx = [r["XXXX flip rate"] for r in rows]
    d3 = all(np.diff(zz) > 0) and all(np.diff(xx) > 0)
    return {"D1": d1, "D2": d2, "D3": d3}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Leung-code syndromes as a noise witness on IBM backends")
    ap.add_argument("--mode", choices=["fake", "ibm"], default="fake")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--yes-i-run-on-hardware", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.mode == "ibm" and not args.yes_i_run_on_hardware:
        print(f"Real IBM device: {2 * len(DELAYS_US)} seven-qubit circuits x {args.shots} shots. "
              "Rerun with --yes-i-run-on-hardware (uses your account's QPU minutes).")
        return None
    backend = get_backend(args.mode, args.backend)
    per_t, layout = run(backend, args.mode, args.shots)
    rows = analyse(per_t, backend, layout)
    v = verdict(rows)
    out = {"backend": getattr(backend, "name", str(backend)), "data qubits": [int(q) for q in layout], "rows": rows}
    print(json.dumps(out, indent=1))
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    if args.out:
        json.dump({"result": out, "verdict": v}, open(args.out, "w"), indent=1)
    return out, v


if __name__ == "__main__":
    main()
