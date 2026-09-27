"""
IonQ noisy simulator (forte-1): the §60 chain in IonQ's native gates,
compiled so that the two-qubit gates are, or are not, small rotations
that stay close to the fixed-N sector.

IonQ does not accept MS and ZZ gates in the same circuit (a first attempt
with MS for XX, YY and a native ZZ for the ZZ term failed preflight), so
the test uses the XY model (Delta = 0, H = sum J (XX + YY)), which both
gate families can express alone, plus the XXZ chain in MS gates only:

  xy_ms      XY model, MS(0,0,a/pi) = exp(-i a XX), MS(1/4,1/4,a/pi) =
             exp(-i a YY): small rotations, no basis change
  xy_zz      XY model from native ZZ(a/pi) = exp(-i a ZZ) wrapped in basis
             changes: GPI2(1/4) ZZ GPI2(3/4) = exp(-i a XX),
             GPI2(0) ZZ GPI2(1/2) = exp(-i a YY)
  xxz_basis  XXZ (Delta = 1) in MS only: XX, YY as above and the ZZ term as
             GPI2(1/4) MS GPI2(3/4)

(all identities checked to 1e-15). n = 6, dt = 0.25, J = 1, Neel start
(GPI(0) on odd qubits), 1, 2, 4 and 6 Trotter steps. Circuits are built
in the native gate set and submitted with gateset="native", so the
service cannot recompile them. From the same shots: raw imbalance and the
qg filter (keep N = 3). Errors are against the noiseless circuit. 12
circuits per run, 2000 shots each (the simulator's cap); 5 runs. Free;
nothing is sent to a QPU. Results in examples/data/ionq_sim_results.json
under "ionq_sim|forte-1|xxz_filter|<tag>".

Prediction, written before the run. §60 E: the compilations differ for
amplitude damping (T1) inside the basis-changed gates, not for
depolarizing noise. The local stand-in (--mode local, exact with 1e6
shots) gives the share of the error removed at 4 and 6 steps listed in
the findings. Trapped ions have negligible T1 on these time scales, so if
the forte-1 model is depolarizing-type we expect the same share for xy_ms
and xy_zz: the pooled share over 4 and 6 steps, averaged over 5 runs,
should differ by less than a factor 1.25. A ratio xy_ms/xy_zz above 1.5
would mean the vendor model contains non-Pauli (T1-like) errors. Raw
errors of xy_ms and xy_zz should be equal unless the model charges
single-qubit gates (xy_zz adds 8 GPI2 per bond). With 2000 shots the
shot-noise std of the imbalance is 0.01-0.02, larger than the error at
1-2 steps, so only 4 and 6 steps are informative.

Findings:

  Local stand-in (exact, 1e6 shots), share of the error the filter
  removes at 4 / 6 steps:
    depolarizing p2 = 0.01 only        xy_ms = xy_zz = 49 / 57 %
    p2 = 0.005 plus T1 gamma = 0.005   xy_ms 66 / 69 %, xy_zz 44 / 54 %
  (with T1 the leak-free MS compilation gains about 1.4x in share).

  forte-1, 5 runs x 12 circuits x 2000 shots. Bias of the imbalance
  averaged over the 5 runs (sd over runs in brackets), 4 and 6 steps:

    xy_ms      raw -0.070 (0.007) / +0.073 (0.011)   filter -0.043 / +0.036   kept 0.72 / 0.61
    xy_zz      raw -0.066 (0.014) / +0.078 (0.015)   filter -0.034 / +0.033   kept 0.70 / 0.59
    xxz_basis  raw -0.037 (0.018) / +0.229 (0.022)   filter -0.011 / +0.161   kept 0.61 / 0.52

  Pooled share of the error removed over 4 and 6 steps, per run:
    xy_ms 0.45 +- 0.08, xy_zz 0.55 +- 0.12   -> ratio 0.82

  * The prediction for a depolarizing-type model holds: the ratio 0.82 is
    inside 0.8-1.25 (at its edge, and the wrong way for a T1 effect), and
    the raw errors are equal. The vendor model shows no sign of T1-like
    errors inside basis-changed gates, and 8 extra GPI2 per bond cost
    nothing visible.
  * The filter removes about half of the error on forte-1 in both XY
    compilations (bias 0.070 -> 0.043 and 0.066 -> 0.034 at 4 steps), as
    the depolarizing stand-in predicts (49-57 %).
  * Consequence for a hardware run: on trapped ions the choice between
    MS and ZZ compilation does not matter for the filter; the §60 leak is
    a T1 effect and belongs on platforms with T1 (superconducting). The
    simulator cannot test it, only real hardware can, and there the
    prediction is the same: ratio within 0.8-1.25 unless the device has
    T1-like errors during the gates.
  * At 6 steps the XXZ imbalance is near a node of the noisy signal
    (raw bias 0.23); the filter cuts it to 0.16 only.
"""

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

N_Q = 6
DT = 0.25
STEPS = (1, 2, 4, 6)
COMPILATIONS = ("xy_ms", "xy_zz", "xxz_basis")
DELTA = {"xy_ms": 0.0, "xy_zz": 0.0, "xxz_basis": 1.0}


def build(steps, comp="xy_ms", n=N_Q, dt=DT, measure=True):
    from qiskit import QuantumCircuit
    from qiskit_ionq import GPI2Gate, GPIGate, MSGate, ZZGate

    th = dt / math.pi

    def wrapped(gate, pre, post, a, b):
        qc.append(GPI2Gate(pre), [a])
        qc.append(GPI2Gate(pre), [b])
        qc.append(gate, [a, b])
        qc.append(GPI2Gate(post), [a])
        qc.append(GPI2Gate(post), [b])

    qc = QuantumCircuit(n)
    for i in range(1, n, 2):
        qc.append(GPIGate(0), [i])
    for _ in range(steps):
        for start in (0, 1):
            for i in range(start, n - 1, 2):
                a, b = i, i + 1
                if comp == "xy_zz":
                    wrapped(ZZGate(th), 0.25, 0.75, a, b)  # XX
                    wrapped(ZZGate(th), 0.0, 0.5, a, b)  # YY
                    continue
                qc.append(MSGate(0, 0, th), [a, b])
                qc.append(MSGate(0.25, 0.25, th), [a, b])
                if comp == "xxz_basis":
                    wrapped(MSGate(0, 0, th), 0.25, 0.75, a, b)  # ZZ
    if measure:
        qc.measure_all()
    return qc


def circuits():
    return [build(s, c) for c in COMPILATIONS for s in STEPS]


def _tables(n=N_Q):
    idx = np.arange(1 << n)
    bits = (idx[:, None] >> np.arange(n)) & 1
    imb = ((1 - 2 * bits) * np.array([(-1) ** i for i in range(n)])).mean(axis=1)
    return imb, bits.sum(axis=1)


def ideal_imbalance(steps, delta=1.0):
    import xxz_trotter_filter_qg as X

    return X.imbalance(X.probabilities(N_Q, steps, DT, delta=delta), N_Q)[0]


def analyse(P):
    imb, N = _tables()
    keep = N == N_Q // 2
    out = {}
    j = 0
    for zz in COMPILATIONS:
        for s in STEPS:
            p = P[j]
            j += 1
            ideal = ideal_imbalance(s, DELTA[zz])
            raw = float((p * imb).sum())
            kept = float(p[keep].sum())
            filt = float((p[keep] * imb[keep]).sum() / kept)
            out[f"{zz}|{s}"] = {"ideal": ideal, "raw": raw, "filter": filt, "kept": kept,
                                "err_raw": abs(raw - ideal), "err_filter": abs(filt - ideal)}
    return out


def _local_counts(circs, p2=0.005, gamma=0.005, shots=2000, seed=11):
    """Generic stand-in for the vendor model: every native gate becomes a
    labelled unitary; two-qubit gates get depolarizing p2 and amplitude
    damping gamma on both qubits."""
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, amplitude_damping_error, depolarizing_error

    nm = NoiseModel()
    err = depolarizing_error(p2, 2)
    if gamma > 0:
        ad = amplitude_damping_error(gamma)
        err = err.compose(ad.tensor(ad))
    nm.add_all_qubit_quantum_error(err, ["g2"])
    out = []
    for qc in circs:
        new = qc.copy_empty_like()
        for inst in qc.data:
            op, qs, cs = inst.operation, inst.qubits, inst.clbits
            if op.name in ("measure", "barrier"):
                new.append(op, qs, cs)
            else:
                new.append(UnitaryGate(Operator(op).data, label="g2" if len(qs) == 2 else "g1"), qs)
        res = AerSimulator(method="density_matrix", noise_model=nm).run(new, shots=shots, seed_simulator=seed).result()
        out.append(res.get_counts())
    return out


def run(mode, noise="forte-1", key=None):
    import ionq_sim_hubbard_qaoa as I

    circs = circuits()
    if mode == "local":
        counts = _local_counts(circs)
    else:
        import ionq_sim_zne_grover as G

        counts = I._run(G.native_backend(), circs, noise, "ionq_sim", key, prebuilt=True)
    return analyse([I._probs(c, N_Q) for c in counts])


def report(out):
    print(f"{'compilation':>11} {'steps':>5} {'ideal':>7} | {'|err| raw':>9} {'filter':>7} {'removed':>8} | kept")
    for zz in COMPILATIONS:
        for s in STEPS:
            r = out[f"{zz}|{s}"]
            rem = 1 - r["err_filter"] / r["err_raw"] if r["err_raw"] > 0 else float("nan")
            print(f"{zz:>11} {s:5d} {r['ideal']:+7.3f} | {r['err_raw']:9.4f} {r['err_filter']:7.4f} {rem:8.1%} | {r['kept']:.3f}")


def main(argv=None):
    import ionq_sim_hubbard_qaoa as I

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("local", "ionq_sim"), default="local")
    ap.add_argument("--noise", choices=("aria-1", "forte-1"), default="forte-1")
    ap.add_argument("--tag", default="run0")
    args = ap.parse_args(argv)
    key = f"{args.mode}|{args.noise if args.mode == 'ionq_sim' else 'generic'}|xxz_filter|{args.tag}"
    res = I.load_results()
    if key in res:
        out = res[key]
    else:
        try:
            out = run(args.mode, args.noise, key)
        except I.Pending as exc:
            print("pending:", key, "-", exc)
            return None
        if args.mode == "ionq_sim":
            res[key] = out
            I.save_results(res)
            print("saved", key)
    report(out)
    return out


if __name__ == "__main__":
    main()
