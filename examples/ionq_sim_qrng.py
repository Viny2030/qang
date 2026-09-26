"""
IonQ noisy cloud simulator: certified quantum random numbers (§41) on
trapped-ion noise models (aria-1, forte-1).

The source. A QRNG qubit q starts in |0> and its output is measured in X
(GPI2(0.75) then Z readout), so ideally each shot is a fair coin. A
second ion e plays the environment / adversary: an MS(0, 0, theta)
interaction lets e learn the X value of q. The output stays unbiased
(the naive estimate keeps saying "1 bit per shot") but the private
randomness falls. For the ideal circuit the reduced state of q has
qg_Z = cos(2 pi theta), qg_Y = 0, so

    H_min(X|E) = -log2[(1 + sqrt(1 - r^2)) / 2],   r = |cos(2 pi theta)|,

the §41 formula with the roles of X and Z exchanged (the coherence
that counts is the one transverse to the output basis: qg_Y, qg_Z).

Circuits per theta (all built in IonQ's native gate set and submitted
with gateset="native", so the service cannot merge the rotations):

  output     GPI2(0.75) on q               -> measures +X (output bits)
  Z+ / Z-    nothing / GPI(0)              -> measures +Z / -Z
  Y+ / Y-    GPI2(0) / GPI2(0.5)           -> measures +Y / -Y

plus two readout-calibration circuits (|0>, GPI(0)|1>) as in §31.
COPIES independent (q, e) pairs run side by side in each circuit, so a
2000-shot job gives 2000 * COPIES samples per setting.

Estimators (as in qrng_qg_certified.py): naive (output bias), qg
one-sided (Z+, Y+ only), qg +/- pairs (offset of the readout cancels),
qg calibrated (+/- pairs divided by the calibrated readout gain b). All
qg estimators use a 3-sigma lower confidence bound on r.

"truth" is the ideal-circuit value. The device noise can only add
classical noise, which a careful certifier must also attribute to the
adversary, so the private randomness of the noisy device is at most the
ideal value: an estimate above it is unsafe; one below it is safe.

Modes: "local" (Aer with the generic all-to-all noise model of §29) and
"ionq_sim" (IonQ's cloud simulator, free; key in IONQ_API_KEY or the
git-ignored .ionq_key). Nothing here submits to a QPU. Results go to
examples/data/ionq_sim_results.json under "ionq_sim|<noise>|qrng|<tag>".

Findings (3 repetitions per noise model, 10^4 samples per setting; mean
certified bits per shot, none of the qg estimates above the truth in
any run, the naive one above it in every run with theta > 0):

    theta  truth   naive   qg one-sided  qg +/-   (aria-1 / forte-1)
    0.00   1.000   0.98    0.69          0.73
    0.04   0.680   0.99    0.53          0.56 / 0.55
    0.08   0.433   0.98    0.37 / 0.36   0.38 / 0.37
    0.12   0.248   0.99    0.22 / 0.21   0.22

  * The naive estimate cannot see the leak: it stays at 0.97-1.00 bits
    while the private randomness falls to a quarter of a bit.
  * The qg estimate is safe and captures 82-90% of the truth with a leak,
    73% without one (the pole at r = 1, as in §41).
  * The simulator's readout is essentially perfect (b = 0.9998-1.0000,
    a < 3e-4), so the §31 calibration adds nothing here, unlike the
    10-50% gain of §41 with hardware-like readout errors.
  * The measured coherence r (0.954, 0.866, 0.72) sits 1-1.5% below the
    ideal |cos 2 pi theta| (0.969, 0.876, 0.729): the MS gate noise, which
    the certifier correctly counts as lost privacy.

These are vendor noise models run on IonQ's simulator, not hardware.
"""

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ionq_sim_hubbard_qaoa as I  # noqa: E402
import qrng_qg_certified as Q  # noqa: E402

THETAS = (0.0, 0.04, 0.08, 0.12)
COPIES = 5
SETTINGS = ("output", "z+", "z-", "y+", "y-")
SIGMAS = 3.0


def truth(theta):
    return Q.h_min_certified(abs(math.cos(2 * math.pi * theta)))


def _circuit(theta, setting):
    from qiskit import QuantumCircuit
    from qiskit_ionq import GPI2Gate, GPIGate, MSGate

    qc = QuantumCircuit(2 * COPIES, COPIES)
    for c in range(COPIES):
        q, e = c, COPIES + c
        if theta > 0:
            qc.append(MSGate(0, 0, theta), [q, e])
        if setting == "output":
            qc.append(GPI2Gate(0.75), [q])
        elif setting == "z-":
            qc.append(GPIGate(0), [q])
        elif setting == "y+":
            qc.append(GPI2Gate(0), [q])
        elif setting == "y-":
            qc.append(GPI2Gate(0.5), [q])
        qc.measure(q, c)
    return qc


def _calibration():
    from qiskit import QuantumCircuit
    from qiskit_ionq import GPIGate

    out = []
    for flip in (False, True):
        qc = QuantumCircuit(COPIES, COPIES)
        for c in range(COPIES):
            if flip:
                qc.append(GPIGate(0), [c])
            qc.measure(c, c)
        out.append(qc)
    return out


def circuits():
    return [_circuit(t, s) for t in THETAS for s in SETTINGS] + _calibration()


def qg_per_copy(counts):
    """<Z> of each classical bit (qg of each copy) and the shot count."""
    n = sum(counts.values())
    z = np.zeros(COPIES)
    for key, c in counts.items():
        k = key.replace(" ", "")
        v = int(k, 16) if k.startswith("0x") else int(k, 2)
        for i in range(COPIES):
            z[i] += c * (1 - 2 * ((v >> i) & 1))
    return z / n, n


def analyse(counts):
    """counts: list in the order of circuits(). Returns per-theta rows."""
    cal0, _ = qg_per_copy(counts[-2])
    cal1, _ = qg_per_copy(counts[-1])
    a = float(np.mean(cal0 + cal1) / 2)
    b = float(np.mean(cal0 - cal1) / 2)
    rows = {}
    for i, t in enumerate(THETAS):
        m = {}
        for j, s in enumerate(SETTINGS):
            z, n = qg_per_copy(counts[i * len(SETTINGS) + j])
            m[s] = float(z.mean())
            shots = n * COPIES
        p0 = (1 + m["output"]) / 2
        n_axis = 2 * shots  # the +/- estimators use both halves
        se_pm = 1 / math.sqrt(n_axis)
        se_one = 1 / math.sqrt(shots)
        z_pm, y_pm = (m["z+"] - m["z-"]) / 2, (m["y+"] - m["y-"]) / 2

        def lcb(x, y, se, scale=1.0):
            return max(0.0, math.hypot(x, y) / scale - SIGMAS * se / scale)

        rows[str(t)] = {
            "theta": t,
            "truth": truth(t),
            "qg": m,
            "naive": Q.h_min_naive(p0),
            "qg one-sided": Q.h_min_certified(lcb(m["z+"], m["y+"], se_one)),
            "qg +/- pairs": Q.h_min_certified(lcb(z_pm, y_pm, se_pm)),
            "qg calibrated": Q.h_min_certified(lcb(z_pm, y_pm, se_pm, b)),
            "r_calibrated": math.hypot(z_pm, y_pm) / b,
        }
    return {"readout_a": a, "readout_b": b, "rows": rows}


def run(noise, mode, key=None):
    circs = circuits()
    if mode == "local":
        return analyse(I._run(None, circs, None, "local"))
    import ionq_sim_zne_grover as G

    return analyse(I._run(G.native_backend(), circs, noise, "ionq_sim", key, prebuilt=True))


def summarize(entries):
    """Mean over repetitions of each estimator, per theta."""
    out = {}
    for t in THETAS:
        rs = [e["rows"][str(t)] for e in entries]
        out[t] = {"truth": truth(t)}
        for k in Q.ESTIMATORS + ("r_calibrated",):
            out[t][k] = float(np.mean([r[k] for r in rs]))
        out[t]["unsafe"] = {k: float(np.mean([r[k] > r["truth"] + 1e-9 for r in rs])) for k in Q.ESTIMATORS}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("local", "ionq_sim"), default="local")
    ap.add_argument("--noise", choices=("aria-1", "forte-1"), default="aria-1")
    ap.add_argument("--tag", default="run0")
    args = ap.parse_args(argv)
    key = f"{args.mode}|{args.noise if args.mode == 'ionq_sim' else 'generic'}|qrng|{args.tag}"
    res = I.load_results()
    if key in res:
        print("already done:", key)
        return res[key]
    try:
        out = run(args.noise, args.mode, key)
    except I.Pending as exc:
        print("pending:", key, "-", exc)
        return None
    if args.mode == "ionq_sim":
        res[key] = out
        I.save_results(res)
        print("saved", key)
    print(f"readout a = {out['readout_a']:+.4f}, b = {out['readout_b']:.4f}")
    print(f"{'theta':>6} {'truth':>6} | " + " ".join(f"{e:>14s}" for e in Q.ESTIMATORS))
    for r in out["rows"].values():
        print(f"{r['theta']:6.2f} {r['truth']:6.3f} | " + " ".join(f"{r[e]:14.3f}" for e in Q.ESTIMATORS))
    return out


if __name__ == "__main__":
    main()
