"""
Coherent MS over-rotation and drift: do the qg filter and native-MS ZNE
conclusions survive the errors IonQ's noise model does not contain?

Why. IonQ's cloud noise models are gate-level stochastic (depolarizing)
with near-ideal readout (§42). Real trapped-ion gates also have coherent
errors: the Molmer-Sorensen angle is over- or under-rotated by a small
fraction eps, and eps drifts between calibrations. Two predictions follow
before running anything:

  * native-MS folding cannot amplify a coherent angle error. The inverse
    MS(phi0 + 1/2, phi1, theta') is the exact inverse of MS(phi0, phi1,
    theta') for the SAME over-rotated theta', so MS MS^-1 MS = MS: the
    folds amplify only the stochastic part, and ZNE extrapolates to the
    coherent-only error, not to zero;
  * the H2 ansatz and the XY-QAOA mixer conserve particle number only if
    every CX (= MS(1/4) plus single-qubit gates) is exact, so an
    over-rotation leaks weight out of the sector and the qg filter
    removes that part, but not the in-sector part.

Model (exact density-matrix simulation of the native circuits that
examples/ionq_hardware_plan.py would send: GPI, GPI2 and MS after
Qiskit's IonQ native transpilation):

  coherent   every MS angle theta -> theta (1 + eps)
  drift      eps drawn per circuit (per job) as eps_bar + sigma N(0, 1),
             frozen within the circuit; 200 realisations
  stochastic 2-qubit depolarizing p2 = 0.005 after each MS, 1-qubit
             p1 = 3e-4 after each GPI/GPI2 (Forte-like)
  readout    symmetric 0.5 % per qubit, corrected with the two
             calibration circuits (as on hardware)

Energies are exact expectations (infinite shots) unless stated.

Findings (python examples/coherent_drift_filter_zne.py):

  Folding check: with any eps, MS MS^-1 MS equals MS to 1e-16, so the
  folds see only the stochastic noise (first prediction, exact).

  H2 at equilibrium, error vs FCI in mHa (HF = 20.3), infinite shots:

      eps   raw     qg filter  ZNE    ZNE+filter | coherent error alone:
                                                 |   raw     filtered
      0     13.54   5.59       3.21   1.62       |   0.00    0.00
      1 %   13.66   5.60       3.33   1.63       |   0.12    0.01
      2 %   14.01   5.63       3.68   1.65       |   0.48    0.03
      5 %   16.47   5.80       6.17   1.81       |   2.97    0.19

  * ZNE passes the coherent error through untouched: at eps = 5 % its
    result moves by +2.96 mHa, the coherent-only error (2.97). The folds
    grow only with the stochastic part (13.5 -> 33.8 -> 53.7 mHa at eps =
    0; 16.5 -> 36.7 -> 56.5 at 5 %: the same slope, shifted).
  * The qg filter removes 94 % of the coherent error (2.97 -> 0.19 mHa at
    5 %): an over-rotated CX leaks weight out of the two-electron sector
    to first order, while the in-sector error is second order because the
    ansatz sits at its energy minimum.
  * ZNE + filter is the most robust combination: 1.6-1.8 mHa over the
    whole range.

  Drift (eps drawn per circuit and per fold, 100 realisations), mean /
  RMS in mHa:

                          raw            filter        ZNE           ZNE+filter
    eps ~ 0 + 2 % N       14.0 / 14.0    5.6 / 5.6     3.6 / 3.9     1.65 / 1.65
    eps ~ 2 % + 2 % N     14.4 / 14.4    5.7 / 5.7     4.1 / 4.8     1.69 / 1.70
    same, 5000 shots      14.9 / 15.4    6.0 / 6.6     5.5 / 10.3    2.3 / 7.2

  * Drift barely moves the filter; it widens ZNE (RMS 3.2 -> 4.8 mHa), and
    with 5000 shots per circuit ZNE's RMS (10.3) is again dominated by
    extrapolated shot noise, as in §39. The filter stays at 6.6.

  XY-QAOA, graph 0 (8 qubits, 150 MS), P(opt) readout -> filter (kept):

      eps 0     0.125 -> 0.238 (0.52)
      eps 2 %   0.115 -> 0.237 (0.49)
      eps 5 %   0.075 -> 0.197 (0.38)

  * The filter's doubling of XY-QAOA survives, and grows to 2.6x at 5 %:
    the over-rotation breaks the XY mixer's number conservation and the
    filter removes most of what it breaks.

For the hardware plan (§46): coherent MS errors and drift do not threaten
the decisive prediction (A1). They do bias ZNE (track B) by the coherent
error, which is why B is compared with the filter and not read alone;
IonQ's debiasing (A3), which randomises the gate frames, is expected to
turn part of the coherent error into stochastic error that ZNE can
extrapolate.

Honest scope. A model, not a device: one coherent error type (MS angle),
Gaussian drift, stochastic noise lower than IonQ's forte-1 model (raw
13.5 vs 34.8 mHa), so the absolute numbers are not predictions; the
qualitative results (ZNE blind to coherent angle errors, filter removing
the number-breaking part) follow from the structure and are the claim.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import chemistry_qg_symmetry_witness as CH  # noqa: E402

P1, P2, READOUT = 3e-4, 0.005, 0.005
SCALES = (1, 3, 5)
RICHARDSON = np.array([15, -10, 3]) / 8  # scales 1, 3, 5
EPS_VALUES = (0.0, 0.01, 0.02, 0.05)


# --------------------------------------------------------------------- #
# native circuits and a small exact density-matrix simulator
# --------------------------------------------------------------------- #
def native_ops(qc):
    """List of (name, qubits, params) after IonQ native transpilation, on the
    circuit's own qubits (the transpiler pads the device, layout is trivial)."""
    from qiskit import transpile
    from qiskit_ionq import IonQProvider

    nb = IonQProvider(token="none").get_backend("simulator", gateset="native")
    c = qc.remove_final_measurements(inplace=False)
    tc = transpile(c, backend=nb, optimization_level=1, seed_transpiler=1)
    layout = tc.layout.final_index_layout()
    inv = {p: v for v, p in enumerate(layout[: c.num_qubits])}
    ops = []
    for ins in tc.data:
        qs = [inv[tc.find_bit(q).index] for q in ins.qubits]
        ops.append((ins.operation.name, qs, [float(p) for p in ins.operation.params]))
    return ops, c.num_qubits


def fold(ops, scale):
    out = []
    for name, qs, pr in ops:
        out.append((name, qs, pr))
        if name == "ms":
            for _ in range((scale - 1) // 2):
                out.append(("ms", qs, [pr[0] + 0.5, pr[1], pr[2]]))
                out.append(("ms", qs, pr))
    return out


_CH1 = {}
_CH2 = {}


def _dep(p, k):
    from qiskit_aer.noise import depolarizing_error

    cache = _CH1 if k == 1 else _CH2
    if p not in cache:
        cache[p] = depolarizing_error(p, k).to_quantumchannel()
    return cache[p]


def run(ops, n, eps=0.0, p1=P1, p2=P2):
    """Outcome probabilities (little-endian index) of the native circuit with
    every MS angle scaled by (1 + eps) and depolarizing noise after each gate."""
    from qiskit.quantum_info import DensityMatrix, Operator
    from qiskit_ionq import GPI2Gate, GPIGate, MSGate

    rho = DensityMatrix.from_label("0" * n)
    for name, qs, pr in ops:
        if name == "ms":
            rho = rho.evolve(Operator(MSGate(pr[0], pr[1], pr[2] * (1 + eps))), qs)
            if p2:
                rho = rho.evolve(_dep(p2, 2), qs)
        else:
            g = GPIGate(pr[0]) if name == "gpi" else GPI2Gate(pr[0])
            rho = rho.evolve(Operator(g), qs)
            if p1:
                rho = rho.evolve(_dep(p1, 1), qs)
    return np.clip(np.real(rho.probabilities()), 0, None)


def readout(p, n, e=READOUT):
    """Symmetric flip e on every qubit (little-endian index)."""
    t = p.reshape([2] * n)
    M = np.array([[1 - e, e], [e, 1 - e]])
    for ax in range(n):
        t = np.moveaxis(np.tensordot(M, t, axes=(1, ax)), 0, ax)
    return t.reshape(-1)


# --------------------------------------------------------------------- #
# H2 (tracks A1 and B)
# --------------------------------------------------------------------- #
_H2 = None


def h2_native():
    global _H2
    if _H2 is None:
        t = CH.optimal_angle()
        labels = ["ZZZZ"] + list(CH.XY_TERMS)
        _H2 = [(l, native_ops(CH._measure_circuit(t, l, 0))[0]) for l in labels]
    return _H2


def h2_energies(eps_per_circuit, scale=1, shots=None, rng=None):
    """Readout-corrected raw and qg-filtered energy errors (mHa); eps_per_circuit
    gives one eps for each measurement circuit (drift between jobs)."""
    n = CH.N_QUBITS
    inv = CH._readout_inverse(readout(np.eye(2**n)[0], n), readout(np.eye(2**n)[-1], n))
    P = {}
    for (label, ops), eps in zip(h2_native(), eps_per_circuit):
        p = readout(run(fold(ops, scale), n, eps), n)
        if shots:
            p = rng.multinomial(shots, p / p.sum()) / shots
        P[label] = inv @ p
    p_z = P.pop("ZZZZ")
    raw = CH._energy(p_z, P) - CH.FCI_ENERGY
    filt = CH._energy(CH.qg_filter(np.clip(p_z, 0, None)), P) - CH.FCI_ENERGY
    return 1e3 * raw, 1e3 * filt


def h2_table():
    k = len(h2_native())
    out = {}
    for eps in EPS_VALUES:
        per = [h2_energies([eps] * k, s) for s in SCALES]
        raw = [r for r, _ in per]
        flt = [f for _, f in per]
        out[eps] = {"raw": raw[0], "filter": flt[0], "raw_by_scale": raw,
                    "zne": float(RICHARDSON @ raw), "zne_filter": float(RICHARDSON @ flt),
                    "coherent_only": coherent_only(eps)}
    return out


def coherent_only(eps):
    n = CH.N_QUBITS
    P = {l: run(ops, n, eps, p1=0.0, p2=0.0) for l, ops in h2_native()}
    p_z = P.pop("ZZZZ")
    return 1e3 * (CH._energy(p_z, P) - CH.FCI_ENERGY), 1e3 * (CH._energy(CH.qg_filter(p_z), P) - CH.FCI_ENERGY)


def h2_drift(eps_bar=0.02, sigma=0.02, reps=200, shots=None, seed=0):
    """Drift: eps drawn per circuit and per fold scale. RMS and mean of each method."""
    rng = np.random.default_rng(seed)
    k = len(h2_native())
    rows = []
    for _ in range(reps):
        per = [h2_energies(eps_bar + sigma * rng.standard_normal(k), s, shots, rng) for s in SCALES]
        raw = np.array([r for r, _ in per])
        flt = np.array([f for _, f in per])
        rows.append((raw[0], flt[0], RICHARDSON @ raw, RICHARDSON @ flt))
    a = np.array(rows)
    names = ("raw", "filter", "zne", "zne_filter")
    return {nm: {"mean": float(a[:, i].mean()), "rms": float(np.sqrt((a[:, i] ** 2).mean())),
                 "std": float(a[:, i].std())} for i, nm in enumerate(names)}


# --------------------------------------------------------------------- #
# XY-QAOA (track C)
# --------------------------------------------------------------------- #
def qaoa_popt(graph=0, eps=0.0):
    import json

    import qaoa_k_constraint_qg as QA

    with open(os.path.join(HERE, "data", "qaoa_k_angles.json"), encoding="utf-8") as fh:
        ang = json.load(fh)
    edges = QA.random_graph(graph)
    qc = QA.qaoa_circuit(edges, "xy", np.array(ang[f"p1_g{graph}_xy"]))
    ops, n = native_ops(qc)
    p = readout(run(ops, n, eps), n)
    raw = QA.sample_metrics(p, edges, False)
    flt = QA.sample_metrics(p, edges, True)
    return {"raw": raw["p_opt"], "filter": flt["p_opt"], "kept": raw["kept"]}


def make_figure(path, table):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    e = np.array(EPS_VALUES)
    style = {"raw": ("#8c8c8c", "o", "readout-corrected"), "filter": ("#1f6fb2", "s", "qg filter"),
             "zne": ("#e0a030", "^", "native-MS ZNE"), "zne_filter": ("#8c2d04", "D", "ZNE + filter")}
    for k, (c, m, lab) in style.items():
        ax.plot(100 * e, [table[x][k] for x in EPS_VALUES], marker=m, color=c, label=lab)
    ax.plot(100 * e, [table[x]["coherent_only"][0] for x in EPS_VALUES], "k:", label="coherent error alone")
    ax.axhline(20.3, color="#bbbbbb", lw=1)
    ax.text(0.2, 21.0, "Hartree-Fock", fontsize=7, color="#777777")
    ax.set_xlabel("MS over-rotation eps (%)")
    ax.set_ylabel("H2 energy error vs FCI (mHa)")
    ax.set_ylim(-1, 23)
    ax.set_title("Coherent MS error: ZNE passes it through, the qg filter removes most of it", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    t = h2_table()
    print("H2 at equilibrium (mHa vs FCI), no drift, infinite shots")
    print(f"{'eps':>5} | {'raw':>7} {'filter':>7} {'ZNE':>7} {'ZNE+f':>7} | {'coh. raw':>8} {'coh. filt':>9} | folds 1,3,5 raw")
    for eps, r in t.items():
        print(f"{eps:5.2f} | {r['raw']:7.2f} {r['filter']:7.2f} {r['zne']:7.2f} {r['zne_filter']:7.2f} | "
              f"{r['coherent_only'][0]:8.2f} {r['coherent_only'][1]:9.2f} | " + " ".join(f"{x:6.1f}" for x in r["raw_by_scale"]))
    for eb, sg in ((0.0, 0.02), (0.02, 0.02)):
        d = h2_drift(eb, sg, reps=100)
        print(f"\ndrift eps ~ {eb} + {sg} N(0,1) per circuit (100 realisations, infinite shots): mean / rms (mHa)")
        for k, v in d.items():
            print(f"  {k:11s} {v['mean']:7.2f} / {v['rms']:6.2f}")
    d = h2_drift(0.02, 0.02, reps=100, shots=5000)
    print("\nsame drift with 5000 shots per circuit: mean / rms (mHa)")
    for k, v in d.items():
        print(f"  {k:11s} {v['mean']:7.2f} / {v['rms']:6.2f}")
    print("\nXY-QAOA graph 0 (8 qubits), P(opt) raw -> filter (kept)")
    for eps in EPS_VALUES:
        q = qaoa_popt(0, eps)
        print(f"  eps {eps:4.2f}: {q['raw']:.3f} -> {q['filter']:.3f}  ({q['kept']:.3f})")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"), t)
