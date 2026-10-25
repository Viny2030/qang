"""Figures of chapter 32 of the book (software architecture, the multi-SDK ecosystem and physical execution), Spanish
and English.

    python tools/make_cap32_figures.py

writes book/es/figuras/cap32_*.png and book/en/figures/cap32_*.png and prints the numbers the chapter quotes.

* One qubit order (§130): the same outcome (only qubit 0 excited, 3 qubits) as Qiskit and PennyLane write it, through
  qang.interop.canonical_counts; a 3-qubit W state sampled with Qiskit's StatevectorSampler and with PennyLane's
  default.qubit (20 000 shots each). Cirq, Braket and pytket are not installed here: their bit order is pinned by
  the tests (tests/test_interop.py), not rerun.
* Filtered observables (§131): on random 4-qubit states, EstimatorV2-style values of
  interop.filtered_observables(4, 2) give the kept fraction and the filtered qg_Z exactly; the number of Z strings
  per observable for n = 4..12.
* The IBM runs of §137, §138, §141, §142, §143 and §150, rescored from their stored files (nothing is re-simulated).
"""

import json
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from make_cap01_figures import BLUE, GREEN, GREY, INK, ORANGE  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)

DATA = os.path.join(ROOT, "examples", "data")
RED = "#8c2d04"
METHODS = ["raw", "readout", "raw + zne", "filtered", "filtered_readout", "filtered + zne"]

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "m_y": r"error medio de $qg_Z$",
        "m_title": "Diez circuitos de peso 1 (§142): sin qang a la izquierda, con qang a la derecha",
        "m_labels": {"raw": "crudo", "readout": "lectura", "raw + zne": "ZNE", "filtered": "filtro",
                     "filtered_readout": "filtro + lectura", "filtered + zne": "filtro + ZNE"},
        "z_x": "ZZ estático (kHz)",
        "z_y": r"error medio de $qg_Z$",
        "z_raw": "crudo",
        "z_fil": "filtrado",
        "z_title": "El ZZ siempre encendido (§150)",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "m_y": r"mean $qg_Z$ error",
        "m_title": "Ten weight-1 circuits (§142): without qang on the left, with qang on the right",
        "m_labels": {"raw": "raw", "readout": "readout", "raw + zne": "ZNE", "filtered": "filter",
                     "filtered_readout": "filter + readout", "filtered + zne": "filter + ZNE"},
        "z_x": "static ZZ (kHz)",
        "z_y": r"mean $qg_Z$ error",
        "z_raw": "raw",
        "z_fil": "filtered",
        "z_title": "Always-on ZZ (§150)",
    },
}


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return json.load(fh)


def bit_order():
    import pennylane as qml
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler

    from qang import circuits as C
    from qang import interop as IO

    same = (IO.canonical_counts({"001": 1}, sdk="qiskit"), IO.canonical_counts({"100": 1}, sdk="pennylane"))
    shots = 20000
    w = C.w_circuit(3)
    wq = w.copy()
    wq.measure_all()
    res = StatevectorSampler(seed=130).run([wq], shots=shots).result()[0]
    qk = IO.canonical_counts(res.data.meas)

    dev = qml.device("default.qubit", wires=3, shots=shots, seed=130)

    @qml.qnode(dev)
    def wstate():
        qml.QubitStateVector(_w_vector(), wires=[0, 1, 2]) if hasattr(qml, "QubitStateVector") else \
            qml.StatePrep(_w_vector(), wires=[0, 1, 2])
        return qml.counts()

    pl = IO.canonical_counts({k: int(v) for k, v in wstate().items()}, sdk="pennylane")
    out = {}
    for name, c in (("qiskit", qk), ("pennylane", pl)):
        out[name] = ([round(z, 3) for z, _ in IO.qg_z_profile(c)], IO.weight_filter(c, 1)[1], IO.joint_qg_s(c))
    return same, out


def _w_vector():
    v = np.zeros(8)
    for k in (1, 2, 4):
        v[k] = 1 / np.sqrt(3)
    return v


def filtered_observables_check(trials=20):
    from qiskit.quantum_info import SparsePauliOp, random_statevector

    from qang import interop as IO

    n, w = 4, 2
    obs = [SparsePauliOp.from_list(list(o.items())) for o in IO.filtered_observables(n, w)]
    idx = np.arange(2 ** n)
    weight = np.array([bin(i).count("1") for i in idx])
    worst = 0.0
    for s in range(trials):
        sv = random_statevector(2 ** n, seed=s)
        p = sv.probabilities()
        kept_ex = p[weight == w].sum()
        z_ex = np.array([(p * (weight == w) * (1 - 2 * ((idx >> i) & 1))).sum() for i in range(n)]) / kept_ex
        kept, z = IO.estimator_filtered_qg([sv.expectation_value(o).real for o in obs])
        worst = max(worst, abs(float(kept) - kept_ex), float(np.abs(z - z_ex).max()))
    terms = {m: max(len(o) for o in IO.filtered_observables(m, m // 2)) for m in (4, 6, 8, 10, 12)}
    return worst, terms


def ibm_runs():
    out = {}
    out["runtime"] = {r["backend"]: r["error"] for r in load("ibm_runtime_qg.json")["rows"]}
    lay = load("ibm_layout_qg.json")
    out["layout"] = {r["backend"]: (r["mae"], r["spearman"], r["best_minus_default"]) for r in lay["rows"]}
    out["layout_verdict"] = lay["verdict"]
    q = load("ibm_qnn_layout_qg.json")
    out["qnn_layout"] = {r["backend"]: (r["kept"]["default"], r["kept"]["best"], r["correct"]) for r in q["rows"]}
    out["qnn_layout_verdict"] = q["verdict"]
    z = load("ibm_zne_qg.json")
    out["zne"] = {r["backend"]: r["error"] for r in z["rows"]}
    out["zne_verdict"] = z["verdict"]
    s = load("ibm_shot_plan_qg.json")
    out["shots"] = {r["backend"]: (r["worst_se"], r["mean_ratio"]) for r in s["rows"]}
    out["shots_verdict"] = s["verdict"]
    zz = load("ibm_static_zz_qg.json")
    out["zz"] = {r["backend"]: r["by_nu"] for r in zz["rows"]}
    out["zz_verdict"] = zz["verdict"]
    return out


def fig_methods(t, zne):
    backs = list(zne)
    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    cols = [GREY, "#c6c6c6", ORANGE, BLUE, GREEN, "#1d4f8f"]
    width = 0.13
    for j, (m, c) in enumerate(zip(METHODS, cols)):
        x = np.arange(len(backs)) + (j - 2.5) * width
        ax.bar(x, [zne[b][m] for b in backs], width, color=c, label=t["m_labels"][m])
    ax.set_xticks(np.arange(len(backs)), backs)
    ax.set_ylabel(t["m_y"], color=INK)
    ax.set_title(t["m_title"], color=INK, fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def fig_zz(t, zz):
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    nus = ["0", "5000", "15000"]
    x = [0, 5, 15]
    for b, c in zip(zz, (BLUE, ORANGE, GREEN)):
        ax.plot(x, [zz[b][n]["raw"] for n in nus], "o--", color=c, lw=1.3, label=f"{b}, {t['z_raw']}")
        ax.plot(x, [zz[b][n]["filtered"] for n in nus], "s-", color=c, lw=2, label=f"{b}, {t['z_fil']}")
    ax.set_xlabel(t["z_x"], color=INK)
    ax.set_ylabel(t["z_y"], color=INK)
    ax.set_title(t["z_title"], color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def main():
    same, w = bit_order()
    print("qubit 0 excited, as Qiskit '001' and PennyLane '100':", same)
    for k, (z, kept, s) in w.items():
        print(f"  W state, {k:9s}: qg_Z {z}, weight-1 share {kept:.3f}, register qg_S {s:.3f} "
              f"(ideal 1/3 and {np.log2(3) / 3:.3f})")
    worst, terms = filtered_observables_check()
    print(f"filtered observables vs exact filter, 20 random states: max error {worst:.1e}")
    print("  Z strings in the largest filtered observable, n = 4..12 (weight n/2):", terms)
    r = ibm_runs()
    print("\n§137 errors:")
    for b, e in r["runtime"].items():
        print(f"  {b:15s} " + "  ".join(f"{k} {v:.3f}" for k, v in e.items()))
    print("§138 kept-fraction prediction (MAE, Spearman, best_chain - default):", r["layout_verdict"])
    for b, v in r["layout"].items():
        print(f"  {b:15s} {v[0]:.3f} {v[1]:.2f} {v[2]:+.3f}")
    print("§141 QNN layout (kept default, best; correct of 189):", r["qnn_layout_verdict"])
    for b, v in r["qnn_layout"].items():
        print(f"  {b:15s} {v[0]:.3f} {v[1]:.3f} {v[2]}")
    print("§142 ZNE:", r["zne_verdict"])
    for b, e in r["zne"].items():
        print(f"  {b:15s} " + "  ".join(f"{k} {e[k]:.3f}" for k in METHODS + ["readout + zne", "filtered_readout + zne"]))
    print("§143 shot plan (worst se, mean ratio):", r["shots_verdict"], r["shots"])
    print("§150 static ZZ:", r["zz_verdict"])
    for b, by in r["zz"].items():
        f0, f15 = by["0"]["filtered"], by["15000"]["filtered"]
        removed = 1 - (f15 - f0) / (by["15000"]["raw"] - by["0"]["raw"])
        print(f"  {b:15s} raw {by['0']['raw']:.3f}->{by['15000']['raw']:.3f}  filtered {f0:.3f}->{f15:.3f} "
              f"(+{f15 - f0:.3f}, {f15 / f0 - 1:.0%})  kept {by['0']['kept']:.3f}->{by['15000']['kept']:.3f}  "
              f"filter removed {removed:.0%} of the added error; raw/filtered at 15 kHz {by['15000']['raw'] / f15:.1f}")
    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        names = {"es": ("cap32_metodos", "cap32_zz"), "en": ("cap32_methods", "cap32_zz")}[lang]
        for name, fig in zip(names, (fig_methods(tx, r["zne"]), fig_zz(tx, r["zz"]))):
            path = os.path.join(tx["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
