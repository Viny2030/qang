"""Figures of chapter 30 of the book (structural limits, equivalences and negative results), Spanish and English.

    python tools/make_cap30_figures.py

writes book/es/figuras/cap30_*.png and book/en/figures/cap30_*.png and prints the numbers the chapter quotes.

* Graph states (§8.1): the ring graph state on 6 qubits and |+>^6 have the same Z-basis distribution (exactly), and
  only the stabilizers X_i Z_{i-1} Z_{i+1}, read in another basis, tell them apart.
* qg_S under T1 (§10.4): 24 quantum-volume circuits of 4 qubits, amplitude damping on every qubit after every
  two-qubit block (gamma per block from 0 to 0.4), exact density matrices. qg_S = Shannon entropy of the Z
  distribution / n; mean qg_Z = average of the qubits' qg_Z. Depolarizing noise for comparison.
* The weight-1 network is a quadratic form (§76 F2): unary loader + orthogonal RBS circuit, qg_Z_i = 1 - 2 (O v)_i^2.
* Where the filter applies (§13): qang's transpiler pass (qang.ibm.WeightPreservationCheck) on QFT, QPE, GHZ, QV,
  the three-flavour neutrino circuit and a unary RBS network.
* The pre-registered runs of §157-§164: passed and failed predictions read from their stored verdicts.
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
RUNS = [("qpon_teleport_qg", "§157"), ("qpon_teleport_dual_rail_qg", "§158"), ("fisher_natural_gradient_qg", "§159"),
        ("pulse_area_qg", "§160"), ("quantum_kernel_qg", "§161"), ("varqte_kraus_qg", "§162"),
        ("neutrino_ordering_majorana_sterile_qg", "§163"), ("postselection_ctc_blackhole_qg", "§164")]

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "t_x": r"$\gamma$ por bloque de dos qubits",
        "t_s": r"$qg_S$ (entropía de Z / n)",
        "t_z": r"$qg_Z$ medio del registro",
        "t_t1": "T1 (amortiguamiento)",
        "t_dep": "despolarizante",
        "t_title_s": r"$qg_S$ sube y baja con T1",
        "t_title_z": r"el $qg_Z$ medio sólo sube",
        "t_band": "24 circuitos",
        "f_x": "predicciones",
        "f_pass": "cumplidas",
        "f_fail": "falladas",
        "f_title": "Corridas pre-registradas de §157–§164",
        "f_runs": ["teleportación QPON", "QPON doble riel", "gradiente natural", "área de pulso",
                   "kernels", "VarQTE", "neutrinos: orden y estéril", "P-CTC y agujero negro"],
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "t_x": r"$\gamma$ per two-qubit block",
        "t_s": r"$qg_S$ (Z entropy / n)",
        "t_z": r"register mean $qg_Z$",
        "t_t1": "T1 (amplitude damping)",
        "t_dep": "depolarizing",
        "t_title_s": r"$qg_S$ rises and falls under T1",
        "t_title_z": r"mean $qg_Z$ only rises",
        "t_band": "24 circuits",
        "f_x": "predictions",
        "f_pass": "passed",
        "f_fail": "failed",
        "f_title": "Pre-registered runs of §157–§164",
        "f_runs": ["QPON teleportation", "QPON dual rail", "natural gradient", "pulse area",
                   "kernels", "VarQTE", "neutrinos: ordering, sterile", "P-CTC and black hole"],
    },
}


def z_label(n, q):
    return "".join("Z" if k == q else "I" for k in reversed(range(n)))


def graph_states():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector

    from qang import circuits as C

    n = 6
    g = Statevector(C.graph_state_circuit(n, [(i, (i + 1) % n) for i in range(n)]))
    p = QuantumCircuit(n)
    p.h(range(n))
    p = Statevector(p)

    def stab(i):
        s = ["I"] * n
        s[i], s[(i - 1) % n], s[(i + 1) % n] = "X", "Z", "Z"
        return SparsePauliOp("".join(reversed(s)))

    return (float(np.abs(g.probabilities() - p.probabilities()).max()),
            [float(g.expectation_value(stab(i)).real) for i in range(n)],
            [float(p.expectation_value(stab(i)).real) for i in range(n)])


def _kraus(kind, x):
    from qiskit.quantum_info import Kraus

    if kind == "t1":
        return Kraus([np.array([[1, 0], [0, np.sqrt(1 - x)]]), np.array([[0, np.sqrt(x)], [0, 0]])])
    I, X, Y, Z = np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])
    return Kraus([np.sqrt(1 - 3 * x / 4) * I] + [np.sqrt(x / 4) * P for P in (X, Y, Z)])


def qv_point(seed, kind, x, n=4):
    from qiskit.circuit.library import quantum_volume
    from qiskit.quantum_info import DensityMatrix, SparsePauliOp

    qv = quantum_volume(n, seed=seed)
    chan = _kraus(kind, x)
    rho = DensityMatrix.from_label("0" * n)
    for inst in qv.data:
        rho = rho.evolve(inst.operation, [qv.find_bit(q).index for q in inst.qubits])
        for q in range(n):
            rho = rho.evolve(chan, [q])
    pr = rho.probabilities()
    pr = pr[pr > 1e-15]
    qg_s = float(-(pr * np.log2(pr)).sum() / n)
    mean_z = float(np.mean([rho.expectation_value(SparsePauliOp(z_label(n, q))).real for q in range(n)]))
    return qg_s, mean_z


def t1_sweep(seeds=24):
    gam = np.linspace(0, 0.4, 21)
    S = np.zeros((seeds, gam.size))
    Z = np.zeros_like(S)
    for s in range(seeds):
        for j, x in enumerate(gam):
            S[s, j], Z[s, j] = qv_point(s, "t1", x)
    dep = np.array([qv_point(0, "dep", x) for x in gam])
    return gam, S, Z, dep


def quadratic_check(trials=20):
    from qiskit.quantum_info import SparsePauliOp, Statevector

    from qang import ibm

    rng = np.random.default_rng(76)
    worst = 0.0
    for _ in range(trials):
        v = rng.normal(size=5)
        v /= np.linalg.norm(v)
        O, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        sv = Statevector(ibm.unary_loader(v).compose(ibm.unary_orthogonal(O)))
        z = np.array([sv.expectation_value(SparsePauliOp(z_label(5, q))).real for q in range(5)])
        worst = max(worst, float(np.abs(z - (1 - 2 * (O @ v) ** 2)).max()))
    return worst


def filter_reach():
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFTGate, TGate, phase_estimation, quantum_volume
    from qiskit.transpiler import PassManager

    from qang import circuits as C
    from qang import ibm
    from qang import neutrino as nu

    qft = QuantumCircuit(4)
    qft.append(QFTGate(4), range(4))
    t = QuantumCircuit(1)
    t.append(TGate(), [0])
    O, _ = np.linalg.qr(np.random.default_rng(30).normal(size=(5, 5)))
    cases = {
        "QFT, 4 qubits": qft.decompose(),
        "QPE of T, 3 counting qubits": phase_estimation(3, t).decompose(),
        "GHZ, 4 qubits": C.ghz_circuit(4),
        "quantum volume, 4 qubits": quantum_volume(4, seed=0),
        "three-flavour neutrinos (§151)": nu.three_flavour_circuit("mu", 1300.0, 2.6, -np.pi / 2),
        "unary loader + RBS network": ibm.unary_loader([1, 2, 3, 4, 1]).compose(ibm.unary_orthogonal(O)),
    }
    out = {}
    for k, qc in cases.items():
        pm = PassManager([ibm.WeightPreservationCheck()])
        pm.run(qc)
        out[k] = (pm.property_set["qg_weight_preserving"], len(pm.property_set["qg_non_preserving_gates"]))
    return out


def stored_runs():
    rows = []
    for name, sec in RUNS:
        with open(os.path.join(DATA, name + ".json"), encoding="utf-8") as fh:
            v = json.load(fh)["verdict"]
        b = {k: x for k, x in v.items() if isinstance(x, bool)}
        rows.append((sec, sum(b.values()), len(b) - sum(b.values()), [k for k, x in b.items() if not x]))
    return rows


def fig_t1(t, gam, S, Z, dep):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, M, col, d in ((a, S, BLUE, dep[:, 0]), (b, Z, ORANGE, dep[:, 1])):
        ax.fill_between(gam, M.min(0), M.max(0), color=col, alpha=0.18, lw=0, label=t["t_band"])
        ax.plot(gam, M[0], color=col, lw=2.2, label=t["t_t1"])
        ax.plot(gam, d, color=GREY, lw=1.6, ls="--", label=t["t_dep"])
        ax.set_xlabel(t["t_x"], color=INK)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8.5)
    a.set_ylabel(t["t_s"], color=INK)
    b.set_ylabel(t["t_z"], color=INK)
    a.set_title(t["t_title_s"], color=INK, fontsize=11)
    b.set_title(t["t_title_z"], color=INK, fontsize=11)
    fig.tight_layout()
    return fig


def fig_runs(t, rows):
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    y = np.arange(len(rows))[::-1]
    ok = np.array([r[1] for r in rows])
    bad = np.array([r[2] for r in rows])
    ax.barh(y, ok, color=GREEN, label=t["f_pass"])
    ax.barh(y, bad, left=ok, color=RED, label=t["f_fail"])
    for yi, r in zip(y, rows):
        if r[3]:
            ax.text(r[1] + r[2] + 0.15, yi, ", ".join(r[3]), va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [f"{lab} ({r[0]})" for lab, r in zip(t["f_runs"], rows)], fontsize=9)
    ax.set_xlabel(t["f_x"], color=INK)
    ax.set_xlim(0, 10.5)
    ax.set_title(t["f_title"], color=INK, fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def main():
    diff, sg, sp = graph_states()
    print(f"graph state (ring, 6) vs |+>^6: max |Delta p| in Z = {diff:.1e}")
    print("  stabilizers X_i Z Z, graph:", np.round(sg, 6), " |+>^6:", np.round(sp, 6))
    gam, S, Z, dep = t1_sweep()
    nonmono = int(np.sum(np.any(np.diff(S, axis=1) < -1e-9, axis=1)))
    mono = int(np.sum(np.all(np.diff(Z, axis=1) > -1e-9, axis=1)))
    print(f"T1 sweep, 24 QV circuits: qg_S non-monotone in {nonmono}, mean qg_Z monotone in {mono}")
    j = int(np.argmax(S[0]))
    k = int(np.argmin(np.abs(S[0, j:] - S[0, 0]))) + j
    print(f"  circuit 0: qg_S {S[0, 0]:.3f} at gamma 0, peak {S[0, j]:.3f} at {gam[j]:.4f}, back to {S[0, k]:.3f} "
          f"at {gam[k]:.4f}; mean qg_Z there {Z[0, 0]:.3f} vs {Z[0, k]:.3f}")
    print(f"  depolarizing, circuit 0: qg_S {dep[0, 0]:.3f} -> {dep[-1, 0]:.3f}, mean qg_Z {dep[0, 1]:.3f} -> "
          f"{dep[-1, 1]:.3f}")
    print(f"weight-1 network as a quadratic form: max error {quadratic_check():.1e}")
    print("qang's weight-preservation pass (preserving, gates that break it):")
    for k2, v in filter_reach().items():
        print(f"  {k2:34s} {v}")
    rows = stored_runs()
    print("stored verdicts §157-§164:", rows, " total", sum(r[1] for r in rows), "of", sum(r[1] + r[2] for r in rows))
    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        names = {"es": ("cap30_t1", "cap30_fallos"), "en": ("cap30_t1", "cap30_failures")}[lang]
        for name, fig in zip(names, (fig_t1(tx, gam, S, Z, dep), fig_runs(tx, rows))):
            path = os.path.join(tx["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
