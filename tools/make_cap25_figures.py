"""Figures of chapter 25 of the book (quantum PCA, molecular chemistry and overlaps read in qg), Spanish and English, on
the chapter 1 base.

    python tools/make_cap25_figures.py

writes book/es/figuras/cap25_*.png and book/en/figures/cap25_*.png and prints the numbers the chapter quotes. Same
style as make_cap01_figures.py. Recomputed here: the §146 PCA circuit putting eigenvalue k on qubit k (exact state
vectors), the H2 Hamiltonian's symmetries N, N_up and N_down and its ansatz reaching FCI (§21, §28), the §149
inversion test and swap test (exact), and the §30 constraint as a register reading with the XY mixer staying feasible.
The device-model results (§146, §149, §153) are read from the stored JSON of the notes' pre-registered runs, and the
§52-§69 ladder of checks on larger molecules is quoted from the notes. Runs in under a minute.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "examples"))
from make_cap01_figures import BLUE, GREEN, GREY, INK, ORANGE  # noqa: E402
from qang import ibm  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

import chemistry_qg_symmetry_witness as H2  # noqa: E402
import chemistry_spin_resolved_qg_filter as SP  # noqa: E402
import ibm_pca_spectrum_qg as PCA  # noqa: E402
import qaoa_k_constraint_qg as QA  # noqa: E402

DATA = os.path.join(ROOT, "examples", "data")
RED = "#8c2d04"

# §69 (H2O, H4) and §55: error against the noiseless circuit (mHa) after each projection, quoted from the notes.
LADDER = {"H2O": (363.9, 54.6, 43.3, 26.7), "H4": (287.5, 79.9, 70.5, 44.7)}

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "p_x": "componente principal (qubit)",
        "p_y": r"autovalor $\lambda_k/\mathrm{tr}\,C$",
        "p_exact": "exacto",
        "p_raw": "crudo (mediana de 21)",
        "p_f": "filtro + lectura (mediana de 21)",
        "p_title": "PCA en el sector de peso 1 (notas §146)",
        "o_y": "error medio de la fidelidad",
        "o_inv": "inversión (5 qubits)",
        "o_swap": "swap test (11 qubits)",
        "o_raw": "crudo",
        "o_f": "filtro + lectura",
        "o_title": "Solapamientos en 6 dispositivos simulados (notas §149)",
        "l_y": "error contra el circuito sin ruido (mHa)",
        "l_steps": ("crudo", r"$N$", r"$N_\uparrow,N_\downarrow$", r"$+\,S^2=0$"),
        "l_title": "Escalera de verificaciones (notas §55, §69)",
        "a_y": "error de la energía (mHa)",
        "a_kinds": ("crudo", r"$N_\uparrow,N_\downarrow$", r"$\Lambda$", "ambos"),
        "a_title": r"LiH con $\Lambda$ (notas §153)",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "p_x": "principal component (qubit)",
        "p_y": r"eigenvalue $\lambda_k/\mathrm{tr}\,C$",
        "p_exact": "exact",
        "p_raw": "raw (median of 21)",
        "p_f": "filter + readout (median of 21)",
        "p_title": "PCA in the weight-1 sector (notes §146)",
        "o_y": "mean fidelity error",
        "o_inv": "inversion (5 qubits)",
        "o_swap": "swap test (11 qubits)",
        "o_raw": "raw",
        "o_f": "filter + readout",
        "o_title": "Overlaps on 6 simulated devices (notes §149)",
        "l_y": "error against the noiseless circuit (mHa)",
        "l_steps": ("raw", r"$N$", r"$N_\uparrow,N_\downarrow$", r"$+\,S^2=0$"),
        "l_title": "Ladder of checks (notes §55, §69)",
        "a_y": "energy error (mHa)",
        "a_kinds": ("raw", r"$N_\uparrow,N_\downarrow$", r"$\Lambda$", "both"),
        "a_title": r"LiH with $\Lambda$ (notes §153)",
    },
}


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return json.load(fh)


def pca_exact():
    """Mixture of the loaded data vectors, rotated by the PCA circuit: P(qubit k excited) against lambda_k / tr C."""
    X = PCA.data()
    O, lam = ibm.pca_rotation(X, center=False)
    w = (X ** 2).sum(axis=1)
    w = w / w.sum()
    rot = ibm.unary_orthogonal(O)
    p = np.zeros(len(lam))
    for x, wj in zip(X, w):
        qc = ibm.unary_loader(x)
        qc.compose(rot, inplace=True)
        sv = Statevector(qc).data
        p += wj * np.array([abs(sv[1 << k]) ** 2 for k in range(len(lam))])
    exact = lam / lam.sum()
    gates = sum(1 for _ in rot.data)
    return p, exact, float(np.abs(p - exact).max()), gates


def h2_checks():
    H = H2.H2_JW.to_matrix()
    idx = np.arange(2 ** H2.N_QUBITS)
    N = np.diag(H2._WEIGHT.astype(float))
    nup = np.diag(sum((idx >> q) & 1 for q in SP.UP).astype(float))
    ndn = np.diag(sum((idx >> q) & 1 for q in SP.DOWN).astype(float))
    comm = {k: float(np.abs(H @ M - M @ H).max()) for k, M in (("N", N), ("N_up", nup), ("N_down", ndn))}
    t = H2.optimal_angle()
    e = H2.exact_energy(t)
    p = np.abs(Statevector(H2.ansatz(t)).data) ** 2
    zbar = float(p @ (1 - 2 * H2._WEIGHT / H2.N_QUBITS))
    xy_terms = len(H2.XY_TERMS)
    return comm, (e - H2.FCI_ENERGY) * 1000, (H2.HF_ENERGY - H2.FCI_ENERGY) * 1000, zbar, xy_terms


def overlaps(pairs=50, n=5, seed=0):
    rng = np.random.default_rng(seed)
    inv, swp = 0.0, 0.0
    for _ in range(pairs):
        x, y = rng.normal(size=n), rng.normal(size=n)
        x, y = x / np.linalg.norm(x), y / np.linalg.norm(y)
        F = float(x @ y) ** 2
        sv = Statevector(ibm.unary_overlap_circuit(x, y, measure=False)).data
        inv = max(inv, abs(abs(sv[1]) ** 2 - F))
        sw = np.abs(Statevector(ibm.swap_test_circuit(x, y, measure=False)).data) ** 2
        p_anc0 = float(sw[(np.arange(sw.size) & 1) == 0].sum())
        swp = max(swp, abs(p_anc0 - (1 + F) / 2))
    qc_inv = ibm.unary_overlap_circuit(np.ones(n), np.ones(n), measure=False)
    qc_sw = ibm.swap_test_circuit(np.ones(n), np.ones(n), measure=False)
    return inv, swp, qc_inv.num_qubits, qc_sw.num_qubits


def qaoa_feasibility(graphs=5, seed=0):
    rng = np.random.default_rng(seed)
    out = {"xy": [], "standard": []}
    for g in range(graphs):
        edges = QA.random_graph(g)
        for v in out:
            params = rng.uniform(-np.pi, np.pi, 2)
            p = QA.ideal_distribution(edges, v, params)
            out[v].append(float(p[QA.FEASIBLE].sum()))
    return out


def fig_pca(t, pca, swap):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    k = np.arange(1, 6)
    raw = np.median([r["estimate"]["raw"] for r in pca["rows"]], axis=0)
    fil = np.median([r["estimate"]["filtered_readout"] for r in pca["rows"]], axis=0)
    ax1.bar(k - 0.27, pca["exact"], 0.26, color=INK, label=t["p_exact"])
    ax1.bar(k, raw, 0.26, color=ORANGE, label=t["p_raw"])
    ax1.bar(k + 0.27, fil, 0.26, color=GREEN, label=t["p_f"])
    ax1.set_xticks(k)
    ax1.set_xlabel(t["p_x"])
    ax1.set_ylabel(t["p_y"])
    ax1.set_title(t["p_title"], fontsize=10)
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(axis="y", alpha=0.3)
    rows = swap["rows"]
    for j, (kind, c) in enumerate((("inversion", BLUE), ("swap", ORANGE))):
        raw = [r[kind]["error"]["raw"] for r in rows]
        fil = [r[kind]["error"]["filtered_readout"] for r in rows]
        for a, b in zip(raw, fil):
            ax2.plot([2 * j, 2 * j + 1], [a, b], color=GREY, lw=0.7, alpha=0.7)
        ax2.plot(np.full(len(raw), 2 * j), raw, "o", color=c, ms=6)
        ax2.plot(np.full(len(fil), 2 * j + 1), fil, "s", color=c, ms=6, mfc="white")
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels([f"{t['o_inv']}\n{t['o_raw']}", f"{t['o_inv']}\n{t['o_f']}",
                         f"{t['o_swap']}\n{t['o_raw']}", f"{t['o_swap']}\n{t['o_f']}"], fontsize=7.5)
    ax2.set_ylabel(t["o_y"])
    ax2.set_ylim(0, None)
    ax2.set_title(t["o_title"], fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_checks(t, lam):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(4)
    for (mol, v), c, m in zip(LADDER.items(), (BLUE, ORANGE), ("o", "s")):
        ax1.semilogy(x, v, m + "-", color=c, label={"H2O": r"H$_2$O", "H4": r"H$_4$"}[mol])
    ax1.set_xticks(x)
    ax1.set_xticklabels(t["l_steps"])
    ax1.set_ylabel(t["l_y"])
    ax1.set_title(t["l_title"], fontsize=10)
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(alpha=0.3, which="both")
    rows = lam["rows"]
    names = [r["backend"].replace("Fake", "") for r in rows]
    xb = np.arange(len(rows))
    for j, (key, c) in enumerate(zip(("raw", "NS", "L", "NSL"), (ORANGE, GREEN, GREY, BLUE))):
        ax2.bar(xb + (j - 1.5) * 0.2, [r[key]["total_mHa"] for r in rows], 0.2, color=c, label=t["a_kinds"][j])
    ax2.set_yscale("log")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(names)
    ax2.set_ylabel(t["a_y"])
    ax2.set_title(t["a_title"], fontsize=10)
    ax2.legend(fontsize=8, frameon=False, ncol=2)
    ax2.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def main():
    p, exact, dmax, gates = pca_exact()
    pca = load("ibm_pca_spectrum_qg.json")
    print("§146 PCA: exact spectrum", np.round(exact, 3), f"; the circuit puts lambda_k on qubit k to {dmax:.1e};"
          f" {gates} RBS in the rotation")
    print("  stored device summary:", json.dumps(pca["summary"]))

    comm, err, hf, zbar, nxy = h2_checks()
    print(f"\n§21/§28 H2: [H, N], [H, N_up], [H, N_down] max entries {comm}; ansatz at its optimal angle"
          f" {err:.1e} mHa above FCI (HF {hf:.1f} mHa); ideal register mean qg_Z {zbar:+.1e};"
          f" {nxy} XX/YY-type terms out of reach of a Z filter")

    inv, swp, nq_inv, nq_sw = overlaps()
    swap = load("ibm_swap_test_qg.json")
    print(f"\n§149 inversion test P(e_0) = F to {inv:.1e} ({nq_inv} qubits); swap test P(anc 0) = (1 + F)/2 to {swp:.1e}"
          f" ({nq_sw} qubits)")
    for r in swap["rows"]:
        print(f"  {r['backend']:15s} inversion {r['inversion']['error']['raw']:.3f} -> "
              f"{r['inversion']['error']['filtered_readout']:.3f}   swap {r['swap']['error']['raw']:.3f} -> "
              f"{r['swap']['error']['filtered_readout']:.3f}   identical: inv f+ro "
              f"{r['inversion']['identical_mean']['filtered_readout']:.2f}, swap raw {r['swap']['identical_mean']['raw']:.2f}")

    feas = qaoa_feasibility()
    print(f"\n§30 exactly K = {QA.K_CHOSEN} of n = {QA.N_NODES}: register mean qg_Z = {QA.QG_BUDGET};"
          f" ideal feasible mass, random angles p = 1: XY mixer {min(feas['xy']):.6f}..{max(feas['xy']):.6f},"
          f" penalty {min(feas['standard']):.3f}..{max(feas['standard']):.3f}")

    lam = load("lih_lambda_filter_qg.json")
    print("\n§153 stored LiH Lambda rows (total mHa raw / NS / L / NSL):")
    for r in lam["rows"]:
        print(f"  {r['backend']:13s} " + " / ".join(f"{r[k]['total_mHa']:.1f}" for k in ("raw", "NS", "L", "NSL")))

    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        names = {"es": ("cap25_pca", "cap25_verificaciones"), "en": ("cap25_pca", "cap25_checks")}[lang]
        for name, fig in zip(names, (fig_pca(tx, pca, swap), fig_checks(tx, lam))):
            path = os.path.join(tx["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
